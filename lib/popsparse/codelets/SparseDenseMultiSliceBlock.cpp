// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#include "SparseMetaInfo.hpp"

#include <cassert>

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

namespace popsparse {

using MetaInfoType = unsigned short;
using BaseTMetaInfoType =
    Vector<Input<Vector<MetaInfoType, ONE_PTR>>, SHORT_SPAN>;

template <typename FPType, typename BaseTNZType, typename SubTType,
          bool isUpdateAdd>
static bool computeSlice(Input<Vector<unsigned, ONE_PTR>> &offsets,
                         const unsigned short numOffsets, BaseTNZType &baseTNZ,
                         BaseTMetaInfoType &baseTMetaInfo, SubTType &subT,
                         const unsigned rowsPerPartition,
                         const MetaInfoType yPartitionToProcess,
                         const unsigned blockRows, const unsigned blockColumns,
                         const unsigned short subColumns, float scale) {

  using MetaInfo = popsparse::BlockMetaInfo<MetaInfoType>;
  // For halves, accumulate in float so that stochastic rounding will take
  // effect.
  using ScaleType =
      std::conditional_t<std::is_same<FPType, half>::value, float, FPType>;
  const auto scaleVal = ScaleType(scale);

  const auto subGroupElements =
      sizeof(MetaInfo::SubGroupEntry) / sizeof(MetaInfoType);
  const auto gradWWorkerEntryElements =
      sizeof(MetaInfo::GradWWorkerEntry) / sizeof(MetaInfoType);

  // Consider each block of rows found in the metaInfo just once as searching
  // that is more complex than checking the content of `offsets`
  for (unsigned bucket = 0; bucket < baseTMetaInfo.size(); bucket++) {
    auto *iter = &baseTMetaInfo[bucket][0];
    auto *nzIter = &baseTNZ[bucket][0];
    // Loop over sub group entries until the id=0 which indicates the end
    while (*iter != 0) {
      auto *subGroupEntry =
          reinterpret_cast<const MetaInfo::SubGroupEntry *>(iter);
      // Only process sub groups belonging to the specified partition in y
      // (equivalent to partitioned columns). This data that belongs in our
      // partition of the input
      if (subGroupEntry->yPartition == yPartitionToProcess) {
        // Skip over the sub group and the GradWWorkerEntry entries (If any).
        iter += subGroupElements +
                gradWWorkerEntryElements * subGroupEntry->numGradWWorkers;

        const auto rowOffset = subGroupEntry->xPartition * rowsPerPartition;
        for (unsigned sparseRowBase = 0; sparseRowBase <= subGroupEntry->numXm1;
             sparseRowBase++) {
          // The block indicates elements spread over these rows are in the NZ
          // data
          const auto startRow = *iter++ + rowOffset;
          const auto endRow = startRow + blockRows;
          const auto blocksInRow = 1 + *iter++;

          // Loop over the rows listed in the offsets, checking if any are
          // in the range of rows covered by the block
          for (unsigned idx = 0; idx < numOffsets; idx++) {
            if (startRow <= offsets[idx] && endRow > offsets[idx]) {
              // Got a match, loop through the blocks of columns in the row
              auto *colBlockIter = iter;
              auto *colNzBlockIter = nzIter;
              const auto offset = idx * subColumns;
              for (unsigned block = 0; block < blocksInRow; block++) {
                // The first column to write
                auto column = *colBlockIter++;
                const auto rowInBlock = (offsets[idx] - startRow);
                auto colIter = colNzBlockIter + blockColumns * rowInBlock;
                for (unsigned i = 0; i < blockColumns; i++) {
                  if constexpr (isUpdateAdd) {
                    *colIter++ += scale * ScaleType(subT[offset + column + i]);
                  } else {
                    subT[offset + column + i] = *colIter++;
                  }
                }
                colNzBlockIter += blockRows * blockColumns;
              }
            }
          }
          iter += blocksInRow;
          nzIter += blocksInRow * blockRows * blockColumns;
        }
      } else {
        // Didn't use the row info so skip to the next subGroup's NZData
        nzIter += subGroupEntry->offsetToNextSubGroupSparseEntries;
      }
      iter = reinterpret_cast<const MetaInfoType *>(subGroupEntry) +
             subGroupEntry->offsetToNextSubGroupMetaInfo;
    }
  }
  return true;
}

// We have buckets of sparse meta information with NZ values.
// Use the `offsets` tensor which references rows within that sparse bucket
// to populate a dense output tensor `subT`.
template <typename FPType, unsigned vectorWidthInBytes>
class SparseDenseMultiSliceBlock
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {

public:
  using BaseTNZType =
      Vector<Input<Vector<FPType, ONE_PTR, vectorWidthInBytes>>, ONE_PTR>;
  using SubTType = InOut<Vector<FPType, ONE_PTR, 4>>;
  SparseDenseMultiSliceBlock();

  IS_EXTERNAL_CODELET(true);
  // The rows to extract from baseT
  Input<Vector<unsigned, ONE_PTR>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  const unsigned short blockRows;
  const unsigned short blockColumns;
  // This vertex will process data with the given yPartitionToProcess, that data
  // has a row partition in its meta data, which is scaled by rowsPerPartition
  const unsigned short rowsPerPartition;
  const MetaInfoType yPartitionToProcess;
  const unsigned short subColumns; // The number of columns found in subT
  const unsigned short numOffsets;

  bool compute() {
    constexpr bool isHalf = std::is_same<FPType, half>::value;
    // Assembler supports column widths that are 32 bit only.  This is
    // beneficial in the poplibs functions too as it avoids copies. Ensure it is
    // the case here
    assert(!(isHalf && (subColumns % 2)));
    const auto function = computeSlice<FPType, BaseTNZType, SubTType, false>;

    return function(offsets, numOffsets, baseTNZ, baseTMetaInfo, subT,
                    rowsPerPartition, yPartitionToProcess, blockRows,
                    blockColumns, subColumns, 1.0f);
  }
};
template class SparseDenseMultiSliceBlock<float, 4>;
template class SparseDenseMultiSliceBlock<float, 8>;

template class SparseDenseMultiSliceBlock<half, 2>;
template class SparseDenseMultiSliceBlock<half, 4>;
template class SparseDenseMultiSliceBlock<half, 8>;

// We have buckets of sparse meta information with NZ values.
// Use the `offsets` tensor which references rows within that sparse bucket
// to update the NZ values in the bucket based on a dense input tensor `subT`
// by applying nzValue = nzValue + scale*subT
template <typename FPType, bool vectorise>
class SparseDenseMultiUpdateAddBlock : public Vertex {
public:
  using BaseTNZType = Vector<InOut<Vector<float, ONE_PTR>>, ONE_PTR>;
  using SubTType = Input<Vector<FPType, ONE_PTR>>;
  SparseDenseMultiUpdateAddBlock();

  IS_EXTERNAL_CODELET(false);
  // The rows to update baseT with
  Input<Vector<unsigned, ONE_PTR>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  const unsigned short blockRows;
  const unsigned short blockColumns;
  const unsigned short subColumns; // The number of columns found in subT
  // This vertex will process data with the given yPartitionToProcess, that data
  // has a row partition in its meta data, which is scaled by rowsPerPartition
  const unsigned short rowsPerPartition;
  // Technically this is being compared to a variable of type MetaInfoType but
  // exchanging and using single 2-byte elements per tile has a copy cost
  // associated with it after exchange and uses 32 bit exchange anyhow
  Input<unsigned> yPartitionToProcess;
  const unsigned short numOffsets;
  Input<float> scale;

  bool compute() {

    const auto function = computeSlice<FPType, BaseTNZType, SubTType, true>;

    return function(offsets, numOffsets, baseTNZ, baseTMetaInfo, subT,
                    rowsPerPartition,
                    static_cast<MetaInfoType>(*yPartitionToProcess), blockRows,
                    blockColumns, subColumns, *scale);
  }
};
template class SparseDenseMultiUpdateAddBlock<float, true>;
template class SparseDenseMultiUpdateAddBlock<half, true>;
template class SparseDenseMultiUpdateAddBlock<float, false>;
template class SparseDenseMultiUpdateAddBlock<half, false>;
} // namespace popsparse
