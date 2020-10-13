// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#include "SparseCodeletMetaInfoScale.hpp"
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
                         const unsigned nzScaleFactor,
                         const unsigned short subColumns, FPType scale) {

  using MetaInfo = popsparse::MetaInfo<MetaInfoType>;
  // For halves, accumulate in float so that stochastic rounding will take
  // effect.
  using ScaleType =
      std::conditional_t<std::is_same<FPType, half>::value, float, FPType>;
  const auto scaleVal = ScaleType(scale);

  // Note that in practice these sizes are used to divide but are all powers
  // of 2, and can be combined with the shift in the expressions where they are
  // used below
  const auto yOffTypeSize =
      getYOffsetTypeScaleFactor(std::is_same<FPType, float>::value);

  // Consider each row found in the metaInfo just once as searching that
  // is more complex than checking the content of `offsets`
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
        // We aren't using the workerEntry information to divide work so
        // skip over the sub group and the worker entries.
        iter += subGroupEntry->offsetToFirstOutputEntryMetaInfo;

        const auto rowOffset = subGroupEntry->xPartition * rowsPerPartition;
        // Loop over the rows found in a sub-group
        for (unsigned sparseRow = 0; sparseRow <= subGroupEntry->numXm1;
             sparseRow++) {
          const auto rowFound = *iter++ + rowOffset;
          const auto columnsInRow = *iter++;
          // Loop over the rows listed in the offsets, a row may be referenced
          // multiple times
          for (unsigned idx = 0; idx < numOffsets; idx++) {
            if (rowFound == offsets[idx]) {
              // If found, copy the NZ values into the dense result or back into
              // the sparse result for update
              auto *colIter = iter;
              auto *colNZIter = nzIter;
              const auto offset = idx * subColumns;
              for (unsigned c = 0; c < columnsInRow; c++) {
                const unsigned column =
                    reciprocalMulDiv(*colIter, nzScaleFactor) / yOffTypeSize;
                if constexpr (isUpdateAdd) {
                  *colNZIter += scaleVal * ScaleType(subT[offset + column]);
                } else {
                  subT[offset + column] = *colNZIter;
                }
                colIter++;
                colNZIter++;
              }
            }
          }
          iter += columnsInRow;
          nzIter += columnsInRow;
        }
      } else {
        // Didn't use the row info so skip to the next subGroup's NZData
        nzIter += subGroupEntry->sparseElementCount;
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
template <typename FPType>
class SparseDenseMultiSliceElementWise
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {

public:
  using BaseTNZType = Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR>;
  using SubTType = InOut<Vector<FPType, ONE_PTR, 4>>;
  SparseDenseMultiSliceElementWise();

  IS_EXTERNAL_CODELET(true);
  // The rows to extract from baseT
  Input<Vector<unsigned, ONE_PTR>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  MetaInfoType nzScaleFactor;
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
                    rowsPerPartition, yPartitionToProcess, nzScaleFactor,
                    subColumns, 1.0f);
  }
};
template class SparseDenseMultiSliceElementWise<float>;
template class SparseDenseMultiSliceElementWise<half>;

// We have buckets of sparse meta information with NZ values.
// Use the `offsets` tensor which references rows within that sparse bucket
// to update the NZ values in the bucket based on a dense input tensor `subT`
// by applying nzValue = nzValue + scale*subT
template <typename FPType>
class SparseDenseMultiUpdateAddElementWise : public Vertex {
public:
  using BaseTNZType = Vector<InOut<Vector<FPType, ONE_PTR>>, ONE_PTR>;
  using SubTType = Input<Vector<FPType, ONE_PTR>>;
  SparseDenseMultiUpdateAddElementWise();

  IS_EXTERNAL_CODELET(false);
  // The rows to update baseT with
  Input<Vector<unsigned, ONE_PTR>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  MetaInfoType nzScaleFactor;
  const unsigned short subColumns; // The number of columns found in subT
  // This vertex will process data with the given yPartitionToProcess, that data
  // has a row partition in its meta data, which is scaled by rowsPerPartition
  const unsigned short rowsPerPartition;
  // Technically this is being compared to a variable of type MetaInfoType but
  // exchanging and using single 2-byte elements per tile has a copy cost
  // associated with it after exchange and uses 32 bit exchange anyhow
  Input<unsigned> yPartitionToProcess;
  const unsigned short numOffsets;
  Input<FPType> scale;

  bool compute() {

    const auto function = computeSlice<FPType, BaseTNZType, SubTType, true>;

    return function(offsets, numOffsets, baseTNZ, baseTMetaInfo, subT,
                    rowsPerPartition,
                    static_cast<MetaInfoType>(*yPartitionToProcess),
                    nzScaleFactor, subColumns, *scale);
  }
};
template class SparseDenseMultiUpdateAddElementWise<float>;
template class SparseDenseMultiUpdateAddElementWise<half>;
} // namespace popsparse
