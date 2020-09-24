// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#include "SparseCodeletMetaInfoScale.hpp"
#include "SparseMetaInfo.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

namespace popsparse {

using MetaInfoType = unsigned short;
using BaseTMetaInfoType =
    Vector<Input<Vector<MetaInfoType, ONE_PTR>>, SHORT_SPAN>;

template <typename FPType, typename BaseTNZType, typename SubTType,
          bool isUpdateAdd>
static bool computeSlice(Input<Vector<unsigned>> &offsets, BaseTNZType &baseTNZ,
                         BaseTMetaInfoType &baseTMetaInfo, SubTType &subT,
                         const unsigned rowOffset,
                         const MetaInfoType subGroupIdToProcess,
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
      getYOffsetTypeFactor(std::is_same<FPType, float>::value);
  const auto xOffTypeSize =
      getXOffsetTypeFactor(std::is_same<FPType, float>::value);

  const auto subGroupElements =
      sizeof(MetaInfo::SubGroupEntry) / sizeof(MetaInfoType);
  const auto workerEntryElements =
      sizeof(MetaInfo::WorkerEntry) / sizeof(MetaInfoType);

  // Consider each row found in the metaInfo just once as searching that
  // is more complex than checking the content of `offsets`
  for (unsigned bucket = 0; bucket < baseTMetaInfo.size(); bucket++) {
    auto *iter = &baseTMetaInfo[bucket][0];
    auto *nzIter = &baseTNZ[bucket][0];
    // Loop over sub group entries until the id=0 which indicates the end
    while (*iter != 0) {
      auto *subGroupEntry =
          reinterpret_cast<const MetaInfo::SubGroupEntry *>(iter);
      auto id = subGroupEntry->id;
      // Only process sub groups with a specific id.  They contain
      // data that belongs in our partition of the input
      if (id == subGroupIdToProcess) {
        // We aren't using the workerEntry information to divide work so
        // skip over the sub group and the worker entries.

        iter += subGroupEntry->offsetToFirstOutputEntryMetaInfo;

        // Loop over the rows found in a sub-group
        for (unsigned sparseRow = 0; sparseRow <= subGroupEntry->numXm1;
             sparseRow++) {
          // Although this could be maintained unscaled the reciprocal to divide
          // method is fairly efficient and rowFound is used to compare to
          // multiple offsets below, which therefore don't need scaling (which
          // could be done my multiplying the offset)
          const auto rowFound =
              reciprocalMulDiv(*iter++, nzScaleFactor) / xOffTypeSize +
              rowOffset;
          const auto columnsInRow = *iter++;
          // Loop over the rows listed in the offsets, a row may be referenced
          // multiple times
          for (unsigned idx = 0; idx < offsets.size(); idx++) {
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
class SparseDenseMultiSliceElementWise : public Vertex {

public:
  using BaseTNZType = Vector<Input<Vector<FPType, ONE_PTR>>, ONE_PTR>;
  using SubTType = InOut<Vector<FPType, ONE_PTR>>;
  SparseDenseMultiSliceElementWise();

  IS_EXTERNAL_CODELET(false);
  // The rows to extract from baseT
  Input<Vector<unsigned>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  MetaInfoType nzScaleFactor;
  // This vertex will process data with the given subGroupIdToProcess, that data
  // had this rowOffset applied to its metadata
  const unsigned rowOffset;
  const MetaInfoType subGroupIdToProcess;
  const unsigned short subColumns; // The number of columns found in subT

  bool compute() {

    const auto function = computeSlice<FPType, BaseTNZType, SubTType, false>;

    return function(offsets, baseTNZ, baseTMetaInfo, subT, rowOffset,
                    subGroupIdToProcess, nzScaleFactor, subColumns, 1.0f);
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
  Input<Vector<unsigned>> offsets;
  BaseTNZType baseTNZ;
  BaseTMetaInfoType baseTMetaInfo;
  SubTType subT;
  MetaInfoType nzScaleFactor;
  // This vertex will process data with the given subGroupIdToProcess, that data
  // had this rowOffset applied to its metadata
  const unsigned rowOffset;
  const MetaInfoType subGroupIdToProcess;
  const unsigned short subColumns; // The number of columns found in subT
  Input<FPType> scale;

  bool compute() {

    const auto function = computeSlice<FPType, BaseTNZType, SubTType, true>;

    return function(offsets, baseTNZ, baseTMetaInfo, subT, rowOffset,
                    subGroupIdToProcess, nzScaleFactor, subColumns, *scale);
  }
};
template class SparseDenseMultiUpdateAddElementWise<float>;
template class SparseDenseMultiUpdateAddElementWise<half>;
} // namespace popsparse
