// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

#include "SparseMetaInfo.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

template <typename FPType, typename AccumType>
static constexpr inline bool hasAssemblyVersion() {
  return std::is_same<AccumType, float>::value;
}

namespace popsparse {

template <typename FPType, typename AccumType>
class SparseDenseMatMulElementWiseTranspose
    : public SupervisorVertexIf<hasAssemblyVersion<FPType, AccumType>() &&
                                ASM_CODELETS_ENABLED> {

  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::MetaInfo<MetaInfoType>;
  constexpr static std::size_t aAlignmentRequirement = alignof(FPType);

  static void workerCompute(const MetaInfoType *firstOutputEntry,
                            const FPType *q, const FPType *r, AccumType *s,
                            unsigned numZ, unsigned numX) {

    const auto yOffTypeSize =
        getYOffsetTypeFactor(std::is_same<FPType, float>::value);
    const auto xOffTypeSize =
        getXOffsetTypeFactor(std::is_same<FPType, float>::value);

    unsigned numRemainingX = numX;
    const auto *it = firstOutputEntry;
    while (numRemainingX-- > 0) {
      const auto *outputEntry =
          reinterpret_cast<const MetaInfo::OutputEntry *>(it);
      const auto *offsetsYOfS =
          reinterpret_cast<const MetaInfoType *>(outputEntry + 1);
      for (unsigned zIndex = 0; zIndex < numZ; ++zIndex) {
        const auto qIndex = outputEntry->offsetXInQ / xOffTypeSize + zIndex;
        for (unsigned yIndex = 0; yIndex < outputEntry->numY; ++yIndex) {
          const auto rIndex = yIndex;
          const auto sIndex = offsetsYOfS[yIndex] / yOffTypeSize + zIndex;
          s[sIndex] += AccumType(r[rIndex] * q[qIndex]);
        }
      }
      r += outputEntry->numY;
      it = offsetsYOfS + outputEntry->numY;
    }
  }

public:
  SparseDenseMatMulElementWiseTranspose();

  // Pointers to buckets of sparse input values in r.
  Vector<Input<Vector<FPType, ONE_PTR, aAlignmentRequirement>>, ONE_PTR> r;
  // Pointers to buckets of meta-info describing how to process the given
  // inputs.
  Vector<Input<Vector<MetaInfoType, ONE_PTR>>, SHORT_SPAN> metaInfo;
  // Single pointer to dense output s. Layout of elements in memory expected to
  // be {Y,Z}.
  InOut<Vector<AccumType, ONE_PTR, 16, true>> s;
  // Single pointer to dense input q. Layout of elements in memory expected to
  // be {X,Z}.
  // We may use this in multiple passes so this needs to be an InOut edge.
  Input<Vector<FPType, ONE_PTR, 8>> q;
  // The sub-group id that should be processed by this vertex.
  const MetaInfoType subGroupIdToProcess;
  // Number of elements in q to zero. Set to zero if no zeroing required.
  const unsigned short zeroInfo;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType, AccumType>()));

  bool compute() {
    // Zero outputs if requested.
    for (unsigned i = 0; i < zeroInfo; ++i) {
      s[i] = 0;
    }

    for (std::size_t bucket = 0; bucket < metaInfo.size(); ++bucket) {
      const MetaInfoType *metaInfoBucketIter = &metaInfo[bucket][0];
      const FPType *rBucketIter = &r[bucket][0];

      for (;;) {
        const auto *subGroupEntry =
            reinterpret_cast<const MetaInfo::SubGroupEntry *>(
                metaInfoBucketIter);
        if (subGroupEntry->id == MetaInfo::endSubGroupId) {
          break;
        }

        if (subGroupEntry->id == subGroupIdToProcess) {
          const auto *firstOutputEntry =
              reinterpret_cast<const MetaInfoType *>(metaInfoBucketIter) +
              subGroupEntry->offsetToFirstOutputEntryMetaInfo;
          // the workers divide the work per row using field in the
          // subGroup
          workerCompute(firstOutputEntry, &q[0], rBucketIter, &s[0],
                        subGroupEntry->numZ, subGroupEntry->numXm1 + 1);
        }
        rBucketIter += subGroupEntry->sparseElementCount;
        metaInfoBucketIter += subGroupEntry->offsetToNextSubGroupMetaInfo;
      }
    }
    return true;
  }
};

template class SparseDenseMatMulElementWiseTranspose<half, float>;
template class SparseDenseMatMulElementWiseTranspose<float, float>;

} // end namespace popsparse
