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
  return std::is_same<AccumType, float>();
}

namespace popsparse {

template <typename FPType, typename AccumType>
class [[poplar::constraint(
    "elem(*s) != elem(**metaInfo)",
    "elem(**r) != elem(**metaInfo)")]] SparseDenseMatMulGradAElementWise
    : public SupervisorVertexIf<hasAssemblyVersion<FPType, AccumType>() &&
                                ASM_CODELETS_ENABLED> {

  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::MetaInfo<MetaInfoType>;
  constexpr static std::size_t rAlignmentRequirement = alignof(FPType);

  static void workerCompute(unsigned wid,
                            const MetaInfo::WorkerEntry *workerEntries,
                            AccumType *q, const FPType *r, const FPType *s) {
    const auto *workerEntry = workerEntries + wid;

    q += workerEntry->offsetZ;
    r += workerEntry->sparseOffset;
    s += workerEntry->offsetZ;

    const auto yOffTypeSize =
        getYOffsetTypeFactor(std::is_same<FPType, float>::value);
    const auto xOffTypeSize =
        getXOffsetTypeFactor(std::is_same<FPType, float>::value);

    unsigned numRemainingX = workerEntry->numXm1 + 1;
    const auto *it = reinterpret_cast<const MetaInfoType *>(workerEntry) +
                     workerEntry->metaInfoOffset;
    while (numRemainingX-- > 0) {
      const auto *outputEntry =
          reinterpret_cast<const MetaInfo::OutputEntry *>(it);
      const auto *offsetsYOfS =
          reinterpret_cast<const MetaInfoType *>(outputEntry + 1);
      for (unsigned zIndex = 0; zIndex < workerEntry->numZ; ++zIndex) {
        const auto qIndex = outputEntry->offsetXInQ / xOffTypeSize + zIndex;
        AccumType sum = q[qIndex];

        for (unsigned yIndex = 0; yIndex < outputEntry->numY; ++yIndex) {
          const auto rIndex = offsetsYOfS[2 * yIndex] / yOffTypeSize;
          const auto sIndex =
              offsetsYOfS[2 * yIndex + 1] / yOffTypeSize + zIndex;
          sum += AccumType(r[rIndex] * s[sIndex]);
        }
        q[qIndex] = sum;
      }
      it = offsetsYOfS + 2 * outputEntry->numY;
    }
  }

public:
  SparseDenseMatMulGradAElementWise();

  // Pointers to buckets of sparse input values in r.
  Vector<Input<Vector<FPType, ONE_PTR, rAlignmentRequirement>>, ONE_PTR> r;
  // Pointers to buckets of meta-info describing how to process the given
  // inputs.
  Vector<Input<Vector<MetaInfoType, ONE_PTR>>, SHORT_SPAN> metaInfo;
  // Single pointer to dense input s. Layout of elements in memory expected to
  // be {Y,Z}.
  Input<Vector<FPType, ONE_PTR, 8>> s;
  // Single pointer to dense output q. Layout of elements in memory expected to
  // be {X,Z}.
  // We may use this in multiple passes so this needs to be an InOut edge.
  InOut<Vector<AccumType, ONE_PTR, 8>> q;
  // The sub-group id that should be processed by this vertex.
  const MetaInfoType subGroupIdToProcess;
  // Number of elements in q to zero. Set to zero if no zeroing required.
  const unsigned short zeroInfo;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType, AccumType>()));

  bool compute() {
    // Zero outputs if requested.
    for (unsigned i = 0; i < zeroInfo; ++i) {
      q[i] = 0;
    }

    for (std::size_t bucket = 0; bucket < metaInfo.size(); ++bucket) {
      const MetaInfoType *metaInfoBucketIter = &metaInfo[bucket][0];
      const FPType *rBucketIter = &r[bucket][0];

      for (;;) {
        // TODO: Find an easier to read way of generating/reading the meta-data
        // in C++. This could be shared between host and C++ codelet at least.
        const auto *subGroupEntry =
            reinterpret_cast<const MetaInfo::SubGroupEntry *>(
                metaInfoBucketIter);
        if (subGroupEntry->id == MetaInfo::endSubGroupId) {
          break;
        }

        if (subGroupEntry->id == subGroupIdToProcess) {
          // NOTE: In reality we will probably launch all workers and rely on
          // those above 'numWorkers' to not do any work.
          const auto numWorkers = subGroupEntry->numWorkers;
          const auto *workerEntries =
              reinterpret_cast<const MetaInfo::WorkerEntry *>(subGroupEntry +
                                                              1);
          for (unsigned wid = 0; wid < numWorkers; ++wid) {
            workerCompute(wid, workerEntries, &q[0], rBucketIter, &s[0]);
          }
        }
        rBucketIter += subGroupEntry->sparseElementCount;
        metaInfoBucketIter += subGroupEntry->offsetToNextSubGroupMetaInfo;
      }
    }
    return true;
  }
};

template class SparseDenseMatMulGradAElementWise<half, float>;
template class SparseDenseMatMulGradAElementWise<float, float>;

} // end namespace popsparse
