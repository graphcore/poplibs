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
    "elem(*qGrad) != elem(*s)")]] SparseDenseMatMulGradWElementWise
    : public SupervisorVertexIf<hasAssemblyVersion<FPType, AccumType>() &&
                                ASM_CODELETS_ENABLED> {
  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::MetaInfo<MetaInfoType>;

  static void workerCompute(
      unsigned wid, const MetaInfo::GradWWorkerEntry *workerEntries,
      AccumType *rGrad, const FPType *s, const FPType *qGrad, unsigned numZ) {
    const auto *workerEntry = workerEntries + wid;

    rGrad += workerEntry->sparseOffset;

    const auto yOffTypeSize =
        getYOffsetTypeScaleFactor(std::is_same<FPType, float>::value);
    const auto xOffTypeSize =
        getXOffsetTypeDivFactor(std::is_same<FPType, float>::value);

    const auto *it = reinterpret_cast<const MetaInfoType *>(workerEntry) +
                     workerEntry->metaInfoOffsetOutputEntry;
    unsigned numYRemaining = workerEntry->totalNumY;
    while (numYRemaining > 0) {
      const auto *outputEntry =
          reinterpret_cast<const MetaInfo::OutputEntry *>(it);
      // Offset if this is the first output entry to process.
      const auto startOffset =
          (numYRemaining == workerEntry->totalNumY
               ? workerEntry->metaInfoOffsetToOffsetsYInSFirst
               : 0);
      const auto *offsetsYOfS =
          reinterpret_cast<const MetaInfoType *>(outputEntry + 1) + startOffset;
      // Potentially process fewer than numY if this is the last output
      // entry to process;
      const auto numYRemainingThisRow = outputEntry->numY - startOffset;
      const auto numY = numYRemaining < numYRemainingThisRow
                            ? numYRemaining
                            : numYRemainingThisRow;

      // Accumulate over numZ.
      for (unsigned yIndex = 0; yIndex < numY; ++yIndex) {
        AccumType sum = *rGrad;
        for (unsigned zIndex = 0; zIndex < numZ; ++zIndex) {
          const auto qGradIndex =
              outputEntry->offsetXInQ * xOffTypeSize + zIndex;
          const auto sIndex = (offsetsYOfS[yIndex] / yOffTypeSize) + zIndex;
          sum += AccumType(s[sIndex] * qGrad[qGradIndex]);
        }
        *rGrad++ = sum;
      }
      numYRemaining -= numY;

      it = reinterpret_cast<const MetaInfoType *>(outputEntry + 1) +
           outputEntry->numY;
    }
  }

public:
  SparseDenseMatMulGradWElementWise();

  // Single pointer to dense gradients at output of forward pass q.
  // Layout in memory expected to be {X,Z}.
  Input<Vector<FPType, ONE_PTR, 8>> qGrad;
  // Single pointer to bucket with sparse gradients of r.
  InOut<Vector<AccumType, ONE_PTR, 8>> rGrad;
  // Single point to bucket with meta-info describing how to process the given
  // inputs.
  Input<Vector<MetaInfoType, ONE_PTR>> metaInfo;
  // Single pointer to dense input to forward pass s.
  // Layout in memory expected to be {Y,Z}.
  Input<Vector<FPType, ONE_PTR, 8>> s;
  // The sub-group id that should be processed.
  Input<MetaInfoType> subGroupIdToProcess;
  // Number of elements in each rGrad bucket to zero. Set to zero if no zeroing
  // required.
  const unsigned short zeroInfo;
  // Number of elements in Z dimension.
  const unsigned short numZ;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType, AccumType>()));

  bool compute() {
    // Zero outputs if requested.
    for (unsigned i = 0; i < zeroInfo; ++i) {
      rGrad[i] = 0;
    }

    const MetaInfoType *metaInfoIt = &metaInfo[0];
    AccumType *rGradIt = &rGrad[0];
    for (;;) {
      const auto *subGroupEntry =
          reinterpret_cast<const MetaInfo::SubGroupEntry *>(metaInfoIt);
      if (subGroupEntry->id == MetaInfo::endSubGroupId) {
        break;
      }

      if (subGroupEntry->id == *subGroupIdToProcess) {
        // NOTE: In reality we will launch all workers and rely on those above
        // 'numWorkers' to not do any work.
        const auto fwdNumWorkers = subGroupEntry->numWorkers;
        const auto *fwdWorkerEntries =
            reinterpret_cast<const MetaInfo::WorkerEntry *>(subGroupEntry + 1);
        const auto &numWorkers = *reinterpret_cast<const MetaInfoType *>(
            fwdWorkerEntries + fwdNumWorkers);
        const auto *workerEntries =
            reinterpret_cast<const MetaInfo::GradWWorkerEntry *>(&numWorkers +
                                                                 1);
        for (unsigned wid = 0; wid < numWorkers; ++wid) {
          workerCompute(wid, workerEntries, rGradIt, &s[0], &qGrad[0], numZ);
        }
      }
      rGradIt += subGroupEntry->sparseElementCount;
      metaInfoIt += subGroupEntry->offsetToNextSubGroupMetaInfo;
    }

    return true;
  }
};

template class SparseDenseMatMulGradWElementWise<half, float>;
template class SparseDenseMatMulGradWElementWise<float, float>;

} // end namespace popsparse
