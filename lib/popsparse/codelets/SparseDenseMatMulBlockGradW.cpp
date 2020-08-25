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

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
static constexpr inline bool hasAssemblyVersion() {
  constexpr bool is4x4 = BlockRows == 4 && BlockCols == 4;
  constexpr bool is16x16 =
      BlockRows == 16 && BlockCols == 16 && std::is_same<FPType, half>();
  return is4x4 || is16x16;
}

namespace popsparse {

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
class SparseDenseMatMulBlockGradW
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>() &&
          ASM_CODELETS_ENABLED> {
  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::BlockMetaInfo<MetaInfoType>;

  static constexpr auto fpTypeSize = std::is_same<FPType, float>::value ? 4 : 2;
  static constexpr auto accumTypeSize =
      std::is_same<AccumType, float>::value ? 4 : 2;

  static void workerCompute(unsigned wid,
                            const MetaInfo::GradWWorkerEntry *workerEntries,
                            AccumType *rGrad, const FPType *s,
                            const FPType *qGrad, unsigned numZ,
                            unsigned zStrideInQ, unsigned zStrideInS) {
    const auto *workerEntry = workerEntries + wid;

    rGrad += workerEntry->sparseOffset;

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
      const auto numYRemainingThisRow = outputEntry->numYm1 + 1 - startOffset;
      const auto numY = numYRemaining < numYRemainingThisRow
                            ? numYRemaining
                            : numYRemainingThisRow;

      // Accumulate over numZ.
      // TODO: Just need an inner loop here per block
      for (unsigned yIndex = 0; yIndex < numY; ++yIndex) {
        for (unsigned bRow = 0; bRow < BlockRows; ++bRow) {
          for (unsigned bCol = 0; bCol < BlockCols; ++bCol) {
            const auto rIndex = bRow * BlockCols + bCol;
            AccumType sum = rGrad[rIndex];
            for (unsigned zIndex = 0; zIndex < numZ; ++zIndex) {
              const auto qGradIndex =
                  outputEntry->offsetXInQ + zIndex * zStrideInQ + bRow;
              const auto sIndex =
                  offsetsYOfS[yIndex] + zIndex * zStrideInS + bCol;
              sum += AccumType(s[sIndex] * qGrad[qGradIndex]);
            }
            rGrad[rIndex] = sum;
          }
        }
        rGrad += BlockRows * BlockCols;
      }
      numYRemaining -= numY;

      it = reinterpret_cast<const MetaInfoType *>(outputEntry + 1) +
           outputEntry->numYm1 + 1;
    }
  }

public:
  SparseDenseMatMulBlockGradW();

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
  // zStrideInQ : Stride in multiples of 64-bits between elements of Z in Q
  unsigned short zStrideInQ;
  // zStrideInS : Stride in multiples of 64-bits between elements of Z in S
  unsigned short zStrideInS;

  IS_EXTERNAL_CODELET(
      (hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>()));

  bool compute() {

    // Zero outputs if requested.
    static_assert((accumTypeSize * BlockRows * BlockCols) % 4 == 0,
                  "Size in bytes of q must be divisible by 4");
    for (unsigned i = 0; i < zeroInfo * (4 / accumTypeSize); ++i) {
      rGrad[i] = 0;
    }

    static constexpr auto fpTypeSize =
        std::is_same<FPType, float>::value ? 4 : 2;
    const auto qStride = zStrideInQ * (8 / fpTypeSize);
    const auto sStride = zStrideInS * (8 / fpTypeSize);

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
        const auto numWorkers = subGroupEntry->numGradWWorkers;
        const auto *workerEntries =
            reinterpret_cast<const MetaInfo::GradWWorkerEntry *>(subGroupEntry +
                                                                 1);
        for (unsigned wid = 0; wid < numWorkers; ++wid) {
          workerCompute(wid, workerEntries, rGradIt, &s[0], &qGrad[0], numZ,
                        qStride, sStride);
        }
      }
      rGradIt += subGroupEntry->offsetToNextSubGroupSparseEntries;
      metaInfoIt += subGroupEntry->offsetToNextSubGroupMetaInfo;
    }

    return true;
  }
};

template class SparseDenseMatMulBlockGradW<half, float, 4, 4>;
template class SparseDenseMatMulBlockGradW<half, half, 4, 4>;
template class SparseDenseMatMulBlockGradW<float, float, 4, 4>;

template class SparseDenseMatMulBlockGradW<half, float, 8, 8>;
template class SparseDenseMatMulBlockGradW<half, half, 8, 8>;
template class SparseDenseMatMulBlockGradW<float, float, 8, 8>;

template class SparseDenseMatMulBlockGradW<half, float, 16, 16>;
template class SparseDenseMatMulBlockGradW<half, half, 16, 16>;
template class SparseDenseMatMulBlockGradW<float, float, 16, 16>;

} // end namespace popsparse
