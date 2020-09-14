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
  const auto is16x16 = BlockRows == 16 && BlockRows == 16;
  const auto is8x8 = BlockRows == 8 && BlockRows == 8;
  const auto is4x4 = BlockRows == 4 && BlockRows == 4;

  return (is16x16 && std::is_same<FPType, half>()) || is8x8 || is4x4;
}

namespace popsparse {

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
class [[poplar::constraint(
    "elem(*qGrad) != elem(*rGrad)")]] SparseDenseMatMulBlockAmpGradW
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>() &&
          ASM_CODELETS_ENABLED> {
  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::BlockMetaInfo<MetaInfoType>;

  static constexpr auto accumTypeSize =
      std::is_same<AccumType, float>::value ? 4 : 2;

  static void workerCompute(AccumType * rGrad, const FPType *s,
                            const FPType *qGrad, unsigned numZ) {
    // Accumulate over numZ.
    for (unsigned bRow = 0; bRow < BlockRows; ++bRow) {
      for (unsigned bCol = 0; bCol < BlockCols; ++bCol) {
        const auto rIndex = bRow * BlockCols + bCol;
        AccumType sum = rGrad[rIndex];
        for (unsigned zIndex = 0; zIndex < numZ; ++zIndex) {
          const auto qGradIndex = bRow * numZ + zIndex;
          const auto sIndex = bCol * numZ + zIndex;
          sum += AccumType(s[sIndex] * qGrad[qGradIndex]);
          rGrad[rIndex] = sum;
        }
      }
    }
  }

public:
  SparseDenseMatMulBlockAmpGradW();

  // Single pointer to dense gradients at output of forward pass q.
  // Layout in memory expected to be {X,Z}.
  Input<Vector<FPType, ONE_PTR, 8>> qGrad;
  // Single pointer to bucket with sparse gradients of r.
  InOut<Vector<AccumType, ONE_PTR, 8, true>> rGrad;
  // Single point to bucket with meta-info describing how to process the given
  // inputs.
  Input<Vector<MetaInfoType, ONE_PTR>> metaInfo;
  // Single pointer to dense input to forward pass s.
  // Layout in memory expected to be {Y,Z}.
  Input<Vector<FPType, ONE_PTR, 16, true>> s;
  // The sub-group id that should be processed.
  Input<MetaInfoType> subGroupIdToProcess;
  // Number of elements in each rGrad bucket to zero. Set to zero if no zeroing
  // required.
  const unsigned short zeroInfo;
  // Number of elements in Z dimension.
  const unsigned short numZ;

  IS_EXTERNAL_CODELET(
      (hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>()));

  bool compute() {

    // Zero outputs if requested.
    const unsigned bytesPerElem = 8;
    static_assert(
        (accumTypeSize * BlockRows * BlockCols) % bytesPerElem == 0,
        "Size in bytes of rGrad not divisible by bytes per zero elem");
    for (unsigned i = 0; i < zeroInfo * (bytesPerElem / accumTypeSize); ++i) {
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
        // TODO: check if grad entries would be part of metainfo
        const auto numWorkers = subGroupEntry->numGradWWorkers;
        const auto *workerEntries =
            reinterpret_cast<const MetaInfo::GradWWorkerEntry *>(subGroupEntry +
                                                                 1);
        const auto numX = subGroupEntry->numXm1 + 1;
        const auto *it =
            reinterpret_cast<const MetaInfoType *>(workerEntries + numWorkers);
        auto *rGradItSubgroup = rGradIt;
        for (unsigned x = 0; x != numX; ++x) {
          const auto *outputEntry =
              reinterpret_cast<const MetaInfo::OutputEntry *>(it);
          const auto numY = outputEntry->numYm1 + 1;
          const auto offsetInX = outputEntry->offsetXInQ;
          it = reinterpret_cast<const MetaInfoType *>(outputEntry + 1);
          for (unsigned y = 0; y != numY; ++y) {
            const auto offsetInY = *it++;
            // process one block at a time with every worker processing part
            // of the same block.
            workerCompute(rGradItSubgroup, &s[offsetInY * numZ],
                          &qGrad[offsetInX * numZ], numZ);
            rGradItSubgroup += BlockRows * BlockCols;
          }
        }
      }
      rGradIt += subGroupEntry->offsetToNextSubGroupSparseEntries;
      metaInfoIt += subGroupEntry->offsetToNextSubGroupMetaInfo;
    }
    return true;
  }
};

template class SparseDenseMatMulBlockAmpGradW<half, float, 4, 4>;
template class SparseDenseMatMulBlockAmpGradW<half, half, 4, 4>;
template class SparseDenseMatMulBlockAmpGradW<float, float, 4, 4>;

template class SparseDenseMatMulBlockAmpGradW<half, float, 8, 8>;
template class SparseDenseMatMulBlockAmpGradW<half, half, 8, 8>;
template class SparseDenseMatMulBlockAmpGradW<float, float, 8, 8>;

template class SparseDenseMatMulBlockAmpGradW<half, float, 16, 16>;
template class SparseDenseMatMulBlockAmpGradW<half, half, 16, 16>;
template class SparseDenseMatMulBlockAmpGradW<float, float, 16, 16>;

} // end namespace popsparse
