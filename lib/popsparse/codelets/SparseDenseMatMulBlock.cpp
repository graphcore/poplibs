// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "SparseMetaInfo.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
static constexpr inline bool hasAssemblyVersion() {
  return false;
}

namespace popsparse {

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
class SparseDenseMatMulBlock
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>() &&
          ASM_CODELETS_ENABLED> {

  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::BlockMetaInfo<MetaInfoType>;
  // TODO: Dependent on the block size specialisations, this alignment needs
  // increasing.
  constexpr static std::size_t rAlignmentRequirement = alignof(FPType);

public:
  SparseDenseMatMulBlock();

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

  // Some extra info for block-sparsity that does not change dependent on
  // meta-info.
  const unsigned short numWorkers;
  // Stride in multiples of 64-bits between elements of Z in Q
  const unsigned short zStrideInQ;
  // Stride in multiples of 64-bits between elements of Z in S
  const unsigned short zStrideInS;
  // TODO: The above 2 fields can be packed into a single register for use
  // with stride instructions.
  Input<Vector<unsigned short, ONE_PTR>> offsetAndNumZByWorker;

  IS_EXTERNAL_CODELET(
      (hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>()));

  static constexpr auto fpTypeSize = std::is_same<FPType, float>::value ? 4 : 2;
  static constexpr auto accumTypeSize =
      std::is_same<AccumType, float>::value ? 4 : 2;

  static void workerCompute(unsigned wid, unsigned zStrideInQ,
                            unsigned zStrideInS, unsigned offsetZ,
                            unsigned numZ, unsigned offsetXInQ,
                            unsigned offsetYInS, AccumType *q, const FPType *r,
                            const FPType *s) {
    // The following pointers could be calculated once and retained for each
    // worker
    q += (zStrideInQ * (8 / accumTypeSize) * offsetZ);
    s += (zStrideInS * (8 / fpTypeSize) * offsetZ);

    for (std::size_t z = 0; z < numZ; ++z) {
      for (std::size_t blockRow = 0; blockRow < BlockRows; ++blockRow) {
        auto sum = q[blockRow + offsetXInQ];
        for (std::size_t blockCol = 0; blockCol < BlockCols; ++blockCol) {
          sum += static_cast<float>(r[blockRow * BlockCols + blockCol]) *
                 static_cast<float>(s[offsetYInS + blockCol]);
        }
        q[blockRow + offsetXInQ] = sum;
      }
      q += zStrideInQ * (8 / accumTypeSize);
      s += zStrideInS * (8 / fpTypeSize);
    }
  }

  bool compute() {
    // Zero outputs if requested.
    for (unsigned i = 0; i < zeroInfo; ++i) {
      q[i] = 0;
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
          const auto *r = rBucketIter;
          const auto *outputEntry =
              reinterpret_cast<const MetaInfo::OutputEntry *>(subGroupEntry +
                                                              1);
          for (std::size_t xBlock = 0; xBlock < subGroupEntry->numXm1 + 1;
               ++xBlock) {
            const auto *inputEntry =
                reinterpret_cast<const MetaInfo::InputEntry *>(outputEntry + 1);
            for (std::size_t yBlock = 0; yBlock < outputEntry->numY; ++yBlock) {
              // NOTE: In reality we will probably launch all workers and rely
              // on those above 'numWorkers' to not do any work. Also the line
              // between supervisor and worker is more blurred as we can use
              // worker state retention to good effect.
              for (unsigned wid = 0; wid < numWorkers; ++wid) {
                const auto offsetZ = offsetAndNumZByWorker[wid * 2];
                const auto numZ = offsetAndNumZByWorker[wid * 2 + 1];
                workerCompute(wid, zStrideInQ, zStrideInS, offsetZ, numZ,
                              outputEntry->offsetXInQ, inputEntry->offsetYInS,
                              &q[0], r, &s[0]);
              }
              ++inputEntry;
              r += BlockRows * BlockCols;
            }
            outputEntry =
                reinterpret_cast<const MetaInfo::OutputEntry *>(inputEntry);
          }
        }
        rBucketIter += subGroupEntry->offsetToNextSubGroupSparseEntries;
        metaInfoBucketIter += subGroupEntry->offsetToNextSubGroupMetaInfo;
      }
    }
    return true;
  }
};

template class SparseDenseMatMulBlock<half, float, 4, 4>;
template class SparseDenseMatMulBlock<float, float, 4, 4>;
template class SparseDenseMatMulBlock<half, float, 16, 16>;
template class SparseDenseMatMulBlock<float, float, 16, 16>;

} // end namespace popsparse
