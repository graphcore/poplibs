// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// matrix multiply Q = R * S where R is sparse and Q is dense

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "SparseMetaInfo.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;
static constexpr auto SHORT_SPAN = VectorLayout::SHORT_SPAN;

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
static constexpr inline bool hasAssemblyVersion() {
  constexpr bool is4x4 = BlockRows == 4 && BlockCols == 4;
  constexpr bool is8x8 = BlockRows == 8 && BlockCols == 8;
  constexpr bool is16x16 = BlockRows == 16 && BlockCols == 16;

  return (std::is_same<FPType, half>() && std::is_same<AccumType, half>() &&
          (is4x4 || is8x8 || is16x16)) ||
         (std::is_same<FPType, float>() && std::is_same<AccumType, float>() &&
              is4x4 ||
          is8x8 || is16x16);
}

namespace popsparse {

template <typename FPType, typename AccumType, std::size_t BlockRows,
          std::size_t BlockCols>
class [[poplar::constraint(
    "elem(*q) != elem(*s)")]] StaticSparseDenseMatMulBlock
    : public SupervisorVertexIf<
          hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>() &&
          ASM_CODELETS_ENABLED> {

  using MetaInfoType = unsigned short;
  using MetaInfo = popsparse::StaticBlockMetaInfo<MetaInfoType>;
  constexpr static std::size_t rAlignmentRequirement = 8;
  constexpr static std::size_t qAlignmentRequirement = 16;
  constexpr static bool qInInterleavedMem = true;

public:
  StaticSparseDenseMatMulBlock();

  // sparse input values in r
  Input<Vector<FPType, ONE_PTR, rAlignmentRequirement>> r;
  // meta-info describing how to process the given inputs.
  Input<Vector<MetaInfoType, ONE_PTR>> metaInfo;
  // Single pointer to dense input s. Layout of elements in memory expected to
  // be {Y,Z}.
  Input<Vector<FPType, ONE_PTR, 8>> s;
  // Single pointer to dense output q. Layout of elements in memory expected to
  // be {X,Z}.
  Output<Vector<AccumType, ONE_PTR, qAlignmentRequirement, qInInterleavedMem>>
      q;
  // Multiple of 64-bits in q to zero. Set to zero if no zeroing required.
  const unsigned short zeroInfo;
  // used by assembler codelets to avoid NaNs when over-read data is fed to the
  // AMP. The size is in multiples of 8 bytes.
  const unsigned short sSize;

  IS_EXTERNAL_CODELET(
      (hasAssemblyVersion<FPType, AccumType, BlockRows, BlockCols>()));

  static constexpr auto fpTypeSize = std::is_same<FPType, float>::value ? 4 : 2;
  static constexpr auto accumTypeSize =
      std::is_same<AccumType, float>::value ? 4 : 2;

  static void workerCompute(unsigned wid, unsigned offsetZ, unsigned numZ,
                            unsigned offsetXInQ, unsigned offsetYInS,
                            AccumType *q, const FPType *r, const FPType *s) {
    // The following pointers could be calculated once and retained for each
    // worker
    q += BlockRows * offsetZ;
    s += BlockCols * offsetZ;

    // Split columns for float blockSize 16
    static constexpr unsigned colSplits =
        (BlockCols == 16 && std::is_same<AccumType, float>()) ? 2 : 1;

    for (std::size_t z = 0; z < numZ; ++z) {
      for (std::size_t blockRow = 0; blockRow < BlockRows; ++blockRow) {
        auto sum = q[blockRow + offsetXInQ];
        for (unsigned cSplit = 0; cSplit != colSplits; ++cSplit) {

          for (std::size_t blockCol = 0; blockCol < BlockCols / colSplits;
               ++blockCol) {
            const unsigned rIndex = BlockRows * BlockCols / colSplits * cSplit +
                                    blockRow * BlockCols / colSplits + blockCol;
            const unsigned sIndex = BlockCols / colSplits * cSplit + blockCol;
            sum += static_cast<float>(r[rIndex]) *
                   static_cast<float>(s[offsetYInS + sIndex]);
          }
        }
        q[blockRow + offsetXInQ] = sum;
      }
      q += BlockRows;
      s += BlockCols;
    }
  }

  bool compute() {
    // Zero outputs if requested.
    const unsigned bytesPerElem = 8;
    static_assert((accumTypeSize * BlockRows * BlockCols) % bytesPerElem == 0,
                  "Size in bytes of q must be divisible by byes per zero elem");
    unsigned numElems = popsparse::static_::block::convertFromImplOffset(
        accumTypeSize, zeroInfo);
    for (unsigned i = 0; i < numElems; ++i) {
      q[i] = 0;
    }

    const auto *workList =
        reinterpret_cast<const MetaInfo::WorkListEntry *>(&metaInfo[0]);
    const auto *output =
        reinterpret_cast<const MetaInfo::Output *>(workList + CTXT_WORKERS);
    const unsigned numX = output->numXm1 + 1;
    const auto *rPtr = &r[0];
    const auto *outputEntry =
        reinterpret_cast<const MetaInfo::OutputEntry *>(output + 1);

    for (std::size_t xBlock = 0; xBlock < numX; ++xBlock) {
      const auto offsetXInQ = popsparse::static_::block::convertFromImplOffset(
                                  accumTypeSize, outputEntry->offsetXInQ) *
                              BlockRows;
      const auto numY = outputEntry->numYm1 + 1;
      const auto *inputEntry =
          reinterpret_cast<const MetaInfo::InputEntry *>(outputEntry + 1);
      for (std::size_t yBlock = 0; yBlock < numY; ++yBlock) {
        const auto offsetYInS =
            popsparse::static_::block::convertFromImplOffset(
                fpTypeSize, inputEntry->offsetYInS) *
            BlockCols;
        for (unsigned wid = 0; wid < CTXT_WORKERS; ++wid) {
          const auto offsetZ = workList[wid].offsetZ;
          const auto numZ = workList[wid].numZ;
          workerCompute(wid, offsetZ, numZ, offsetXInQ, offsetYInS, &q[0], rPtr,
                        &s[0]);
        }
        ++inputEntry;
        rPtr += BlockRows * BlockCols;
      }
      outputEntry = reinterpret_cast<const MetaInfo::OutputEntry *>(inputEntry);
    }
    return true;
  }
};

template class StaticSparseDenseMatMulBlock<half, half, 4, 4>;
template class StaticSparseDenseMatMulBlock<float, float, 4, 4>;
template class StaticSparseDenseMatMulBlock<half, half, 8, 8>;
template class StaticSparseDenseMatMulBlock<float, float, 8, 8>;
template class StaticSparseDenseMatMulBlock<half, half, 16, 16>;
template class StaticSparseDenseMatMulBlock<float, float, 16, 16>;

} // end namespace popsparse
