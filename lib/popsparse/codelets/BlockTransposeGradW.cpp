// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Rearrange input operand with layout expected to be [Z,XorY] to layout that
// can be loaded into CWEI in AMP block sparse matmul codelets without changing
// input pointer.

#include <poplar/AvailableVTypes.h>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;

template <typename FPType> static constexpr inline bool hasAssemblyVersion() {
  return true;
}

namespace popsparse {

template <typename FPType>
class [[poplar::constraint("elem(*in) != elem(*out)")]] BlockTransposeGradW
    : public MultiVertex {
public:
  BlockTransposeGradW();
  // S or QGrad in dimension [numZ, XorY]
  Input<Vector<FPType, ONE_PTR, 8>> in;
  // Block transpose S or QGrad of size [(XorY)/BXY, numZ/BZ, BXY, BZ]
  // Where BZ is 8 for float and 16 for half
  Output<Vector<FPType, ONE_PTR, 8>> out;
  // Block size BXY
  unsigned short blockSizeXOrY;
  unsigned short numXOrYBlocks;
  unsigned short numZ;
  // used only by asm codelet
  unsigned short maxXOrYBlocksPerWorker;

  IS_EXTERNAL_CODELET((hasAssemblyVersion<FPType>()));

  void compute(unsigned wid) {
    if (wid == 0) {
      const auto blockSizeZ = std::is_same<FPType, float>() ? 8 : 16;
      const auto numBlocksZ = numZ / blockSizeZ;
      const auto numXOrY = numXOrYBlocks * blockSizeXOrY;

      for (auto blockXOrY = 0U; blockXOrY != numXOrYBlocks; ++blockXOrY) {
        for (auto blockZ = 0U; blockZ != numZ / blockSizeZ; ++blockZ) {
          for (auto elemXOrY = 0U; elemXOrY != blockSizeXOrY; ++elemXOrY) {
            for (auto elemZ = 0U; elemZ != blockSizeZ; ++elemZ) {
              const auto z = blockZ * blockSizeZ + elemZ;
              const auto inIndex =
                  z * numXOrY + blockXOrY * blockSizeXOrY + elemXOrY;
              const auto outIndex = blockXOrY * blockSizeXOrY * numZ +
                                    blockZ * blockSizeZ * blockSizeXOrY +
                                    elemXOrY * blockSizeZ + elemZ;
              out[outIndex] = in[inIndex];
            }
          }
        }
      }
    }
  }
};

template class BlockTransposeGradW<half>;
template class BlockTransposeGradW<float>;

} // end namespace popsparse
