// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "elemwiseBinaryOps.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popops/ExprOp.hpp"

using namespace poplar;

namespace popops {

// Overview by example:
// Given: B = [a b c d e f g h]
// bSliceLen = 4
// bBroadcastFactor = 3
// bSlicesM1 = 1
// So the pattern of elements from `B` that get applied to `data` is:
// a b c d   (1st slice of `B` is bSliceLen elements long)
// a b c d
// a b c d   (Repeat by bBroadcastFactor)
// e f g h   (bSlicesM1 = 1 so bSlices = 2 - we use 2 slices of `B`)
// e f g h
// e f g h

template <expr::BinaryOpType op, class FPType>
class [[poplar::constraint("elem(*data) != elem(*out)")]] BroadcastVectorInner1D
    : public MultiVertex {
public:
  BroadcastVectorInner1D();
  // B has size bSliceLen * (bSlicesM1 + 1)
  Input<Vector<FPType, ONE_PTR, 8>> B;
  // data has size bSliceLen * (bSlicesM1 + 1) * bBroadcastFactor
  Input<Vector<FPType, ONE_PTR, 8>> data;
  const uint16_t bSlicesM1;
  const uint16_t bSliceLen;
  const uint16_t bBroadcastFactor;
  Output<Vector<FPType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) {
    // Each worker will process a contiguous span from data[offs]

    const unsigned rowsPerWorkerToAvoidSubWordWrites = 2;
    unsigned numBlocks = rowsPerWorkerToAvoidSubWordWrites *
                         divideWork(bBroadcastFactor + 1, 1, 0);
    unsigned offs = numBlocks * wid;
    if (offs >= bBroadcastFactor) {
      // This worker has no work
      return true;
    }

    if (numBlocks * (wid + 1) > bBroadcastFactor) {
      // This worker has less than the maximum work
      numBlocks = bBroadcastFactor - offs;
    }

    for (unsigned i = 0; i < (bSlicesM1 + 1); i++) {
      const FPType *dataPtr =
          &data[i * (bBroadcastFactor * bSliceLen) + offs * bSliceLen];
      FPType *outPtr =
          &out[i * (bBroadcastFactor * bSliceLen) + offs * bSliceLen];

      for (unsigned k = 0; k != bSliceLen; ++k) {
        for (unsigned j = 0; j != numBlocks; ++j) {
          const auto idx = j * bSliceLen + k;
          outPtr[idx] = BinaryOpFn<op, FPType, architecture::active>::fn(
              dataPtr[idx], B[k + i * bSliceLen]);
        }
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInner1D<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::DIVIDE, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::MULTIPLY, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::MULTIPLY, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::SUBTRACT, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::SUBTRACT, half>;

} // namespace popops
