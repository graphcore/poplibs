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

template <expr::BinaryOpType op, class FPType>
class BroadcastVectorInner1DInPlace : public MultiVertex {
public:
  BroadcastVectorInner1DInPlace();

  // The half float code for the ADD and SUBTRACT inplace operation requires
  // the 'acts' tensor to be in an interleaved region, to be able to use the
  // ldst64pace instruction. This is really needed only if addend.size() is a
  // multiple of four (fast optimized code), but we cannot keep that into
  // account, as the 'interleave' flag is a compile time constant.
  static const bool needsInterleave =
      std::is_same<FPType, half>::value &&
      (op == expr::BinaryOpType::ADD || op == expr::BinaryOpType::MULTIPLY ||
       op == expr::BinaryOpType::SUBTRACT);

  Input<Vector<FPType, SPAN, 8>> B;
  InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>> data;
  // See comment in BroadcastVectorInner1D.cpp for the meaning of this field
  const uint16_t dataBlockCountPacked;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) {
    unsigned BSize = B.size();
    // See comment in BroadcastVectorInner1D.cpp for this calculation
    unsigned dataBlockCount = (dataBlockCountPacked >> 3);
    const unsigned remainingBlocks = dataBlockCountPacked & 0x07;
    unsigned offs = wid * dataBlockCount +
                    ((wid < remainingBlocks) ? wid : remainingBlocks);
    unsigned numBlocks = dataBlockCount + (wid < remainingBlocks);

    FPType *dataPtr = &data[offs * BSize];

    for (unsigned k = 0; k != BSize; ++k) {
      for (unsigned j = 0; j != numBlocks; ++j) {
        dataPtr[j * BSize + k] =
            BinaryOpFn<op, FPType, architecture::active>::fn(
                dataPtr[j * BSize + k], B[k]);
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::DIVIDE, float>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::MULTIPLY,
                                             float>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::MULTIPLY,
                                             half>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::SUBTRACT,
                                             float>;
template class BroadcastVectorInner1DInPlace<expr::BinaryOpType::SUBTRACT,
                                             half>;

} // namespace popops
