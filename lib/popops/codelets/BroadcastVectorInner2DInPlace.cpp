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
class BroadcastVectorInner2DInPlace : public Vertex {
public:
  BroadcastVectorInner2DInPlace();

  // The half float code for this inplace operation requires the 'acts' tensor
  // to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code)
  static const bool needsInterleave = std::is_same<FPType, half>::value;

  // n is equal to B.size(), B.size(), data.size()
  // and dataBlockCount.size()
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Input<Vector<uint16_t, ONE_PTR>> workList;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>>, ONE_PTR> data;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    const auto n = 1 + workList[0];
    const auto *lenAndBlockCount = &workList[1];

    for (unsigned i = 0; i != n; ++i) {
      unsigned len = *lenAndBlockCount++;
      unsigned blockCount = *lenAndBlockCount++;

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
          data[i][b * len + a] =
              BinaryOpFn<op, FPType, architecture::active>::fn(
                  data[i][b * len + a], B[i][a]);
        }
      }
    }

    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::DIVIDE, float>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::MULTIPLY,
                                             float>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::MULTIPLY,
                                             half>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::SUBTRACT,
                                             float>;
template class BroadcastVectorInner2DInPlace<expr::BinaryOpType::SUBTRACT,
                                             half>;

} // namespace popops
