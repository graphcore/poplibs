// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "elemwiseBinaryOps.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ExprOp.hpp"

using namespace poplar;

namespace popops {

template <expr::BinaryOpType op, class FPType>
class [[poplar::constraint(
    "elem(**data) != elem(**out)")]] BroadcastVectorInner2D : public Vertex {
public:
  BroadcastVectorInner2D();

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
      unsigned blockCount = dataBlockCount[i];
      unsigned len = BLen[i];

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
          out[i][b * len + a] =
              BinaryOpFn<op, FPType, architecture::active>::fn(
                  data[i][b * len + a], B[i][a]);
        }
      }
    }

    return true;
  }
};

// The ADD and SUBTRACT codelets are implemented with low level code that
// performs the operation:
//      out = data +  scale * B
// where 'scale' is either +1 or -1 (hard coded for ADD and SUBTRACT). There
// used to be a more general vertex where 'scale' could be specified as an
// arbitrary value. This was done using the specialization for operator
// BroadcastOpType::SCALED_ADD which was removed in D29735 as it was never used.
// Refer to that diff in case the functionality is needed again in the future.
// This was also implemented for:
//    BroadcastVectorInner2DInPlace
//    BroadcastVectorInnerSupervisor
//    BroadcastVectorInnerInPlaceSupervisor
template class BroadcastVectorInner2D<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInner2D<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInner2D<expr::BinaryOpType::DIVIDE, float>;
template class BroadcastVectorInner2D<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInner2D<expr::BinaryOpType::MULTIPLY, float>;
template class BroadcastVectorInner2D<expr::BinaryOpType::MULTIPLY, half>;
template class BroadcastVectorInner2D<expr::BinaryOpType::SUBTRACT, float>;
template class BroadcastVectorInner2D<expr::BinaryOpType::SUBTRACT, half>;

} // namespace popops
