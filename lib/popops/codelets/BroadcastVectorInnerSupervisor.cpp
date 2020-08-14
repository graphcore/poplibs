// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "ElementOp.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"
#include "popops/ExprOp.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;

namespace popops {

template <expr::BinaryOpType op, class FPType>
class [[poplar::constraint(
    "elem(*data) != elem(*out)")]] BroadcastVectorInnerSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  BroadcastVectorInnerSupervisor();

  Input<Vector<FPType, SPAN, 8>> B;
  Input<Vector<FPType, ONE_PTR, 8>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (dataBlockCount/6 << 3) | (dataBlockCount % 6)
  const uint16_t dataBlockCountPacked;
  Output<Vector<FPType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount =
        (dataBlockCountPacked >> 3) * 6 + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        out[j * chansPerGroup + k] =
            ElementOp<op, FPType>::fn(data[j * chansPerGroup + k], B[k]);
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::MULTIPLY,
                                              float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::MULTIPLY,
                                              half>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::SUBTRACT,
                                              float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::SUBTRACT,
                                              half>;

} // namespace popops
