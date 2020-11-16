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
    "elem(*data) != elem(*out)")]] BroadcastVectorInnerSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  BroadcastVectorInnerSupervisor();

  Input<Vector<FPType, SPAN, 8>> B;
  Input<Vector<FPType, ONE_PTR, 8>> data;
  // The division of work among 6 workers has been done when creating the vertex
  // (contrary to other types of vertices that do that in the device code).
  //
  // The amount of work to do is expressed by:
  //        dataBlockCount = data.size() / B.size();
  // i.e. how many times the 'B' vector fits inside 'data'
  // This has been divided by 6; the quotient and remainder of this division
  // has been packed into 'dataBlockCountPacked'
  //
  //                         31 30 29 28 27 26            4  3  2  1  0
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  // dataBlockCountPacked:  |           29 bits               | 3 bits |
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  //
  //                        |                                 |        |
  //                        +---------------+-----------------+----+---+
  //                                        |                      |
  //                            floor(dataBlockCount/6)    dataBlockCount % 6
  //
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
            BinaryOpFn<op, FPType, architecture::active>::fn(
                data[j * chansPerGroup + k], B[k]);
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::DIVIDE,
                                              float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::MULTIPLY,
                                              float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::MULTIPLY,
                                              half>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::SUBTRACT,
                                              float>;
template class BroadcastVectorInnerSupervisor<expr::BinaryOpType::SUBTRACT,
                                              half>;

} // namespace popops
