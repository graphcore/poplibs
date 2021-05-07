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
class BroadcastVectorInnerInPlaceSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  BroadcastVectorInnerInPlaceSupervisor();

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

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount =
        (dataBlockCountPacked >> 3) * 6 + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        data[j * chansPerGroup + k] =
            BinaryOpFn<op, FPType, architecture::active>::fn(
                data[j * chansPerGroup + k], B[k]);
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInnerInPlaceSupervisor<expr::BinaryOpType::ADD,
                                                     float>;
template class BroadcastVectorInnerInPlaceSupervisor<expr::BinaryOpType::ADD,
                                                     half>;
template class BroadcastVectorInnerInPlaceSupervisor<expr::BinaryOpType::DIVIDE,
                                                     float>;
template class BroadcastVectorInnerInPlaceSupervisor<expr::BinaryOpType::DIVIDE,
                                                     half>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BinaryOpType::MULTIPLY, float>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BinaryOpType::MULTIPLY, half>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BinaryOpType::SUBTRACT, float>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BinaryOpType::SUBTRACT, half>;

} // namespace popops
