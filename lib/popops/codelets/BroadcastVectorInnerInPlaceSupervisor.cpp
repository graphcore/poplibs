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

template <expr::BroadcastOpType op, class FPType>
class BroadcastVectorInnerInPlaceSupervisor
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  BroadcastVectorInnerInPlaceSupervisor();

  // The half float code for the ADD inplace operation requires the 'acts'
  // tensor to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code).
  static const bool needsInterleave =
      std::is_same<FPType, half>::value && op == expr::BroadcastOpType::ADD;

  Input<Vector<FPType, SPAN, 8>> B;
  InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (actsBlockCount/6 << 3) | (actsBlockCount % 6)
  const uint16_t dataBlockCountPacked;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount =
        (dataBlockCountPacked >> 3) * 6 + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        data[j * chansPerGroup + k] =
            ElementOp<op, FPType>::fn(data[j * chansPerGroup + k], B[k]);
      }
    }
    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class BroadcastVectorInnerInPlaceSupervisor<expr::BroadcastOpType::SCALED_ADD,
                                            FPType>
    : public SupervisorVertexIf<ASM_CODELETS_ENABLED> {
public:
  BroadcastVectorInnerInPlaceSupervisor();

  // The half float code for this inplace operation requires the 'acts' tensor
  // to be in an interleaved region, to be able to use the ldst64pace
  // instruction. This is really needed only if addend.size() is a multiple of
  // of four (fast optimized code).
  static const bool needsInterleave = std::is_same<FPType, half>::value;

  Input<Vector<FPType, SPAN, 8>> B;
  InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>> data;
  // dataBlockCount = data.size() / B.size();
  // dataBlockCountPacked = (dataBlockCount/6 << 3) | (dataBlockCount % 6)
  const uint16_t dataBlockCountPacked;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    unsigned chansPerGroup = B.size();
    unsigned dataBlockCount =
        (dataBlockCountPacked >> 3) * 6 + (dataBlockCountPacked & 0x07);
    for (unsigned j = 0; j != dataBlockCount; ++j) {
      for (unsigned k = 0; k != chansPerGroup; ++k) {
        data[j * chansPerGroup + k] += B[k] * scale;
      }
    }
    return true;
  }
};

template class BroadcastVectorInnerInPlaceSupervisor<expr::BroadcastOpType::ADD,
                                                     float>;
template class BroadcastVectorInnerInPlaceSupervisor<expr::BroadcastOpType::ADD,
                                                     half>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BroadcastOpType::MULTIPLY, float>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BroadcastOpType::MULTIPLY, half>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BroadcastOpType::SCALED_ADD, float>;
template class BroadcastVectorInnerInPlaceSupervisor<
    expr::BroadcastOpType::SCALED_ADD, half>;

} // namespace popops
