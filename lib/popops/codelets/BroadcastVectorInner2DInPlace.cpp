// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
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
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
      unsigned blockCount = dataBlockCount[i];
      unsigned len = BLen[i];

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
          data[i][b * len + a] =
              ElementOp<op, FPType>::fn(data[i][b * len + a], B[i][a]);
        }
      }
    }

    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::SCALED_ADD, FPType>
    : public Vertex {
public:
  BroadcastVectorInner2DInPlace();

  static const bool needsInterleave = std::is_same<FPType, half>::value;

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<InOut<Vector<FPType, ONE_PTR, 8, needsInterleave>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  const FPType scale;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
      unsigned blockCount = dataBlockCount[i];
      unsigned len = BLen[i];

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
          data[i][b * len + a] += B[i][a] * scale;
        }
      }
    }

    return true;
  }
};

template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::ADD, float>;
template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::ADD, half>;
template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::MULTIPLY,
                                             float>;
template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::MULTIPLY,
                                             half>;
template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::SCALED_ADD,
                                             float>;
template class BroadcastVectorInner2DInPlace<expr::BroadcastOpType::SCALED_ADD,
                                             half>;

} // namespace popops
