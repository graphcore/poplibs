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
class [
    [poplar::constraint("elem(**data) != elem(**out)")]] BroadcastVectorInner2D
    : public Vertex {
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
              ElementOp<op, FPType>::fn(data[i][b * len + a], B[i][a]);
        }
      }
    }

    return true;
  }
};

// Partial specialization for SCALED_ADD
template <class FPType>
class [[poplar::constraint(
    "elem(**data) != "
    "elem(**out)")]] BroadcastVectorInner2D<expr::BroadcastOpType::SCALED_ADD,
                                            FPType> : public Vertex {
public:
  BroadcastVectorInner2D();

  // n is equal to B.size(), BLen.size(), data.size()
  // and dataBlockCount.size()
  const uint32_t n;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> B;
  Vector<uint16_t, ONE_PTR> BLen;
  Vector<Input<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> data;
  Vector<uint16_t, ONE_PTR> dataBlockCount;
  const FPType scale;
  Vector<Output<Vector<FPType, ONE_PTR, 8>>, ONE_PTR> out;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    for (unsigned i = 0; i != n; ++i) {
      unsigned blockCount = dataBlockCount[i];
      unsigned len = BLen[i];

      for (unsigned b = 0; b != blockCount; ++b) {
        for (unsigned a = 0; a != len; ++a) {
          out[i][b * len + a] = data[i][b * len + a] + B[i][a] * scale;
        }
      }
    }

    return true;
  }
};

template class BroadcastVectorInner2D<expr::BroadcastOpType::ADD, float>;
template class BroadcastVectorInner2D<expr::BroadcastOpType::ADD, half>;
template class BroadcastVectorInner2D<expr::BroadcastOpType::MULTIPLY, float>;
template class BroadcastVectorInner2D<expr::BroadcastOpType::MULTIPLY, half>;
template class BroadcastVectorInner2D<expr::BroadcastOpType::SCALED_ADD, float>;
template class BroadcastVectorInner2D<expr::BroadcastOpType::SCALED_ADD, half>;

} // namespace popops
