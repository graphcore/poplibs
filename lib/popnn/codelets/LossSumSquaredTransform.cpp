// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifdef VECTOR_AVAIL_SCALED_PTR32
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::SCALED_PTR32;
#else
static constexpr auto PTR_ALIGN32 = poplar::VectorLayout::ONE_PTR;
#endif

using namespace poplar;
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;

namespace popnn {

template <typename FPType> class LossSumSquaredTransform : public Vertex {
public:
  LossSumSquaredTransform();

  Input<Vector<FPType, PTR_ALIGN32, 4>> probs;
  Input<Vector<FPType, PTR_ALIGN32, 4>> expected;
  Output<Vector<FPType, PTR_ALIGN32, 4>> deltas;
  Output<Vector<FPType, PTR_ALIGN32, 4>> transformed;
  const unsigned short size;

  IS_EXTERNAL_CODELET(true);

  void compute() {
    for (std::size_t i = 0; i < size; i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      FPType delta = (actual - expect);
      deltas[i] = delta;
      transformed[i] = FPType(0.5) * delta * delta;
    }
  }
};

template class LossSumSquaredTransform<float>;
template class LossSumSquaredTransform<half>;

} // namespace popnn
