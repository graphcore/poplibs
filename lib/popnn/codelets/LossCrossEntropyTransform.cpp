// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popops/EncodingConstants.hpp"
#include <cmath>
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

template <typename FPType> class LossCrossEntropyTransform : public Vertex {
public:
  LossCrossEntropyTransform();

  Input<Vector<FPType, PTR_ALIGN32, 4>> probs;
  Input<Vector<FPType, PTR_ALIGN32, 4>> expected;
  Output<Vector<FPType, PTR_ALIGN32, 4>> deltas;
  Output<Vector<FPType, PTR_ALIGN32, 4>> transformed;
  const unsigned short size;
  Input<FPType> deltasScale;
  Input<FPType> modelOutputScaling;

  IS_EXTERNAL_CODELET(true);

  bool compute() {
    float eps =
        std::is_same<FPType, float>() ? EPS_LOG_N_FLOAT : EPS_LOG_N_HALF;
    const FPType scale = *deltasScale / *modelOutputScaling;
    const FPType logModelOutputScaling =
        FPType(log(float(*modelOutputScaling)));
    for (std::size_t i = 0; i < size; i++) {
      FPType expect = expected[i];
      FPType actual = probs[i];
      // Returned deltas are scaled by deltasScale to
      // maintain accuracy (actual is already assumed to be scaled by
      // modelOutputScaling)

      deltas[i] = scale * (actual - expect * (*modelOutputScaling));
      // Returned transformed is adjusted to no longer be scaled
      transformed[i] =
          -expect * (FPType(log(float(actual) + eps)) - logModelOutputScaling);
    }
    return true;
  }
};

template class LossCrossEntropyTransform<float>;
template class LossCrossEntropyTransform<half>;

} // namespace popnn
