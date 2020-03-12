// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "poplibs_support/ExternalCodelet.hpp"
#include "poplibs_support/TileConstants.hpp"

using namespace poplar;

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

namespace poplin {

template <class MeanType, class PowerType, class OutType, bool stableAlgo>
class InverseStdDeviation : public Vertex {
public:
  InverseStdDeviation();

  // inner loop will process two elements at a time;
  // output can be written as a single fp32 or pair of f16 plus a trailing
  // aligned element
  Vector<Input<Vector<MeanType, SPAN, sizeof(MeanType) * 2>>> mean;
  Vector<Input<Vector<PowerType, ONE_PTR, sizeof(PowerType) * 2>>, ONE_PTR>
      power;
  Vector<Output<Vector<OutType, ONE_PTR, 4>>, ONE_PTR> iStdDev;
  const float scaleVar;
  const float eps;

  bool compute() {
    for (unsigned i = 0; i != mean.size(); ++i) {
      for (unsigned j = 0; j != mean[i].size(); ++j) {
        if (stableAlgo) {
          // If stable algorithm is used, the power estimate is the variance
          // estimate and guaranteed to be >= 0
          float varianceEst = (float(power[i][j]) + eps) * scaleVar;
          float invStdDev = 1.0f / sqrt(varianceEst);
          iStdDev[i][j] = invStdDev;
        } else {
          float elem = float(mean[i][j]);
          float varianceEst = float(power[i][j]) - elem * elem;
          // rounding can cause this estimate to become negative
          if (varianceEst < 0.0f)
            varianceEst = 0.0f;
          varianceEst += eps;
          varianceEst *= scaleVar;
          float invStdDev = 1.0f / sqrt(varianceEst);
          iStdDev[i][j] = invStdDev;
        }
      }
    }
    return true;
  }
};

#define INSTANTIATE_INVERSE_STD_DEV(stableAlgo)                                \
  template class InverseStdDeviation<float, float, float, stableAlgo>;         \
  template class InverseStdDeviation<float, float, half, stableAlgo>;          \
  template class InverseStdDeviation<half, float, half, stableAlgo>;           \
  template class InverseStdDeviation<half, half, half, stableAlgo>;

INSTANTIATE_INVERSE_STD_DEV(true)
INSTANTIATE_INVERSE_STD_DEV(false)

} // end namespace poplin
