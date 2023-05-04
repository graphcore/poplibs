// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popnn/NonLinearity.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto DELTANELEMENTS = poplar::VectorListLayout::DELTANELEMENTS;

// Macros to instantiate a template class for non linear operations
#define INSTANTIATE_NL(v)                                                      \
  template class v<float, popnn::NonLinearityType::GELU>;                      \
  template class v<half, popnn::NonLinearityType::GELU>;                       \
  template class v<float, popnn::NonLinearityType::SWISH>;                     \
  template class v<half, popnn::NonLinearityType::SWISH>;

#define INSTANTIATE_NL_GRAD(v)                                                 \
  template class v<float, popnn::NonLinearityType::SIGMOID>;                   \
  template class v<half, popnn::NonLinearityType::SIGMOID>;                    \
  template class v<float, popnn::NonLinearityType::RELU>;                      \
  template class v<half, popnn::NonLinearityType::RELU>;                       \
  template class v<float, popnn::NonLinearityType::TANH>;                      \
  template class v<half, popnn::NonLinearityType::TANH>;                       \
  template class v<float, popnn::NonLinearityType::GELU>;                      \
  template class v<half, popnn::NonLinearityType::GELU>;                       \
  template class v<float, popnn::NonLinearityType::GELU_ERF>;                  \
  template class v<half, popnn::NonLinearityType::GELU_ERF>;                   \
  template class v<float, popnn::NonLinearityType::SWISH>;                     \
  template class v<half, popnn::NonLinearityType::SWISH>;

namespace popnn {

/****************************************************************************/
/*            Auxiliary math functions                                      */
/****************************************************************************/
static float sigmoid(float x) { return (1.0f / (1.0f + exp(-x))); }

static float sigmoid_derivative(float activation) {
  return activation * (1.0f - activation);
}

static float relu(float x) {
  if (x > 0.0f)
    return x;
  return 0.0f;
}

static float swish(float x) { return x * sigmoid(x); }

static float swish_derivative(float x) {
  const auto sigm = sigmoid(x);
  return sigm * (1 + x * (1 - sigm));
}

static float relu_derivative(float activation) {
  if (activation > 0.0f)
    return 1.0f;
  return 0.0f;
}

static float tanh_derivative(float activation) {
  return 1.0f - activation * activation;
}

// This gives the factor in the computation of the approximation of the CDF of
// a normal distribution.
// The actual approximation is 0.5 * cdfFactorForNormalDist(x).
// Note that several approximations exists and this one is widely used in the
// ML community. Actual values can be computed using erf (or erfc)
static const float alphaPhi = 0.7978845608f;
static const float betaPhi = 0.044715f;

static float cdfFactorForNormalDist(float x) {
  return tanh(x * alphaPhi * (1 + betaPhi * x * x));
}

static float gelu_gradient(float x) {
  float tanhx = cdfFactorForNormalDist(x);
  float g =
      1 + tanhx +
      (1 - tanhx * tanhx) * x * (alphaPhi + 3 * x * x * alphaPhi * betaPhi);
  return 0.5f * g;
}

static constexpr float gelu_erf_gradient(float x) {
  constexpr float twoDivByDqrtOfPi = 1.1283791670955126f; // 2 / sqrt(PI)
  const float y = x * 0.7071067811865475f;                // x * 1/sqrt(2)
  return 0.5f * (1.0f + erf(y) + y * twoDivByDqrtOfPi * exp(-y * y));
}

static float nonlinearity(popnn::NonLinearityType t, float x) {
  switch (t) {
  case popnn::NonLinearityType::SIGMOID:
    return sigmoid(x);
  case popnn::NonLinearityType::RELU:
    return relu(x);
  case popnn::NonLinearityType::TANH:
    return tanh(x);
  case popnn::NonLinearityType::GELU:
    return 0.5f * x * (1 + cdfFactorForNormalDist(x));
  case popnn::NonLinearityType::GELU_ERF:
    return 0.5f * x * (1 + erf(x * 0.7071067811865475f));
  case popnn::NonLinearityType::SWISH:
    return swish(x);
  case popnn::NonLinearityType::HARD_SIGMOID:
  case popnn::NonLinearityType::SOFTMAX:
  case popnn::NonLinearityType::SOFTMAX_STABLE:
  case popnn::NonLinearityType::SOFTMAX_SCALED:
    assert(0 && "Non linearity not supported");
    return x;
  }
}

static float nonlinearity_derivative(popnn::NonLinearityType t,
                                     float activation) {
  switch (t) {
  case popnn::NonLinearityType::SIGMOID:
    return sigmoid_derivative(activation);
  case popnn::NonLinearityType::RELU:
    return relu_derivative(activation);
  case popnn::NonLinearityType::TANH:
    return tanh_derivative(activation);
  case popnn::NonLinearityType::GELU:
    return gelu_gradient(activation);
  case popnn::NonLinearityType::GELU_ERF:
    return gelu_erf_gradient(activation);
  case popnn::NonLinearityType::SWISH:
    return swish_derivative(activation);
  case popnn::NonLinearityType::HARD_SIGMOID:
  case popnn::NonLinearityType::SOFTMAX:
  case popnn::NonLinearityType::SOFTMAX_STABLE:
  case popnn::NonLinearityType::SOFTMAX_SCALED:
    assert(0 && "Non linearity not supported");
    return activation;
  }
}

} // namespace popnn
