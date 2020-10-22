// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplar/AvailableVTypes.h"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popnn/NonLinearity.hpp"
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
#ifdef VECTOR_AVAIL_SCALED_PTR32
static constexpr auto SCALED_PTR32 = poplar::VectorLayout::SCALED_PTR32;
#endif
#ifdef VECTOR_AVAIL_SCALED_PTR64
static constexpr auto SCALED_PTR64 = poplar::VectorLayout::SCALED_PTR64;
#endif
#ifdef VECTORLIST_AVAIL_DELTAN
static constexpr auto DELTAN = poplar::VectorListLayout::DELTAN;
#endif
static constexpr auto DELTANELEMENTS = poplar::VectorListLayout::DELTANELEMENTS;

// Macros to instantiate a template class for non linear operations
#define INSTANTIATE_NL(v)                                                      \
  template class v<float, popnn::NonLinearityType::GELU>;                      \
  template class v<half, popnn::NonLinearityType::GELU>;

#define INSTANTIATE_NL_GRAD(v)                                                 \
  template class v<float, popnn::NonLinearityType::SIGMOID>;                   \
  template class v<half, popnn::NonLinearityType::SIGMOID>;                    \
  template class v<float, popnn::NonLinearityType::RELU>;                      \
  template class v<half, popnn::NonLinearityType::RELU>;                       \
  template class v<float, popnn::NonLinearityType::TANH>;                      \
  template class v<half, popnn::NonLinearityType::TANH>;                       \
  template class v<float, popnn::NonLinearityType::GELU>;                      \
  template class v<half, popnn::NonLinearityType::GELU>;

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
  case popnn::NonLinearityType::SOFTMAX:
  case popnn::NonLinearityType::SOFTMAX_STABLE:
  case popnn::NonLinearityType::SOFTMAX_SCALED:
    assert(0 && "Non linearity not supported");
    return activation;
  }
}

} // namespace popnn
