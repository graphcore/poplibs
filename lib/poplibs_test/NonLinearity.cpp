#include <poplibs_test/NonLinearity.hpp>
#include <poplibs_support/Compiler.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

using namespace popnn;

static double sigmoid(double x) {
  return (1.0 / (1.0 + exp(-x)));
}

static double nonLinearity(NonLinearityType nonLinearityType,
                           double x) {
  switch (nonLinearityType) {
  case NonLinearityType::SIGMOID:
    return sigmoid(x);
  case NonLinearityType::RELU:
    return std::max(0.0, x);
  case NonLinearityType::TANH:
    return tanh(x);
  case NonLinearityType::SOFTMAX:
  case NonLinearityType::SOFTMAX_STABLE:
  case NonLinearityType::SOFTMAX_SCALED:
    throw poplibs_test::poplibs_test_error("softmax not supported");
  }
  POPLIB_UNREACHABLE();
}

static void softmax(boost::multi_array_ref<double, 2> &array,
                    bool stableAlgo) {
  for (auto b = 0u; b != array.shape()[0]; ++b) {
    if (stableAlgo) {
      double max = std::numeric_limits<double>::min();
      for (auto c = 0u; c != array.shape()[1]; ++c) {
        max = std::max(max, array[b][c]);
      }
      for (auto c = 0u; c != array.shape()[1]; ++c) {
        array[b][c] -= max;
      }
    }
    double sum = 0;
    for (auto c = 0u; c != array.shape()[1]; ++c) {
      sum += exp(array[b][c]);
    }
    for (auto c = 0u; c != array.shape()[1]; ++c) {
      array[b][c] = exp(array[b][c]) / sum;
    }
  }
}

// Calculate the gradient at the input to a softmax operation from
// the output from the operation and the gradient at that output.
//
// out - input: output from a softmax operation
// outGradient - input:  gradient at output of the softmax operation.
//               output: gradient at input of the softmax operation.
static void softmaxGradient(const boost::multi_array_ref<double, 2> &out,
                            boost::multi_array_ref<double, 2> &outGradient) {
  for (unsigned n = 0; n < out.size(); ++n) {
    double sumgXy = 0.0;
    for (unsigned i = 0; i < out[n].size(); ++i) {
      outGradient[n][i] = out[n][i] * outGradient[n][i];
      sumgXy += outGradient[n][i];
    }
    for (unsigned i = 0; i < out[n].size(); ++i) {
      outGradient[n][i] = outGradient[n][i] + sumgXy * -out[n][i];
    }
  }
}

void poplibs_test::nonLinearity(NonLinearityType nonLinearityType,
                                const double *inputData, double *outputData,
                                std::size_t n) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE ||
      nonLinearityType == NonLinearityType::SOFTMAX_SCALED) {
    throw poplibs_test::poplibs_test_error("softmax not supported, use "
                                           "shaped functions instead");
  }
  while (n-- > 0) {
    *outputData++ = ::nonLinearity(nonLinearityType, *inputData++);
  }
}

void poplibs_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array_ref<double, 2> array) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE ||
      nonLinearityType == NonLinearityType::SOFTMAX_SCALED) {
    bool stableAlgo = nonLinearityType == NonLinearityType::SOFTMAX_STABLE;
    softmax(array, stableAlgo);
  } else {
    nonLinearity(nonLinearityType,
                 array.data(), array.data(),
                 array.num_elements());
  }
}

void poplibs_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array<double, 4> &array) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE ||
      nonLinearityType == NonLinearityType:: SOFTMAX_SCALED) {
    throw poplibs_test::poplibs_test_error("softmax not supported for 4D "
                                           "tensor");
  }
  nonLinearity(nonLinearityType,
               array.data(), array.data(),
               array.num_elements());
}

static double nonLinearityDerivative(NonLinearityType nonLinearityType,
                                     double act) {
  switch (nonLinearityType) {
  case NonLinearityType::SIGMOID:
    return act * (1.0 - act);
  case NonLinearityType::RELU:
    return (act > 0) ? 1 : 0;
  case NonLinearityType::TANH:
    return 1 - act * act;
  case NonLinearityType::SOFTMAX:
  case NonLinearityType::SOFTMAX_STABLE:
  case NonLinearityType::SOFTMAX_SCALED:
    throw poplibs_test::poplibs_test_error("softmax derivative not "
                                           "implemented");
  }
  POPLIB_UNREACHABLE();
}

static double bwdNonLinearity(NonLinearityType nonLinearityType,
                              double delta, double act) {
  return delta * nonLinearityDerivative(nonLinearityType, act);
}

void poplibs_test::bwdNonLinearity(
  NonLinearityType nonLinearityType,
  const double *activations, double *deltas,
  std::size_t n) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE ||
      nonLinearityType == NonLinearityType::SOFTMAX_SCALED) {
    throw poplibs_test::poplibs_test_error("softmax not supported, use "
                                           "shaped functions instead");
  }
  while (n-- > 0) {
    *deltas = ::bwdNonLinearity(nonLinearityType, *deltas, *activations++);
    ++deltas;
  }
}

void poplibs_test::bwdNonLinearity(
    NonLinearityType nonLinearityType,
    const boost::multi_array<double, 4> &activations,
    boost::multi_array<double, 4> &deltas) {
  assert(std::equal(activations.shape(), activations.shape() + 4,
                    deltas.shape()));
  bwdNonLinearity(nonLinearityType,
                  activations.data(), deltas.data(),
                  deltas.num_elements());
}

void poplibs_test::bwdNonLinearity(
    NonLinearityType nonLinearityType,
    const boost::multi_array<double, 2> &activations,
    boost::multi_array<double, 2> &deltas) {
  assert(std::equal(activations.shape(), activations.shape() + 2,
                    deltas.shape()));
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE ||
      nonLinearityType == NonLinearityType::SOFTMAX_SCALED ) {
    softmaxGradient(activations, deltas);
  } else {
    bwdNonLinearity(nonLinearityType,
                    activations.data(), deltas.data(),
                    deltas.num_elements());
  }
}
