#include <poplib_test/NonLinearity.hpp>
#include <util/Compiler.hpp>

#include <algorithm>
#include <cmath>

using namespace popnn;

static double sigmoid(double x) {
  return (1.0 / (1.0 + exp(-x)));
}

static double nonLinearity(NonLinearityType nonLinearityType,
                           double x) {
  switch (nonLinearityType) {
  case NON_LINEARITY_SIGMOID:
    return sigmoid(x);
  case NON_LINEARITY_RELU:
    return std::max(0.0, x);
  case NON_LINEARITY_TANH:
    return tanh(x);
  case NON_LINEARITY_SOFTMAX:
    throw poplib_test::poplib_test_error("softmax not supported");
  }
  POPLIB_UNREACHABLE();
}

static void softmax(boost::multi_array_ref<double, 2> &array) {
  for (auto b = 0; b != array.shape()[0]; ++b) {
    double sum = 0;
    for (auto c = 0; c != array.shape()[1]; ++c) {
      sum += exp(array[b][c]);
    }
    for (auto c = 0; c != array.shape()[1]; ++c) {
      array[b][c] = exp(array[b][c]) / sum;
    }
  }
}

void poplib_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array_ref<double, 2> array) {
  if (nonLinearityType == NON_LINEARITY_SOFTMAX) {
    softmax(array);
  } else {
    for (auto it = array.data(), end = array.data() + array.num_elements();
         it != end; ++it) {
      *it = ::nonLinearity(nonLinearityType, *it);
    }
  }
}

void poplib_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array<double, 4> &array) {
  if (nonLinearityType == NON_LINEARITY_SOFTMAX) {
    throw poplib_test::poplib_test_error("softmax not supported for 4D tensor");
  }
  for (auto it = array.data(), end = array.data() + array.num_elements();
       it != end; ++it) {
    *it = ::nonLinearity(nonLinearityType, *it);
  }
}

static double nonLinearityDerivative(NonLinearityType nonLinearityType,
                                     double act) {
  switch (nonLinearityType) {
  case NON_LINEARITY_SIGMOID:
    return act * (1.0 - act);
  case NON_LINEARITY_RELU:
    return (act > 0) ? 1 : 0;
  case NON_LINEARITY_TANH:
    return 1 - act * act;
  case NON_LINEARITY_SOFTMAX:
    throw poplib_test::poplib_test_error("softmax derivative not implemented");
  }
  POPLIB_UNREACHABLE();
}

static double bwdNonLinearity(NonLinearityType nonLinearityType,
                              double delta, double act) {
  return delta * nonLinearityDerivative(nonLinearityType, act);
}

void poplib_test::bwdNonLinearity(
    NonLinearityType nonLinearityType,
    const boost::multi_array<double, 4> &activations,
    boost::multi_array<double, 4> &deltas) {
  assert(std::equal(activations.shape(), activations.shape() + 4,
                    deltas.shape()));
  auto actIt = activations.data();
  for (auto it = deltas.data(), end = deltas.data() + deltas.num_elements();
       it != end; ++it, ++actIt) {
    *it = ::bwdNonLinearity(nonLinearityType, *it, *actIt);
  }
}

void poplib_test::bwdNonLinearity(
    NonLinearityType nonLinearityType,
    const boost::multi_array<double, 2> &activations,
    boost::multi_array<double, 2> &deltas) {
  assert(std::equal(activations.shape(), activations.shape() + 2,
                    deltas.shape()));
  auto actIt = activations.data();
  for (auto it = deltas.data(), end = deltas.data() + deltas.num_elements();
       it != end; ++it, ++actIt) {
    *it = ::bwdNonLinearity(nonLinearityType, *it, *actIt);
  }
}
