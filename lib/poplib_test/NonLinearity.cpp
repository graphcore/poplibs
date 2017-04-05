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
  }
  POPLIB_UNREACHABLE();
}

void poplib_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array_ref<double, 2> array) {
  for (auto it = array.data(), end = array.data() + array.num_elements();
       it != end; ++it) {
    *it = ::nonLinearity(nonLinearityType, *it);
  }
}

void poplib_test::nonLinearity(NonLinearityType nonLinearityType,
                               boost::multi_array<double, 4> &array) {
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
