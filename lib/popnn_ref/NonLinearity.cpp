#include <popnn_ref/NonLinearity.hpp>

#include <algorithm>
#include <cmath>

static double sigmoid(double x) {
  return (1.0 / (1.0 + exp(-x)));
}

void ref::fwdNonLinearity(NonLinearityType nonLinearityType,
                          boost::multi_array<double, 3> &array) {
  if (nonLinearityType == NON_LINEARITY_NONE)
    return;
  for (auto it = array.data(), end = array.data() + array.num_elements();
       it != end; ++it) {
    switch (nonLinearityType) {
    case NON_LINEARITY_NONE:
      break;
    case NON_LINEARITY_SIGMOID:
      *it = sigmoid(*it);
      break;
    case NON_LINEARITY_RELU:
      *it = std::max(0.0, *it);
      break;
    }
  }
}
