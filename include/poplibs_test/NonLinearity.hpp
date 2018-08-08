// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_NonLinearity_hpp
#define poplibs_test_NonLinearity_hpp

#include "popnn/NonLinearity.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_test/exceptions.hpp"
#include <boost/multi_array.hpp>

namespace poplibs_test {
inline const char *asString(const popnn::NonLinearityType &type) {
  switch (type) {
  case popnn::NonLinearityType::RELU: return "relu";
  case popnn::NonLinearityType::SIGMOID: return "sigmoid";
  case popnn::NonLinearityType::TANH: return "tanh";
  case popnn::NonLinearityType::SOFTMAX: return "softmax";
  }
  POPLIB_UNREACHABLE();
}

inline std::ostream &operator<<(std::ostream &os,
                                const popnn::NonLinearityType &type) {
  return os << asString(type);
}

inline std::istream &operator>>(std::istream &in,
                                popnn::NonLinearityType &type) {
  std::string token;
  in >> token;
  if (token == "relu")
    type = popnn::NonLinearityType::RELU;
  else if (token == "sigmoid")
    type = popnn::NonLinearityType::SIGMOID;
  else if (token == "tanh")
    type = popnn::NonLinearityType::TANH;
  else if (token == "softmax")
    type = popnn::NonLinearityType::SOFTMAX;
  else
    throw poplibs_test::poplibs_test_error(
        "Unsupported nonlinearity <" + token + ">");

  return in;
}

// input/output can be pointers to same memory
void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  const double *inputData, double *outputData,
                  std::size_t n);

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array_ref<double, 2> array);

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array<double, 4> &array);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const double *activations, double *deltas,
                     std::size_t n);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 4> &activations,
                     boost::multi_array<double, 4> &deltas);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 2> &activations,
                     boost::multi_array<double, 2> &deltas);

} // End namespace poplibs_test.

#endif // poplibs_test_NonLinearity_hpp
