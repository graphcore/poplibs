#ifndef _poplib_test_NonLinearity_hpp_
#define _poplib_test_NonLinearity_hpp_

#include "popnn/NonLinearity.hpp"
#include "util/Compiler.hpp"
#include "poplib_test/exceptions.hpp"
#include <boost/multi_array.hpp>

namespace poplib_test {
inline const char *asString(const popnn::NonLinearityType &type) {
  switch (type) {
  case popnn::NonLinearityType::NON_LINEARITY_RELU: return "relu";
  case popnn::NonLinearityType::NON_LINEARITY_SIGMOID: return "sigmoid";
  case popnn::NonLinearityType::NON_LINEARITY_TANH: return "tanh";
  case popnn::NonLinearityType::NON_LINEARITY_SOFTMAX: return "softmax";
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
    type = popnn::NonLinearityType::NON_LINEARITY_RELU;
  else if (token == "sigmoid")
    type = popnn::NonLinearityType::NON_LINEARITY_SIGMOID;
  else if (token == "tanh")
    type = popnn::NonLinearityType::NON_LINEARITY_TANH;
  else if (token == "softmax") {
    type = popnn::NonLinearityType::NON_LINEARITY_SOFTMAX;
  } else
    throw poplib_test::poplib_test_error(
        "Unsupported nonlinearity <" + token + ">");

  return in;
}

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array_ref<double, 2> array);

void nonLinearity(popnn::NonLinearityType nonLinearityType,
                  boost::multi_array<double, 4> &array);

void bwdNonLinearity(popnn::NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 4> &activations,
                     boost::multi_array<double, 4> &deltas);

} // End namespace poplib_test.

#endif  // _poplib_test_NonLinearity_hpp_
