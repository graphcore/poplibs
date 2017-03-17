#ifndef popnn_ref_NonLinearity_hpp_
#define popnn_ref_NonLinearity_hpp_

#include "popnn/NonLinearityDef.hpp"
#include "popnn/Compiler.hpp"
#include "popnn_ref/exceptions.hpp"
#include <boost/multi_array.hpp>

namespace ref {

const char *asString(const NonLinearityType &type) {
  switch (type) {
  case NON_LINEARITY_RELU: return "relu";
  case NON_LINEARITY_SIGMOID: return "sigmoid";
  }
  POPNN_UNREACHABLE();
}

inline std::ostream &operator<<(std::ostream &os,
                                const NonLinearityType &type) {
  return os << asString(type);
}

inline std::istream &operator>>(std::istream &in, NonLinearityType &type) {
  std::string token;
  in >> token;
  if (token == "relu")
    type = NON_LINEARITY_RELU;
  else if (token == "sigmoid")
    type = NON_LINEARITY_SIGMOID;
  else
    throw popnn_ref::popnn_ref_error(
        "Unsupported nonlinearity <" + token + ">");

  return in;
}

void nonLinearity(NonLinearityType nonLinearityType,
                  boost::multi_array<double, 4> &array);

void bwdNonLinearity(NonLinearityType nonLinearityType,
                     const boost::multi_array<double, 4> &activations,
                     boost::multi_array<double, 4> &deltas);

} // End namespace ref.

#endif  // popnn_ref_NonLinearity_hpp_
