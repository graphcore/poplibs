// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poplibs_NonLinearityDefUtil_hpp_
#define poplibs_NonLinearityDefUtil_hpp_

#include <popnn/NonLinearityDef.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

namespace popnn {

inline const char *asString(const popnn::NonLinearityType &type) {
  switch (type) {
  case popnn::NonLinearityType::RELU:
    return "relu";
  case popnn::NonLinearityType::SIGMOID:
    return "sigmoid";
  case popnn::NonLinearityType::TANH:
    return "tanh";
  case popnn::NonLinearityType::GELU:
    return "gelu";
  case popnn::NonLinearityType::SOFTMAX:
    return "softmax";
  case popnn::NonLinearityType::SOFTMAX_STABLE:
    return "softmax (stable)";
  case popnn::NonLinearityType::SOFTMAX_SCALED:
    return "softmax (scaled, stable)";
  default:
    throw poputil::poplibs_error("Unsupported non-linearity type");
  }
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
  else if (token == "gelu")
    type = popnn::NonLinearityType::GELU;
  else if (token == "softmax")
    type = popnn::NonLinearityType::SOFTMAX;
  else if (token == "softmax_stable")
    type = popnn::NonLinearityType::SOFTMAX_STABLE;
  else if (token == "softmax_scaled")
    type = popnn::NonLinearityType::SOFTMAX_SCALED;
  else
    throw poputil::poplibs_error("Unsupported non-linearity type \'" + token +
                                 "\'");
  return in;
}

} // end namespace popnn

namespace poputil {

/// Specialise vertex template stringification for non-linearity type.
template <> struct VertexTemplateToString<popnn::NonLinearityType> {
  static std::string to_string(const popnn::NonLinearityType &nlType) {
    switch (nlType) {
    case popnn::NonLinearityType::SIGMOID:
      return "popnn::NonLinearityType::SIGMOID";
    case popnn::NonLinearityType::RELU:
      return "popnn::NonLinearityType::RELU";
    case popnn::NonLinearityType::TANH:
      return "popnn::NonLinearityType::TANH";
    case popnn::NonLinearityType::GELU:
      return "popnn::NonLinearityType::GELU";
    case popnn::NonLinearityType::SOFTMAX:
    case popnn::NonLinearityType::SOFTMAX_STABLE:
    case popnn::NonLinearityType::SOFTMAX_SCALED:
    default:
      throw poputil::poplibs_error("Unsupported non-linearity type");
    }
  }
};

} // end namespace poputil

#endif // poplibs_NonLinearityDefUtil_hpp_
