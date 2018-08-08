// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_NonLinearityDefUtil_hpp_
#define poplibs_NonLinearityDefUtil_hpp_

#include <popnn/NonLinearityDef.hpp>
#include <poputil/exceptions.hpp>
#include <poputil/VertexTemplates.hpp>

// Specialize vertex template stringification for non-linearity type.
namespace poputil {

template <>
struct VertexTemplateToString<popnn::NonLinearityType> {
  static std::string to_string(const popnn::NonLinearityType &nlType) {
    switch (nlType) {
      case popnn::NonLinearityType::SIGMOID:
        return "popnn::NonLinearityType::SIGMOID";
      case popnn::NonLinearityType::RELU:
        return "popnn::NonLinearityType::RELU";
      case popnn::NonLinearityType::TANH:
        return "popnn::NonLinearityType::TANH";
      case popnn::NonLinearityType::SOFTMAX:
      default:
        throw poputil::poplib_error("Unsupported non-linearity type");
    }
  }
};

} // end namespace poputil

#endif // poplibs_NonLinearityDefUtil_hpp_
