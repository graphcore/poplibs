// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef _popfloat_gfloat_expr_util_hpp_
#define _popfloat_gfloat_expr_util_hpp_

#include <experimental/popfloat/GfloatExpr.hpp>

#include <poputil/VertexTemplates.hpp>

namespace experimental {
namespace popfloat {

std::string roundTypeToString(RoundType rmode);
std::string formatTypeToString(FormatType fmt);
std::string srDensityTypeToString(SRDensityType dist);
poplar::Type specTypeToPoplarType(SpecType specType);

} // end namespace popfloat
} // end namespace experimental

// Specialize vertex template stringification for expr ops
namespace poputil {
template<>
  struct VertexTemplateToString<experimental::popfloat::RoundType>{
  static std::string to_string(const experimental::popfloat::RoundType &rmode) {
    return "experimental::popfloat::RoundType::" +
    experimental::popfloat::roundTypeToString(rmode);
  }
};

template<>
struct VertexTemplateToString<experimental::popfloat::FormatType> {
  static std::string to_string(const experimental::popfloat::FormatType &fmt) {
    return "experimental::popfloat::FormatType::" +
      experimental::popfloat::formatTypeToString(fmt);
  }
};

template<>
struct VertexTemplateToString<experimental::popfloat::SRDensityType> {
  static std::string to_string(
     const experimental::popfloat::SRDensityType &dist) {
    return "experimental::popfloat::SRDensityType::" +
           experimental::popfloat::srDensityTypeToString(dist);
  }
};
} // end namespace poputil

#endif // _popfloat_gfloat_expr_hpp_
