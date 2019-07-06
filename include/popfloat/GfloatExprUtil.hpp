// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef _popfloat_gfloat_expr_util_hpp_
#define _popfloat_gfloat_expr_util_hpp_

#include <popfloat/GfloatExpr.hpp>

#include <poputil/VertexTemplates.hpp>

namespace popfloat {
namespace gfexpr {

std::string gfloatCastOpTypeToString(popfloat::gfexpr::GfloatCastOpType op);
std::string gfloatRoundTypeToString(popfloat::gfexpr::GfloatRoundType rmode);
std::string gfloatFormatTypeToString(GfloatFormatType fmt);
std::string GfloatSRDensityTypeToString(GfloatSRDensityType dist);
}
} // end namespace popfloat::gfexpr

// Specialize vertex template stringification for expr ops
namespace poputil {
template<>
  struct VertexTemplateToString<popfloat::gfexpr::GfloatCastOpType>{
  static std::string to_string(const popfloat::gfexpr::GfloatCastOpType &op) {
    return "popfloat::gfexpr::GfloatCastOpType::" +
    popfloat::gfexpr::gfloatCastOpTypeToString(op);
  }
};

template<>
  struct VertexTemplateToString<popfloat::gfexpr::GfloatRoundType>{
  static std::string to_string(const popfloat::gfexpr::GfloatRoundType &rmode) {
    return "popfloat::gfexpr::GfloatRoundType::" +
    popfloat::gfexpr::gfloatRoundTypeToString(rmode);
  }
};

template<>
struct VertexTemplateToString<popfloat::gfexpr::GfloatFormatType> {
  static std::string to_string(const popfloat::gfexpr::GfloatFormatType &fmt) {
    return "popfloat::gfexpr::GfloatFormatType::" +
           popfloat::gfexpr::gfloatFormatTypeToString(fmt);
  }
};

template<>
struct VertexTemplateToString<popfloat::gfexpr::GfloatSRDensityType> {
  static std::string to_string(
     const popfloat::gfexpr::GfloatSRDensityType &dist) {
    return "popfloat::gfexpr::GfloatSRDensityType::" +
           popfloat::gfexpr::GfloatSRDensityTypeToString(dist);
  }
};
} // end namespace poputil

#endif // _popfloat_gfloat_expr_hpp_
