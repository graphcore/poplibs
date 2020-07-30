// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparseDenseMatMulBlock
#include <poplibs_support/TestDevice.hpp>

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

#include "SparseDenseUtils.hpp"

using namespace poplibs_support;
using namespace poputil;

std::ostream &operator<<(std::ostream &os, const VertexType &vt) {
  switch (vt) {
  case VertexType::Forward:
    os << "Forward";
    break;
  case VertexType::GradA:
    os << "GradA";
    break;
  case VertexType::Transposed:
    os << "Transposed";
    break;
  case VertexType::GradW:
    os << "GradW";
    break;
  default:
    throw poplibs_error(
        "Unrecognised vertex type " +
        std::to_string(std::underlying_type<VertexType>::type(vt)));
  }
  return os;
}

std::istream &operator>>(std::istream &is, VertexType &vt) {
  std::string token;
  is >> token;
  if (token == "Forward") {
    vt = VertexType::Forward;
  } else if (token == "GradA") {
    vt = VertexType::GradA;
  } else if (token == "Transposed") {
    vt = VertexType::Transposed;
  } else if (token == "GradW") {
    vt = VertexType::GradW;
  } else {
    throw poplibs_error("Unrecognised vertex type '" + token + "'");
  }
  return is;
}
