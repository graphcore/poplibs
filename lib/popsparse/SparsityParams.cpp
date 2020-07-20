// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparsityParams.hpp"
#include "poputil/exceptions.hpp"
#include <tuple>

#include "poplibs_support/StructHelper.hpp"

namespace popsparse {
namespace dynamic {

std::ostream &operator<<(std::ostream &os, const SparsityType &t) {
  switch (t) {
  case SparsityType::Element:
    os << "Element";
    break;
  case SparsityType::Block:
    os << "Block";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised SparsityType");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const SparsityStructure &s) {
  switch (s) {
  case SparsityStructure::Unstructured:
    os << "Unstructured";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised SparsityStructure");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const SparsityParams &p) {
  os << "{type: " << p.type << ", structure: " << p.structure;
  if (p.type == SparsityType::Block) {
    os << ", block dimensions: [" << p.blockDimensions[0] << ",";
    os << p.blockDimensions[1] << "] ";
  }
  os << "}";
  return os;
}

static constexpr auto comparisonHelper = poplibs_support::makeStructHelper(
    &SparsityParams::type, &SparsityParams::structure,
    &SparsityParams::blockDimensions);

bool operator<(const SparsityParams &a, const SparsityParams &b) {
  return comparisonHelper.lt(a, b);
}

bool operator==(const SparsityParams &a, const SparsityParams &b) {
  return comparisonHelper.eq(a, b);
}

bool operator!=(const SparsityParams &a, const SparsityParams &b) {
  return !(a == b);
}

} // end namespace dynamic
} // end namespace popsparse
