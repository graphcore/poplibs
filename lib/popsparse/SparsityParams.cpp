// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparsityParams.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"
#include <tuple>

#include "poplibs_support/StructHelper.hpp"

namespace poputil {

template <>
poplar::ProfileValue toProfileValue(const popsparse::dynamic::SparsityType &t) {
  switch (t) {
  case popsparse::dynamic::SparsityType::Element:
    return poplar::ProfileValue("Element");
  case popsparse::dynamic::SparsityType::Block:
    return poplar::ProfileValue("Block");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}

template <>
poplar::ProfileValue
toProfileValue(const popsparse::dynamic::SparsityStructure &t) {
  switch (t) {
  case popsparse::dynamic::SparsityStructure::Unstructured:
    return poplar::ProfileValue("Unstructured");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}

template <>
poplar::ProfileValue
toProfileValue(const popsparse::dynamic::SparsityParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"type", toProfileValue(t.type)});
  v.insert({"structure", toProfileValue(t.structure)});
  v.insert({"blockDimensions", toProfileValue(t.blockDimensions)});
  return v;
}
} // namespace poputil

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
