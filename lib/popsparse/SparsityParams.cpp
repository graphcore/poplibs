// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/SparsityParams.hpp"
#include "poputil/exceptions.hpp"
#include <tuple>

namespace popsparse {
namespace dynamic {

std::ostream &operator<<(std::ostream &os, const SparsityType &t) {
  switch (t) {
  case SparsityType::Element:
    os << "Element";
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
  os << "{type: " << p.type << ", structure: " << p.structure << "}";
  return os;
}

bool operator<(const SparsityParams &a, const SparsityParams &b) {
  return std::tie(a.type, a.structure) < std::tie(b.type, b.structure);
}

} // end namespace dynamic
} // end namespace popsparse
