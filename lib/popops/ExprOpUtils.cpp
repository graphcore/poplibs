// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "popops/ExprOpUtils.hpp"
#include "ExprOpUtil.hpp"

#include <iostream>

namespace popops::expr {
namespace ostream_ext {

std::ostream &operator<<(std::ostream &os, const UnaryOpType &t) {
  os << unaryOpTypeToString(t);
  return os;
}

std::ostream &operator<<(std::ostream &os, const BinaryOpType &t) {
  os << binaryOpTypeToString(t);
  return os;
}

std::ostream &operator<<(std::ostream &os, const TernaryOpType &t) {
  os << ternaryOpTypeToString(t);
  return os;
}

} // end namespace ostream_ext
} // end namespace popops::expr
