// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/Operation.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/exceptions.hpp"
#include <iostream>
#include <sstream>
#include <string>

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popops::Operation &op) {

  std::stringstream ss;
  ss << op;
  return poplar::ProfileValue(ss.str());
}
} // namespace poputil

namespace popops {

std::istream &operator>>(std::istream &in, Operation &op) {
  std::string opStr;
  in >> opStr;

  if (opStr == "ADD") {
    op = Operation::ADD;
  } else if (opStr == "MUL") {
    op = Operation::MUL;
  } else if (opStr == "MIN") {
    op = Operation::MIN;
  } else if (opStr == "MAX") {
    op = Operation::MAX;
  } else if (opStr == "LOGICAL_AND") {
    op = Operation::LOGICAL_AND;
  } else if (opStr == "LOGICAL_OR") {
    op = Operation::LOGICAL_OR;
  } else if (opStr == "SQUARE_ADD") {
    op = Operation::SQUARE_ADD;
  } else {
    throw poputil::poplibs_error("Unrecognised operation " + opStr);
  }

  return in;
}

std::ostream &operator<<(std::ostream &os, const Operation &op) {
  switch (op) {
  case Operation::ADD:
    os << "ADD";
    break;
  case Operation::MUL:
    os << "MUL";
    break;
  case Operation::MIN:
    os << "MIN";
    break;
  case Operation::MAX:
    os << "MAX";
    break;
  case Operation::LOGICAL_AND:
    os << "LOGICAL_AND";
    break;
  case Operation::LOGICAL_OR:
    os << "LOGICAL_OR";
    break;
  case Operation::SQUARE_ADD:
    os << "SQUARE_ADD";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised operation.");
  }
  return os;
}

} // End namespace popops
