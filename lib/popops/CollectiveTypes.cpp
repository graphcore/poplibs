// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "popops/CollectiveTypes.hpp"
#include "poputil/exceptions.hpp"

namespace popops {

std::istream &operator>>(std::istream &in, CollectiveOperator &op) {
  std::string opStr;
  in >> opStr;

  if (opStr == "ADD") {
    op = CollectiveOperator::ADD;
  } else if (opStr == "MUL") {
    op = CollectiveOperator::MUL;
  } else if (opStr == "MIN") {
    op = CollectiveOperator::MIN;
  } else if (opStr == "MAX") {
    op = CollectiveOperator::MAX;
  } else if (opStr == "LOGICAL_AND") {
    op = CollectiveOperator::LOGICAL_AND;
  } else if (opStr == "LOGICAL_OR") {
    op = CollectiveOperator::LOGICAL_OR;
  } else if (opStr == "SQUARE_ADD") {
    op = CollectiveOperator::SQUARE_ADD;
  } else if (opStr == "LOCAL") {
    op = CollectiveOperator::LOCAL;
  } else {
    throw poputil::poplibs_error("Unrecognised operation " + opStr);
  }

  return in;
}

std::ostream &operator<<(std::ostream &os, const CollectiveOperator &op) {
  switch (op) {
  case CollectiveOperator::ADD:
    return os << "ADD";
  case CollectiveOperator::MUL:
    return os << "MUL";
  case CollectiveOperator::MIN:
    return os << "MIN";
  case CollectiveOperator::MAX:
    return os << "MAX";
  case CollectiveOperator::LOGICAL_AND:
    return os << "LOGICAL_AND";
  case CollectiveOperator::LOGICAL_OR:
    return os << "LOGICAL_OR";
  case CollectiveOperator::SQUARE_ADD:
    return os << "SQUARE_ADD";
  case CollectiveOperator::LOCAL:
    return os << "LOCAL";
  }
  throw poputil::poplibs_error("Unrecognised operation.");
}

CollectiveOperator operationToCollectiveOperator(const Operation &col) {
  switch (col) {
  case Operation::ADD:
    return CollectiveOperator::ADD;
  case Operation::MUL:
    return CollectiveOperator::MUL;
  case Operation::MIN:
    return CollectiveOperator::MIN;
  case Operation::MAX:
    return CollectiveOperator::MAX;
  case Operation::LOGICAL_AND:
    return CollectiveOperator::LOGICAL_AND;
  case Operation::LOGICAL_OR:
    return CollectiveOperator::LOGICAL_OR;
  case Operation::SQUARE_ADD:
    return CollectiveOperator::SQUARE_ADD;
  }
  throw poputil::poplibs_error("Unrecognised operation.");
}
} // namespace popops