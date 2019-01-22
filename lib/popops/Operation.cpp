#include "popops/Operation.hpp"

#include "poputil/exceptions.hpp"
#include <iostream>
#include <string>

namespace popops {

std::istream &operator>>(std::istream &in, Operation &op) {
  std::string opStr;
  in >> opStr;

  if (opStr == "add") {
    op = Operation::ADD;
  } else if (opStr == "mul") {
    op = Operation::MUL;
  } else if (opStr == "min") {
    op = Operation::MIN;
  } else if (opStr == "max") {
    op = Operation::MAX;
  } else if (opStr == "logical-and") {
    op = Operation::LOGICAL_AND;
  } else if (opStr == "logical-or") {
    op = Operation::LOGICAL_OR;
  } else if (opStr == "square-add") {
    op = Operation::SQUARE_ADD;
  } else {
    throw poputil::poplibs_error("Unrecognised operation " + opStr);
  }

  return in;

}

std::ostream &operator<<(std::ostream &os, const Operation &op) {
  switch(op) {
  case Operation::ADD:
    os << "add";
    break;
  case Operation::MUL:
    os << "mul";
    break;
  case Operation::MIN:
    os << "min";
    break;
  case Operation::MAX:
    os << "max";
    break;
  case Operation::LOGICAL_AND:
    os << "logical-and";
    break;
  case Operation::LOGICAL_OR:
    os << "logical-or";
    break;
  case Operation::SQUARE_ADD:
    os << "square-add";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised operation.");
  }
  return os;
}

} // End namespace popops
