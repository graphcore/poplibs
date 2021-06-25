// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Utilities for Operation types
 *
 */

#ifndef popops_OperationDefUtil_hpp
#define popops_OperationDefUtil_hpp

#include <poplibs_support/Compiler.hpp>
#include <popops/Operation.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

namespace poputil {

inline const char *asString(const popops::Operation &op) {
  switch (op) {
  case popops::Operation::ADD:
    return "Add";
  case popops::Operation::MUL:
    return "Mul";
  case popops::Operation::MIN:
    return "Min";
  case popops::Operation::MAX:
    return "Max";
  case popops::Operation::LOGICAL_AND:
    return "Logical And";
  case popops::Operation::LOGICAL_OR:
    return "Logical Or";
  case popops::Operation::SQUARE_ADD:
    return "Square Add";
  case popops::Operation::LOG_ADD:
    return "Log Add";
  }
  throw poputil::poplibs_error("Unsupported operation type");
}

/// Specialise vertex template stringification for operation type.
template <> struct VertexTemplateToString<popops::Operation> {
  static std::string to_string(const popops::Operation &op) {
    switch (op) {
    case popops::Operation::ADD:
      return "popops::Operation::ADD";
    case popops::Operation::MUL:
      return "popops::Operation::MUL";
    case popops::Operation::MIN:
      return "popops::Operation::MIN";
    case popops::Operation::MAX:
      return "popops::Operation::MAX";
    case popops::Operation::LOGICAL_AND:
      return "popops::Operation::LOGICAL_AND";
    case popops::Operation::LOGICAL_OR:
      return "popops::Operation::LOGICAL_OR";
    case popops::Operation::SQUARE_ADD:
      return "popops::Operation::SQUARE_ADD";
    case popops::Operation::LOG_ADD:
      return "popops::Operation::LOG_ADD";
    }
    POPLIB_UNREACHABLE();
  }
};

} // end namespace poputil
#endif // popops_OperationDefUtil_hpp
