// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplibs_ExprOpUtil_hpp_
#define poplibs_ExprOpUtil_hpp_
#include <popops/ExprOp.hpp>
#include <poputil/VertexTemplates.hpp>
#include <string>

namespace popops {
namespace expr {

std::string unaryOpTypeToString(UnaryOpType op);
std::string binaryOpTypeToString(BinaryOpType op);
std::string ternaryOpTypeToString(TernaryOpType op);
std::string broadcastOpTypeToString(BroadcastOpType op);

// Is this binary op a special case. I.E doesn't conform to either function
// semantics of OP(arg1, arg2) or the normal arg1 OP arg2, and as such should be
// handled separately.
bool isSpecialCase(BinaryOpType op);
std::string handleSpecialCase(BinaryOpType op, const std::string &param1,
                              const std::string &param2);

// Get the C++ code representation of this operation.
poplar::StringRef getBinaryOpAsString(BinaryOpType op, poplar::Type t);

// Should this operation be processed as OP(arg1, arg2)?
bool hasFunctionSemantics(BinaryOpType op);

// Get the return type of the operation given the type of the input. We assume
// the type of arg1 and arg2 are the same.
poplar::Type getReturnType(BinaryOpType op,
                           const std::pair<std::string, poplar::Type> &lhs,
                           const std::pair<std::string, poplar::Type> &rhs);

// Is this a bitwise operation.
bool isBitwiseOperation(BinaryOpType op);

// We assume most unary ops can be handled like functions, I.E OP(arg) so if the
// operation doesn't meet that semantic it is a special case which must be
// handled separately.
bool isSpecialCase(UnaryOpType op);

// Whether or not this operation supports vector types as only some unary
// operations support vectorization.
bool supportsVectorization(UnaryOpType op);

std::string handleSpecialCase(UnaryOpType op, const std::string &param1);

// Get the return type of the operation given the type of the input.
poplar::Type getReturnType(UnaryOpType op, poplar::Type inType);

// Get the c++ code representation of this operation.
poplar::StringRef getUnaryOpAsString(UnaryOpType op, poplar::Type type);

} // namespace expr
} // namespace popops

// Specialize vertex template stringification for expr ops
namespace poputil {
template <> struct VertexTemplateToString<popops::expr::UnaryOpType> {
  static std::string to_string(const popops::expr::UnaryOpType &op) {
    return "popops::expr::UnaryOpType::" +
           popops::expr::unaryOpTypeToString(op);
  }
};
template <> struct VertexTemplateToString<popops::expr::BinaryOpType> {
  static std::string to_string(const popops::expr::BinaryOpType &op) {
    return "popops::expr::BinaryOpType::" +
           popops::expr::binaryOpTypeToString(op);
  }
};
template <> struct VertexTemplateToString<popops::expr::TernaryOpType> {
  static std::string to_string(const popops::expr::TernaryOpType &op) {
    return "popops::expr::TernaryOpType::" +
           popops::expr::ternaryOpTypeToString(op);
  }
};
template <> struct VertexTemplateToString<popops::expr::BroadcastOpType> {
  static std::string to_string(const popops::expr::BroadcastOpType &op) {
    return "popops::expr::BroadcastOpType::" +
           popops::expr::broadcastOpTypeToString(op);
  }
};

} // end namespace poputil

#endif // poplibs_ExprOpUtil_hpp_
