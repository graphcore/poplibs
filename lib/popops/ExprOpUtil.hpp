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

}} // end namespace popops::expr

// Specialize vertex template stringification for expr ops
namespace poputil {
template <>
struct VertexTemplateToString<popops::expr::UnaryOpType> {
  static std::string to_string(const popops::expr::UnaryOpType &op) {
    return "popops::expr::UnaryOpType::" +
            popops::expr::unaryOpTypeToString(op);
  }
};
template <>
struct VertexTemplateToString<popops::expr::BinaryOpType> {
  static std::string to_string(const popops::expr::BinaryOpType &op) {
    return "popops::expr::BinaryOpType::" +
            popops::expr::binaryOpTypeToString(op);
  }
};
template <>
struct VertexTemplateToString<popops::expr::TernaryOpType> {
  static std::string to_string(const popops::expr::TernaryOpType &op) {
    return "popops::expr::TernaryOpType::" +
            popops::expr::ternaryOpTypeToString(op);
  }
};
template <>
struct VertexTemplateToString<popops::expr::BroadcastOpType> {
  static std::string to_string(const popops::expr::BroadcastOpType &op) {
    return "popops::expr::BroadcastOpType::" +
            popops::expr::broadcastOpTypeToString(op);
  }
};

} // end namespace poputil

#endif // poplibs_ExprOpUtil_hpp_
