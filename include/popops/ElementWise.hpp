 // Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_ElementWise_hpp
#define popops_ElementWise_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <popops/Expr.hpp>
#include <string>

namespace popops {

// Element-wise operations.

/** Map an expression across tensors.
 *
 *  \param graph   The graph to update.
 *  \param expr    The expression to map across the tensors. The placeholders
 *                 in the expressions will be substituted with corresponding
 *                 elements from the tensors in \ts.
 *  \param ts      The list of tensors to map the expression across.
 *                 If elements from these tensors are used in binary/ternary
 *                 operations in the expression the numpy-style broadcast rules
 *                 are used to match the shapes of the tensors (see
 *                 poputil::broadcastToMatch()).
 *  \param prog    The seqeuence to extend with the execution of the expression
 *                 evaluation.
 *  \param debugPrefix
 *                 A debug prefix to be added to debug strings in compute sets
 *                 and variables created by this function
 *  \param options A list of flags to pass to the expression evaluator.
 *
 *
 *  \returns A tensor containing the elements resulting from the application of
 *           the expression across the tensors.
 */
poplar::Tensor map(poplar::Graph &graph,
                   const expr::Expr &expr,
                   const std::vector<poplar::Tensor> &ts,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "",
                   const poplar::OptionFlags &options = {});

inline poplar::Tensor map(poplar::Graph &graph,
                          expr::UnaryOpType op, const poplar::Tensor &t,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOp(op, expr::_1), {t}, prog, debugPrefix,
             options);
}

inline poplar::Tensor map(poplar::Graph &graph, expr::BinaryOpType op,
                          const poplar::Tensor &a, const poplar::Tensor &b,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOp(op, expr::_1, expr::_2), {a, b}, prog,
             debugPrefix, options);
}

inline poplar::Tensor map(poplar::Graph &graph, expr::TernaryOpType op,
                          const poplar::Tensor &a, const poplar::Tensor &b,
                          const poplar::Tensor &c,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::TernaryOp(op, expr::_1, expr::_2, expr::_3),
             {a, b, c}, prog, debugPrefix, options);
}

/** Map an expression across tensors and assign it back to the first tensor
 *  given.
 *
 *  \param graph   The graph to update.
 *  \param expr    The expression to map across the tensors. The placeholders
 *                 in the expressions will be substituted with corresponding
 *                 elements from the tensors in \ts. The result of the
 *                 expression is then written to the elements of the first
 *                 tensor in \ts.
 *  \param ts      The list of tensors to map the expression across.
 *                 If elements from these tensors are used in binary/ternary
 *                 operations in the expression the numpy-style broadcast rules
 *                 are used to match the shapes of the tensors (see
 *                 poputil::broadcastToMatch()).
 *  \param prog    The seqeuence to extend with the execution of the expression
 *                 evaluation.
 *  \param debugPrefix
 *                 A debug prefix to be added to debug strings in compute sets
 *                 and variables created by this function
 *
 *
 *  \returns A tensor containing the elements resulting from the application of
 *           the expression across the tensors.
 */
void mapInPlace(poplar::Graph &graph,
                const expr::Expr &expr,
                const std::vector<poplar::Tensor> &ts,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix = "",
                const poplar::OptionFlags &options = {});

inline void mapInPlace(poplar::Graph &graph,
                       expr::UnaryOpType op, const poplar::Tensor &t,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOp(op, expr::_1), {t}, prog, debugPrefix,
             options);
}

inline void mapInPlace(poplar::Graph &graph, expr::BinaryOpType op,
                       const poplar::Tensor &a, const poplar::Tensor &b,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOp(op, expr::_1, expr::_2), {a, b}, prog,
             debugPrefix, options);
}

inline void mapInPlace(poplar::Graph &graph, expr::TernaryOpType op,
                       const poplar::Tensor &a, const poplar::Tensor &b,
                       const poplar::Tensor &c,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::TernaryOp(op, expr::_1, expr::_2, expr::_3),
             {a, b, c}, prog, debugPrefix, options);
}

#define POPLIBS_DEFINE_UNARY_OPERATOR_FN(fn, op) \
  inline \
  poplar::Tensor fn(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    return map(graph, expr::UnaryOpType::op, A, prog, debugPrefix, \
               options); \
  } \
  inline \
  void fn ## InPlace( \
      poplar::Graph &graph, \
      const poplar::Tensor &A, \
      poplar::program::Sequence &prog, \
      const std::string &debugPrefix = "", \
      const poplar::OptionFlags &options = {}) { \
    mapInPlace(graph, expr::UnaryOpType::op, A, prog, debugPrefix, \
               options); \
  }

POPLIBS_DEFINE_UNARY_OPERATOR_FN(abs, ABSOLUTE)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(bitwiseNot, BITWISE_NOT)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(ceil, CEIL)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(countLeadingZeros, COUNT_LEADING_ZEROS)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(cos, COS)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(exp, EXPONENT)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(expm1, EXPONENT_MINUS_ONE)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(floor, FLOOR)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(inv, INVERSE)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(log, LOGARITHM)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(log1p, LOGARITHM_ONE_PLUS)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(logicalNot, LOGICAL_NOT)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(neg, NEGATE)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(popcount, POPCOUNT)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(signum, SIGNUM)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(sin, SIN)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(tanh, TANH)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(round, ROUND)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(sqrt, SQRT)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(square, SQUARE)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(sigmoid, SIGMOID)
POPLIBS_DEFINE_UNARY_OPERATOR_FN(rsqrt, RSQRT)

inline
poplar::Tensor isFinite(poplar::Graph &graph,
                        const poplar::Tensor &A,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::IS_FINITE, A, prog, debugPrefix,
             options);
}

template <typename constType>
inline void checkTypes(poplar::Type elementType, constType constant) {
  if (elementType != poplar::equivalent_device_type<constType>().value) {
    throw std::runtime_error("Type mismatch between Binary op Tensor "
                             "and constant");
  }
}
template <>
inline void checkTypes<float>(poplar::Type elementType, float constant){
  if (elementType != poplar::FLOAT && elementType != poplar::HALF) {
    throw std::runtime_error("Type mismatch between Binary op Tensor "
                             "and constant");
  }
  return;
}
template <>
inline void checkTypes<double>(poplar::Type elementType, double constant){
  if (elementType != poplar::FLOAT && elementType != poplar::HALF) {
    throw std::runtime_error("Type mismatch between Binary op Tensor "
                             "and constant");
  }
  return;
}

#define POPLIBS_DEFINE_BINARY_OPERATOR_FN(fn, op) \
  inline \
  poplar::Tensor fn(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    const poplar::Tensor &B, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    return map(graph, expr::BinaryOpType::op, A, B, prog, debugPrefix, \
               options); \
  } \
  template <typename constType> \
  inline \
  poplar::Tensor fn(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    const constType B, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    checkTypes(A.elementType(), B); \
    const auto BTensor = graph.addConstant(A.elementType(), {}, B); \
    graph.setTileMapping(BTensor, 0); \
    return map(graph, expr::BinaryOpType::op, A, BTensor, prog, debugPrefix, \
               options); \
  } \
  template <typename constType> \
  inline \
  poplar::Tensor fn(poplar::Graph &graph, \
                    const constType A, \
                    const poplar::Tensor &B, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    checkTypes(B.elementType(), A); \
    const auto ATensor = graph.addConstant(B.elementType(), {}, A); \
    graph.setTileMapping(ATensor, 0); \
    return map(graph, expr::BinaryOpType::op, ATensor, B, prog, debugPrefix, \
               options); \
  } \
  inline \
  void fn ## InPlace(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    const poplar::Tensor &B, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    mapInPlace(graph, expr::BinaryOpType::op, A, B, prog, debugPrefix, \
               options); \
  } \
  template <typename constType> \
  inline \
  void fn ## InPlace(poplar::Graph &graph, \
                     const poplar::Tensor &A, \
                     const constType B, \
                     poplar::program::Sequence &prog, \
                     const std::string &debugPrefix = "", \
                     const poplar::OptionFlags &options = {}) { \
    checkTypes(A.elementType(), B); \
    const auto BTensor = graph.addConstant(A.elementType(), {}, B); \
    graph.setTileMapping(BTensor, 0); \
    mapInPlace(graph, expr::BinaryOpType::op, A, BTensor, prog, debugPrefix, \
               options); \
  }


POPLIBS_DEFINE_BINARY_OPERATOR_FN(add, ADD)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(atan2, ATAN2)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(bitwiseAnd, BITWISE_AND)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(bitwiseOr, BITWISE_OR)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(bitwiseXor, BITWISE_XOR)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(bitwiseXnor, BITWISE_XNOR)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(div, DIVIDE)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(eq, EQUAL)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(gteq, GREATER_THAN_EQUAL)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(gt, GREATER_THAN)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(invStdDevToVariance, INV_STD_DEV_TO_VARIANCE)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(lteq, LESS_THAN_EQUAL)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(logicalAnd, LOGICAL_AND)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(logicalOr, LOGICAL_OR)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(lt, LESS_THAN)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(max, MAXIMUM)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(min, MINIMUM)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(mul, MULTIPLY)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(neq, NOT_EQUAL)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(pow, POWER)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(rem, REMAINDER)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(shiftLeft, SHIFT_LEFT)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(shiftRight, SHIFT_RIGHT)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(shiftRightSignExtend,
                                  SHIFT_RIGHT_SIGN_EXTEND)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(sub, SUBTRACT)
POPLIBS_DEFINE_BINARY_OPERATOR_FN(varianceToInvStdDev, VARIANCE_TO_INV_STD_DEV)

#define POPLIBS_DEFINE_TERNARY_OPERATOR_FN(fn, op) \
  inline \
  poplar::Tensor fn(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    const poplar::Tensor &B, \
                    const poplar::Tensor &C, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    return map(graph, expr::TernaryOpType::op, A, B, C, prog, debugPrefix, \
               options); \
  } \
  inline \
  void fn ## InPlace(poplar::Graph &graph, \
                    const poplar::Tensor &A, \
                    const poplar::Tensor &B, \
                    const poplar::Tensor &C, \
                    poplar::program::Sequence &prog, \
                    const std::string &debugPrefix = "", \
                    const poplar::OptionFlags &options = {}) { \
    mapInPlace(graph, expr::TernaryOpType::op, A, B, C, prog, debugPrefix, \
               options); \
  }

POPLIBS_DEFINE_TERNARY_OPERATOR_FN(select, SELECT)
POPLIBS_DEFINE_TERNARY_OPERATOR_FN(clamp, CLAMP)

} // end namespace popops

#endif // popops_ElementWise_hpp
