// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popops_ElementWise_hpp
#define popops_ElementWise_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <popops/Expr.hpp>
#include <string>

namespace popops {

// Element-wise operations.

/* Variance conversion operations can be created using the map functions
 * below, but that requires the input and output to be of the same type.
 * It can be an advantage to maintain variance in full precision and
 * inverse standard deviation in half precision.  These supplementary functions
 * make that possible.
 *
 *  \param graph   The graph to update.
 *  \param src     The source Tensor
 *  \param epsilon A tensor initialised with the epsilon parameter used in
 *                 conversion.  Must have a single element and have the same
 *                 type as the input type.  Alternatively a float value can be
 *                 used and the appropriate tensor will be created.
 *  \param prog    The sequence to extend with the execution of conversion.
 *  \param dstType The type of the tensor to be output. Must be FLOAT when
 *                 outputting variance, HALF whe outputting standard deviation,
 *                 or equal to the input type.
 *  \param debugPrefix
 *                 A debug prefix to be added to debug strings in compute sets
 *                 and variables created by this function.
 *
 *  \returns       A tensor containing the elements resulting from the
 *                 variance to/from standard deviation conversion.
 */
poplar::Tensor varianceToInvStdDev(poplar::Graph &graph,
                                   const poplar::Tensor &src,
                                   const poplar::Tensor &epsilon,
                                   poplar::program::Sequence &prog,
                                   const poplar::Type dstType = poplar::HALF,
                                   const std::string &debugPrefix = "");

poplar::Tensor invStdDevToVariance(poplar::Graph &graph,
                                   const poplar::Tensor &src,
                                   const poplar::Tensor &epsilon,
                                   poplar::program::Sequence &prog,
                                   const poplar::Type dstType = poplar::FLOAT,
                                   const std::string &debugPrefix = "");

poplar::Tensor varianceToInvStdDev(poplar::Graph &graph,
                                   const poplar::Tensor &src,
                                   const float epsilon,
                                   poplar::program::Sequence &prog,
                                   const poplar::Type dstType = poplar::HALF,
                                   const std::string &debugPrefix = "");

poplar::Tensor invStdDevToVariance(poplar::Graph &graph,
                                   const poplar::Tensor &src,
                                   const float epsilon,
                                   poplar::program::Sequence &prog,
                                   const poplar::Type dstType = poplar::FLOAT,
                                   const std::string &debugPrefix = "");

/** Map an expression across tensors.
 *
 * **Element Wise Options**
 *
 *    * `enableGenerateCodelet` (true, false) [=true]
 *
 *      If true (and all of the inputs are the same size and do not alias), a
 *      codelet is generated to execute this map operation. A codelet will not
 *      be generated if there is only a single operation unless
 *      `forceGenerateCodelet` is true.
 */
/*[INTERNAL]
 *    * `enableVectorBroadcastOptimisations` (true, false) [=true]
 *
 *      This option is only applicable if `expr` is a binary operation. If true,
 *      vector broadcasts are optimised by attempting to find the most efficient
 *      way to perform the binary operation on each tile.
 *
 *    * `forceGenerateCodelet` (true, false) [=false]
 *
 *      See `enableGenerateCodelet`. Intended for testing only.
 */
/**
 *  \param graph   The graph to update.
 *  \param expr    The expression to map across the tensors. The placeholders
 *                 in the expressions will be substituted with corresponding
 *                 elements from the tensors in \p ts.
 *  \param ts      The list of tensors to map the expression across.
 *                 If elements from these tensors are used in binary/ternary
 *                 operations in the expression the numpy-style broadcast rules
 *                 are used to match the shapes of the tensors (see
 *                 poputil::broadcastToMatch()).
 *  \param prog    The sequence to extend with the execution of the expression
 *                 evaluation.
 *  \param debugPrefix
 *                 A debug prefix to be added to debug strings in compute sets
 *                 and variables created by this function
 *  \param options A list of flags to pass to the expression evaluator.
 *
 *  \returns A tensor containing the elements resulting from the application of
 *           the expression across the tensors.
 */
poplar::Tensor map(poplar::Graph &graph, const expr::Expr &expr,
                   const std::vector<poplar::Tensor> &ts,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "",
                   const poplar::OptionFlags &options = {});

inline poplar::Tensor map(poplar::Graph &graph, expr::UnaryOpType op,
                          const poplar::Tensor &t,
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
 *                 elements from the tensors in \p ts. The result of the
 *                 expression is then written to the elements of the first
 *                 tensor in \p ts.
 *  \param ts      The list of tensors to map the expression across.
 *                 If elements from these tensors are used in binary/ternary
 *                 operations in the expression the numpy-style broadcast rules
 *                 are used to match the shapes of the tensors (see
 *                 poputil::broadcastToMatch()).
 *  \param prog    The sequence to extend with the execution of the expression
 *                 evaluation.
 *  \param debugPrefix
 *                 A debug prefix to be added to debug strings in compute sets
 *                 and variables created by this function
 * \param options  Element wise options. See map().
 *
 *  \returns A tensor containing the elements resulting from the application of
 *           the expression across the tensors.
 */
void mapInPlace(poplar::Graph &graph, const expr::Expr &expr,
                const std::vector<poplar::Tensor> &ts,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix = "",
                const poplar::OptionFlags &options = {});

inline void mapInPlace(poplar::Graph &graph, expr::UnaryOpType op,
                       const poplar::Tensor &t, poplar::program::Sequence &prog,
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
                       const poplar::Tensor &c, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::TernaryOp(op, expr::_1, expr::_2, expr::_3),
             {a, b, c}, prog, debugPrefix, options);
}

// Unary operations

inline poplar::Tensor abs(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::ABSOLUTE, A, prog, debugPrefix, options);
}
inline void absInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::ABSOLUTE, A, prog, debugPrefix, options);
}
inline poplar::Tensor asin(poplar::Graph &graph, const poplar::Tensor &A,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::ASIN, A, prog, debugPrefix, options);
}
inline void asinInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::ASIN, A, prog, debugPrefix, options);
}
inline poplar::Tensor bitwiseNot(poplar::Graph &graph, const poplar::Tensor &A,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::BITWISE_NOT, A, prog, debugPrefix,
             options);
}
inline void bitwiseNotInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::BITWISE_NOT, A, prog, debugPrefix,
             options);
}
inline poplar::Tensor ceil(poplar::Graph &graph, const poplar::Tensor &A,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::CEIL, A, prog, debugPrefix, options);
}
inline void ceilInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::CEIL, A, prog, debugPrefix, options);
}
inline poplar::Tensor
countLeadingZeros(poplar::Graph &graph, const poplar::Tensor &A,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "",
                  const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::COUNT_LEADING_ZEROS, A, prog,
             debugPrefix, options);
}
inline void countLeadingZerosInPlace(poplar::Graph &graph,
                                     const poplar::Tensor &A,
                                     poplar::program::Sequence &prog,
                                     const std::string &debugPrefix = "",
                                     const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::COUNT_LEADING_ZEROS, A, prog,
             debugPrefix, options);
}
inline poplar::Tensor cos(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::COS, A, prog, debugPrefix, options);
}
inline void cosInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::COS, A, prog, debugPrefix, options);
}
inline poplar::Tensor exp(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::EXPONENT, A, prog, debugPrefix, options);
}
inline void expInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::EXPONENT, A, prog, debugPrefix, options);
}
inline poplar::Tensor expm1(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::EXPONENT_MINUS_ONE, A, prog, debugPrefix,
             options);
}
inline void expm1InPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::EXPONENT_MINUS_ONE, A, prog, debugPrefix,
             options);
}
inline poplar::Tensor floor(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::FLOOR, A, prog, debugPrefix, options);
}
inline void floorInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::FLOOR, A, prog, debugPrefix, options);
}
inline poplar::Tensor inv(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::INVERSE, A, prog, debugPrefix, options);
}
inline void invInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::INVERSE, A, prog, debugPrefix, options);
}
inline poplar::Tensor log(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::LOGARITHM, A, prog, debugPrefix,
             options);
}
inline void logInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::LOGARITHM, A, prog, debugPrefix,
             options);
}
inline poplar::Tensor log1p(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::LOGARITHM_ONE_PLUS, A, prog, debugPrefix,
             options);
}
inline void log1pInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::LOGARITHM_ONE_PLUS, A, prog, debugPrefix,
             options);
}
inline poplar::Tensor logicalNot(poplar::Graph &graph, const poplar::Tensor &A,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::LOGICAL_NOT, A, prog, debugPrefix,
             options);
}
inline void logicalNotInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::LOGICAL_NOT, A, prog, debugPrefix,
             options);
}
inline poplar::Tensor neg(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::NEGATE, A, prog, debugPrefix, options);
}
inline void negInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::NEGATE, A, prog, debugPrefix, options);
}
inline poplar::Tensor popcount(poplar::Graph &graph, const poplar::Tensor &A,
                               poplar::program::Sequence &prog,
                               const std::string &debugPrefix = "",
                               const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::POPCOUNT, A, prog, debugPrefix, options);
}
inline void popcountInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::POPCOUNT, A, prog, debugPrefix, options);
}
inline poplar::Tensor signum(poplar::Graph &graph, const poplar::Tensor &A,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::SIGNUM, A, prog, debugPrefix, options);
}
inline void signumInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::SIGNUM, A, prog, debugPrefix, options);
}
inline poplar::Tensor sin(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::SIN, A, prog, debugPrefix, options);
}
inline void sinInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::SIN, A, prog, debugPrefix, options);
}
inline poplar::Tensor tan(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::TAN, A, prog, debugPrefix, options);
}
inline void tanInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::TAN, A, prog, debugPrefix, options);
}
inline poplar::Tensor tanh(poplar::Graph &graph, const poplar::Tensor &A,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::TANH, A, prog, debugPrefix, options);
}
inline void tanhInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::TANH, A, prog, debugPrefix, options);
}
inline poplar::Tensor round(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::ROUND, A, prog, debugPrefix, options);
}
inline void roundInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::ROUND, A, prog, debugPrefix, options);
}
inline poplar::Tensor sqrt(poplar::Graph &graph, const poplar::Tensor &A,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::SQRT, A, prog, debugPrefix, options);
}
inline void sqrtInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::SQRT, A, prog, debugPrefix, options);
}
inline poplar::Tensor square(poplar::Graph &graph, const poplar::Tensor &A,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::SQUARE, A, prog, debugPrefix, options);
}
inline void squareInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::SQUARE, A, prog, debugPrefix, options);
}
inline poplar::Tensor sigmoid(poplar::Graph &graph, const poplar::Tensor &A,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::SIGMOID, A, prog, debugPrefix, options);
}
inline void sigmoidInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::SIGMOID, A, prog, debugPrefix, options);
}
inline poplar::Tensor rsqrt(poplar::Graph &graph, const poplar::Tensor &A,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::UnaryOpType::RSQRT, A, prog, debugPrefix, options);
}
inline void rsqrtInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::UnaryOpType::RSQRT, A, prog, debugPrefix, options);
}

inline poplar::Tensor isFinite(poplar::Graph &graph, const poplar::Tensor &A,
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
inline void checkTypes<float>(poplar::Type elementType, float constant) {
  if (elementType != poplar::FLOAT && elementType != poplar::HALF) {
    throw std::runtime_error("Type mismatch between Binary op Tensor "
                             "and constant");
  }
  return;
}
template <>
inline void checkTypes<double>(poplar::Type elementType, double constant) {
  if (elementType != poplar::FLOAT && elementType != poplar::HALF) {
    throw std::runtime_error("Type mismatch between Binary op Tensor "
                             "and constant");
  }
  return;
}

// Binary operations

inline poplar::Tensor add(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::ADD, A, B, prog, debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor add(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::ADD, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
add(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::ADD, ATensor, B, prog, debugPrefix,
             options);
}
inline void addInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::ADD, A, B, prog, debugPrefix, options);
}
template <typename constType>
inline void addInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::ADD, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor atan2(poplar::Graph &graph, const poplar::Tensor &A,
                            const poplar::Tensor &B,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::ATAN2, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor atan2(poplar::Graph &graph, const poplar::Tensor &A,
                            const constType B, poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::ATAN2, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
atan2(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
      poplar::program::Sequence &prog, const std::string &debugPrefix = "",
      const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::ATAN2, ATensor, B, prog, debugPrefix,
             options);
}
inline void atan2InPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         const poplar::Tensor &B,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::ATAN2, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void atan2InPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         const constType B, poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::ATAN2, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor bitwiseAnd(poplar::Graph &graph, const poplar::Tensor &A,
                                 const poplar::Tensor &B,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::BITWISE_AND, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
bitwiseAnd(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_AND, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
bitwiseAnd(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_AND, ATensor, B, prog,
             debugPrefix, options);
}
inline void bitwiseAndInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const poplar::Tensor &B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::BITWISE_AND, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void bitwiseAndInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const constType B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::BITWISE_AND, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor bitwiseOr(poplar::Graph &graph, const poplar::Tensor &A,
                                const poplar::Tensor &B,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix = "",
                                const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::BITWISE_OR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
bitwiseOr(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_OR, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
bitwiseOr(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_OR, ATensor, B, prog,
             debugPrefix, options);
}
inline void bitwiseOrInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::BITWISE_OR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void bitwiseOrInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const constType B, poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::BITWISE_OR, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor bitwiseXor(poplar::Graph &graph, const poplar::Tensor &A,
                                 const poplar::Tensor &B,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::BITWISE_XOR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
bitwiseXor(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_XOR, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
bitwiseXor(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_XOR, ATensor, B, prog,
             debugPrefix, options);
}
inline void bitwiseXorInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const poplar::Tensor &B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::BITWISE_XOR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void bitwiseXorInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const constType B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::BITWISE_XOR, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor bitwiseXnor(poplar::Graph &graph, const poplar::Tensor &A,
                                  const poplar::Tensor &B,
                                  poplar::program::Sequence &prog,
                                  const std::string &debugPrefix = "",
                                  const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::BITWISE_XNOR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor bitwiseXnor(poplar::Graph &graph, const poplar::Tensor &A,
                                  const constType B,
                                  poplar::program::Sequence &prog,
                                  const std::string &debugPrefix = "",
                                  const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_XNOR, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor bitwiseXnor(poplar::Graph &graph, const constType A,
                                  const poplar::Tensor &B,
                                  poplar::program::Sequence &prog,
                                  const std::string &debugPrefix = "",
                                  const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::BITWISE_XNOR, ATensor, B, prog,
             debugPrefix, options);
}
inline void bitwiseXnorInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                               const poplar::Tensor &B,
                               poplar::program::Sequence &prog,
                               const std::string &debugPrefix = "",
                               const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::BITWISE_XNOR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void bitwiseXnorInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                               const constType B,
                               poplar::program::Sequence &prog,
                               const std::string &debugPrefix = "",
                               const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::BITWISE_XNOR, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor div(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::DIVIDE, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor div(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::DIVIDE, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
div(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::DIVIDE, ATensor, B, prog, debugPrefix,
             options);
}
inline void divInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::DIVIDE, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void divInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::DIVIDE, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor eq(poplar::Graph &graph, const poplar::Tensor &A,
                         const poplar::Tensor &B,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::EQUAL, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor eq(poplar::Graph &graph, const poplar::Tensor &A,
                         const constType B, poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::EQUAL, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
eq(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
   poplar::program::Sequence &prog, const std::string &debugPrefix = "",
   const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::EQUAL, ATensor, B, prog, debugPrefix,
             options);
}
inline void eqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::EQUAL, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void eqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const constType B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::EQUAL, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor gteq(poplar::Graph &graph, const poplar::Tensor &A,
                           const poplar::Tensor &B,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::GREATER_THAN_EQUAL, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor gteq(poplar::Graph &graph, const poplar::Tensor &A,
                           const constType B, poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::GREATER_THAN_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
gteq(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
     poplar::program::Sequence &prog, const std::string &debugPrefix = "",
     const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::GREATER_THAN_EQUAL, ATensor, B, prog,
             debugPrefix, options);
}
inline void gteqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        const poplar::Tensor &B,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::GREATER_THAN_EQUAL, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline void gteqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        const constType B, poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::GREATER_THAN_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor gt(poplar::Graph &graph, const poplar::Tensor &A,
                         const poplar::Tensor &B,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::GREATER_THAN, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor gt(poplar::Graph &graph, const poplar::Tensor &A,
                         const constType B, poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::GREATER_THAN, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
gt(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
   poplar::program::Sequence &prog, const std::string &debugPrefix = "",
   const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::GREATER_THAN, ATensor, B, prog,
             debugPrefix, options);
}
inline void gtInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::GREATER_THAN, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void gtInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const constType B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::GREATER_THAN, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor
invStdDevToVariance(poplar::Graph &graph, const poplar::Tensor &A,
                    const poplar::Tensor &B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
invStdDevToVariance(poplar::Graph &graph, const poplar::Tensor &A,
                    const constType B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE, A, BTensor,
             prog, debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
invStdDevToVariance(poplar::Graph &graph, const constType A,
                    const poplar::Tensor &B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE, ATensor, B,
             prog, debugPrefix, options);
}
inline void invStdDevToVarianceInPlace(
    poplar::Graph &graph, const poplar::Tensor &A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline void
invStdDevToVarianceInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                           const constType B, poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE, A, BTensor,
             prog, debugPrefix, options);
}

inline poplar::Tensor lteq(poplar::Graph &graph, const poplar::Tensor &A,
                           const poplar::Tensor &B,
                           poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::LESS_THAN_EQUAL, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor lteq(poplar::Graph &graph, const poplar::Tensor &A,
                           const constType B, poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::LESS_THAN_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
lteq(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
     poplar::program::Sequence &prog, const std::string &debugPrefix = "",
     const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::LESS_THAN_EQUAL, ATensor, B, prog,
             debugPrefix, options);
}
inline void lteqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        const poplar::Tensor &B,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::LESS_THAN_EQUAL, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline void lteqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                        const constType B, poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::LESS_THAN_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor logicalAnd(poplar::Graph &graph, const poplar::Tensor &A,
                                 const poplar::Tensor &B,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::LOGICAL_AND, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
logicalAnd(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::LOGICAL_AND, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
logicalAnd(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::LOGICAL_AND, ATensor, B, prog,
             debugPrefix, options);
}
inline void logicalAndInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const poplar::Tensor &B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::LOGICAL_AND, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void logicalAndInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const constType B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::LOGICAL_AND, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor logicalOr(poplar::Graph &graph, const poplar::Tensor &A,
                                const poplar::Tensor &B,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix = "",
                                const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::LOGICAL_OR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
logicalOr(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::LOGICAL_OR, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
logicalOr(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::LOGICAL_OR, ATensor, B, prog,
             debugPrefix, options);
}
inline void logicalOrInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::LOGICAL_OR, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void logicalOrInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const constType B, poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::LOGICAL_OR, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor lt(poplar::Graph &graph, const poplar::Tensor &A,
                         const poplar::Tensor &B,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::LESS_THAN, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor lt(poplar::Graph &graph, const poplar::Tensor &A,
                         const constType B, poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::LESS_THAN, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
lt(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
   poplar::program::Sequence &prog, const std::string &debugPrefix = "",
   const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::LESS_THAN, ATensor, B, prog,
             debugPrefix, options);
}
inline void ltInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const poplar::Tensor &B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::LESS_THAN, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void ltInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                      const constType B, poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::LESS_THAN, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor max(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::MAXIMUM, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor max(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::MAXIMUM, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
max(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::MAXIMUM, ATensor, B, prog, debugPrefix,
             options);
}
inline void maxInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::MAXIMUM, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void maxInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::MAXIMUM, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor min(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::MINIMUM, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor min(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::MINIMUM, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
min(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::MINIMUM, ATensor, B, prog, debugPrefix,
             options);
}
inline void minInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::MINIMUM, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void minInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::MINIMUM, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor mul(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::MULTIPLY, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor mul(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::MULTIPLY, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
mul(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::MULTIPLY, ATensor, B, prog, debugPrefix,
             options);
}
inline void mulInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::MULTIPLY, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void mulInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::MULTIPLY, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor neq(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::NOT_EQUAL, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor neq(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::NOT_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
neq(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::NOT_EQUAL, ATensor, B, prog,
             debugPrefix, options);
}
inline void neqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::NOT_EQUAL, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void neqInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::NOT_EQUAL, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor pow(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::POWER, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor pow(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::POWER, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
pow(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::POWER, ATensor, B, prog, debugPrefix,
             options);
}
inline void powInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::POWER, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void powInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::POWER, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor rem(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::REMAINDER, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor rem(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::REMAINDER, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
rem(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::REMAINDER, ATensor, B, prog,
             debugPrefix, options);
}
inline void remInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::REMAINDER, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void remInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::REMAINDER, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor shiftLeft(poplar::Graph &graph, const poplar::Tensor &A,
                                const poplar::Tensor &B,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix = "",
                                const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::SHIFT_LEFT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
shiftLeft(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_LEFT, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
shiftLeft(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "",
          const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_LEFT, ATensor, B, prog,
             debugPrefix, options);
}
inline void shiftLeftInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::SHIFT_LEFT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void shiftLeftInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                             const constType B, poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::SHIFT_LEFT, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor shiftRight(poplar::Graph &graph, const poplar::Tensor &A,
                                 const poplar::Tensor &B,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
shiftRight(poplar::Graph &graph, const poplar::Tensor &A, const constType B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT, A, BTensor, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
shiftRight(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
           poplar::program::Sequence &prog, const std::string &debugPrefix = "",
           const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT, ATensor, B, prog,
             debugPrefix, options);
}
inline void shiftRightInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const poplar::Tensor &B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::SHIFT_RIGHT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void shiftRightInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                              const constType B,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::SHIFT_RIGHT, A, BTensor, prog,
             debugPrefix, options);
}

inline poplar::Tensor
shiftRightSignExtend(poplar::Graph &graph, const poplar::Tensor &A,
                     const poplar::Tensor &B, poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "",
                     const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
shiftRightSignExtend(poplar::Graph &graph, const poplar::Tensor &A,
                     const constType B, poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "",
                     const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, A, BTensor,
             prog, debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
shiftRightSignExtend(poplar::Graph &graph, const constType A,
                     const poplar::Tensor &B, poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "",
                     const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, ATensor, B,
             prog, debugPrefix, options);
}
inline void shiftRightSignExtendInPlace(
    poplar::Graph &graph, const poplar::Tensor &A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline void
shiftRightSignExtendInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                            const constType B, poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::SHIFT_RIGHT_SIGN_EXTEND, A, BTensor,
             prog, debugPrefix, options);
}

inline poplar::Tensor sub(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::SUBTRACT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor sub(poplar::Graph &graph, const poplar::Tensor &A,
                          const constType B, poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::SUBTRACT, A, BTensor, prog, debugPrefix,
             options);
}
template <typename constType>
inline poplar::Tensor
sub(poplar::Graph &graph, const constType A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::SUBTRACT, ATensor, B, prog, debugPrefix,
             options);
}
inline void subInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const poplar::Tensor &B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::SUBTRACT, A, B, prog, debugPrefix,
             options);
}
template <typename constType>
inline void subInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                       const constType B, poplar::program::Sequence &prog,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::SUBTRACT, A, BTensor, prog, debugPrefix,
             options);
}

inline poplar::Tensor
varianceToInvStdDev(poplar::Graph &graph, const poplar::Tensor &A,
                    const poplar::Tensor &B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  return map(graph, expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
varianceToInvStdDev(poplar::Graph &graph, const poplar::Tensor &A,
                    const constType B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  return map(graph, expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV, A, BTensor,
             prog, debugPrefix, options);
}
template <typename constType>
inline poplar::Tensor
varianceToInvStdDev(poplar::Graph &graph, const constType A,
                    const poplar::Tensor &B, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {}) {
  checkTypes(B.elementType(), A);
  const auto ATensor = graph.addConstant(B.elementType(), {}, A);
  graph.setTileMapping(ATensor, 0);
  return map(graph, expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV, ATensor, B,
             prog, debugPrefix, options);
}

inline void varianceToInvStdDevInPlace(
    poplar::Graph &graph, const poplar::Tensor &A, const poplar::Tensor &B,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV, A, B, prog,
             debugPrefix, options);
}
template <typename constType>
inline void
varianceToInvStdDevInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                           const constType B, poplar::program::Sequence &prog,
                           const std::string &debugPrefix = "",
                           const poplar::OptionFlags &options = {}) {
  checkTypes(A.elementType(), B);
  const auto BTensor = graph.addConstant(A.elementType(), {}, B);
  graph.setTileMapping(BTensor, 0);
  mapInPlace(graph, expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV, A, BTensor,
             prog, debugPrefix, options);
}

// Ternary operations

inline poplar::Tensor select(poplar::Graph &graph, const poplar::Tensor &A,
                             const poplar::Tensor &B, const poplar::Tensor &C,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             const poplar::OptionFlags &options = {}) {
  return map(graph, expr::TernaryOpType::SELECT, A, B, C, prog, debugPrefix,
             options);
}
inline void selectInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                          const poplar::Tensor &B, const poplar::Tensor &C,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "",
                          const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::TernaryOpType::SELECT, A, B, C, prog, debugPrefix,
             options);
}
inline poplar::Tensor clamp(poplar::Graph &graph, const poplar::Tensor &A,
                            const poplar::Tensor &B, const poplar::Tensor &C,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "",
                            const poplar::OptionFlags &options = {}) {
  return map(graph, expr::TernaryOpType::CLAMP, A, B, C, prog, debugPrefix,
             options);
}
inline void clampInPlace(poplar::Graph &graph, const poplar::Tensor &A,
                         const poplar::Tensor &B, const poplar::Tensor &C,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "",
                         const poplar::OptionFlags &options = {}) {
  mapInPlace(graph, expr::TernaryOpType::CLAMP, A, B, C, prog, debugPrefix,
             options);
}

} // end namespace popops

#endif // popops_ElementWise_hpp
