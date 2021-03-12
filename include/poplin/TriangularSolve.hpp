// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Solving linear equations using triangular matrices.
 *
 */

#ifndef poplin_TriangularSolve_hpp
#define poplin_TriangularSolve_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplin/MatMul.hpp>

namespace poplin {

/**
 * Create a tensor that is used as the left operand of triangular solve.
 *
 * This will create a 2D/3D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a triangular solver with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the triangular solver.
 *                        Supported options:
 *                          'blockSize' - blockSize hint
 *                        See matMul() for additional options.
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p aShape. The
 *                        tensor will have been mapped to tiles.
 */

poplar::Tensor createTriangularSolveInputLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options = {},
    matmul::PlanningCache *cache = nullptr);

/**
 * \deprecated Create a tensor that is used as the left operand of triangular
 * solve.
 *
 * This will create a 2D/3D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a triangular solver with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param blockSize       Block size for blocked solver.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the triangular solver.
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p aShape. The
 *                        tensor will have been mapped to tiles.
 */

poplar::Tensor createTriangularSolveInputLHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    std::size_t blockSize, const poplar::DebugContext &debugContext,
    poplar::OptionFlags options = {}, matmul::PlanningCache *cache = nullptr);

/**
 * Create a tensor that is used as the right operand of triangular solve.
 *
 * This will create a 2D/3D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a triangular solver with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the triangular solver.
 *                        Supported options:
 *                          'blockSize' - blockSize hint
 *                        See matMul() for additional options.
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p bShape. The
 *                        tensor will have been mapped to tiles.
 */

poplar::Tensor createTriangularSolveInputRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options = {},
    matmul::PlanningCache *cache = nullptr);

/**
 * \deprecated Create a tensor that is used as the right operand of triangular
 * solve.
 *
 * This will create a 2D/3D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a triangular solver with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param inputType       The input data type.
 * \param outputType      The data type of the returned tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param blockSize       Block size for blocked solver.
 * \param debugContext    Debug information.
 * \param options         The implementation options of the triangular solver.
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p bShape. The
 *                        tensor will have been mapped to tiles.
 */

poplar::Tensor createTriangularSolveInputRHS(
    poplar::Graph &graph, const poplar::Type &inputType,
    const poplar::Type &outputType, const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide,
    std::size_t blockSize, const poplar::DebugContext &debugContext,
    poplar::OptionFlags options, matmul::PlanningCache *cache = nullptr);

/**
 * Masks the unused components of the input tensor with zeroes, optionally
 * allowing for a unit diagonal.
 *
 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param lower          Whether to use the upper or lower triangle of \p a.
 *  \param unitDiagonal   If true, the diagonal elements of \p a are assumed to
 *                        be 1 and not accessed.
 *  \param prog           A reference to a program sequence which the code
 *                        to perform the arrangement will be appended to.
 *  \param debugContext   Optional debug information.
 *  \returns              A tensor with the same shape as \p a with all unused
 *                        values masked.
 */
poplar::Tensor triangularMask(poplar::Graph &graph, const poplar::Tensor &a,
                              bool lower, bool unitDiagonal,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext = {});

/**
 * Solves systems of linear equations with lower or upper triangular
 * coefficients.
 *
 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param b              Tensor of the same type with shape [...,  N, K] if
 *                        left_side is true, [...,K, N] otherwise.
 *  \param leftSide       Solve AX = B if true, XA = B overwise.
 *  \param lower          Use the upper or lower triangle of \p a.
 *  \param unitDiagonal   If true, the diagonal elements of \p a are assumed to
 *                        be 1 and not accessed.
 *  \param prog           A reference to a program sequence which the code
 *                        to perform the arrangement will be appended to.
 *  \param debugContext   Optional debug information.
 *  \param options        The implementation options of the triangular solver.
 *                        Supported options:
 *                          'blockSize' - blockSize hint
 *                        See matMul() for additional options.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              Tensor with shape of \p b with linear system solution.
 */
poplar::Tensor triangularSolve(poplar::Graph &graph, const poplar::Tensor &a,
                               const poplar::Tensor &b, bool leftSide,
                               bool lower, bool unitDiagonal,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {},
                               const poplar::OptionFlags &options = {},
                               matmul::PlanningCache *cache = nullptr);

/**
 * \deprecated Solves systems of linear equations with lower or upper triangular
 * coefficients.
 *
 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param b              Tensor of the same type with shape [...,  N, K] if
 *                        left_side is true, [...,K, N] otherwise.
 *  \param leftSide       Solve AX = B if true, XA = B overwise.
 *  \param lower          Use the upper or lower triangle of \p a.
 *  \param unitDiagonal   If true, the diagonal elements of \p a are assumed to
 *                        be 1 and not accessed.
 *  \param blockSize      Block size for blocked solver.
 *  \param prog           A reference to a program sequence which the code
 *                        to perform the arrangement will be appended to.
 *  \param debugContext   Optional debug information.
 *  \param options        A structure describing options on how the
 *                        triangular solver should be implemented.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              Tensor with shape of \p b with linear system solution.
 */

poplar::Tensor triangularSolve(poplar::Graph &graph, const poplar::Tensor &a,
                               const poplar::Tensor &b, bool leftSide,
                               bool lower, bool unitDiagonal,
                               std::size_t blockSize,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {},
                               poplar::OptionFlags options = {},
                               matmul::PlanningCache *cache = nullptr);

/**
 * Plan matrix multiplication for given triangular solver
 *
 * \param inputType       The data type of the lhs tensor.
 * \param outputType      The data type of the rhs tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param options         The implementation options of the triangular solver.
 *  \returns              Mat mul preplan parameters.
 */

std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getTriangularSolveMatMulPrePlanParameters(
    const poplar::Type &inputType, const poplar::Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide, bool lower,
    const poplar::OptionFlags &options);

/**
 * \deprecated Plan matrix multiplication for given triangular solver
 *
 * \param inputType       The data type of the lhs tensor.
 * \param outputType      The data type of the rhs tensor.
 * \param aShape          The shape of the left operand.
 * \param bShape          The shape of the right operand.
 * \param leftSide        Solve AX = B if true, XA = B overwise.
 * \param blockSize       Block size for blocked solver.
 * \param options         The implementation options of the triangular solver.
 *  \returns              Mat mul preplan parameters.
 */
std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getTriangularSolveMatMulPrePlanParameters(
    const poplar::Type &inputType, const poplar::Type &outputType,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, bool leftSide, bool lower,
    std::size_t blockSize, poplar::OptionFlags options);

} // namespace poplin

#endif // poplin_TriangularSolve_hpp
