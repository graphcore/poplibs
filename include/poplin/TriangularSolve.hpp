// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Solving linear equations using triangular matrices.
 *
 */

#ifndef popops_TriangularSolve_hpp
#define popops_TriangularSolve_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poplin {

namespace matmul {
class PlanningCache;
}

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
 *  \param debugPrefix    A debug prefix added to compute set and tensor
 *                        names.
 *  \returns              A tensor with the same shape as \p a with all unused
 *                        values masked.
 */
poplar::Tensor triangularMask(poplar::Graph &graph, const poplar::Tensor &a,
                              bool lower, bool unitDiagonal,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "");

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
 *  \param blockSize      Block size for blocked solver.
 *  \param prog           A reference to a program sequence which the code
 *                        to perform the arrangement will be appended to.
 *  \param debugPrefix    A debug prefix added to compute set and tensor
 *                        names.
 *  \param options        A structure describing options on how the
 *                        multiplication should be implemented.
 *                        See matMul() for details.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              Tensor with shape of \p b with linear system solution.
 */
poplar::Tensor triangularSolve(
    poplar::Graph &graph, const poplar::Tensor &a, const poplar::Tensor &b,
    bool leftSide, bool lower, bool unitDiagonal, std::size_t blockSize,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    poplar::OptionFlags options = {}, matmul::PlanningCache *cache = nullptr);

} // namespace poplin

#endif // poplin_TriangularSolve_hpp
