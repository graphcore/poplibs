// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popops_TriangularSolve_hpp
#define popops_TriangularSolve_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poplin {

namespace matmul {
class PlanningCache;
}

/**
Masks unused components of input tensor with zeroes, optionally providing
unit diagonal if required.

 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param lower          Whether to use the upper or lower triangle of a.
 *  \param unitDiagonal   If true, the diagonal elements of a are assumed to
 *                        be 1 and not accessed.
 *  \param prog           A reference to a program sequence which will
 *                        be appended with the code to perform the
 *                        arrangement.
 *  \param debugPrefix    A debug prefix added to compute set and tensor
 *                        names.
 *  \returns              Tensor with the same shape as A with all unused values
 *                        masked.
 */
poplar::Tensor triangularMask(poplar::Graph &graph, const poplar::Tensor &a,
                              bool lower, bool unitDiagonal,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "");

/**
 * Solves systems of linear equations with lower or upper triangular
 coefficient.

 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param b              Tensor of the same type with shape [...,  N, K] if
 *                        left_side is true, [...,K, N] otherwise.
 *  \param leftSide       Solve AX = B if true, XA = B overwise.
 *  \param lower          Use the upper or lower triangle of A.
 *  \param unitDiagonal   If true, the diagonal elements of a are assumed to
 *                        be 1 and not accessed.
 *  \param blockSize      Block size for blocked solver.
 *  \param prog           A reference to a program sequence which will
 *                        be appended with the code to perform the
 *                        arrangement.
 *  \param debugPrefix    A debug prefix added to compute set and tensor
 *                        names.
 *  \param options        The structure describing options on how the
 *                        multiplication should be implemented.
 *                        See MatMul.hpp for details.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              Tensor with shape of b with linear system solution.
 */

poplar::Tensor triangularSolve(
    poplar::Graph &graph, const poplar::Tensor &a, const poplar::Tensor &b,
    bool leftSide, bool lower, bool unitDiagonal, std::size_t blockSize,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    poplar::OptionFlags options = {}, matmul::PlanningCache *cache = nullptr);

} // namespace poplin

#endif // poplin_TriangularSolve_hpp
