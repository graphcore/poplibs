// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
/**
 *
 * Decomposition of a matrix into an lower triangular matrix L and upper
 * triangular matrix U.
 *
 */

#ifndef poplin_LuFactorization_hpp
#define poplin_LuFactorization_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace poplin {

namespace experimental {

/**
 * Calculates the LU factorization for the given matrix.
 *
 *
 *  \param graph          The Poplar graph.
 *  \param input          Input Tensor of floating-point type [M, N].
 *  \param prog           A reference to a program sequence to which the code
 *                        to perform the arrangement will be appended.
 *  \param debugContext   Optional debug information.
 *
 *  \returns              The matrices L and U, where ->
 *                        L = Lower triangular fp32 matrix [M, M].
 *                        U = Upper triangular fp32 matrix [M, N].
 */
std::pair<poplar::Tensor, poplar::Tensor>
LUFactorization(poplar::Graph &graph, poplar::Tensor &input,
                poplar::program::Sequence &seq,
                const poplar::DebugContext &debugContext = {});

} // namespace experimental

} // namespace poplin

#endif // poplin_LuFactorization_hpp
