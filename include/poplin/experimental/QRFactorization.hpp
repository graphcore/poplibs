// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Decomposition of a matrix into an orthogonal matrix Q and upper triangular
 * matrix R.
 *
 */

#ifndef poplin_QRFactorization_hpp
#define poplin_QRFactorization_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poplin {

namespace experimental {

/**
 * Create a tensor that is used as the output Q operand of QR factorization.
 *
 * This will create matrices A and Q (2D tensors) in the graph. The number of
 * matrix A rows must be greater or equal to the number of columns. Matrix Q is
 * a square matrix. The ordering and tile mapping of the matrices will be set to
 * make a QR factorization with these matrices as the input and output
 * efficiency.
 *
 * \param graph           The Poplar graph.
 * \param type            The data type of the returned tensor.
 * \param m               The number of rows of the input matrix A. This is also
 *                        used to shape the square Q matrix.
 * \param n               The number of columns of the input matrix A.
 * \param debugContext    Debug information.
 *
 * \returns               The matrices A and Q of type \p type and shape \p
 *                        dimensionA and \p dimensionQ respectively. The
 *                        matrices will be include a mapping to tiles.
 */
std::array<poplar::Tensor, 2>
createQRFactorizationMatrices(poplar::Graph &graph, const poplar::Type &type,
                              const std::size_t m, const std::size_t n,
                              const poplar::DebugContext &debugContext);

/**
 * Calculates the QR factorization for the given matrix.
 *
 * This will compute the QR factorization for the input matrix A. The number of
 * matrix A rows must be greater or equal to the number of columns. The output
 * matrix R will overwrite matrix A. For matrix Q, the identity matrix should be
 * passed.
 *
 * Supported options:
 *
 * * `rowsPerIteration`
 *
 *   A hint for the size of rows block to be calculated per iteration. The
 *   QRFactorization() function uses a hybrid approach, balancing the number of
 *   programs generated and the number of iterations performed by each program.
 *   A higher value of this parameter results in a shorter compilation time at
 *   the expense of performance.
 *
 *   The default value is 32.
 *
 *  \param graph          The Poplar graph.
 *  \param A              Tensor of floating-point type with shape [M, N].
 *  \param Q              Identity matrix of the same type with shape [M, M].
 *  \param prog           A reference to a program sequence which the code
 *                        to perform the arrangement will be appended to.
 *  \param debugContext   Optional debug information.
 *  \param options        The implementation options of the triangular solver.
 */
void QRFactorization(poplar::Graph &graph, poplar::Tensor &A, poplar::Tensor &Q,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext = {},
                     const poplar::OptionFlags &options = {});

} // namespace experimental

} // namespace poplin

#endif // poplin_QRFactorization_hpp
