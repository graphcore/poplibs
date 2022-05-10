// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Factorise a positive definite matrix using Cholesky decomposition.
 *
 */

#ifndef poplin_Cholesky_hpp
#define poplin_Cholesky_hpp
#include "poplin/MatMul.hpp"
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace poplin {

class PlanningCache;

/**
 * Plan matrix multiplication for the Cholesky factoriser.
 *
 * Supported options:
 *
 * * `blockSize`
 *
 *   A hint for the size of block to be used.
 *
 * See matMul() for additional options.
 *
 *
 * \param type       The data type of the input tensor.
 * \param shape      The shape of the input tensor.
 * \param lower      Lower triangular matrix if true, else upper triangular.
 * \param options    A structure describing options on how the
 *                   decomposition should be implemented.
 * \returns          Preplan parameters for matMul().
 */
std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getCholeskyMatMulPrePlanParameters(const poplar::Type &type,
                                   const std::vector<std::size_t> &shape,
                                   bool lower, poplar::OptionFlags options);

/**
 * Create a tensor that is used as the input for the Cholesky factoriser.
 *
 * Supported options:
 *
 * * `blockSize`
 *
 *   A hint for the size of block to be used.
 *
 * See matMul() for additional options.
 *
 * This will create a 2D/3D tensor in the graph. The ordering and tile mapping
 * of the tensor will be set to make a triangular factoriser with this
 * tensor as the left argument efficient.
 *
 * \param graph           The Poplar graph.
 * \param type            The input data type.
 * \param shape           The shape of the tensor.
 * \param debugContext    Debug information.
 * \param options         A structure describing options on how the
 *                        decomposition should be implemented.
 * \param cache           Optional pointer to a planning cache to use.
 *
 * \returns               A matrix of type \p type and shape \p shape. The
 *                        tensor will have been mapped to tiles.
 */
poplar::Tensor createCholeskyInput(poplar::Graph &graph,
                                   const poplar::Type &type,
                                   const std::vector<std::size_t> &shape,
                                   bool lower,
                                   const poplar::DebugContext &debugContext,
                                   const poplar::OptionFlags &options = {},
                                   PlanningCache *cache = nullptr);

/**
 * Computes Cholesky factor for a symmetric positive definite matrix.
 *
 * Supported options:
 *
 * * `blockSize`
 *
 *   A hint for the size of block to be used.
 *
 * See matMul() for additional options.
 *
 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param lower          If true, return a lower triangular matrix, else return
 *                        an upper triangular matrix.
 *  \param prog           A reference to a program
 *                        sequence which the code to perform the arrangement
 *                        will be appended to.
 *  \param debugContext   Optional debug information.
 *  \param options        A structure describing options on how the
 *                        decomposition should be implemented.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              A tensor with the same shape as \p a with a
 *                        triangular factor.
 */
poplar::Tensor cholesky(poplar::Graph &graph, const poplar::Tensor &a,
                        bool lower, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        poplar::OptionFlags options = {},
                        PlanningCache *cache = nullptr);

/**
 * Computes Cholesky factor in place for a symmetric positive definite matrix.
 *
 * Supported options:
 *
 * * `blockSize`
 *
 *   A hint for the size of block to be used.
 *
 * See matMul() for additional options.
 *
 *  \param graph          The Poplar graph.
 *  \param a              Tensor of floating-point type with shape [..., N,N].
 *  \param lower          If true, return a lower triangular matrix, else return
 *                        an upper triangular matrix.
 *  \param prog           A reference to a program
 *                        sequence which the code to perform the arrangement
 *                        will be appended to.
 *  \param debugContext   Optional debug information.
 *  \param options        A structure describing options on how the
 *                        decomposition should be implemented.
 *  \param cache          Optional pointer to a planning cache to use.
 *  \returns              None
 */
void choleskyInPlace(poplar::Graph &graph, const poplar::Tensor &a, bool lower,
                     poplar::program::Sequence &prog,
                     const poplar::DebugContext &debugContext = {},
                     poplar::OptionFlags options = {},
                     PlanningCache *cache = nullptr);

} // namespace poplin

#endif // poplin_Cholesky_hpp
