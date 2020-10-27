// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions for updating values in tensors.
 *
 */

#ifndef popops_UpdateScalarInRows_hpp
#define popops_UpdateScalarInRows_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/ExprOp.hpp>

#include <string>

namespace popops {

/** Update in-place one scalar per row of the tensor \p params. For each row,
 * the index of the value to update is specified by the tensor \p indices.
 * If the index from \p indices is equal to \c MASKED_LABEL_CODE then no update
 * is carried out.
 *
 * Pseudo-code:
 * \code
 *   for each row r
 *     if indices[r] != MASKED_LABEL_CODE
 *       params[r][indices[r]] = params[r][indices[r]] - 1.f
 * \endcode
 *
 * If the ith index is less than 0 or greater than the size of the row then the
 * whole row of the \p param tensor is set to NaN. This is to match the
 * interface of the backward phase of
 * \c tf.nn.sparse_softmax_cross_entropy_with_logits, see
 * https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
 *
 * \param graph     The Poplar graph.
 * \param params The 2D tensor to be updated, the element type must be either
 * float or half.
 * \param indices 1D tensor, the element-type must be unsigned integer.
 * \param program    The program to be extended.
 * \param debugContext Optional debug information.
 */
void updateScalarInRows(poplar::Graph &graph, const poplar::Tensor &params,
                        const poplar::Tensor &indices,
                        poplar::program::Sequence &program,
                        const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_UpdateScalarInRows_hpp
