// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popops_UpdateScalarInRows_hpp
#define popops_UpdateScalarInRows_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/ExprOp.hpp>

#include <string>

namespace popops {

/** Update in-place one scalar per row of the tensor \p params. For each row,
 * the index of the value to update is specified by the tensor \p indices.
 * If the index from \p indices is equal to MASKED_LABEL_CODE then no update
 * is carried out.
 *
 * Pseudo-code
 * for each row r
 *   if indices[r] != MASKED_LABEL_CODE
 *     params[r][indices[r]] = params[r][indices[r]] - 1.f
 *
 * If the ith index is less than 0 or greater than width then the whole row of
 * the param tensor is set to NAN.
 * This is to match the interface of the backward phase of
 * tf.nn.sparse_softmax_cross_entropy_with_logits (see the link above).
 *
 * \param params The 2D tensor to be updated, element-type must be either float
 * or half.
 * \param indices 1D tensor, element-type must be unsigned integer.
 */
void updateScalarInRows(poplar::Graph &graph, const poplar::Tensor &params,
                        const poplar::Tensor &indices,
                        poplar::program::Sequence &program,
                        const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_UpdateScalarInRows_hpp
