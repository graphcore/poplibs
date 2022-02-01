// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Functions for finding the top k elements.
 */

#ifndef _popops_TopK_hpp_
#define _popops_TopK_hpp_

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include <popops/SortOrder.hpp>

namespace popops {

/** Parameters for topK* APIs
 */
struct TopKParams {
  /// The number of outputs from the top k operation.
  /// This must be less or equal the number of elements in the innermost
  /// dimension of the tensor used as input to the operation.
  unsigned k;
  /// If true, return the top k largest elements. Otherwise return the
  /// top k smallest elements.
  bool largest;
  /// The required ordering of elements in the resulting tensor.
  SortOrder sortOrder;
  /// When sortOrder != SortOrder::NONE and stableSort is
  /// true, the relative order of values that compare equal are
  /// guaranteed not to change in the output.
  bool stableSort;

  TopKParams(unsigned k, bool largest, SortOrder sortOrder,
             bool stableSort = false) noexcept;
};

std::ostream &operator<<(std::ostream &os, const TopKParams &p);

/** Create an return a new tensor laid out optimally to be used as
 *  an input to a topK operation with the given parameters.
 *
 *  \param graph        The Poplar graph to add the tensor to.
 *  \param type         The Poplar type of elements in the returned tensor.
 *  \param shape        The shape of the returned tensor.
 *  \param params       The parameters of the top k that the returned tensor
 *                      will be used as input to.
 *  \param debugContext Optional debug information.
 *
 *  \returns A newly created tensor with shape \p shape and full tile mapping.
 */
poplar::Tensor createTopKInput(poplar::Graph &graph, const poplar::Type &type,
                               const std::vector<std::size_t> &shape,
                               const TopKParams &params,
                               const poplar::DebugContext &debugContext = {});

/** Return the top k values in the innermost dimension of a tensor.
 *
 *  \param graph        The Poplar graph to add the operation to.
 *  \param prog         The Poplar sequence to add the operation to.
 *  \param t            The tensor in which to find the top-k values in
 *                      the innermost dimension.
 *  \param params       The parameters of the top k.
 *  \param debugContext Optional debug information.
 *
 *  \returns A tensor with the top k values found in the innermost dimension
 *           of \p t.
 */
poplar::Tensor topK(poplar::Graph &graph, poplar::program::Sequence &prog,
                    const poplar::Tensor &t, const TopKParams &params,
                    const poplar::DebugContext &debugContext = {});

/** Return the top k values in the innermost dimension of a tensor along
 *  with the permutation of another tensor with respect to the values.
 *
 *  \param graph        The Poplar graph to add the operation to.
 *  \param prog         The Poplar sequence to add the operation to.
 *  \param key          The tensor in which to find the top-k values in
 *                      the innermost dimension.
 *  \param value        A tensor with the same shape as \p key for which to
 *                      get the permutation with respect to \p key.
 *  \param params       The parameters of the top k.
 *  \param debugContext Optional debug information.
 *
 *  \returns A pair of tensors. The first contains the top k values found
 *           in the innermost dimension of \p key. The second contains the
 *           permutation of the tensor \p value with respect to the tensor
 *           \p key.
 */
std::pair<poplar::Tensor, poplar::Tensor>
topKKeyValue(poplar::Graph &graph, poplar::program::Sequence &prog,
             const poplar::Tensor &keys, const poplar::Tensor &values,
             const TopKParams &params,
             const poplar::DebugContext &debugContext = {});

/** Return the top k values in the innermost dimension of a tensor along
 *  with the indices of those values in the input tensor in the innermost
 *  dimension.
 *
 *  \param graph        The Poplar graph to add the operation to.
 *  \param prog         The Poplar sequence to add the operation to.
 *  \param t            The tensor in which to find the top-k values in
 *                      the innermost dimension.
 *  \param params       The parameters of the top k.
 *  \param debugContext Optional debug information.
 *
 *  \returns A pair of tensors. The first contains the top k values found
 *           in the innermost dimension of \p t. The second contains the
 *           indices of those values in the innermost dimension of \p t in
 *           the original input.
 */
std::pair<poplar::Tensor, poplar::Tensor>
topKWithPermutation(poplar::Graph &graph, poplar::program::Sequence &prog,
                    const poplar::Tensor &t, const TopKParams &params,
                    const poplar::DebugContext &debugContext = {});

} // end namespace popops

#endif // _popops_TopK_hpp_
