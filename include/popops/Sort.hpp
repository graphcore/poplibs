// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Sort_hpp
#define popops_Sort_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Sort a given tensor along the given dimension.
 *
 *  This will return a tensor that is a permutation of the input tensor with the
 *  property that all 1D slices in the chosen dimensions are in ascending order.
 *
 *  This aims to match TensorFlow's XLA sort
 *  https://www.tensorflow.org/xla/operation_semantics#sort
 *
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 *  \returns           A tensor which is a permutation of `t` such that all
 *                     elements in the given dimension are in order.
 */
poplar::Tensor sort(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

/** In-place sort a given tensor along the given dimension.
 *
 *  This will permute the input tensor so that all 1D slices in the chosen
 *  dimensions are in ascending order.
 *
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 */
void sortInPlace(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Sort a given tensor by a key tensor along the given dimension.
 *
 *  This will return a tensor that is a permutation of the input value tensor
 *  with the property that all 1D slices in the chosen dimensions are in
 *  ascending order with respect to the key tensor.
 *
 *  This aims to match TensorFlow's XLA sort
 *  https://www.tensorflow.org/xla/operation_semantics#sort
 *
 *  \param graph       The Poplar graph.
 *  \param k           The key tensor to sort on.
 *  \param v           The value tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 *  \returns           A tensor which is a permutation of `v` such that it is in
 *                     order with respect to the tensor `k` in the given
 *                     dimension.
 *
 *  \note If `k` and `v` alias, the result is undefined.
 */
poplar::Tensor sortKeyValue(poplar::Graph &graph, const poplar::Tensor &k,
                            const poplar::Tensor &v, unsigned dim,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");

/** In-place sort a given tensor by a key tensor along the given dimension.
 *
 *  This will permute the key and value tensors so that all 1D slices in the
 *  chosen dimensions are in ascending order with respect to the key tensor.
 *
 *  \param graph       The Poplar graph.
 *  \param k           The key tensor to sort on.
 *  \param v           The value tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 *  \note the 'k' tensor is also sorted by this in-place operation.
 *  \note If the `k` tensor and the `v` tensor alias, the result is undefined.
 */
void sortKeyValueInPlace(poplar::Graph &graph, const poplar::Tensor &k,
                         const poplar::Tensor &v, unsigned dim,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "");

} // namespace popops

#endif
