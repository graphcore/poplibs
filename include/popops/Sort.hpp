// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions for sorting tensors.
 *
 */

#ifndef popops_Sort_hpp
#define popops_Sort_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Sort a tensor along the given dimension.
 *
 * This will return a tensor that is a permutation of the input tensor \p v with
 * all the elements of the 1D slices in the chosen dimension in ascending order.
 *
 *  This aims to match TensorFlow's XLA sort:
 *  https://www.tensorflow.org/xla/operation_semantics#sort
 *
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 *  \returns           A tensor which is a permutation of \p t such that all
 *                     elements in the given dimension are in order.
 *
 *  \throw poputil::poplibs_error If \p dim is not a valid dimension of \p v.
 */
poplar::Tensor sort(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

/** In-place sort a tensor along the given dimension.
 *
 *  This will permute the input tensor so that all the elements of 1D slices in
 *  the chosen dimension are in ascending order.
 *
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 *  \throw poputil::poplibs_error If \p dim is not a valid dimension of \p v.
 */
void sortInPlace(poplar::Graph &graph, const poplar::Tensor &t, unsigned dim,
                 poplar::program::Sequence &prog,
                 const std::string &debugPrefix = "");

/** Sort a tensor by a key tensor along the given dimension.
 *
 *  This will return a tensor that is a permutation of the input tensor \p v
 *  with the property that all 1D slices in the chosen dimensions are in
 *  ascending order with respect to the key tensor \p k.
 *
 *  This aims to match TensorFlow's XLA sort:
 *  https://www.tensorflow.org/xla/operation_semantics#sort
 *
 *  \param graph       The Poplar graph.
 *  \param k           The key tensor to sort on.
 *  \param v           The value tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *  \returns           A tensor which is a permutation of \p v such that it is
 *                     in order with respect to the tensor \p k in the given
 *                     dimension.
 *
 *  \note If \p k and \p v alias, the result is undefined.
 *
 *  \throw poputil::poplibs_error If \p dim is not a valid dimension of \p v.
 *  \throw poputil::poplibs_error If \p v and \p k are not the same shape.
 */
poplar::Tensor sortKeyValue(poplar::Graph &graph, const poplar::Tensor &k,
                            const poplar::Tensor &v, unsigned dim,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");

/** In-place sort a given tensor by a key tensor along the given dimension.
 *
 *  This will permute the key and value tensors so that all the elements of the
 *  1D slices in the chosen dimension are in ascending order with respect to the
 *  key tensor.
 *
 *  \param graph       The Poplar graph.
 *  \param k           The key tensor to sort on.
 *  \param v           The value tensor to be sorted.
 *  \param dim         The dimension to sort on.
 *  \param prog        The program to be extended.
 *  \param debugPrefix The prefix prepended to debugging info.
 *
 * \note The \p k tensor is also sorted by this in-place operation.
 * \note If the \p k tensor and the \p v tensor alias, the result is undefined.
 *
 *  \throw poputil::poplibs_error If \p dim is not a valid dimension of \p v.
 *  \throw poputil::poplibs_error If \p v and \p k are not the same shape.
 */
void sortKeyValueInPlace(poplar::Graph &graph, const poplar::Tensor &k,
                         const poplar::Tensor &v, unsigned dim,
                         poplar::program::Sequence &prog,
                         const std::string &debugPrefix = "");

} // namespace popops

#endif
