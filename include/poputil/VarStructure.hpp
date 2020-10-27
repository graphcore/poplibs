// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file VarStructure.hpp
 *
 * Manage partitioning and grouping in tensors.
 *
 */

#ifndef poputil_VarStructure_hpp
#define poputil_VarStructure_hpp

#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

#include <poplar/Graph.hpp>

namespace poputil {

/** Detect if the tensor \p t has a grouping in
 * its innermost dimension.
 *  \param graph    The graph to add the function to.
 *  \param t        The tensor to check for grouping.
 *  \return         The size of the group. Zero if there is no grouping.
 *  \throw poputil::poplibs_error If the rank of \p t is zero.
 */
unsigned detectInnermostGrouping(const poplar::Graph &graph,
                                 const poplar::Tensor &t);

using GroupingInfo = std::pair<unsigned, unsigned>;

/** Find all grouped dimensions from the innermost grouped dimension
 * moving outwards, returning groupings for each. The same dimension may appear
 * more than once. This uses detectInnermostGrouping() iteratively.
 *  \param graph  The graph to add the function to.
 *  \param t      The tensor to check for grouping.
 *  \return       A list of the grouped dimensions starting with the innermost.
 *  \throw poputil::poplibs_error If the rank of \p t is zero.
 */
std::vector<GroupingInfo> detectDimGroupings(const poplar::Graph &graph,
                                             const poplar::Tensor &t);

/** Create a tensor with the given shape, so that when it
 *  is partitioned into slices according to the given number of
 *  partitions in each dimension, each slice is a single contiguous region.
 *
 *  This partitions the tensor so that the maximum number of elements in
 *  each partition of a dimension is minimised as well as the number
 *  of partitions. That is, if a dimension has \c n elements, and the number of
 *  partitions in that dimension is \c d then:
 *
 *      a * ceil(n/d) + 1 * (n%d) = n
 *
 *  There will be \c a partitions with \c ceil(n/d) elements followed
 *  by \c b partitions with \c floor(n/d) elements and possibly some
 *  number of partitions with 0 elements.
 *
 *  The returned tensor has no tile mapping set.
 *
 *  \param graph        The graph to add the variable to.
 *  \param type         The type of the elements in the returned tensor.
 *  \param shape        The shape of the returned tensor.
 *  \param nPartitions  The number of partitions the shape will be partitioned
 *                      into in each dimension.
 *  \param debugContext Optional debug information.
 *
 *  \return A tensor with the given shape where each partition is contiguous.
 * \throw poputil::poplibs_error If the size of \p shape and \p nPartitions are
 *        not equal.
 */
poplar::Tensor
createPartitionableTensor(poplar::Graph &graph, const poplar::Type &type,
                          const std::vector<std::size_t> &shape,
                          const std::vector<std::size_t> &nPartitions,
                          const poplar::DebugContext &debugContext = {});

/** Iterate a function over the partitions of a tensor.
 *
 *  Partitioning follows the same definition as described for
 *  createPartitionableTensor().
 *
 * \param t           The tensor to iterate over.
 * \param nPartitions The number of partitions the tensor is partitioned into
 *                    in each dimension.
 * \param i
 * \param f           A function taking the indices of the partition in
 *                    the range [0, splits[d]) in each dimension of
 *                    the tensor as well as the slice of the tensor
 *                    corresponding to that partition.
 * \throw poputil::poplibs_error If the rank of \p t and the size of
 *        \p nPartitions are not equal.
 */
void iterateTensorPartitions(
    const poplar::Tensor &t, const std::vector<std::size_t> &nPartitions,
    const std::function<void(const std::vector<std::size_t> &i,
                             const poplar::Tensor &s)> &f);

} // end namespace poputil

#endif // poputil_VarStructure_hpp
