// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef poputil_VarStructure_hpp
#define poputil_VarStructure_hpp

#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

#include <poplar/Graph.hpp>

namespace poputil {

/** Create a tensor with the given shape such that when it
 *  is partitioned into slices according to the given number of
 *  partitions in each dimension, each slice is a single contiguous region.
 *
 *  This partitions such that the maximum number of elements in
 *  each partition of a dimension is minimised as well as the number
 *  of partitions. i.e. if a dimension has `n` elements, and the number of
 *  partitions in that dimension is `d` then:
 *
 *    a * ceil(n/d) + 1 * (n%d) = n
 *
 *  There will be `a` partitions with ceil(n/d) elements followed
 *  by `b` partitions with floor(n/d) elements and possibly some
 *  number of partitions with 0 elements.
 *
 *  The returned tensor has no tile mapping set.
 *
 *  \param graph        The graph to add the variable to.
 *  \param type         The type of the elements in the returned tensor.
 *  \param shape        The shape of the returned tensor.
 *  \param nPartitions  How many partitions the given shape will be partitioned
 *                      into in each dimension.
 *  \param debugName    The debug name associated with the returned tensor.
 *
 *  \return A tensor with the given shape where each partition is contiguous.
 */
poplar::Tensor
createPartitionableTensor(poplar::Graph &graph,
                          const poplar::Type &type,
                          const std::vector<std::size_t> &shape,
                          const std::vector<std::size_t> &nPartitions,
                          const std::string &debugName = "");

/** Iterate the partitions of a tensor.
 *
 *  Partitioning follows the same definition as described above in
 *  `addVariableWithSplits`.
 *
 * \param t           The tensor to iterate.
 * \param nPartitions How many partitions the given tensor is partitioned into
 *                    in each dimension.
 * \param f           A function taking the indices of the partition in
 *                    the range [0,splits[d]) in each dimension of
 *                    the tensor as well as the slice of the tensor
 *                    corresponding to that partition.
 */
void
iterateTensorPartitions(const poplar::Tensor &t,
                        const std::vector<std::size_t> &nPartitions,
                        const std::function<
                          void(const std::vector<std::size_t> &i,
                               const poplar::Tensor &s)> &f);


} // end namespace poputil

#endif // poputil_VarStructure_hpp
