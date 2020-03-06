// Copyright (c) 2016 Graphcore Ltd, All rights reserved.

#ifndef poputil_Util_hpp
#define poputil_Util_hpp

#include <algorithm>
#include <cassert>
#include <climits>
#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Target.hpp>
#include <poplar/Tensor.hpp>
#include <string>
#include <vector>

namespace poputil {

void mergeAdjacentRegions(std::vector<poplar::Interval> &regions);

void mergeAdjacentRegions(std::vector<std::vector<poplar::Interval>> &mapping);

/// Given a set of contiguous regions, partition these regions trying to
/// balance the number of elements in each partition, respecting the specified
/// grain. At most maxPartitions partitions are created. Regions may be split to
/// achieve a better balance.
std::vector<std::vector<poplar::Interval>>
splitRegions(const std::vector<poplar::Interval> &regions, unsigned grainSize,
             unsigned maxPartitions, unsigned minElementsPerPartition = 0,
             unsigned maxElementsPerPartition = UINT_MAX,
             unsigned maxElementsPerRegion = UINT_MAX);

/// Given a set of contiguous regions per tile, partition these regions
/// between workers on that tile, respecting the specified grain size.
/// Regions may be split to balance the work across workers.
std::vector<std::vector<poplar::Interval>> splitRegionsBetweenWorkers(
    const poplar::Target &target, const std::vector<poplar::Interval> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0,
    unsigned maxElementsPerPartition = UINT_MAX,
    unsigned maxElementsPerRegion = UINT_MAX);

// Given a set of sequences of regions, partition these sequences trying to
// balance the number of elements in each partition, respecting the specified
// grain. At most maxPartitions partitions are created. Sequences (and regions
// within them may be split to achieve a better balance.
std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegions(const std::vector<std::vector<poplar::Interval>> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition = 0,
             unsigned maxElementsPerPartition = UINT_MAX,
             unsigned maxElementsPerRegion = UINT_MAX);

/// Given a set of sequences of regions per tile, partition these sequences
/// between workers on that tile, respecting the specified grain size.
/// Regions may be split to balance the work across workers.
std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegionsBetweenWorkers(
    const poplar::Target &target,
    const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0,
    unsigned maxElementsPerPartition = UINT_MAX,
    unsigned maxElementsPerRegion = UINT_MAX);

/// Given an index into a flattened tensor returns the indices into the
/// dimensions of the original tensor.
template <class T>
std::vector<T> unflattenIndex(const std::vector<T> &shape, std::size_t index) {
  std::vector<T> coord(shape.size());

  for (std::size_t i = shape.size(); i > 0; --i) {
    coord[i - 1] = index % shape[i - 1];
    index /= shape[i - 1];
  }

  assert(index == 0);
  return coord;
}

/// Given an list of indices into a tensor return the corresponding index in a
/// flattened version of the tensor.
template <class T>
std::size_t flattenIndex(const std::vector<T> &shape,
                         const std::vector<T> &indices) {
  auto rank = shape.size();
  assert(indices.size() == rank);
  std::size_t index = 0;
  for (unsigned i = 0; i != rank; ++i) {
    index = index * shape[i] + indices[i];
  }
  return index;
}

/// Total number of elements in the interval sequence
std::size_t intervalSequenceNumElements(
    const std::vector<std::vector<poplar::Interval>> &seq);

/// Copy a tensor's data to a new tensor. The duplicated tensor has the same
/// tile mapping as the original tensor.
poplar::Tensor
duplicate(poplar::Graph &graph, const poplar::Tensor &in,
          poplar::program::Sequence &p, const std::string &name = "",
          poplar::TensorCloneMethod method =
              poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Clone a tensor N times.
 *
 *  Given a tensor of shape [D1, D2, ... Dn], this function will create a new
 *  tensor of shape [N, D1, D2, ..., Dn] where each of the N sub-tensors
 *  is a clone of the original tensor (i.e. has the same layout).
 *
 *  \param graph   The Poplar graph
 *  \param t       The tensor to clone
 *  \param N       The replication factor to clone with
 *  \param name    The name for the new variables created
 *  \param method  The tensor cloning method (see Graph::clone)
 */
poplar::Tensor
cloneN(poplar::Graph &graph, const poplar::Tensor &t, unsigned N,
       poplar::StringRef name = "",
       poplar::TensorCloneMethod method =
           poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/// Utility function to split a range [0, `rangeUpperBound`] into `splitCount`
/// slices as evenly as possible. If `splitCount` does not divide
/// `rangeUpperBound` evenly then output slices are assigned more units in
/// round-robin.
std::vector<int> balancedPartition(int rangeUpperBound, int splitCount);

// Utility function to check if \input can be cast from \inputType to
// \outputType without error in its accuracy, or overflowing
bool checkAccuracyWhenCast(const poplar::Target &target, double input,
                           poplar::Type inputType, poplar::Type outputType,
                           double tolerance);
} // end namespace poputil

#endif // poputil_Util_hpp
