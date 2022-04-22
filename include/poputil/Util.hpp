// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file Util.hpp
 *
 * General operations on tensors.
 *
 */

#ifndef poputil_Util_hpp
#define poputil_Util_hpp

#include <algorithm>
#include <cassert>
#include <climits>
#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <poplar/Quarter.hpp>
#include <poplar/Target.hpp>
#include <poplar/Tensor.hpp>
#include <string>
#include <vector>

/// General utility functions for building graphs
namespace poputil {

void mergeAdjacentRegions(std::vector<poplar::Interval> &regions);

void mergeAdjacentRegions(std::vector<std::vector<poplar::Interval>> &mapping);

/// Flatten a vector of vectors of intervals to a vector, maintaining
/// ordering.
std::vector<poplar::Interval>
flattenIntervals(const std::vector<std::vector<poplar::Interval>> &intervals);

/// Given a set of contiguous regions, partition these regions while trying to
/// balance the number of elements in each partition and respecting the
/// specified grain size. At most \p maxPartitions partitions are created.
/// Regions may be split to achieve a better balance.
std::vector<std::vector<poplar::Interval>>
splitRegions(const std::vector<poplar::Interval> &regions, unsigned grainSize,
             unsigned maxPartitions, unsigned minElementsPerPartition = 0,
             unsigned maxElementsPerPartition = UINT_MAX,
             unsigned maxElementsPerRegion = UINT_MAX);

/// Given a set of contiguous regions per tile, partition these regions
/// between workers on that tile while respecting the specified grain size.
/// Regions may be split to balance the work across workers.
std::vector<std::vector<poplar::Interval>> splitRegionsBetweenWorkers(
    const poplar::Target &target, const std::vector<poplar::Interval> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0,
    unsigned maxElementsPerPartition = UINT_MAX,
    unsigned maxElementsPerRegion = UINT_MAX);

/// Given a set of sequences of regions, partition these sequences while trying
/// to balance the number of elements in each partition and respecting the
/// specified grain size. At most \p maxPartitions partitions are created.
/// Sequences, and regions within them, may be split to achieve a better
/// balance.
std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegions(const std::vector<std::vector<poplar::Interval>> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition = 0,
             unsigned maxElementsPerPartition = UINT_MAX,
             unsigned maxElementsPerRegion = UINT_MAX);

/// Given a set of sequences of regions per tile, partition these sequences
/// between workers on that tile while respecting the specified grain size.
/// Regions may be split to balance the work across workers.
std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegionsBetweenWorkers(
    const poplar::Target &target,
    const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0,
    unsigned maxElementsPerPartition = UINT_MAX,
    unsigned maxElementsPerRegion = UINT_MAX);

/// Given an index into a flattened tensor, returns the indices into the
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

/// Given a list of indices into a tensor, return the corresponding index in a
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

/// Return the total number of elements in the interval sequence.
std::size_t intervalSequenceNumElements(
    const std::vector<std::vector<poplar::Interval>> &seq);

/// Copy a tensor's data to a new tensor. The duplicated tensor has the same
/// tile mapping as the original tensor.
poplar::Tensor
duplicate(poplar::Graph &graph, const poplar::Tensor &in,
          poplar::program::Sequence &p,
          const poplar::DebugContext &debugContext = {},
          poplar::TensorCloneMethod method =
              poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Clone a tensor N times.
 *
 *  Given a tensor of shape [D1, D2, ... Dn], this function will create a new
 *  tensor of shape [N, D1, D2, ..., Dn] where each of the N sub-tensors
 *  is a clone of the original tensor (that is, it has the same layout).
 *
 *  \param graph   The Poplar graph.
 *  \param t       The tensor to clone.
 *  \param N       The replication factor to clone with.
 *  \param name    The name for the new variables created.
 *  \param method  The tensor cloning method (see Graph::clone()).
 */
poplar::Tensor
cloneN(poplar::Graph &graph, const poplar::Tensor &t, unsigned N,
       const poplar::DebugContext &debugContext = {},
       poplar::TensorCloneMethod method =
           poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);

/** Split a range.
 * Utility function to split a range [0, \p rangeUpperBound] into \p splitCount
 * slices as evenly as possible. If \p splitCount does not divide
 * \p rangeUpperBound evenly then output slices are assigned more units in
 * round-robin.
 */
std::vector<int> balancedPartition(int rangeUpperBound, int splitCount);

/** Cast a double precision value to a value exactly representable in device
 *  HALF type.
 *
 * \param target        The target device that the cast will be performed on.
 * \param input         Input value.
 * \return              Value cast to HALF type on device.
 */
double castToDeviceHalfValue(const poplar::Target &target, double input);

/** Check accuracy of a cast operation.
 * Utility function to check if \p input can be cast from \p inputType to
 * \p outputType without an error in its accuracy, or causing an overflow.
 *
 * \param target        The target device that the cast will be performed on.
 * \param input         Input value.
 * \param inputType     Input type before the cast operation.
 * \param outputType    Output type after the cast operation.
 * \param tolerance     Allowed tolerance in error from cast operation.
 * \return              Boolean tensor indicating the error will be less than
 *                      \p tolerance.
 * \throw poputil::poplibs_error If either \p inputType or \p outputType
 * are not either half or float.
 */
bool checkAccuracyWhenCast(const poplar::Target &target, double input,
                           poplar::Type inputType, poplar::Type outputType,
                           double tolerance);

/** Factors the outermost dimensions of tensor \p t by the values given in
 *  \p factors. For each value \c f in \p factors, the corresponding outer
 *  dimension is split into two parts of sizes \c size(dim)/f and \c f. The
 *  second of these becomes a dimension inside all the factored dimensions. For
 *  example, given a tensor with shape [4,6,4] and factors [1,2], we first
 *  divide the shape into [4/1,1,6/2,2,4] and then shuffle it to
 *  [4/1,6/2,1,2,4].
 *
 *  \param t The tensor to be factored.
 *  \param factors The values to factor each dimension by.
 *  \param startDim The outermost dimension to start at.
 *  \return The refactored tensor.
 */
poplar::Tensor factorDims(const poplar::Tensor &t,
                          const std::vector<std::size_t> &factors,
                          unsigned startDim = 0);

/** The opposite of factorDims(). This does not need information for each
 *  dimension because that is present in the tensor. It just needs the number of
 *  dimensions.
 *
 *  \param t The tensor to be refactored.
 *  \param numDims The number of dimensions to be refactored.
 *  \param startDim The outermost dimension to start at.
 *  \return The refactored tensor.
 */
poplar::Tensor unfactorDims(const poplar::Tensor &t, unsigned numDims,
                            unsigned startDim = 0);

// Create metadata for use with FP8 data types
poplar::Tensor createMetadataTensor(poplar::Graph &graph,
                                    poplar::QuarterMetadata::Format fp8Format,
                                    int fp8Scale);

poplar::Tensor createMetadataTensor(poplar::Graph &graph,
                                    poplar::QuarterMetadata::Format fp8Format,
                                    int fp8Scale,
                                    poplar::program::Sequence &prog);

/** Calculate the un-shuffling intervals based on the given intervals.
 *
 * Given a vector of intervals, one could use these intervals to shuffle a
 * tensor. For example:
 *
 *   poplar::Tensor shuffled = poplar::concat(tensor.slices(intervals));
 *
 * Another vector of intervals exists that can be applied in the same way to the
 * shuffled tensor to undo the shuffling. This function calculates these
 * intervals. The time complexity is `nlog(n)` with `n` the number of intervals.
 *
 * Note: This function assumes that the intervals are non-overlapping and form
 *       one contiguous interval.
 *
 * \param intervals A vector of intervals that shuffle a tensor.
 *
 * \returns         A vector of intervals that unshuffle a tensor.
 */
std::vector<poplar::Interval>
calculateUnshufflingIntervals(const std::vector<poplar::Interval> &intervals);

} // end namespace poputil

#endif // poputil_Util_hpp
