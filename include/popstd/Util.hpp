#ifndef _popstd_Util_hpp_
#define _popstd_Util_hpp_

#include <poplar/Device.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

namespace popstd {

void mergeAdjacentRegions(
    std::vector<poplar::Interval<std::size_t>> &regions);

void mergeAdjacentRegions(
    std::vector<std::vector<poplar::Interval<std::size_t>>> &mapping);

// Given a set of contiguous regions, partition these regions trying to
// balance the number of elements in each partition, respecting the specified
// grain. At most maxPartitions partitions are created. Regions may be split to
// achieve a better balance.
std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegions(const std::vector<poplar::Interval<std::size_t>> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition = 0);

// Given a set of contiguous regions per tile, partition these regions
// between workers on that tile, respecting the specified grain size.
// Regions may be split to balance the work across workers.
std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<poplar::Interval<std::size_t>> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0);

// Given a set of sequences of regions, partition these sequences trying to
// balance the number of elements in each partition, respecting the specified
// grain. At most maxPartitions partitions are created. Sequences (and regions
// within them may be split to achieve a better balance.
std::vector<std::vector<std::vector<poplar::Interval<std::size_t>>>>
splitRegions(
    const std::vector<std::vector<poplar::Interval<std::size_t>>> &regions,
    unsigned grainSize, unsigned maxPartitions,
    unsigned minElementsPerPartition = 0);

// Given a set of sequences of regions per tile, partition these sequences
// between workers on that tile, respecting the specified grain size.
// Regions may be split to balance the work across workers.
std::vector<std::vector<std::vector<poplar::Interval<std::size_t>>>>
splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<std::vector<poplar::Interval<std::size_t>>> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0);

/// Given an index into a flattened tensor returns the indices into the
/// dimensions of the original tensor.
std::vector<std::size_t> unflattenIndex(const std::vector<std::size_t> &shape,
                                        std::size_t index);

/// Given an list of indices into a tensor return the corresponding index in a
/// flattened version of the tensor.
std::size_t flattenIndex(const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &indices);

// Total number of elements in the interval sequence
std::size_t intervalSequenceNumElements(
    const std::vector<std::vector<poplar::Interval<std::size_t>>> &seq);

// Copy a tensors data to a new tensor. The duplicated tensor has the same tile
// mapping as the original tensor.
poplar::Tensor duplicate(poplar::Graph &graph, const poplar::Tensor &in,
                         poplar::program::Sequence &p);

} // end namespace popstd


#endif // _popstd_Util_hpp_
