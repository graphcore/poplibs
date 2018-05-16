#include "poputil/Util.hpp"

#include <algorithm>
#include <numeric>

namespace poputil {

void mergeAdjacentRegions(
    std::vector<poplar::Interval> &regions) {
  std::vector<poplar::Interval> newRegions;
  std::sort(regions.begin(), regions.end());
  for (const auto &region : regions) {
    if (region.begin() == region.end())
      continue;
    assert(newRegions.empty() || newRegions.back().end() <= region.begin());
    if (!newRegions.empty() && newRegions.back().end() == region.begin()) {
      newRegions.back() = {newRegions.back().begin(), region.end()};
    } else {
      newRegions.push_back(region);
    }
  }
  std::swap(regions, newRegions);
}

void mergeAdjacentRegions(
    std::vector<std::vector<poplar::Interval>> &mapping) {
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    mergeAdjacentRegions(mapping[tile]);
  }
}

// Take a vector of items, which each has a size and split them into sets. If
// both minSizePerPartition and maxSizePerPartition can be satisfied there will
// be `maxPartitions` different sets. Typically the item is an interval
// and its size is the size of the interval (end - begin).
//
// Each partition will have a total size of at least `minSizePerPartition`
// and the size will be a multiple of `grainSize` (except possibly the last
// one).
//
// This takes two template function - one to tell the size of an item,
// and one to add part of an item to a partition.
//
// Consider the following example.
//
//   T = poplar::Interval
//
//   std::size_t size(const poplar::Interval &ival) {
//     return ival.end() - ival.begin();
//   }
//   void extend(std::vector<poplar::Interval> &partition,
//               const poplar::Interval &ival,
//               unsigned offset,
//               unsigned len) {
//     partition.emplace_back(ival.begin() + offset,
//                            ival.begin() + offset + len);
//   }
//
//    item: [{0, 10}, {15, 20}, {33, 60}]
//    grainSize: 4
//    maxPartitions: 8
//    minSizePerPartition: 6
//
// In this case, the function will calculate the total size using the supplied
// size() function:
//
//   (10-0) + (20-15) + (60-33) = 42
//
// Then it will work out how many grains that is:
//
//   ceil(42/4) = 11
//
// And then divide those into up to 8 partitions, respecting the
// minSizePerPartition (6 here, therefore at least 2 grains per partition).
// In this case it means we actually get 4 partitions with this many grains:
// (3,3,3,2). You might expect 5 (3,2,2,2,2) but the algorithm tries to
// minimise the number of partitions if it doesn't affect the maximum partition
// size (i.e. the execution time).
//
// Once the number of grains per partition is decided, it loops through
// the partitions and adds grainSize * numGrains worth of the items to
// each partition, using the supplied extend() function. In this example it
// would call extend like this:
//
//   extend(partition1, {0, 10}, 0, 10);  // size(partition1) = 10
//   extend(partition1, {15, 20}, 0, 2);  // size(partition1) = 12 (3 grains)
//   extend(partition2, {15, 20}, 2, 3);  // size(partition2) = 3
//   extend(partition2, {33, 60}, 0, 9);  // size(partition2) = 12 (3 grains)
//   extend(partition3, {33, 60}, 9, 12); // size(partition3) = 12 (3 grains)
//   extend(partition4, {33, 60}, 21, 6); // size(partition4) = 6
//
// And the final output would be
//
//   [
//      [{0, 10}, {15, 17}],
//      [{17, 20}, {33, 42}],
//      [{42, 54}],
//      [{54, 60}],
//   ]
//
template <typename T, std::size_t size(const T &),
          void extend(std::vector<T> &, const T&, unsigned, unsigned)>
std::vector<std::vector<T>>
splitRegionsAux(const std::vector<T> &items,
                unsigned grainSize, unsigned maxPartitions,
                unsigned minSizePerPartition,
                unsigned maxSizePerPartition) {

  // The list of regions (items) for each vertex (partition).
  std::vector<std::vector<T>> vertexItems;

  // The total size of all the regions.
  std::size_t totalSize =
      std::accumulate(items.begin(), items.end(), 0UL,
                      [](std::size_t totalSize, const T &item) {
    return totalSize + size(item);
  });

  if (totalSize == 0)
    return vertexItems;

  // Divide a by b, rounding up.
  auto udiv = [](std::size_t a, std::size_t b) { return (a + b - 1) / b; };

  // The number of grains required. E.g. if grainSize is 8 and totalSize
  // is 20 there are 3 grains.
  const auto numGrains = udiv(totalSize, grainSize);

  // The maximum number of grains in each vertex. For example if there
  // are 10 grains and 3 vertices this is 4.
  auto maxGrainsPerPartition = udiv(numGrains, maxPartitions);

  // Ensure that maxSizePerPartition is honoured
  if (maxGrainsPerPartition > maxSizePerPartition / grainSize)
    maxGrainsPerPartition = maxSizePerPartition / grainSize;

  // Adjust maxGrainsPerPartition if minSizePerPartitions is not 0.
  if (minSizePerPartition != 0) {
    const auto minGrainsPerPartition = udiv(minSizePerPartition, grainSize);
    const auto maxVerticesToCreate =
      std::max(1UL, numGrains / minGrainsPerPartition);
    maxGrainsPerPartition =
        std::max(maxGrainsPerPartition, udiv(numGrains, maxVerticesToCreate));
  }

  // The number of vertices to create.
  const auto verticesToCreate = udiv(numGrains, maxGrainsPerPartition);

  // Pointer to the item we are currently processing, and the offset into it.
  auto it = items.begin();
  unsigned offset = 0;

  vertexItems.resize(verticesToCreate);

  // Distribute the grains among verticesToCreate vertices.
  for (std::size_t vertex = 0; vertex != verticesToCreate; ++vertex) {
    // The first and last+1 grains in this vertex.
    const auto grainBegin = (vertex * numGrains) / verticesToCreate;
    const auto grainEnd = ((vertex + 1) * numGrains) / verticesToCreate;

    // The corresponding elements.
    const auto elemBegin = grainBegin * grainSize;
    const auto elemEnd = std::min(totalSize, grainEnd * grainSize);

    // Total size in this vertex.
    auto vertexSize = elemEnd - elemBegin;

    // While there are still elements to add to this vertex...
    while (vertexSize > 0) {

      // If we have finished adding the current item, go to the start of
      // the next one.
      if (offset == size(*it)) {
        offset = 0;
        ++it;
      }

      // Get the size of the item we are adding, starting from
      // the current offset and going to the end, or until we have enough
      // for vertexSize.
      const auto vertexItemSize =
          std::min(static_cast<std::size_t>(vertexSize),
                   size(*it) - offset);

      // Add (the part of) the item to the end of the list of items for this
      // vertex.
      extend(vertexItems[vertex], *it, offset, vertexItemSize);

      offset += vertexItemSize;
      vertexSize -= vertexItemSize;
    }
  }
  return vertexItems;
}

static std::size_t intervalSize(const poplar::Interval &i) {
  return i.size();
}

static void
extendIntervalVector(
    std::vector<poplar::Interval> &xs,
    const poplar::Interval &region,
    unsigned offset, unsigned size) {
  xs.emplace_back(region.begin() + offset, region.begin() + offset + size);
}

std::vector<std::vector<poplar::Interval>>
splitRegions(const std::vector<poplar::Interval> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition,
             unsigned maxElementsPerPartition) {
  return splitRegionsAux<poplar::Interval,
                         intervalSize,
                         extendIntervalVector>(
    regions, grainSize, maxPartitions, minElementsPerPartition,
    maxElementsPerPartition);
}

static std::size_t
intervalSequenceSize(const std::vector<poplar::Interval> &is) {
  return std::accumulate(is.begin(), is.end(), 0UL,
                         [](std::size_t size,
                            const poplar::Interval &i) {
                           return size + i.size();
                         });
}

static void
extendIntervalSequenceVector(
    std::vector<std::vector<poplar::Interval>> &xs,
    const std::vector<poplar::Interval> &regions,
    unsigned offset, unsigned size) {
  std::vector<poplar::Interval> slice;
  auto it = regions.begin();
  while (offset >= it->size()) {
    offset -= it->size();
    ++it;
  }
  while (size) {
    auto begin = it->begin() + offset;
    auto end = std::min(it->end(), it->begin() + offset + size);
    size -= (end - begin);
    slice.emplace_back(begin, end);
    offset = 0;
    ++it;
  }
  xs.emplace_back(std::move(slice));
}

std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegions(
  const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned grainSize, unsigned maxPartitions,
    unsigned minElementsPerPartition,
    unsigned maxElementsPerPartition) {
  return splitRegionsAux<std::vector<poplar::Interval>,
                         intervalSequenceSize,
                         extendIntervalSequenceVector>(
    regions, grainSize, maxPartitions, minElementsPerPartition,
    maxElementsPerPartition);
}

std::vector<std::vector<poplar::Interval>>
splitRegionsBetweenWorkers(
    const poplar::Target &target,
    const std::vector<poplar::Interval> &regions,
    unsigned grainSize,
    unsigned minElementsPerVertex,
    unsigned maxElementsPerVertex) {
  const auto workersPerTile = target.getNumWorkerContexts();
  return splitRegions(regions, grainSize, workersPerTile,
                      minElementsPerVertex, maxElementsPerVertex);
}

std::vector<std::vector<std::vector<poplar::Interval>>>
splitRegionsBetweenWorkers(
    const poplar::Target &target,
    const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned grainSize,
    unsigned minElementsPerVertex,
    unsigned maxElementsPerVertex) {
  const auto workersPerTile = target.getNumWorkerContexts();
  return splitRegions(regions, grainSize, workersPerTile,
                      minElementsPerVertex, maxElementsPerVertex);
}

std::size_t flattenIndex(const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &indices) {
  auto rank = shape.size();
  assert(indices.size() == rank);
  std::size_t index = 0;
  for (unsigned i = 0; i != rank; ++i) {
    index = index * shape[i] + indices[i];
  }
  return index;
}

std::size_t intervalSequenceNumElements(
    const std::vector<std::vector<poplar::Interval>> &seq) {
  std::size_t numElements = 0;
  for (const auto &s : seq) {
    numElements += intervalSequenceSize(s);
  }
  return numElements;
}

poplar::Tensor duplicate(poplar::Graph &graph, const poplar::Tensor &in,
                         poplar::program::Sequence &p) {
  poplar::Tensor out = graph.clone(in);
  p.add(poplar::program::Copy(in, out));
  return out;
}

} // end namespace popops
