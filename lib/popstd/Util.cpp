#include "popstd/Util.hpp"

#include <algorithm>
#include <numeric>

namespace popstd {

void mergeAdjacentRegions(
    std::vector<poplar::Interval<std::size_t>> &regions) {
  std::vector<poplar::Interval<std::size_t>> newRegions;
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
    std::vector<std::vector<poplar::Interval<std::size_t>>> &mapping) {
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    mergeAdjacentRegions(mapping[tile]);
  }
}

template <typename T, std::size_t size(const T &),
          void extend(std::vector<T> &, const T&, unsigned, unsigned)>
std::vector<std::vector<T>>
splitRegionsAux(const std::vector<T> &items,
                unsigned grainSize, unsigned maxPartitions,
                unsigned minSizePerPartition) {
  std::vector<std::vector<T>> vertexItems;
  std::size_t totalSize =
      std::accumulate(items.begin(), items.end(), 0UL,
                      [](std::size_t totalSize, const T &item) {
    return totalSize + size(item);
  });
  if (totalSize == 0)
    return vertexItems;
  const auto numGroups = (totalSize + grainSize - 1) / grainSize;
  auto maxGroupsPerPartition =
    (numGroups + maxPartitions - 1) / maxPartitions;
  if (minSizePerPartition) {
    const auto minGroupsPerPartition =
      (minSizePerPartition + grainSize - 1) / grainSize;
    const auto maxVerticesToCreate =
      std::max(1UL, numGroups / minGroupsPerPartition);
    maxGroupsPerPartition =
        std::max(maxGroupsPerPartition,
                 (numGroups + maxVerticesToCreate - 1) / maxVerticesToCreate);
  }
  const auto verticesToCreate =
    (numGroups + maxGroupsPerPartition - 1) / maxGroupsPerPartition;
  auto it = items.begin();
  unsigned offset = 0;
  vertexItems.resize(verticesToCreate);
  for (std::size_t vertex = 0; vertex != verticesToCreate; ++vertex) {
    const auto groupBegin = (vertex * numGroups) / verticesToCreate;
    const auto groupEnd = ((vertex + 1) * numGroups) / verticesToCreate;
    const auto elemBegin = groupBegin * grainSize;
    const auto elemEnd = std::min(totalSize, groupEnd * grainSize);
    auto vertexSize = elemEnd - elemBegin;
    while (vertexSize) {
      if (offset == size(*it)) {
        offset = 0;
        ++it;
      }
      const auto vertexItemSize =
          std::min(static_cast<std::size_t>(vertexSize),
                   size(*it) - offset);
      extend(vertexItems[vertex], *it, offset, vertexItemSize);
      offset += vertexItemSize;
      vertexSize -= vertexItemSize;
    }
  }
  return vertexItems;
}

static std::size_t intervalSize(const poplar::Interval<std::size_t> &i) {
  return i.size();
}

static void
extendIntervalVector(
    std::vector<poplar::Interval<std::size_t>> &xs,
    const poplar::Interval<std::size_t> &region,
    unsigned offset, unsigned size) {
  xs.emplace_back(region.begin() + offset, region.begin() + offset + size);
}

std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegions(const std::vector<poplar::Interval<std::size_t>> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition) {
  return splitRegionsAux<poplar::Interval<std::size_t>,
                         intervalSize,
                         extendIntervalVector>(
    regions, grainSize, maxPartitions, minElementsPerPartition);
}

static std::size_t
intervalSequenceSize(const std::vector<poplar::Interval<std::size_t>> &is) {
  return std::accumulate(is.begin(), is.end(), 0UL,
                         [](std::size_t size,
                            const poplar::Interval<std::size_t> &i) {
                           return size + i.size();
                         });
}

static void
extendIntervalSequenceVector(
    std::vector<std::vector<poplar::Interval<std::size_t>>> &xs,
    const std::vector<poplar::Interval<std::size_t>> &regions,
    unsigned offset, unsigned size) {
  std::vector<poplar::Interval<std::size_t>> slice;
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

std::vector<std::vector<std::vector<poplar::Interval<std::size_t>>>>
splitRegions(
  const std::vector<std::vector<poplar::Interval<std::size_t>>> &regions,
    unsigned grainSize, unsigned maxPartitions,
    unsigned minElementsPerPartition) {
  return splitRegionsAux<std::vector<poplar::Interval<std::size_t>>,
                         intervalSequenceSize,
                         extendIntervalSequenceVector>(
    regions, grainSize, maxPartitions, minElementsPerPartition);
}

std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<poplar::Interval<std::size_t>> &regions,
    unsigned grainSize,
    unsigned minElementsPerVertex) {
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  return splitRegions(regions, grainSize, workersPerTile,
                      minElementsPerVertex);
}

std::vector<std::vector<std::vector<poplar::Interval<std::size_t>>>>
splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<std::vector<poplar::Interval<std::size_t>>> &regions,
    unsigned grainSize,
    unsigned minElementsPerVertex) {
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  return splitRegions(regions, grainSize, workersPerTile,
                      minElementsPerVertex);
}

std::vector<std::size_t> unflattenIndex(const std::vector<std::size_t> &shape,
                                        std::size_t index) {
  std::vector<std::size_t> coord;
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    const auto dim = *it;
    coord.push_back(index % dim);
    index /= dim;
  }
  std::reverse(coord.begin(), coord.end());
  return coord;
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

} // end namespace popstd
