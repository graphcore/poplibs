#include "Util.hpp"

#include <algorithm>
#include <numeric>

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

std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegions(const std::vector<poplar::Interval<std::size_t>> &regions,
             unsigned grainSize, unsigned maxPartitions,
             unsigned minElementsPerPartition) {
  std::vector<std::vector<poplar::Interval<std::size_t>>> vertexRegions;
  const auto numElements =
      std::accumulate(regions.begin(), regions.end(), 0U,
                      [](unsigned numElements,
                         const poplar::Interval<std::size_t> &region) {
    return numElements + region.end() - region.begin();
  });
  if (numElements == 0)
    return vertexRegions;
  const auto numGroups = (numElements + grainSize - 1) / grainSize;
  auto maxGroupsPerPartition =
    (numGroups + maxPartitions - 1) / maxPartitions;
  if (minElementsPerPartition) {
    const auto minGroupsPerPartition =
      (minElementsPerPartition + grainSize - 1) / grainSize;
    const auto maxVerticesToCreate =
      std::max(1U, numGroups / minGroupsPerPartition);
    maxGroupsPerPartition =
        std::max(maxGroupsPerPartition,
                 (numGroups + maxVerticesToCreate - 1) / maxVerticesToCreate);
  }
  const auto verticesToCreate =
    (numGroups + maxGroupsPerPartition - 1) / maxGroupsPerPartition;
  auto it = regions.begin();
  unsigned count = 0;
  vertexRegions.resize(verticesToCreate);
  for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
    const auto groupBegin = (vertex * numGroups) / verticesToCreate;
    const auto groupEnd = ((vertex + 1) * numGroups) / verticesToCreate;
    const auto elemBegin = groupBegin * grainSize;
    const auto elemEnd = std::min(numElements, groupEnd * grainSize);
    auto vertexElements = elemEnd - elemBegin;
    while (vertexElements) {
      if (count == it->end() - it->begin()) {
        count = 0;
        ++it;
      }
      const auto vertexRegionSize =
          std::min(static_cast<std::size_t>(vertexElements),
                   it->end() - it->begin() - count);
      const auto vertexRegionBegin = it->begin() + count;
      const auto vertexRegionEnd = vertexRegionBegin + vertexRegionSize;
      vertexRegions[vertex].emplace_back(vertexRegionBegin, vertexRegionEnd);
      count += vertexRegionSize;
      vertexElements -= vertexRegionSize;
    }
  }
  return vertexRegions;
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
