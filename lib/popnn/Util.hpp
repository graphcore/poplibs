#ifndef _Util_hpp_
#define _Util_hpp_

#include <poplar/Device.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Graph.hpp>
#include <vector>

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
// between vertices on that tile, respecting the specified grain size.
// Regions may be split to balance the work across vertices.
std::vector<std::vector<poplar::Interval<std::size_t>>>
splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<poplar::Interval<std::size_t>> &regions,
    unsigned grainSize, unsigned minElementsPerPartition = 0);

/// Given a mapping of data to tiles, use the specified builder function to
/// create vertices that operate on that data. Each vertex operates on data
/// that is local to the tile it runs on. The number of vertices per tile is
/// decided based on the number of worker contexts.
template <class Builder>
void buildTransform(const std::vector<unsigned> &tileMapping,
                    const poplar::Graph &graph, Builder &&builder) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.numWorkerContexts;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileElementBegin = tileMapping[tile];
    const auto tileElementEnd = tileMapping[tile + 1];
    const auto tileNumElements = tileElementEnd - tileElementBegin;
    if (tileNumElements == 0)
      continue;
    const auto maxElementsPerWorker =
      (tileNumElements + workersPerTile - 1) / workersPerTile;
    const auto verticesToCreate =
      (tileNumElements + maxElementsPerWorker - 1) / maxElementsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto elementBegin =
          tileElementBegin +
          (vertex * tileNumElements) / verticesToCreate;
      const auto elementEnd =
          tileElementBegin +
          ((vertex + 1) * tileNumElements) / verticesToCreate;
      builder(elementBegin, elementEnd, tile);
    }
  }
}

/// Given a list of regions on a tile, split the regions between workers and
/// call the builder with the list of regions for each worker.
template <class Builder>
void buildTransform2D(
    poplar::Graph &graph,
    const std::vector<poplar::Interval<std::size_t>> &regions,
    unsigned grainSize,
    Builder &&builder) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto vertexRegions =
      splitRegionsBetweenWorkers(deviceInfo, regions,grainSize);
  for (const auto &regions : vertexRegions) {
    if (regions.empty())
      continue;
    builder(regions);
  }
}

/// Given a tile mapping, split the regions on each tile between workers on that
/// tile and call for each worker call the builder with the list of regions for
/// that worker and the tile number.
template <class Builder>
void buildTransform2D(
    poplar::Graph &graph,
    const std::vector<
      std::vector<poplar::Interval<std::size_t>>
    > &mapping,
    unsigned grainSize,
    Builder &&builder) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    buildTransform2D(
      graph, mapping[tile], grainSize,
      [&builder,tile](const std::vector<poplar::Interval<std::size_t>> &
                      vertexRegions) {
      builder(vertexRegions, tile);
    });
  }
}

#endif // _Util_hpp_
