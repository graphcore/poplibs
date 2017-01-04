#ifndef _Util_hpp_
#define _Util_hpp_

#include <poplar/Device.hpp>
#include <poplar/Graph.hpp>
#include <vector>
#include <utility>

void mergeAdjacentRegions(
    std::vector<std::pair<unsigned, unsigned>> &regions);

void mergeAdjacentRegions(
    std::vector<std::vector<std::pair<unsigned, unsigned>>> &mapping);

// Given a set of contiguous regions per tile, partition these regions
// between vertices on that tile, respecting the specified grain size.
// Regions may be split to balance the work across vertices.
void splitRegionsBetweenWorkers(
    const poplar::DeviceInfo &deviceInfo,
    const std::vector<std::pair<unsigned, unsigned>> &regions,
    std::vector<std::vector<std::pair<unsigned, unsigned>>> &vertexRegions,
    unsigned grainSize);

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

#endif // _Util_hpp_
