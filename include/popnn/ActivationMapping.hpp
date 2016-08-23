#ifndef __ActivationMapping_hpp__
#define __ActivationMapping_hpp__
#include <vector>
#include "poplar/Graph.hpp"

std::vector<unsigned> computeActivationsMapping(const poplar::Graph &graph,
                                                poplar::Tensor t);

void mapActivations(poplar::Graph &graph, poplar::Tensor t);


void mapTensor(poplar::Graph &graph, poplar::Tensor t);

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

#endif // __ActivationMapping_hpp__
