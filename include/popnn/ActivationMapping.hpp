#ifndef __ActivationMapping_hpp__
#define __ActivationMapping_hpp__
#include <vector>
#include "DeviceInfo.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/IPUModelEngine.hpp"

std::vector<unsigned> computeActivationsMapping(poplar::Tensor t,
                                                const DeviceInfo &deviceInfo);

void mapActivations(poplar::Tensor t,
                    poplar::IPUModelEngineBuilder::TileMapping &mapping,
                    const DeviceInfo &deviceInfo);


void mapTensor(poplar::Tensor t,
               poplar::IPUModelEngineBuilder::TileMapping &mapping,
               const DeviceInfo &deviceInfo);

/// Given a mapping of data to tiles, use the specified builder function to
/// create vertices that operate on that data. Each vertex operates on data
/// that is local to the tile it runs on. The number of vertices per tile is
/// decided based on the number of worker contexts.
template <class Builder>
void buildTransform(const std::vector<unsigned> &tileMapping,
                    const DeviceInfo &deviceInfo, Builder &&builder) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto workersPerTile = deviceInfo.getNumWorkerContexts();
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
