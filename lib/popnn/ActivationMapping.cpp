#include "popnn/ActivationMapping.hpp"
#include <cassert>

void applyTensorMapping(poplar::Graph &graph, poplar::Tensor t,
                        const std::vector<unsigned> &mapping) {
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  assert(mapping.size() == numTiles + 1);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    graph.setTileMapping(t.flatten().slice(mapping[tile], mapping[tile + 1]),
                         tile);
  }
}

std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph, poplar::Tensor act,
                          unsigned batchNum, unsigned batchSize) {
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  const auto numActivations = act.numElements();
  unsigned chansPerGroup, beginTile, endTile;
  if (act.rank() == 1) {
    chansPerGroup = 1;
    beginTile = 0;
    endTile = numTiles;
  } else {
    assert(act.rank() == 4);
    chansPerGroup = act.dim(3);
    const auto batchElemsPerTile = (batchSize + numTiles - 1) / numTiles;
    const auto numBatchGroups =
        (batchSize + batchElemsPerTile - 1) / batchElemsPerTile;
    const auto tilesPerBatchGroup =
        numTiles / numBatchGroups;
    beginTile = batchNum / batchElemsPerTile * tilesPerBatchGroup;
    endTile = beginTile + tilesPerBatchGroup;
  }
  const auto numBatchTiles = endTile - beginTile;
  std::vector<unsigned> mapping;
  mapping.resize(numTiles + 1);
  const auto numGroups = numActivations / chansPerGroup;
  // Instead of spreading activations across all tiles, compute the maximum
  // number of activations that would need to be stored on a tile if activations
  // were spread evenly and use the minimum number of tiles necessary to ensure
  // that this maximum is not exceeded.
  // This strategy reduces the number of tiles that activations are spread over.
  // This reduces the amount of exchange code needed in the next layer
  // as input data is spread over fewer tiles and therefore fewer set receive
  // mux / set receive pointer instructions are required.
  // The amount of work a tile has to perform during the reduce and complete
  // phases is proportional to the number of activations. Because this strategy
  // does not increase the maximum number of activations on a tile, the
  // execution time of the reduce and complete phases should remain roughly the
  // same.
  const auto maxGroupsPerTile = (numGroups + numBatchTiles - 1) / numBatchTiles;
  const auto tilesToUse = (numGroups + maxGroupsPerTile - 1) / maxGroupsPerTile;
  for (unsigned tile = 0; tile != tilesToUse; ++tile) {
    const auto groupEnd = (tile * numGroups) / tilesToUse;
    mapping[tile + beginTile] = groupEnd * chansPerGroup;
  }
  for (unsigned tile = beginTile + tilesToUse; tile != numTiles + 1; ++tile)
    mapping[tile] = numGroups * chansPerGroup;
  return mapping;
}

void mapActivations(poplar::Graph &graph, poplar::Tensor act) {
  auto batchSize = act.dim(0);
  for (unsigned i = 0; i != batchSize; ++i) {
    auto actMapping = computeActivationsMapping(graph, act[i], i, batchSize);
    applyTensorMapping(graph, act[i], actMapping);
  }
}

std::vector<unsigned> computeTensorMapping(const poplar::Graph &graph,
                                           poplar::Tensor t,
                                           unsigned grainSize)
{
  const auto numElements = t.numElements();
  const auto numGroups = (numElements + grainSize - 1) / grainSize;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  std::vector<unsigned> mapping;
  mapping.reserve(numTiles + 1);
  mapping.emplace_back(0);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto end = std::min((numGroups * (tile + 1)) / numTiles * grainSize,
                              numElements);
    mapping.emplace_back(end);
  }
  mapping.resize(numTiles + 1, mapping.back());
  return mapping;
}

void mapTensor(poplar::Graph &graph, poplar::Tensor t) {
  applyTensorMapping(graph, t, computeTensorMapping(graph, t));
}
