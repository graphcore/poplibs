#include "popnn/ActivationMapping.hpp"
#include <cassert>

std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph, poplar::Tensor act) {
  const auto numActivations = act.numElements();
  unsigned chansPerGroup;
  if (act.getDimensionality() == 1) {
    chansPerGroup = 1;
  } else {
    assert(act.getDimensionality() == 4);
    chansPerGroup = act.dim(3);
  }
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  std::vector<unsigned> mapping;
  mapping.reserve(numTiles + 1);
  mapping.emplace_back(0);
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
  const auto maxGroupsPerTile = (numGroups + numTiles - 1) / numTiles;
  const auto tilesToUse = (numGroups + maxGroupsPerTile - 1) / maxGroupsPerTile;
  for (unsigned tile = 0; tile != tilesToUse; ++tile) {
    const auto groupEnd = ((tile + 1) * numGroups) / tilesToUse;
    mapping.emplace_back(groupEnd * chansPerGroup);
  }
  mapping.resize(numTiles + 1, mapping.back());
  return mapping;
}

void mapActivations(poplar::Graph &graph, poplar::Tensor act) {
  auto actMapping = computeActivationsMapping(graph, act);
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  assert(actMapping.size() == numTiles + 1);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    graph.setTileMapping(act.flatten().slice(actMapping[tile],
                                             actMapping[tile + 1]),
                         tile);
  }
}

void mapTensor(poplar::Graph &graph, poplar::Tensor t) {
  std::uint64_t size = t.numElements();
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  for (unsigned i = 0; i < numTiles; ++i) {
    const auto begin = (size * i) / numTiles;
    const auto end = (size * (i + 1)) / numTiles;
    if (begin == end)
      continue;
    graph.setTileMapping(t.flatten().slice(begin, end), i);
  }
}
