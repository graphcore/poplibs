#include "poputil/ActivationMapping.hpp"

#include "poputil/Util.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include "poplibs_support/gcd.hpp"

namespace popstd {

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
computeActivationsMapping(const poplar::Graph &graph,
                          const std::string &actType,
                          const std::vector<std::size_t> &shape,
                          unsigned batchNum, unsigned batchSize) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numActivations = std::accumulate(shape.begin(), shape.end(),
                                              1UL,
                                              std::multiplies<std::size_t>());
  const auto actTypeSize = "float" ? 4 : 2;
  unsigned grainSize = actType == "float" ? deviceInfo.getFloatVectorWidth() :
                                            deviceInfo.getHalfVectorWidth();
  // Limit the minimum number of activation bytes per tile to reduce the
  // amount of exchange code. Increasing this constant reduces exchange code
  // size and increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + actTypeSize - 1) / minBytesPerTile;
  unsigned beginTile, endTile;
  const auto rank = shape.size();
  if (rank == 1) {
    beginTile = 0;
    endTile = numTiles;
  } else {
    assert(rank == 4);
    const unsigned chansPerGroup = shape[3];
    // The grain size is chosen to avoid splitting the tensor at a point
    // that will require the incoming pointer to be changed if the messages from
    // the source tiles are received in the wrong order. The convolution layers
    // access elements in groups of chansPerGroup. Other layers (e.g.
    // FwdNonLinearity) flatten the tensor and access elements in groups of the
    // vector width. We don't know which layer comes next so hedge our bets by
    // using least common multiple of the vector width and chansPerGroup.
    grainSize = lcm(grainSize, chansPerGroup);
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
  std::vector<poplar::Interval> regions = {
    {0, numActivations}
  };
  const auto perTileRegions =
      splitRegions(regions, grainSize, numBatchTiles, minElementsPerTile);
  for (unsigned tile = beginTile; tile != numTiles; ++tile) {
    if (tile - beginTile < perTileRegions.size() &&
        !perTileRegions[tile - beginTile].empty()) {
      assert(perTileRegions[tile - beginTile].size() == 1);
      const auto &region = perTileRegions[tile - beginTile].front();
      assert(mapping[tile] == region.begin());
      mapping[tile + 1] = region.end();
    } else {
      mapping[tile + 1] = mapping[tile];
    }
  }
  assert(mapping[endTile] == numActivations);
  return mapping;
}

std::vector<unsigned>
computeActivationsMapping(const poplar::Graph &graph, poplar::Tensor act,
                          unsigned batchNum, unsigned batchSize) {
  return computeActivationsMapping(graph, act.elementType(),
                                   act.shape(), batchNum, batchSize);
}

void mapActivations(poplar::Graph &graph, poplar::Tensor act) {
  poplar::Tensor actExt = act;
  if (actExt.rank() == 4) {
    // In cases when there is no channel grouping, extend tensor such that
    // number of groups is 1
    actExt = actExt.reshape({act.dim(0), act.dim(1), act.dim(2), 1, act.dim(3)})
                   .dimShuffle({0, 3, 1, 2, 4});
  }
  auto batchSize = actExt.dim(0);
  for (unsigned i = 0; i != batchSize; ++i) {
    auto actMapping = computeActivationsMapping(graph, actExt[i], i, batchSize);
    applyTensorMapping(graph, actExt[i], actMapping);
  }
}

std::vector<unsigned>
computeTensorMapping(const poplar::Graph &graph,
                     const std::vector<std::size_t> &shape,
                     unsigned grainSize)
{
  const auto numElements = std::accumulate(shape.begin(), shape.end(),
                                           1UL, std::multiplies<std::size_t>());
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

std::vector<unsigned>
computeTensorMapping(const poplar::Graph &graph,
                     poplar::Tensor t,
                     unsigned grainSize) {
  return computeTensorMapping(graph, t.shape(), grainSize);
}

void mapTensor(poplar::Graph &graph, poplar::Tensor t) {
  applyTensorMapping(graph, t, computeTensorMapping(graph, t));
}

} // end namespace popstd
