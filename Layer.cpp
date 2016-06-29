#include "Layer.hpp"

#include "Net.hpp"
#include <random>
#include <cmath>
unsigned Layer::getWorkerContextsPerTile() const {
  return net.getWorkerContextsPerTile();
}

IPUModelEngineBuilder &Layer::
getIPUModelEngineBuilder() const { return net.getIPUModelEngineBuilder(); }

unsigned Layer::getNumIPUs() const { return net.getNumIPUs(); }

unsigned Layer::getTilesPerIPU() const { return net.getTilesPerIPU(); }

float Layer::getLearningRate() const { return net.getLearningRate(); }

Layer *Layer::getPrevLayer() const { return net.getPrevLayer(index); }

Layer *Layer::getNextLayer() const { return net.getNextLayer(index); }

const std::string &Layer::getDType() const { return net.getDType(); }

unsigned Layer::getDTypeSize() const {
  if (getDType() == "float")
    return 4;
  assert(getDType() == "half");
  return 2;
}

unsigned Layer::getBatchSize() const { return net.getBatchSize(); }

enum NetType Layer::getNetType() const { return net.getNetType(); }

const NetOptions &Layer::getNetOptions() const { return net.options; }

void Layer::mapTensor(Tensor t, IPUModelEngineBuilder::TileMapping *mapping) {
  if (!mapping)
    return;
  std::uint64_t size = t.numElements();
  const auto numTiles = getTilesPerIPU() * getNumIPUs();
  for (unsigned i = 0; i < numTiles; ++i) {
    const auto begin = (size * i) / numTiles;
    const auto end = (size * (i + 1)) / numTiles;
    if (begin == end)
      continue;
    mapping->setMapping(t.flatten().slice(begin, end), i);
  }
}

void Layer::mapComputeSet(const Graph &graph, ComputeSet c,
                          IPUModelEngineBuilder::TileMapping *mapping) {
  if (!mapping)
    return;
  auto cs = graph.getComputeSet(c);
  std::uint64_t size = cs.size();
  const auto numTiles = getTilesPerIPU() * getNumIPUs();
  for (unsigned i = 0; i < numTiles; ++i) {
    const auto begin = (size * i) / numTiles;
    const auto end = (size * (i + 1)) / numTiles;
    if (begin == end)
      continue;
    for (unsigned j = begin; j != end; ++j) {
      mapping->setMapping(cs[j], i);
    }
  }
}

std::vector<unsigned> Layer::computeActivationsMapping(Tensor act) {
  const auto numActivations = act.numElements();
  const auto chansPerGroup = act.dim(3);
  const auto numTiles = getTilesPerIPU() * getNumIPUs();
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

void Layer::mapActivations(Tensor act,
                           IPUModelEngineBuilder::TileMapping *mapping) {
  if (!mapping)
    return;
  auto actMapping = computeActivationsMapping(act);
  const auto numTiles = getTilesPerIPU() * getNumIPUs();
  assert(actMapping.size() == numTiles + 1);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    mapping->setMapping(act.flatten().slice(actMapping[tile],
                                            actMapping[tile + 1]),
                        tile);
  }
}

std::string Layer::makeLayerName(const std::string &name) {
  return name + ".layer" + std::to_string(index);
}

std::unique_ptr<float[]>
createRandomWeightInitializers(Tensor t, float mean, float variance,
                               std::mt19937 &randomEngine) {
  const auto numWeights = t.numElements();
  auto inits = std::unique_ptr<float[]>(new float[numWeights]);

  std::normal_distribution<> dist(mean, variance);
  for (unsigned i = 0; i < numWeights; ++i)
    inits[i] = dist(randomEngine);

  return inits;
}
