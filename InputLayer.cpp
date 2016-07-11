#include "InputLayer.hpp"

void InputLayer::
init(Graph &graph, std::mt19937 &randomEngine,
     IPUModelEngineBuilder::TileMapping &mapping) {
  const auto dType = getDType();
  Layer *next = getNextLayer();
  // Re-arrange so that the channels are the major
  auto numGroups = next->getNumChannelGroupsIn(data.dim[0], data.dim[1],
                                             data.dim[2]);
  if (!numGroups)
    numGroups = 1;
  const auto dim = std::vector<size_t>({numGroups, data.dim[0], data.dim[1],
                                        data.dim[2]/numGroups});
  out = graph.addTensor(dType, dim, makeLayerName("input"));
  mapTensor(out, mapping);
}
