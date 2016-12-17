#include "Zero.hpp"

#include <poplar/Graph.hpp>
#include "Util.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;

void
zero(Graph &graph,
     Tensor t,
     const std::vector<std::pair<unsigned, unsigned>> &tileRegions,
     unsigned tile,
     ComputeSet zeroCS) {
  t = t.flatten();
  const auto dType = graph.getTensorElementType(t);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned vectorWidth;
  if (dType == "float")
    vectorWidth = deviceInfo.getFloatVectorWidth();
  else
    vectorWidth = deviceInfo.getHalfVectorWidth();
  std::vector<std::vector<std::pair<unsigned, unsigned>>> vertexRegions;
  splitRegionsBetweenWorkers(deviceInfo, tileRegions, vertexRegions,
                             vectorWidth);
  for (const auto &regions : vertexRegions) {
    VertexRef v;
    const auto numRegions = regions.size();
    if (numRegions == 0)
      continue;
    if (numRegions == 1) {
      v = graph.addVertex(zeroCS, templateVertex("popnn::Zero", dType));
      const auto &region = regions.front();
      const auto regionBegin = region.first;
      const auto regionEnd = region.second;
      auto out = t.slice(regionBegin, regionEnd);
      graph.connect(v["out"], out);
    } else {
      v = graph.addVertex(zeroCS, templateVertex("popnn::Zero2D", dType));
      graph.setFieldSize(v["out"], regions.size());
      for (unsigned i = 0; i != numRegions; ++i) {
        const auto &region = regions[i];
        const auto regionBegin = region.first;
        const auto regionEnd = region.second;
        auto out = t.slice(regionBegin, regionEnd);
        graph.connect(v["out"][i], out);
      }
    }
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    graph.setTileMapping(v, tile);
  }
}

void
zero(Graph &graph,
     const Tensor &t,
     const std::vector<
       std::vector<std::pair<unsigned, unsigned>>
     > &mapping,
     ComputeSet zeroCS) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    zero(graph, t, mapping[tile], tile, zeroCS);
  }
}
