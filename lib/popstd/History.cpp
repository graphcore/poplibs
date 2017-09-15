#include <popstd/History.hpp>
#include <popstd/Util.hpp>
#include <popstd/VertexTemplates.hpp>
#include <numeric>
#include <algorithm>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popstd {

History::History(Graph &graph, const std::string &dataType,
                 unsigned size, const std::vector<std::size_t> &shape) :
  graph(graph), size_(size), shape(shape) {
 auto N = std::accumulate(shape.begin(), shape.end(), 1UL,
                          std::multiplies<std::size_t>());
 hist = graph.addTensor(dataType, {N, size});
 auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
 auto regions = splitRegions({{0, N}}, 2, numTiles);
 for (unsigned tile = 0; tile < regions.size(); ++tile) {
   const auto &tileRegions = regions[tile];
   for (const auto &r : tileRegions) {
      graph.setTileMapping(hist.slice(r), tile);
   }
 }
 index = graph.addTensor("unsigned", {1}, "index");
 graph.setInitialValue(index[0], 0);
 graph.setTileMapping(index, 0);
}

Tensor History::prev(unsigned i, Sequence &seq,
                     const std::string &debugPrefix) {
  if (i > size_)
    std::abort();
  auto t = graph.addTensor(hist.elementType(), shape);
  auto N = t.numElements();
  graph.setTileMapping(t, graph.getTileMapping(hist.slice({0, 0}, {N, 1})));
  const auto dType = t.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/HistorySelect");
  auto mapping = graph.getTileMapping(t);
  const auto numTiles = deviceInfo.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto vertexRegions = splitRegionsBetweenWorkers(deviceInfo, mapping[tile],
                                                    vectorWidth,
                                                    2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      std::vector<Tensor> histSlices;
      for (const auto &region : regions) {
        histSlices.push_back(hist.slice(region).flatten());
      }
      auto v = graph.addVertex(cs, templateVertex("popstd::HistSelect",
                                                  dType),
                                {{"index", index[0]},
                                 {"in", histSlices},
                                 {"out", tFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["offset"], i);
      graph.setTileMapping(v, tile);
    }
  }
  seq.add(Execute(cs));
  return t;
}

void History::add(Tensor in, Sequence &seq, const std::string &debugPrefix) {
  auto t = graph.addTensor(hist.elementType(), shape);
  auto N = t.numElements();
  graph.setTileMapping(t, graph.getTileMapping(hist.slice({0, 0}, {N, 1})));
  seq.add(Copy(in, t));
  const auto dType = t.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/HistorySet");
  auto mapping = graph.getTileMapping(t);
  const auto numTiles = deviceInfo.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    auto vertexRegions = splitRegionsBetweenWorkers(deviceInfo, mapping[tile],
                                                    vectorWidth,
                                                    2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      std::vector<Tensor> histSlices;
      for (const auto &region : regions) {
        histSlices.push_back(hist.slice(region).flatten());
      }
      auto v = graph.addVertex(cs, templateVertex("popstd::HistSet",
                                                  dType),
                                {{"index", index[0]},
                                 {"out", histSlices},
                                 {"in", tFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  ComputeSet csIndexIncr = graph.addComputeSet(debugPrefix + "/HistorySet");
  auto v = graph.addVertex(csIndexIncr, "popstd::HistIncrIndex",
                            {{"index", index[0]}});
  graph.setInitialValue(v["hSize"], size_);
  graph.setTileMapping(v, 0);
  seq.add(Execute(csIndexIncr));
  seq.add(Execute(cs));
}

}
