#include "Cast.hpp"

#include <poplar/Graph.hpp>
#include "Util.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"

using namespace poplar;
using namespace poplar::program;

void
cast(Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     Tensor src, Tensor dst, ComputeSet cs) {
  auto srcType = graph.getTensorElementType(src);
  auto dstType = graph.getTensorElementType(dst);

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  buildTransform(dstActivationMapping, graph, [&](unsigned begin,
                                                  unsigned end,
                                                  unsigned tile) {
    auto v = graph.addVertex(cs,
                             templateVertex("popnn::Cast", srcType, dstType),
                             {{"src", src.flatten().slice(begin, end)},
                              {"dst", dst.flatten().slice(begin, end)}});
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
}


Program
cast(Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     Tensor src, Tensor dst, const std::string &debugPrefix) {
  auto srcType = graph.getTensorElementType(src);
  auto dstType = graph.getTensorElementType(dst);
  if (srcType == dstType)
    return Copy(dst, src);
  auto cs = graph.createComputeSet(debugPrefix + "/Cast");
  cast(graph, dstActivationMapping, src, dst, cs);
  return Execute(cs);
}

void
cast(poplar::Graph &graph,
     const std::vector<
       std::vector<std::pair<unsigned, unsigned>>
     > &mapping,
     poplar::Tensor src, poplar::Tensor dst,
     poplar::ComputeSet cs) {
  assert(src.shape() == dst.shape());
  src = src.flatten();
  dst = dst.flatten();
  const auto srcType = graph.getTensorElementType(src);
  const auto dstType = graph.getTensorElementType(dst);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto vectorWidth = deviceInfo.getFloatVectorWidth();
  buildTransform2D(
    graph, mapping, vectorWidth,
    [&](const std::vector<std::pair<unsigned, unsigned>> &regions,
        unsigned tile) {
    const auto numRegions = regions.size();
    assert(numRegions != 0);
    VertexRef v;
    if (numRegions == 1) {
      v = graph.addVertex(cs, templateVertex("popnn::Cast", srcType, dstType));
      const auto &region = regions.front();
      const auto regionBegin = region.first;
      const auto regionEnd = region.second;
      graph.connect(v["src"], src.slice(regionBegin, regionEnd));
      graph.connect(v["dst"], dst.slice(regionBegin, regionEnd));
    } else {
      v = graph.addVertex(cs, templateVertex("popnn::Cast2D", srcType,
                                             dstType));
      graph.setFieldSize(v["src"], numRegions);
      graph.setFieldSize(v["dst"], numRegions);
      for (unsigned i = 0; i != numRegions; ++i) {
        const auto &region = regions[i];
        const auto regionBegin = region.first;
        const auto regionEnd = region.second;
        graph.connect(v["src"][i], src.slice(regionBegin, regionEnd));
        graph.connect(v["dst"][i], dst.slice(regionBegin, regionEnd));
      }
    }
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    graph.setTileMapping(v, tile);
  });
}
