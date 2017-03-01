#include "Cast.hpp"

#include <poplar/Graph.hpp>
#include "Util.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"

using namespace poplar;
using namespace poplar::program;

Program
cast(Graph &graph, Tensor src, Tensor dst, const std::string &debugPrefix) {
  auto srcType = graph.getTensorElementType(src);
  auto dstType = graph.getTensorElementType(dst);
  if (srcType == dstType)
    return Copy(src, dst);
  auto cs = graph.addComputeSet(debugPrefix + "/Cast");
  cast(graph, src, dst, cs);
  return Execute(cs);
}

#ifndef NDEBUG
static bool
mappingIsComplete(const Tensor &t,
                  const std::vector<
                    std::vector<Interval<std::size_t>>
                  > &mapping) {
  unsigned mappedElements = 0;
  for (const auto &regions : mapping) {
    for (const auto &region : regions) {
      mappedElements += region.end - region.begin;
    }
  }
  return mappedElements == t.numElements();
}
#endif

void
cast(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
  assert(src.shape() == dst.shape());
  src = src.flatten();
  dst = dst.flatten();
  const auto srcType = graph.getTensorElementType(src);
  const auto dstType = graph.getTensorElementType(dst);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto vectorWidth = deviceInfo.getFloatVectorWidth();
  std::vector<std::vector<Interval<std::size_t>>> mapping;
  if (srcType == "float") {
    mapping = graph.getTileMapping(src);
    assert(mappingIsComplete(src, mapping));
  } else {
    mapping = graph.getTileMapping(dst);
    assert(mappingIsComplete(dst, mapping));
  }
  buildTransform2D(
    graph, mapping, vectorWidth,
    [&](const std::vector<Interval<std::size_t>> &regions,
        unsigned tile) {
    const auto numRegions = regions.size();
    assert(numRegions != 0);
    VertexRef v;
    if (numRegions == 1) {
      v = graph.addVertex(cs, templateVertex("popnn::Cast", srcType, dstType));
      const auto &region = regions.front();
      const auto regionBegin = region.begin;
      const auto regionEnd = region.end;
      graph.connect(v["src"], src.slice(regionBegin, regionEnd));
      graph.connect(v["dst"], dst.slice(regionBegin, regionEnd));
    } else {
      v = graph.addVertex(cs, templateVertex("popnn::Cast2D", srcType,
                                             dstType));
      graph.setFieldSize(v["src"], numRegions);
      graph.setFieldSize(v["dst"], numRegions);
      for (unsigned i = 0; i != numRegions; ++i) {
        const auto &region = regions[i];
        const auto regionBegin = region.begin;
        const auto regionEnd = region.end;
        graph.connect(v["src"][i], src.slice(regionBegin, regionEnd));
        graph.connect(v["dst"][i], dst.slice(regionBegin, regionEnd));
      }
    }
    graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    graph.setTileMapping(v, tile);
  });
}
