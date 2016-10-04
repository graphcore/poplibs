#include "Cast.hpp"
#include <poplar/Graph.hpp>
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"

using namespace poplar;
using namespace poplar::program;

Program
cast(Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     Tensor src, Tensor dst) {
  auto srcType = graph.getTensorElementType(src);
  auto dstType = graph.getTensorElementType(dst);
  if (srcType == dstType)
    return Copy(dst, src);

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto cs = graph.createComputeSet("cast");
  buildTransform(dstActivationMapping, graph, [&](unsigned begin,
                                                  unsigned end,
                                                  unsigned tile) {
    auto v = graph.addVertex(cs,
                             templateVertex("Cast", srcType, dstType),
                             {{"src", src.flatten().slice(begin, end)},
                              {"dst", dst.flatten().slice(begin, end)}});
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  return Execute(cs);
}
