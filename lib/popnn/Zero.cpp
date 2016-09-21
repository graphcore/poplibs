#include "Zero.hpp"

#include <cstdlib>
#include <poplar/Graph.hpp>
#include <popnn/ActivationMapping.hpp>
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

poplar::program::Program
zero(poplar::Graph &graph,
     poplar::Tensor t,
     const std::vector<unsigned> &tileMapping) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  ComputeSet cs = graph.createComputeSet("zero");
  const auto dType = graph.getTensorElementType(t);
  buildTransform(tileMapping, graph,
                 [&](unsigned elementBegin, unsigned elementEnd,
                     unsigned tile) {
    auto v = graph.addVertex(
      cs, templateVertex("Zero", dType),
      {{"out", t.flatten().slice(elementBegin, elementEnd)}}
    );
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  return Execute(cs);
}
