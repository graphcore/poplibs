#include "popnn/NonLinearity.hpp"
#include "popnn/ActivationMapping.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

Program
bwdNonLinearity(Graph &graph,
                Tensor activations, Tensor deltasIn,
                Tensor zDeltas,
                NonLinearityType nonLinearityType) {
  const auto dType = graph.getTensorElementType(activations);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto deltasInMapping = computeActivationsMapping(graph, activations);
  deltasIn = deltasIn.flatten();
  auto bwdNonLinearityCS = graph.createComputeSet("NonLinearity.bwd");
  buildTransform(deltasInMapping, graph, [&](unsigned deltaBegin,
                                                  unsigned deltaEnd,
                                                  unsigned tile) {
    auto v =
        graph.addVertex(bwdNonLinearityCS,
                        templateVertex("NonLinearityBwd", dType),
                        {{"deltasIn",
                          deltasIn.flatten().slice(deltaBegin, deltaEnd)},
                         {"activations",
                          activations.flatten().slice(deltaBegin, deltaEnd)},
                         {"deltasOut",
                          zDeltas.flatten().slice(deltaBegin, deltaEnd)},
                        });
    graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  return Execute(bwdNonLinearityCS);
}
