#include "popnn/NonLinearity.hpp"
#include "popnn/ActivationMapping.hpp"
#include "VertexTemplates.hpp"
#include "Regroup.hpp"

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
  auto prog = Sequence();
  Tensor regroupedActs;
  // TODO: This could possible be made more efficient by merging the
  // regrouping with the calculation of the non linearity derivative.
  if (activations.getDimensionality() == 1 ||
      activations.dim(3) == zDeltas.dim(3)) {
    regroupedActs = activations;
  } else {
    auto dType = graph.getTensorElementType(activations);
    regroupedActs = graph.addTensor(dType, zDeltas.dims(), "regoupedActs");
    mapActivations(graph, regroupedActs);
    prog.add(regroup(graph, "NonLinearity.bwd", dType, dType, deltasInMapping,
                     activations, regroupedActs));
  }
  buildTransform(deltasInMapping, graph, [&](unsigned deltaBegin,
                                                  unsigned deltaEnd,
                                                  unsigned tile) {
    auto v =
        graph.addVertex(bwdNonLinearityCS,
                        templateVertex("NonLinearityBwd", dType),
                        {{"deltasIn",
                          deltasIn.flatten().slice(deltaBegin, deltaEnd)},
                         {"activations",
                          regroupedActs.flatten().slice(deltaBegin, deltaEnd)},
                         {"deltasOut",
                          zDeltas.flatten().slice(deltaBegin, deltaEnd)},
                        });
    graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
    graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
    graph.setTileMapping(v, tile);
  });
  prog.add(Execute(bwdNonLinearityCS));
  return prog;
}
