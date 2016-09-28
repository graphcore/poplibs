#include "popnn/NonLinearity.hpp"
#include "popnn/ActivationMapping.hpp"
#include "VertexTemplates.hpp"
#include "Regroup.hpp"

using namespace poplar;
using namespace poplar::program;

Program
bwdNonLinearity(Graph &graph,
                Tensor batchActivations, Tensor batchDeltasIn,
                Tensor batchZDeltas,
                NonLinearityType nonLinearityType) {
  if (batchActivations.dim(0) != 1) {
    std::cerr << "Batch size != 1 not implemented for backwards pass\n";
    std::abort();
  }
  auto activations = batchActivations[0];
  auto deltasIn = batchDeltasIn[0];
  auto zDeltas = batchZDeltas[0];
  const auto dType = graph.getTensorElementType(activations);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto deltasInMapping = computeActivationsMapping(graph, activations, 0, 1);
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
    regroupedActs = graph.addTensor(dType, zDeltas.dims(), "regroupedActs");
    auto mapping = computeActivationsMapping(graph, regroupedActs, 0, 1);
    applyTensorMapping(graph, regroupedActs, mapping);
    prog.add(Copy(regroupedActs, regroup(activations, zDeltas.dim(3))));
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
