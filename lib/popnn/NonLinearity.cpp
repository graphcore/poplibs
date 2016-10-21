#include "popnn/NonLinearity.hpp"
#include "popnn/ActivationMapping.hpp"
#include "popnn/exceptions.hpp"
#include "VertexTemplates.hpp"
#include "Regroup.hpp"

using namespace poplar;
using namespace poplar::program;

Program
bwdNonLinearity(Graph &graph,
                Tensor batchActivations, Tensor batchDeltasIn,
                Tensor batchZDeltas,
                NonLinearityType nonLinearityType) {
  auto bwdNonLinearityCS = graph.createComputeSet("NonLinearity.bwd");
  auto prog = Sequence();
  const auto batchSize = batchActivations.dim(0);
  for (unsigned b = 0; b != batchSize; ++b) {
    auto activations = batchActivations[b];
    auto deltasIn = batchDeltasIn[b];
    auto zDeltas = batchZDeltas[b];
    const auto dType = graph.getTensorElementType(activations);
    const auto &deviceInfo = graph.getDevice().getDeviceInfo();
    const auto dataPathWidth = deviceInfo.dataPathWidth;
    auto deltasInMapping = computeActivationsMapping(graph, activations,
                                                     b, batchSize);
    deltasIn = deltasIn.flatten();
    Tensor regroupedActs;
    // TODO: This could possible be made more efficient by merging the
    // regrouping with the calculation of the non linearity derivative.
    if (activations.getDimensionality() == 1 ||
        activations.dim(3) == zDeltas.dim(3)) {
      regroupedActs = activations;
    } else {
      auto dType = graph.getTensorElementType(activations);
      regroupedActs = graph.addTensor(dType, zDeltas.dims(), "regroupedActs");
      auto mapping = computeActivationsMapping(graph, regroupedActs,
                                               b, batchSize);
      applyTensorMapping(graph, regroupedActs, mapping);
      prog.add(Copy(regroupedActs, regroup(activations, zDeltas.dim(3))));
    }
    buildTransform(deltasInMapping, graph, [&](unsigned deltaBegin,
                   unsigned deltaEnd,
                   unsigned tile) {
      auto v =
          graph.addVertex(bwdNonLinearityCS,
                          templateVertex("popnn::NonLinearityBwd", dType),
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
  }
  prog.add(Execute(bwdNonLinearityCS));
  return prog;
}

Program
fwdNonLinearity(Graph &graph,
                Tensor activations,
                NonLinearityType nonLinearityType) {
  auto prog = Sequence();
  const auto dType = graph.getTensorElementType(activations);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  ComputeSet nonLinCs = graph.createComputeSet("FwdNonlinearity");
  prog.add(Execute(nonLinCs));
  const auto batchSize = activations.dim(0);
  for (unsigned b = 0; b < batchSize; b++) {

    const auto &activationMapping =
      computeActivationsMapping(graph, activations[b], b, batchSize);
    buildTransform(activationMapping, graph, [&](unsigned deltaBegin,
                                                 unsigned deltaEnd,
                                                 unsigned tile)
      {
        auto v =
          graph.addVertex(
              nonLinCs,
              templateVertex("popnn::NonLinearityFwd", dType),
              {{"activationIn", activations[b].flatten().slice(deltaBegin,
                                                            deltaEnd)},
               {"activationOut", activations[b].flatten().slice(deltaBegin,
                                                             deltaEnd)}});
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
        graph.setTileMapping(v, tile);
      });
  }
  return prog;
}
