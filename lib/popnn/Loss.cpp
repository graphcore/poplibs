#include "poplar/Graph.hpp"
#include "popnn/Loss.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor& activations,
         const poplar::Tensor& expected,
         const poplar::Tensor& loss,
         const poplar::Tensor& deltas,
         const poplar::Tensor& numCorrect,
         const std::string& activationType,
         const std::string& expectedType,
         LossType lossType) {
  auto lossCS = graph.createComputeSet("LossLayer");
  auto v = graph.addVertex(lossCS, templateVertex("popnn::CalcLoss",
                                                  activationType,
                                                  expectedType),
                          {{"batchIn", activations},
                           {"batchDeltaOut", deltas},
                           {"label", expected},
                           {"loss", loss[0]},
                           {"numCorrect", numCorrect[0]}});
  graph.setTileMapping(v, 0);
  graph.setFieldSize(v["probs"], activations[0].numElements());
  graph.setInitialValue(v["lossType"], lossType);
  return Execute(lossCS);
}
