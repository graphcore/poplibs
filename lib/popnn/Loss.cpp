#include "popnn/Loss.hpp"

#include "poplar/Graph.hpp"
#include "popstd/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popnn {

Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor& activations,
         const poplar::Tensor& expected,
         const poplar::Tensor& loss,
         const poplar::Tensor& deltas,
         const poplar::Tensor& numCorrect,
         const std::string& activationType,
         const std::string& expectedType,
         LossType lossType,
         const std::string &debugPrefix) {
  auto lossCS = graph.addComputeSet(debugPrefix + "/LossLayer");
  auto v = graph.addVertex(lossCS, templateVertex("popnn::CalcLoss",
                                                  activationType,
                                                  expectedType),
                          {{"batchIn", activations},
                           {"batchDeltaOut", deltas},
                           {"label", expected},
                           {"loss", loss},
                           {"numCorrect", numCorrect[0]}});
  graph.setTileMapping(v, 0);
  graph.setFieldSize(v["probs"], activations[0].numElements());
  graph.setInitialValue(v["lossType"], lossType);
  return Execute(lossCS);
}

} // end namespace popnn
