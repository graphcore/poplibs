#include "LossLayer.hpp"

void LossLayer::
init(Graph &graph, std::mt19937 &randomEngine,
     IPUModelEngineBuilder::TileMapping &mapping) {
    const auto dType = getDType();
    Layer *prev = getPrevLayer();
    assert(prev);
    deltas = graph.addTensor(dType, {prev->getFwdActivations().dims()},
                             makeLayerName("deltas"));
    expected = graph.addTensor("unsigned", {1}, makeLayerName("expected"));
    lossTypeTensor = graph.addTensor("LossType", {1}, makeLayerName("lossType"));
    graph.setInitialValue(lossTypeTensor[0], lossType);
    loss = graph.addTensor(dType, {1}, makeLayerName("loss"));
    numCorrect = graph.addTensor("unsigned", {1}, makeLayerName("numCorrect"));
    mapTensor(deltas, mapping);
    mapTensor(expected, mapping);
    mapTensor(lossTypeTensor, mapping);
    mapTensor(loss, mapping);
    mapTensor(numCorrect, mapping);
    fwd = graph.createComputeSet("LossLayer");
  }

Program LossLayer::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping) {
  Layer *prev = getPrevLayer();
  auto v = graph.addVertex(fwd, templateVertex("CalcLoss", getDType()),
                           {{"zIn", prev->getFwdZs().flatten()},
                            {"deltaOut", deltas.flatten()},
                            {"label", expected[0]},
                            {"lossType", lossTypeTensor[0]},
                            {"loss", loss[0]},
                            {"numCorrect", numCorrect[0]}});
  graph.setFieldSize(v["probs"], prev->getFwdActivations().numElements());
  graph.setInitialValue(v["nonLinearityType"], prev->getNonLinearityType());
  mapComputeSet(graph, fwd, mapping);
  return Sequence(Copy(numCorrect, &hNumCorrect),
                  Execute(fwd),
                  Copy(&hNumCorrect, numCorrect));
}
