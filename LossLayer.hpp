#ifndef _loss_layer_hpp_
#define _loss_layer_hpp_

#include "Layer.hpp"

class LossLayer : public Layer {
  DataSet &data;
  LossType lossType;
  Tensor deltas, expected, lossTypeTensor, loss, numCorrect;
  unsigned hNumCorrect;
  ComputeSet fwd;
public:
  LossLayer(const Net &net, int index,
            DataSet &data, LossType lossType) :
    Layer(net, index), data(data), lossType(lossType) {}

  void init(Graph &graph, std::mt19937 &randomEngine,
            IPUModelEngineBuilder::TileMapping &mapping) override;

  void resetNumCorrect() {
    hNumCorrect = 0;
  }

  unsigned getNumCorrect() {
    return hNumCorrect;
  }

  Program initParams(Graph &graph) override { return Sequence(); }
  Program forward(Graph &graph,
                  IPUModelEngineBuilder::TileMapping &mapping) override;
  Program loadLabels(Graph &graph, bool isTraining) {
    if (isTraining) {
      return Copy(expected,
                  &data.trainingLabels[0],
                  &data.trainingLabels[data.numTraining]);
    } else {
      return Copy(expected,
                  &data.testLabels[0],
                  &data.testLabels[data.numTest]);
    }
  }
  Program backward(Graph &graph,
                   IPUModelEngineBuilder::TileMapping &mapping) override {
    return Sequence();
  }
  Program weightUpdate(Graph &graph,
                       IPUModelEngineBuilder::TileMapping &mapping) override {
    return Sequence();
  }
  void describe(std::ostream &out) override {}
  std::uint64_t getNumberOfFlops() override { return 0; }
  virtual double getPerfectCycleCount() override { return 0.0; }
  Tensor getFwdActivations() const override { return {}; }
  Tensor getFwdZs() const override { return {}; }
  Tensor getBwdDeltas() const override { return deltas; }
};

#endif //_loss_layer_hpp_
