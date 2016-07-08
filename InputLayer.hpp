#ifndef _input_layer_hpp_
#define _input_layer_hpp_

#include "Layer.hpp"

class InputLayer : public Layer {
  DataSet &data;
  Tensor out, z;
public:
  InputLayer(const Net &net, int index, DataSet &data) :
    Layer(net, index), data(data) {}

  void init(Graph &graph, std::mt19937 &randomEngine,
            IPUModelEngineBuilder::TileMapping &mapping) override;

  Program initParams(Graph &graph) override { return Sequence(); }
  Program forward(Graph &graph,
                  IPUModelEngineBuilder::TileMapping &mapping) override {
    return Sequence();
  }

  Program loadData(Graph &graph, bool isTraining) {
    if (isTraining) {
      size_t trainingDataSize = data.numTraining * data.dataSize;
      return Copy(out, &data.trainingData[0],
                  &data.trainingData[trainingDataSize]);
    } else {
      size_t testDataSize = data.numTest * data.dataSize;
      return Copy(out, &data.testData[0],
                  &data.testData[testDataSize]);
    }
  }
  Program backward(Graph &graph) override { return Sequence(); }
  Program weightUpdate(Graph &graph) override { return Sequence(); }
  void describe(std::ostream &out) override {}
  std::uint64_t getNumberOfFlops() override { return 0; }
  virtual double getPerfectCycleCount() override { return 0.0; }
  Tensor getFwdActivations() const override { return out; }
  Tensor getFwdZs() const override { return out; }
  Tensor getBwdDeltas() const override { return {}; }
};

#endif // _input_layer_hpp_
