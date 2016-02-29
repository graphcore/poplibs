#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

class FullyConnectedLayer : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, bwdWeights, z,
    activations, errors, activationRecord, errorRecord,
    actRecordIndex, errorRecordIndex;

  std::unique_ptr<float []> hWeights;

  NetType netType;
  float eta;
  unsigned batchSize;
  std::string dType;

  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size),
    nonLinearityType(nonLinearityType) {
  }

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return z;
  }

  Tensor getBwdErrors() const {
    return errors;
  }

  NonLinearityType getNonLinearityType() const {
    return nonLinearityType;
  }

  void describe(std::ostream &out) {
    std::cout << "   -- Fully connected layer:\n"
              << "        Input: "  << weights.dim(1)-1 << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << weights.numElements() << "\n";
  }

  void init(Graph &graph, Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize, const std::string &dType) {
    this->netType = netType;
    this->eta = eta;
    this->batchSize = batchSize;
    this->dType = dType;
    prevSize = prev->getFwdActivations().numElements();
    weights = graph.addTensor(dType, {size, prevSize + 1});

    z = graph.addTensor(dType, {size});
    activations = graph.addTensor(dType, {size});
    if (netType == TrainingNet) {
      errors = graph.addTensor(dType, {prevSize});
      activationRecord = graph.addTensor(dType, {prevSize, batchSize});
      actRecordIndex = graph.addTensor("unsigned", {1});
      errorRecord = graph.addTensor(dType, {size, batchSize});
      errorRecordIndex = graph.addTensor("unsigned", {1});
      bwdWeights = graph.addTensor(dType, {prevSize + 1, size});
    }
    hWeights = std::unique_ptr<float[]>(new float[(prevSize+1) * size]);
    unsigned seed = time(0);
    boost::variate_generator< boost::mt19937, boost::normal_distribution<> >
      generator(boost::mt19937(seed), boost::normal_distribution<>(0, 1));
    for (unsigned i = 0; i < (prevSize+1)*size; ++i)
      hWeights[i] = generator();
  }

  Program initParams(Graph &graph) {
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    return Sequence();
  }

  Program forward(Graph &graph, Layer *prev)  {
    Tensor in = prev->getFwdActivations().flatten();
    ComputeSet fwd = graph.createComputeSet();
    for (unsigned i = 0; i < size; ++i) {
      auto v = graph.addVertex(fwd, "FullyConnected",
                               {{"activationIn", in},
                                {"weights", weights[i]},
                                {"zOut", z[i]},
                                {"activationOut", activations[i]}});
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
    }
    return Execute(fwd);
  }

  Program backward(Graph &graph, Layer *prev, Layer *next) {
    // TODO
    return Sequence();
  }

  Program weightSync(Graph &graph) {
    // TODO
    return Sequence();
  }
};


#endif // _fully_connected_layer_hpp_
