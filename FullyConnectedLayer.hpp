#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>

#define USE_PARTIAL_SUMS 1

class FullyConnectedLayer : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, biases, bwdWeights, z,
    activations, errors, activationRecord, errorRecord,
    actRecordIndex, errorRecordIndex;

  std::unique_ptr<float []> hWeights;

  NetType netType;
  float eta;
  unsigned batchSize;
  unsigned numIPUs, tilesPerIPU;
  std::string dType;
  std::string layerName;

  size_t numPartials, elemsPerPartial;

  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size),
    nonLinearityType(nonLinearityType) {
    layerName = "FullyConnected" + std::to_string(size);
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
              << "        Input: "  << prevSize << "\n"
              << "        Output: " << size << "\n"
              << "        Params: " << size * (prevSize + 1) << "\n";
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
            Layer *prev, Layer *next, NetType netType, float eta,
            unsigned batchSize, unsigned numIPUs, unsigned tilesPerIPU,
            const std::string &dType) {
    Layer::init(numIPUs, tilesPerIPU);
    this->numIPUs = numIPUs;
    this->tilesPerIPU = tilesPerIPU;
    this->netType = netType;
    this->eta = eta;
    this->batchSize = batchSize;
    this->dType = dType;
    prevSize = prev->getFwdActivations().numElements();
    if (USE_PARTIAL_SUMS) {
      numPartials = sqrt(tilesPerIPU * numIPUs);
      elemsPerPartial = prevSize/numPartials;
      while (numPartials * elemsPerPartial != prevSize) {
        numPartials--;
        elemsPerPartial = prevSize/numPartials;
      }
      assert(numPartials * elemsPerPartial == prevSize);
      weights = graph.addTensor(dType, {numPartials, size, elemsPerPartial});
    } else {
      weights = graph.addTensor(dType, {size, prevSize});
    }
    biases = graph.addTensor(dType, {size});
    z = graph.addTensor(dType, {size});
    activations = graph.addTensor(dType, {size});
    mapTensor(weights, mapping);
    mapTensor(biases, mapping);
    mapTensor(z, mapping);
    mapTensor(activations, mapping);
    if (netType == TrainingNet) {
      errors = graph.addTensor(dType, {prevSize});
      activationRecord = graph.addTensor(dType, {prevSize, batchSize});
      actRecordIndex = graph.addTensor("unsigned", {1});
      errorRecord = graph.addTensor(dType, {size, batchSize});
      errorRecordIndex = graph.addTensor("unsigned", {1});
      bwdWeights = graph.addTensor(dType, {prevSize + 1, size});
      mapTensor(errors, mapping);
      mapTensor(activationRecord, mapping);
      mapTensor(actRecordIndex, mapping);
      mapTensor(errorRecord, mapping);
      mapTensor(errorRecordIndex, mapping);
      mapTensor(bwdWeights, mapping);
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

  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping,
                  Layer *prev)  {
    Tensor in = prev->getFwdActivations().flatten();

    if (USE_PARTIAL_SUMS) {
      Tensor partials = graph.addTensor(dType, {size, numPartials});
      mapTensor(partials, mapping);
      ComputeSet fwd1 = graph.createComputeSet(layerName + ".fwd"),
                 fwd2 = graph.createComputeSet(layerName + ".fwd.reduce");
      for (unsigned j = 0; j < numPartials; j++) {
        for (unsigned i = 0; i < size; ++i) {
          Tensor partialIn = in.slice(j * elemsPerPartial,
                                      (j + 1) * elemsPerPartial);
          Tensor partialWeights = weights[j][i];
          auto v = graph.addVertex(fwd1, "FullyConnectedPartial",
                                   {{"in", partialIn},
                                    {"weights", partialWeights},
                                    {"out", partials[i][j]}});
        }
      }
      for (unsigned i = 0; i < size; ++i) {
        auto v = graph.addVertex(fwd2, "FullyConnectedReduce",
                                 {{"partials", partials[i]},
                                  {"bias", biases[i]},
                                  {"zOut", z[i]},
                                  {"activationOut", activations[i]}});
          graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      }
      mapComputeSet(graph, fwd1, mapping);
      mapComputeSet(graph, fwd2, mapping);
      return Sequence(Execute(fwd1), Execute(fwd2));
    } else {
      ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
      for (unsigned i = 0; i < size; ++i) {
        auto v = graph.addVertex(fwd, "FullyConnected",
                                 {{"activationIn", in},
                                  {"weights", weights[i]},
                                  {"bias", biases[i]},
                                  {"zOut", z[i]},
                                  {"activationOut", activations[i]}});
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      }
      mapComputeSet(graph, fwd, mapping);
      return Sequence(Execute(fwd));
    }
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
