#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <cmath>

class FullyConnectedLayerImpl : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, biases, bwdWeights, z,
    activations, errors, activationRecord, errorRecord,
    actRecordIndex, errorRecordIndex;

  std::unique_ptr<float []> hWeights;
  std::string layerName;

  FullyConnectedLayerImpl(Net &net, int index,
                          unsigned size,
                          NonLinearityType nonLinearityType) :
    Layer(net, index),
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

  void describe(std::ostream &out);

  std::uint64_t getNumberOfFlops();

  double getPerfectCycleCount();

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping);

  Program initParams(Graph &graph) {
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    return Sequence();
  }

  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping);

  Program backward(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program weightSync(Graph &graph) {
    // TODO
    return Sequence();
  }
};

class FullyConnectedLayer : public LayerSpec {
  unsigned size;
  NonLinearityType nonLinearityType;
public:
  FullyConnectedLayer(unsigned size,
                      NonLinearityType nonLinearityType) :
    size(size), nonLinearityType(nonLinearityType) {}
  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
       new FullyConnectedLayerImpl(net, index, size, nonLinearityType));
  }
};


#endif // _fully_connected_layer_hpp_
