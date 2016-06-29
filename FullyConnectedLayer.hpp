#ifndef _fully_connected_layer_hpp_
#define _fully_connected_layer_hpp_
#include "Net.hpp"
#include <cmath>

class FullyConnectedLayerImpl : public Layer {
public:
  std::size_t size, prevSize;
  NonLinearityType nonLinearityType;

  Tensor weights, biases, bwdWeights, z,
    activations, deltas, zDeltas;

  std::unique_ptr<float []> hWeights, hBiases;
  std::string layerName;

  FullyConnectedLayerImpl(Net &net, int index,
                          unsigned size,
                          NonLinearityType nonLinearityType) :
    Layer(net, index),
    size(size),
    nonLinearityType(nonLinearityType) {
    layerName = "FullyConnected" + std::to_string(size);
  }

  Tensor getFwdActivations() const override {
    return activations;
  }

  Tensor getFwdZs() const override {
    return z;
  }

  Tensor getBwdDeltas() const override {
    return deltas;
  }

  NonLinearityType getNonLinearityType() const override {
    return nonLinearityType;
  }

  void describe(std::ostream &out) override;

  std::uint64_t getNumberOfFlops() override;

  double getPerfectCycleCount() override;

  void init(Graph &graph, std::mt19937 &randomEngine,
            IPUModelEngineBuilder::TileMapping *mapping) override;

  Program initParams(Graph &graph) override {
    if (getDType() != "float") {
      // TODO: host to device tensor copies of half datatype
      return Sequence();
    }
    return Sequence(Copy(weights, &hWeights[0]),
                    Copy(biases, &hBiases[0]));
  }

  Program forward(Graph &graph,
                  IPUModelEngineBuilder::TileMapping *mapping) override;
  Program backward(Graph &graph) override;

  Program weightUpdate(Graph &graph) override;
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
