#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"
#include <cstdlib>

class MaxPoolLayerImpl : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;

  Tensor out, activations, deltas;

  unsigned xDim, yDim, numChannels, xDimOut, yDimOut, numChanGroups;

  std::string layerName;

  MaxPoolLayerImpl(const Net &net,
                   int index,
                   unsigned kernelSize,
                   unsigned stride,
                   unsigned padding);

  Tensor getFwdActivations() const override {
    return activations;
  }

  Tensor getBwdDeltas() const override {
    return deltas;
  }

  void describe(std::ostream &out) override;

  std::uint64_t getNumberOfFlops() override;

  double getPerfectCycleCount() override;

  size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                               size_t zPrev) const override;

  void init(Graph &graph, std::mt19937 &randomEngine,
            IPUModelEngineBuilder::TileMapping &mapping) override;

  Program initParams(Graph &graph) override {
    return Sequence();
  }

  Program forward(Graph &graph,
                  IPUModelEngineBuilder::TileMapping &mapping) override;

  Program backward(Graph &graph,
                   IPUModelEngineBuilder::TileMapping &mapping) override;

  Program weightUpdate(Graph &graph,
                       IPUModelEngineBuilder::TileMapping &mapping) override {
    return Sequence();
  }
};

class MaxPoolLayer : public LayerSpec {
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
public:
  MaxPoolLayer(unsigned kernelSize,
               unsigned stride,
               unsigned padding=0) :
  kernelSize(kernelSize),
  stride(stride),
  padding(padding) {}

  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
      new MaxPoolLayerImpl(net, index, kernelSize, stride, padding));
  }
};



#endif // _max_pool_layer_hpp_
