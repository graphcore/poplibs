#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"
#include <cstdlib>

class MaxPoolLayerImpl : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;

  Tensor out, activations;

  unsigned xDim, yDim, numChannels, xDimOut, yDimOut, numChanGroups;

  std::string layerName;

  MaxPoolLayerImpl(const Net &net,
                   int index,
                   unsigned kernelSize,
                   unsigned stride);

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return activations;
  }

  Tensor getBwdErrors() const {
    // TODO
    std::abort();
  }

  NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  }

  void describe(std::ostream &out);

  std::uint64_t getNumberOfFlops();

  double getPerfectCycleCount();

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping);

  Program initParams(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    // TODO
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

class MaxPoolLayer : public LayerSpec {
  unsigned kernelSize;
  unsigned stride;
public:
  MaxPoolLayer(unsigned kernelSize,
               unsigned stride) :
  kernelSize(kernelSize),
  stride(stride) {}

  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
      new MaxPoolLayerImpl(net, index, kernelSize, stride));
  }
};



#endif // _max_pool_layer_hpp_
