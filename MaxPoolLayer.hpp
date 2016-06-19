#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"
#include <cstdlib>

class MaxPoolLayerImpl : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;

  Tensor out, activations, errors;

  unsigned xDim, yDim, numChannels, xDimOut, yDimOut, numChanGroups;

  std::string layerName;

  MaxPoolLayerImpl(const Net &net,
                   int index,
                   unsigned kernelSize,
                   unsigned stride,
                   unsigned padding);

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return activations;
  }

  Tensor getBwdErrors() const {
    return errors;
  }

  NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  }

  void describe(std::ostream &out);

  std::uint64_t getNumberOfFlops();

  double getPerfectCycleCount();

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping);

  Program initParams(Graph &graph) {
    return Sequence();
  }

  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping);

  Program backward(Graph &graph);

  Program weightUpdate(Graph &graph) {
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
