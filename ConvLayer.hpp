#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

class ConvLayerImpl : public Layer {
  Program forwardByChanGroup(Graph &graph,
                             IPUModelEngineBuilder::TileMapping *mapping);
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned inNumChans, inNumChanGroups, inDimX, inDimY;
  unsigned outNumChans, outNumChanGroups, outDimX, outDimY;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
  Tensor weights, biases, z, activations;

  std::string layerName;

  // This get set if the layer should try the forward pass by calculating the
  // partial result for a group of channels at a time and then reducing these
  // partial sums to get the full result for each neuron.
  //
  // If this isn't set the layer will implement the forward pass with a
  // vertex for each neuron calculating the sum of all the input channels.
  bool tryForwardByChanGroup;

  ConvLayerImpl(Net &net,
                int index,
                unsigned kernelSize,
                unsigned stride,
                unsigned padding,
                unsigned numChannels,
                NonLinearityType nonLinearityType,
                NormalizationType normalizationType);

  std::uint64_t getNumberOfFlops();

  double getPerfectCycleCount();

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return z;
  }

  Tensor getBwdErrors() const {
    // TODO
  }

  NonLinearityType getNonLinearityType() const {
    return nonLinearityType;
  }

  void describe(std::ostream &out);

  size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                               size_t zPrev) const;

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

class ConvLayer : public LayerSpec {
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned numChannels;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
public:
  ConvLayer(unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned numChannels,
            NonLinearityType nonLinearityType,
            NormalizationType normalizationType) :
  kernelSize(kernelSize),
  stride(stride),
  padding(padding),
  numChannels(numChannels),
  nonLinearityType(nonLinearityType),
  normalizationType(normalizationType) {}

  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
       new ConvLayerImpl(net, index, kernelSize, stride, padding, numChannels,
                         nonLinearityType, normalizationType));
  }
};

#endif // _conv_layer_hpp_
