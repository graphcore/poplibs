#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

struct ConvLayerPartition {
  unsigned tilesPerXAxis;
  unsigned tilesPerYAxis;
  unsigned tilesPerZAxis;
  unsigned tilesPerInZGroupAxis;
  unsigned inChansPerGroup;
  ConvLayerPartition() = default;
  ConvLayerPartition(unsigned tilesPerXAxis, unsigned tilesPerYAxis,
                     unsigned tilesPerZAxis, unsigned tilesPerInZGroupAxis,
                     unsigned inChansPerGroup) :
    tilesPerXAxis(tilesPerXAxis), tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis), tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup) {}
};

class ConvLayerImpl : public Layer {
  void
  forwardTile(Graph &graph,
              IPUModelEngineBuilder::TileMapping *mapping,
              unsigned tile, unsigned outXBegin, unsigned outXEnd,
              unsigned outYBegin, unsigned outYEnd,
              unsigned outZBegin, unsigned outZEnd,
              unsigned inZGroupBegin, unsigned inZGroupEnd,
              ComputeSet cs,
              const Tensor &out);
public:
  ConvLayerPartition partition;
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned inNumChans, inNumChanGroups, inDimX, inDimY;
  unsigned outNumChans, outNumChanGroups, outDimX, outDimY;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
  Tensor weights, biases, z, activations;

  std::string layerName;

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
