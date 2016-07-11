#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"
#include <cstdlib>
#include <map>
#include <tuple>

class ConvImplSpec {
  unsigned inNumChans, inNumChanGroups, inDimX, inDimY;
  unsigned outNumChans, outNumChanGroups, outDimX, outDimY;
  unsigned resNumChans, resNumChanGroups, resDimX, resDimY;
  unsigned kernelSize, stride, padding;
public:
  ConvImplSpec(unsigned inNumChans, unsigned inNumChanGroups,
               unsigned inDimX, unsigned inDimY,
               unsigned outNumChans, unsigned outNumChanGroups,
               unsigned outDimX, unsigned outDimY,
               unsigned resNumChans, unsigned resNumChanGroups,
               unsigned resDimX, unsigned resDimY,
               unsigned kernelSize, unsigned stride, unsigned padding) :
    inNumChans(inNumChans), inNumChanGroups(inNumChanGroups),
    inDimX(inDimX), inDimY(inDimY),
    outNumChans(outNumChans), outNumChanGroups(outNumChanGroups),
    outDimX(outDimX), outDimY(outDimY),
    resNumChans(resNumChans), resNumChanGroups(resNumChanGroups),
    resDimX(resDimX), resDimY(resDimY),
    kernelSize(kernelSize), stride(stride), padding(padding) {}

  bool operator<(const ConvImplSpec &other) const {
    auto t1 = std::make_tuple(inNumChans, inNumChanGroups, inDimX, inDimY,
                              outNumChans, outNumChanGroups, outDimX, outDimY,
                              resNumChans, resNumChanGroups, resDimX, resDimY,
                              kernelSize, stride, padding);
    auto t2 = std::make_tuple(other.inNumChans, other.inNumChanGroups,
                              other.inDimX, other.inDimY,
                              other.outNumChans, other.outNumChanGroups,
                              other.outDimX, other.outDimY,
                              other.resNumChans, other.resNumChanGroups,
                              other.resDimX, other.resDimY,
                              other.kernelSize, other.stride, other.padding);
    return t1 < t2;
  }
};

struct ConvLayerPartition {
  unsigned tilesPerXAxis;
  unsigned tilesPerYAxis;
  unsigned tilesPerZAxis;
  unsigned verticesPerTilePerYAxis;
  unsigned tilesPerInZGroupAxis;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  bool floatPartials;
  ConvLayerPartition() = default;
  ConvLayerPartition(unsigned tilesPerXAxis,
                     unsigned tilesPerYAxis,
                     unsigned tilesPerZAxis,
                     unsigned verticesPerTilePerYAxis,
                     unsigned tilesPerInZGroupAxis,
                     unsigned inChansPerGroup,
                     unsigned partialChansPerGroup,
                     bool floatPartials) :
    tilesPerXAxis(tilesPerXAxis),
    tilesPerYAxis(tilesPerYAxis),
    tilesPerZAxis(tilesPerZAxis),
    verticesPerTilePerYAxis(verticesPerTilePerYAxis),
    tilesPerInZGroupAxis(tilesPerInZGroupAxis),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup),
    floatPartials(floatPartials) {}
};

enum ResidualMethod {
  RESIDUAL_PAD,
  RESIDUAL_WEIGHTED_CONV_IF_SIZES_DIFFER,
  RESIDUAL_WEIGHTED_CONV
};

class ConvLayerImpl : public Layer {
  bool useConvolutionInstruction() const;
  bool targetSharedConvWeights() const;

  void
  createConvPartial1x1InOutVertex(Graph &graph,
                                  IPUModelEngineBuilder::TileMapping &mapping,
                                  unsigned tile,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned outZGroup,
                                  unsigned inZGroupBegin, unsigned inZGroupEnd,
                                  const std::string &partialType,
                                  ComputeSet fwdCS,
                                  const Tensor &out);
  void
  forwardTile(Graph &graph,
              IPUModelEngineBuilder::TileMapping &mapping,
              unsigned tile, unsigned outXBegin, unsigned outXEnd,
              unsigned outYBegin, unsigned outYEnd,
              unsigned outZGroupBegin, unsigned outZGroupEnd,
              unsigned inZGroupBegin, unsigned inZGroupEnd,
              ComputeSet zeroCS,
              ComputeSet fwdCS,
              const Tensor &out);
  void mapBiases(Tensor w, const std::vector<unsigned> &activationMapping,
                 IPUModelEngineBuilder::TileMapping &mapping);
  void mapWeights(Graph &graph,
                  IPUModelEngineBuilder::TileMapping &mapping,
                  Tensor w);
  Tensor getInputWeights() const {
    return weightsIn;
  }
  Tensor getInputBiases() const {
    return biasesIn;
  }
  Tensor getInputTensor() const {
    return in;
  }
  Tensor getOutputTensor() const {
    return activations;
  }
  Tensor getOutputZ() const {
    return z;
  }
  void createFwdProg(Graph &graph,
                     IPUModelEngineBuilder::TileMapping &mapping);
  Program
  getOrCreateFwdProg(Graph &graph,
                     IPUModelEngineBuilder::TileMapping &mapping) {
    if (!createdForwardProg)
      createFwdProg(graph, mapping);
    return forwardProg;
  }
  Tensor getInputResidual() const {
    return resIn;
  }
  static std::map<ConvImplSpec, ConvLayerImpl *> implMap;
  void addResidualCalc(Graph &graph,
                       ComputeSet cs,
                       IPUModelEngineBuilder::TileMapping &mapping);
  std::uint64_t getNumberOfMACs();
  std::uint64_t getNumberOfAdds();
public:
  ConvLayerPartition partition;
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned inNumChans, inNumChanGroups, inDimX, inDimY;
  unsigned outNumChans, outNumChanGroups, outDimX, outDimY;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
  Tensor weights, in, weightsIn, biases, biasesIn, fwdZ, z, activations,
         fwdActivations, resIn, deltas, zDeltas;

  std::string layerName;

  ConvLayerImpl *reuseImpl = 0;

  bool createdForwardProg;
  Sequence forwardProg;

  unsigned resIndex;
  enum ResidualMethod resMethod;
  Layer *resLayer = 0;
  Tensor residual;
  unsigned resStrideX, resStrideY;

  bool reuseLayerImplGraphs;

  std::unique_ptr<float []> hWeights, hBiases;

  ConvLayerImpl(Net &net,
                int index,
                unsigned kernelSize,
                unsigned stride,
                unsigned padding,
                unsigned numChannels,
                NonLinearityType nonLinearityType,
                NormalizationType normalizationType,
                unsigned resIndex,
                enum ResidualMethod resMethod);

  const char *getPartialType() const {
    return partition.floatPartials ? "float" : "half";
  }

  std::uint64_t getNumberOfFlops() override;

  double getPerfectCycleCount() override;

  Tensor getFwdActivations() const override {
    return fwdActivations;
  }

  Tensor getBwdDeltas() const override {
    return deltas;
  }

  void describe(std::ostream &out) override;

  size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                               size_t zPrev) const override;

  void init(Graph &graph, std::mt19937 &randomEngine,
            IPUModelEngineBuilder::TileMapping &mapping) override;

  Program initParams(Graph &graph) override {
    if (getDType() != "float") {
      // TODO: host to device tensor copies of half datatype
      return Sequence();
    }
    return Sequence(Copy(weights, &hWeights[0]),
                    Copy(biases, &hBiases[0]));
  }

  Program forward(Graph &graph,
                  IPUModelEngineBuilder::TileMapping &mapping) override;

  Program backward(Graph &graph,
                   IPUModelEngineBuilder::TileMapping &mapping) override;

  Program weightUpdate(Graph &graph,
                       IPUModelEngineBuilder::TileMapping &mapping) override;

};

class ConvLayer : public LayerSpec {
protected:
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
                         nonLinearityType, normalizationType,
                         0, RESIDUAL_PAD));
  }
};

class ConvResLayer : public ConvLayer {
  unsigned resIndex;
  enum ResidualMethod resMethod;
public:
  ConvResLayer(unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned numChannels,
            NonLinearityType nonLinearityType,
            NormalizationType normalizationType,
            unsigned resIndex,
            enum ResidualMethod resMethod) :
  ConvLayer(kernelSize, stride, padding, numChannels, nonLinearityType,
            normalizationType),
  resIndex(resIndex), resMethod(resMethod) {
    if (padding >= kernelSize) {
      // Padding greater than kernelsize is not supported (and weird)
      std::abort();
    }
  }

  std::unique_ptr<Layer>
  makeLayer(Net &net, int index) {
    return std::unique_ptr<Layer>(
       new ConvLayerImpl(net, index, kernelSize, stride, padding, numChannels,
                         nonLinearityType, normalizationType,
                         resIndex, resMethod));
  }
};

#endif // _conv_layer_hpp_
