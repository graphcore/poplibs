#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

class ConvLayerImpl : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned numChannels;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
  Tensor weights, biases, z, activations;
  unsigned xDim, yDim, prevChannels, xDimOut, yDimOut, weightsPerOutputChannel;

  std::string layerName;

  ConvLayerImpl(Net &net,
                int index,
                unsigned kernelSize,
                unsigned stride,
                unsigned padding,
                unsigned numChannels,
                NonLinearityType nonLinearityType,
                NormalizationType normalizationType) :
    Layer(net, index),
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    numChannels(numChannels),
    nonLinearityType(nonLinearityType),
    normalizationType(normalizationType) {
    layerName = "Conv" + std::to_string(kernelSize) + "x" +
                std::to_string(kernelSize);
  }

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

  void describe(std::ostream &out) {
    unsigned numParams = weights.numElements() + biases.numElements();
    out << "   -- Convolutional layer:\n"
        << "        Size: " << kernelSize << "x" << kernelSize << "\n"
        << "        Stride: " << stride << "\n"
        << "        Padding: " << padding << "\n"
        << "        Input: " << xDim << "x" << yDim
                    <<   "x" << prevChannels << "\n"
        << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << numChannels << "\n"
        << "        Params: " << numParams << "\n";
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    if (dType != "float" && numChannels % 2 != 0) {
      throw net_creation_error("Convolution layers with an odd number of channels are not supported for FP16 data");
    }
    Layer *prev = getPrevLayer();
    Tensor in = prev->getFwdActivations();
    xDim = in.dim(0);
    yDim = in.dim(1);
    prevChannels = in.dim(2);
    xDimOut = (xDim + padding - kernelSize) / stride + 1;
    yDimOut = (yDim + padding - kernelSize) / stride + 1;
    weightsPerOutputChannel = kernelSize * kernelSize * prevChannels + 1;
    z = graph.addTensor(dType, {xDimOut, yDimOut, numChannels});
    activations = graph.addTensor(dType, {xDimOut, yDimOut, numChannels});
    weights = graph.addTensor(dType, {numChannels,
                                      kernelSize,
                                      kernelSize * prevChannels});
    biases = graph.addTensor(dType, {numChannels});
    mapTensor(z, mapping);
    mapTensor(activations, mapping);
    mapTensor(weights, mapping);
    mapTensor(biases, mapping);
  }

  Program initParams(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
    Layer *prev = getPrevLayer();
    const auto dType = getDType();
    Tensor in = prev->getFwdActivations();
    ComputeSet fwd =
      graph.createComputeSet(layerName + ".fwd");
    unsigned chansPerVertex = dType == "float" ? 1 : 2;
    assert(numChannels % chansPerVertex == 0);
    for (unsigned chan = 0; chan < numChannels; chan += chansPerVertex) {
      for (unsigned i = 0; i < xDimOut; ++i) {
        for (unsigned j = 0; j < yDimOut; ++j) {
          unsigned width = std::min(i * stride + kernelSize, xDim) - i * stride;
          unsigned height = std::min(j * stride + kernelSize, yDim) - j * stride;
          // Create window into previous layer
          Tensor window =
            in.slice({i * stride, j * stride, 0 },
                     {i * stride + width, j * stride + height, prevChannels})
              .reshape({width, height * prevChannels});
          // Get weights that match window size
          Tensor w =
            weights.slice({chan, 0, 0},
                          {chan + chansPerVertex, width, height * prevChannels})
                   .reshape({chansPerVertex * width, height * prevChannels});
          auto v = graph.addVertex(fwd, "Convolution",
            { {"activationIn", window },
              {"weights", w },
              {"bias", biases.slice(chan, chan + chansPerVertex) },
              {"activationOut", activations[i][j].slice(chan,
                                                        chan + chansPerVertex)}
            });
          graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        }
      }
    }
    mapComputeSet(graph, fwd, mapping);
    return Execute(fwd);
  }

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
