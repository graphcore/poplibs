#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

class ConvLayerImpl : public Layer {
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
                NormalizationType normalizationType) :
    Layer(net, index),
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    outNumChans(numChannels),
    nonLinearityType(nonLinearityType),
    normalizationType(normalizationType) {
    const auto dType = getDType();
    layerName = "Conv" + std::to_string(kernelSize) + "x" +
                std::to_string(kernelSize);
    // TODO: This heuristic is a bit specialized. It would be better to
    // have clearer criteria as to when to do this.
    tryForwardByChanGroup = (dType != "float" && stride == 1 &&
                              (kernelSize == 3 || kernelSize == 5));
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
        << "        Input: " << inDimX << "x" << inDimY
                    <<   "x" << inNumChans << "\n"
        << "        Output: " << outDimX << "x" << outDimY
                     <<   "x" << outNumChans << "\n"
        << "        Params: " << numParams << "\n";
  }

  size_t getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                               size_t zPrev) const {
    if (tryForwardByChanGroup && zPrev % 4 == 0) {
      // If doing the convolution by channel is preferred then try
      // and target the special convolution instructions
      // which require writing back to a 4-element vector.
      return zPrev/4;
    } else {
      return 1;
    }
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    if (dType != "float" && outNumChans % 2 != 0) {
      throw net_creation_error("Convolution layers with an odd number of "
                               "channels are not supported for FP16 data");
    }
    Layer *prev = getPrevLayer();
    Tensor in = prev->getFwdActivations();
    inNumChanGroups = in.dim(0);
    inDimY = in.dim(1);
    inDimX = in.dim(2);
    size_t inChansPerGroup = in.dim(3);
    inNumChans = inChansPerGroup * inNumChanGroups;
    outDimX = (inDimX + padding - kernelSize) / stride + 1;
    outDimY = (inDimY + padding - kernelSize) / stride + 1;
    Layer *next = getNextLayer();
    outNumChanGroups = next->getNumChannelGroupsIn(inDimX, inDimY, outNumChans);
    if (!outNumChanGroups)
      outNumChanGroups = inNumChanGroups;
    size_t outChansPerGroup = outNumChans / outNumChanGroups;
    assert(outNumChanGroups * outChansPerGroup == outNumChans);
    z = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                outChansPerGroup});
    activations = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                          outChansPerGroup});
    weights = graph.addTensor(dType, {inNumChanGroups,
                                      outNumChans,
                                      kernelSize,
                                      kernelSize *
                                      inChansPerGroup});
    biases = graph.addTensor(dType, {outNumChans});
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

  Program forwardByChanGroup(Graph &graph,
                             IPUModelEngineBuilder::TileMapping *mapping) {
    // The strategy applied here computes the forward pass in three phases.
    // The first phase computes partial sums over
    // each group of channels for each output in a row-by-row fashion
    // which can target the special convolution instructions.
    // A second phase reduces these partial sums down to a complete sum.
    // The final phase applies the non-linearity and rearranges the output
    // tensor into the form required by the next layer.
    Layer *prev = getPrevLayer();
    const auto dType = getDType();
    unsigned inChansPerGroup = inNumChans / inNumChanGroups;
    unsigned outChansPerGroup = outNumChans / outNumChanGroups;
    assert(outChansPerGroup % 4 == 0);
    Tensor partials = graph.addTensor("float",
                                      {inNumChanGroups,
                                       outNumChans,
                                       outDimY,
                                       outDimX});
    mapTensor(partials, mapping);
    Tensor in = prev->getFwdActivations();
    ComputeSet fwdCS = graph.createComputeSet(layerName + ".fwd");

    auto numTiles = getNumIPUs() * getTilesPerIPU();
    // TODO: how to set this?
    unsigned rowsPerVertex = (outDimX * inNumChanGroups * outNumChans) /
                                 (numTiles * 16);
    if (rowsPerVertex == 0)
      rowsPerVertex = 1;
    if (rowsPerVertex > outDimX)
      rowsPerVertex = outDimX;

    for (unsigned inChanGroup = 0;
         inChanGroup < inNumChanGroups;
         ++inChanGroup) {
      for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
        for (unsigned y = 0; y < outDimY; y += rowsPerVertex) {
          unsigned numRows = std::min(y + rowsPerVertex, outDimY) - y;
          unsigned height =
            std::min((y + numRows) * stride + kernelSize, inDimY) - y * stride;
          Tensor inSlice = in[inChanGroup].slice(y * stride,
                                                y * stride + height).flatten();
          Tensor w = weights[inChanGroup][outChan].flatten();
          Tensor out = partials[inChanGroup][outChan]
                               .slice(y, y + numRows);
          auto v = graph.addVertex(fwdCS, "ConvPartial",
                                   {{ "in", inSlice},
                                    { "weights", w },
                                    { "out", out}});
          graph.setInitialValue(v["kernelSize"], kernelSize);
          graph.setInitialValue(v["stride"], stride);
          graph.setInitialValue(v["chans"], inChansPerGroup);
          graph.setInitialValue(v["inputCols"], inDimX);
        }
      }
    }
    mapComputeSet(graph, fwdCS, mapping);
    auto fwd = Execute(fwdCS);

    Tensor reduced = graph.addTensor("float",
                                     {outNumChans, outDimY, outDimX});
    mapTensor(reduced, mapping);
    ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
    for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
      for (unsigned y = 0; y < outDimY; ++y) {
        unsigned xStep = 6;
        for (unsigned x = 0; x < outDimX; x += xStep) {
          unsigned xEnd = std::min(x + xStep, outDimX);
          Tensor p = partials.slice({0, outChan, y, x},
                                    {inNumChanGroups, outChan + 1, y + 1, xEnd})
                             .reshape({inNumChanGroups, xEnd - x});
          Tensor out = reduced[outChan][y].slice(x, xEnd);
          graph.addVertex(reduceCS, "ConvReduce",
                          {{ "out", out  },
                           { "partials", p }});
        }
      }
    }
    mapComputeSet(graph, reduceCS, mapping);
    auto reduce = Execute(reduceCS);

    ComputeSet completionCS =
       graph.createComputeSet(layerName + ".fwd.complete");
    for (unsigned outChanGroup = 0;
         outChanGroup < outNumChanGroups;
         ++outChanGroup) {
      for (unsigned y = 0; y < outDimY; ++y) {
        unsigned xStep = 4;
        for (unsigned x = 0; x < outDimX; x += xStep) {
          unsigned xEnd = std::min(outDimX, x + xStep);
          Tensor actOut = activations[outChanGroup][y].slice(x, xEnd).flatten();
          Tensor biasSlice = biases.slice(outChanGroup * outChansPerGroup,
                                          (outChanGroup+1) * outChansPerGroup);
          Tensor reduced1 = reduced.reshape({outNumChanGroups, outChansPerGroup,
                                           outDimY, outDimX});
          Tensor in = reduced1.slice({outChanGroup, 0, y, x},
                                     {outChanGroup+1, outChansPerGroup,
                                      y + 1, xEnd})
                              .reshape({outChansPerGroup, xEnd - x});
          auto v = graph.addVertex(completionCS, "ConvComplete",
                                   {{ "in", in },
                                    { "bias", biasSlice },
                                    { "out", actOut} });
          graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        }
      }
    }
    mapComputeSet(graph, completionCS, mapping);
    auto complete = Execute(completionCS);

    return Sequence(fwd, reduce, complete);
  }


  Program forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
    unsigned inChansPerGroup = inNumChans / inNumChanGroups;
    if (tryForwardByChanGroup && inChansPerGroup == 4) {
      return forwardByChanGroup(graph, mapping);
    }
    // If the convolution is not done by channels groups then
    // the input should be set up to only have one channel group which all
    // the input tensors channels vectorized. The calculation of each
    // output can vectorize of the y and z axes of the receptive field.
    assert(inNumChanGroups == 1);
    Layer *prev = getPrevLayer();
    const auto dType = getDType();
    Tensor in = prev->getFwdActivations();
    ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
    unsigned outChansPerVertex = dType == "float" ? 1 : 2;
    assert(outNumChans % outChansPerVertex == 0);
    for (unsigned inChan = 0;
         inChan < outNumChans;
         inChan += outChansPerVertex) {
      for (unsigned x = 0; x < outDimX; ++x) {
        for (unsigned y = 0; y < outDimY; ++y) {
          unsigned width =
            std::min(x * stride + kernelSize, inDimX) - x * stride;
          unsigned height =
            std::min(y * stride + kernelSize, inDimY) - y * stride;
          // Create window into previous layer
          Tensor window =
            in[0].slice({x * stride, y * stride, 0 },
                        {x * stride + width, y * stride + height, inNumChans})
                 .reshape({width, height * inNumChans});
          // Get weights that match window size
          Tensor w =
            weights[0].slice({inChan, 0, 0},
                             {inChan + outChansPerVertex,
                              width,
                              height * inNumChans})
                      .reshape({outChansPerVertex * width,
                                height * inNumChans});
          unsigned outChansPerGroup = outNumChans / outNumChanGroups;
          unsigned outChanGroup = inChan / outChansPerGroup;
          unsigned outChanGroupElement = inChan % outChansPerGroup;
          Tensor out =
            activations[outChanGroup][x][y]
              .slice(outChanGroupElement,
                     outChanGroupElement + outChansPerVertex);
          auto v = graph.addVertex(fwd, "Convolution",
            { {"activationIn", window },
              {"weights", w },
              {"bias", biases.slice(inChan, inChan + outChansPerVertex) },
              {"activationOut", out}
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
