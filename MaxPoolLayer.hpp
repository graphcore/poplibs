#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"

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
                   unsigned stride)  :
    Layer(net, index),
    kernelSize(kernelSize),
    stride(stride) {
    layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
      std::to_string(kernelSize);
  }

  Tensor getFwdActivations() const {
    return activations;
  }

  Tensor getFwdZs() const {
    return activations;
  }

  Tensor getBwdErrors() const {
    // TODO
  }

  NonLinearityType getNonLinearityType() const {
    return NON_LINEARITY_NONE;
  }

  void describe(std::ostream &out) {
    out << "   -- Max pooling layer:\n"
        << "        Size: " << kernelSize << "x" << kernelSize << "\n"
        << "        Stride: " << stride << "\n"
        << "        Input: " << xDim << "x" << yDim
                     <<   "x" << numChannels << "\n"
        << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << numChannels << "\n";
  }

  void init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
    const auto dType = getDType();
    Layer *prev = getPrevLayer();
    auto in = prev->getFwdActivations();
    xDim = in.dim(1);
    yDim = in.dim(2);
    numChannels = in.dim(0) * in.dim(3);
    xDimOut = (xDim - kernelSize) / stride + 1;
    yDimOut = (yDim - kernelSize) / stride + 1;
    Layer *next = getNextLayer();
    numChanGroups = next->getNumChannelGroupsIn(xDimOut, yDimOut, numChannels);
    if (!numChanGroups)
      numChanGroups = in.dim(0);
    size_t chansPerGroup = numChannels / numChanGroups;
    activations = graph.addTensor(dType, {numChanGroups, xDimOut, yDimOut,
                                          chansPerGroup});
    mapTensor(activations, mapping);
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
    Tensor in = prev->getFwdActivations();
    unsigned prevChanGroups = in.dim(0);
    unsigned prevChansPerGroup = numChannels / prevChanGroups;
    unsigned chansPerGroup = numChannels / numChanGroups;
    ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
    for (unsigned i = 0; i < xDimOut; ++i) {
      for (unsigned j = 0; j < yDimOut; ++j) {
        for (unsigned chan = 0; chan < numChannels; ++chan) {
          unsigned width = std::min(i * stride + kernelSize, xDim) - i * stride;
          unsigned height = std::min(j * stride + kernelSize, yDim) - j * stride;
          // Create window into previous layer
          unsigned prevChanGroup = chan / prevChansPerGroup;
          unsigned prevChanInGroup = chan % prevChansPerGroup;
          unsigned chanGroup = chan / chansPerGroup;
          unsigned chanInGroup = chan % chansPerGroup;
          Tensor window =
            in[prevChanGroup].slice({i * stride, j * stride, prevChanInGroup},
                                    {i * stride + width, j * stride + height,
                                      prevChanInGroup+1})
              .flatten();
          graph.addVertex(fwd, "MaxPooling",
            { {"activationIn", window},
              {"activationOut", activations[chanGroup][i][j][chanInGroup]} });
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
