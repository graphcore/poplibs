#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"

class MaxPoolLayer : public Layer {
public:
  unsigned kernelSize;
  unsigned stride;

  Tensor out, activations;

  std::string dType;

  unsigned xDim, yDim, numChannels, xDimOut, yDimOut;

  MaxPoolLayer(unsigned kernelSize,
               unsigned stride)  :
    kernelSize(kernelSize),
    stride(stride) { }

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

  void init(Graph &graph, Layer *prev, Layer *next, NetType netType,
            float eta, unsigned batchSize, const std::string &dType) {
    this->dType = dType;
    Tensor in = prev->getFwdActivations();
    xDim = in.dim(0);
    yDim = in.dim(1);
    numChannels = in.dim(2);
    xDimOut = (xDim - kernelSize) / stride + 1;
    yDimOut = (yDim - kernelSize) / stride + 1;
    activations = graph.addTensor(dType, {xDimOut, yDimOut, numChannels});
  }

  Program initParams(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program startBatch(Graph &graph) {
    // TODO
    return Sequence();
  }

  Program forward(Graph &graph, Layer *prev)  {
    Tensor in = prev->getFwdActivations();
    ComputeSet fwd = graph.createComputeSet();
    for (unsigned i = 0; i < xDimOut; ++i) {
      for (unsigned j = 0; j < yDimOut; ++j) {
        for (unsigned chan = 0; chan < numChannels; ++chan) {
          unsigned width = std::min(i * stride + kernelSize, xDim) - i * stride;
          unsigned height = std::min(j * stride + kernelSize, yDim) - j * stride;
          // Create window into previous layer
          Tensor window =
            in.slice({i * stride, j * stride, chan },
                     {i * stride + width, j * stride + height, chan + 1})
              .reshape({width, height});
          graph.addVertex(fwd, "MaxPooling",
            { {"activationIn", window},
              {"activationOut", activations[i][j][chan]} });
        }
      }
    }
    return Execute(fwd);
  }

  Program backward(Graph &graph, Layer *prev, Layer *next) {
    // TODO
    return Sequence();
  }

  Program weightSync(Graph &graph) {
    // TODO
    return Sequence();
  }
};



#endif // _max_pool_layer_hpp_
