#ifndef _max_pool_layer_hpp_
#define _max_pool_layer_hpp_
#include "Net.hpp"

class MaxPoolLayer : public HiddenLayer {
public:
  unsigned kernelSize;
  unsigned stride;

  MaxPoolLayer(unsigned kernelSize,
               unsigned stride)  :
    kernelSize(kernelSize),
    stride(stride) { }

  virtual bool requiresLayeredInput() {return true;}
  virtual bool providesLayeredOutput() {return true;}

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    unsigned xDim = net.xDim;
    unsigned yDim = net.yDim;

    std::vector<VertexRef> fwd;
    for (unsigned i = 0; i <= yDim - kernelSize; i += stride) {
      for (unsigned j = 0; j <= xDim - kernelSize; j += stride) {
        VertexRef v = builder.addVertex("MaxPoolFwdVertex");
        vertices.push_back(v);
        fwd.push_back(v);
        builder.addToComputeSet(net.trainCS, v);
        builder.addToComputeSet(net.testCS, v);

        builder.addEdge(net.stateField, v["state"], false);
        builder.setFieldSize(v["activationOut"], net.prevLayers);
        builder.setFieldSize(v["zOut"], net.prevLayers);
        builder.setFieldSize(v["activationIn"], kernelSize * kernelSize);

        builder.addEdge(net.fwd[i * xDim + j]["indexOut"],
                        v["indexIn"],
                        true);

        unsigned aIndex = 0;
        for (unsigned k1 = 0; k1 < kernelSize; k1++) {
          for (unsigned k2 = 0; k2 < kernelSize; k2++) {
            unsigned y = i + k1;
            unsigned x = j + k2;
            builder.addEdge(net.fwd[y * xDim + x]["activationOut"],
                            v["activationIn"][aIndex++],
                            true);
          }
        }
      }
    }


    unsigned xDimOut = (xDim - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim - kernelSize) / stride + 1;

    std::cout << "   -- Added max pooling layer:\n"
              << "        Size: " << kernelSize << "x" << kernelSize << "\n"
              << "        Stride: " << stride << "\n"
              << "        Input: " << xDim << "x" << yDim
                     <<   "x" << net.prevLayers << "\n"
              << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << net.prevLayers << "\n";

    net.xDim = xDimOut;
    net.yDim = yDimOut;
    assert(fwd.size() == net.xDim * net.yDim);
    net.prevNonLinearityType = NON_LINEARITY_NONE;
    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
  }
};



#endif // _max_pool_layer_hpp_
