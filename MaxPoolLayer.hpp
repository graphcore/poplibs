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
    unsigned xDimOut = (xDim - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim - kernelSize) / stride + 1;

    unsigned prevLayersPerChunk = net.prevLayers / net.prevChunks;

    std::vector<VertexRef> fwd;
        for (unsigned prevChunk = 0; prevChunk < net.prevChunks; prevChunk++) {


    for (unsigned ii = 0; ii <= yDim - kernelSize; ii += stride) {
      for (unsigned jj = 0; jj <= xDim - kernelSize; jj += stride) {
        VertexRef v = builder.addVertex("MaxPoolFwdVertex");
        unsigned d = (ii/stride) * xDimOut + (jj/stride);
        int i, j;
        d2xy(xDimOut, d, &i, &j);
        vertices.push_back(v);
        fwd.push_back(v);
        builder.addToComputeSet(net.trainCS, v);
        builder.addToComputeSet(net.testCS, v);

        builder.addEdge(net.stateField, v["state"], false);
        builder.setFieldSize(v["activationOut"], prevLayersPerChunk);
        //builder.setFieldSize(v["zOut"], net.prevLayers);
        builder.setFieldSize(v["activationIn"], kernelSize * kernelSize);

        builder.addEdge(net.fwd[i * xDim + j]["indexOut"],
                        v["indexIn"],
                        true);

        unsigned aIndex = 0;
        for (unsigned k1 = 0; k1 < kernelSize; k1++) {
          for (unsigned k2 = 0; k2 < kernelSize; k2++) {
            unsigned y = i + k1;
            unsigned x = j + k2;
            unsigned offset = prevChunk * xDim * yDim;
            builder.addEdge(net.fwd[offset + y * xDim + x]["activationOut"],
                            v["activationIn"][aIndex++],
                            true);
          }
        }
      }
    }
    }

    std::cout << "   -- Added max pooling layer:\n"
              << "        Size: " << kernelSize << "x" << kernelSize << "\n"
              << "        Stride: " << stride << "\n"
              << "        Input: " << xDim << "x" << yDim
                     <<   "x" << net.prevLayers << "\n"
              << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << net.prevLayers << "\n";

    net.xDim = xDimOut;
    net.yDim = yDimOut;
    assert(fwd.size() == net.xDim * net.yDim * net.prevChunks);
    net.prevNonLinearityType = NON_LINEARITY_NONE;
    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
  }
};



#endif // _max_pool_layer_hpp_
