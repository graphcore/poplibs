#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

class ConvLayer : public HiddenLayer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned numLayers;
  NonLinearityType nonLinearityType;

  ConvLayer(unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned numLayers,
            NonLinearityType nonLinearityType) :
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    numLayers(numLayers),
    nonLinearityType(nonLinearityType) { }

  virtual bool requiresLayeredInput() {return true;}
  virtual bool providesLayeredOutput() {return true;}

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    unsigned xDim = net.xDim;
    unsigned yDim = net.yDim;

    unsigned numWeights =
      (kernelSize * kernelSize * net.prevLayers) * numLayers;
    VertexRef pv = builder.addVertex("ConvParamsFwdOnlyVertex");
    vertices.push_back(pv);
    builder.addToComputeSet(net.trainCS, pv);
    builder.addToComputeSet(net.testCS, pv);
    builder.setFieldSize(pv["weights"], numWeights);
    builder.setFieldSize(pv["bias"], numLayers);

    VertexRef padDataVertex = builder.addVertex("ConvPaddingVertex");
    builder.setFieldSize(padDataVertex["activationOut"], net.prevLayers);

    std::vector<VertexRef> fwd;
    for (unsigned i = 0; i <= yDim + padding - kernelSize; i += stride) {
      for (unsigned j = 0; j <= xDim + padding - kernelSize; j += stride) {
        VertexRef v = builder.addVertex("ConvLayerFwdVertex");
        vertices.push_back(v);
        fwd.push_back(v);
        builder.addToComputeSet(net.trainCS, v);
        builder.addToComputeSet(net.testCS, v);

        builder.addEdge(net.stateField, v["state"], false);
        builder.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                       nonLinearityType);
        builder.setFieldSize(v["activationOut"], net.prevLayers);
        builder.setFieldSize(v["zOut"], numLayers);
        builder.setFieldSize(v["activationIn"], kernelSize * kernelSize);

        builder.addEdge(net.fwd[i * xDim + j]["indexOut"],
                        v["indexIn"],
                        true);

        unsigned aIndex = 0;
        for (unsigned k1 = 0; k1 < kernelSize; k1++) {
          for (unsigned k2 = 0; k2 < kernelSize; k2++) {
            unsigned y = i + k1;
            unsigned x = j + k2;
            FieldRef src;
            if (x > xDim || y > yDim) {
              src = padDataVertex["activationOut"];
            } else {
              src = net.fwd[y * xDim + x]["activationOut"];
            }
            builder.addEdge(src,
                            v["activationIn"][aIndex++],
                            true);
          }
        }
        builder.addEdge(pv["weights"], v["weights"], false);
        builder.addEdge(pv["bias"], v["bias"], false);
      }
    }

    unsigned xDimOut = (xDim + padding - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim + padding - kernelSize) / stride + 1;

    std::cout << "   -- Added convolutional layer:\n"
              << "        Size: " << kernelSize << "x" << kernelSize << "\n"
              << "        Stride: " << stride << "\n"
              << "        Padding: " << padding << "\n"
              << "        Input: " << xDim << "x" << yDim
                     <<   "x" << net.prevLayers << "\n"
              << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << numLayers << "\n"
              << "        Params: " << numWeights + numLayers << "\n";

    net.xDim = xDimOut;
    net.yDim = yDimOut;
    net.prevLayers = numLayers;
    assert(fwd.size() == net.xDim * net.yDim);
    net.prevNonLinearityType = nonLinearityType;
    net.fwd = fwd;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    //TODO
  }
};



#endif // _conv_layer_hpp_
