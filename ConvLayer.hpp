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

  unsigned idealParamsPerVertex = (256*1024/10/4);

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

    unsigned layersPerVertex =
      idealParamsPerVertex / ((kernelSize * kernelSize * net.prevLayers) + 1);

    if (layersPerVertex == 0) {
      layersPerVertex = 1;
    } else if (numLayers % layersPerVertex != 0) {
      layersPerVertex = numLayers / ( numLayers / layersPerVertex + 1);
    }

    VertexRef padDataVertex = builder.addVertex("ConvPaddingVertex");
    builder.setFieldSize(padDataVertex["activationOut"], net.prevLayers);

    unsigned xDimOut = (xDim + padding - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim + padding - kernelSize) / stride + 1;

    std::vector<VertexRef> gatherVertices;
    std::vector<VertexRef> fwd;
    for (unsigned layer = 0; layer < numLayers / layersPerVertex; ++layer) {
      VertexRef pv = builder.addVertex("ConvParamsFwdOnlyVertex");
      vertices.push_back(pv);
      builder.addToComputeSet(net.trainCS, pv);
      builder.addToComputeSet(net.testCS, pv);
      builder.setFieldSize(pv["weights"], kernelSize * kernelSize *
                                          net.prevLayers * layersPerVertex);
      builder.setFieldSize(pv["bias"], layersPerVertex);

      unsigned outIndex = 0;
      for (unsigned i = 0; i <= yDim + padding - kernelSize; i += stride) {
        for (unsigned j = 0; j <= xDim + padding - kernelSize; j += stride) {
          VertexRef gv;
          if (layer == 0) {
            gv = builder.addVertex("InnerProductFwdLayeredGatherVertex");
            fwd.push_back(gv);
            vertices.push_back(gv);
            builder.setFieldSize(gv["activationIn"],
                                 numLayers / layersPerVertex);
            builder.setFieldSize(gv["activationOut"], numLayers);
            builder.addEdge(net.fwd[0]["indexOut"], gv["indexIn"], true);
            builder.addEdge(net.stateField, gv["state"], false);
          } else {
            gv = fwd[outIndex++];
          }

          VertexRef v = builder.addVertex("ConvLayerFwdVertex");
          vertices.push_back(v);
          builder.addToComputeSet(net.trainCS, v);
          builder.addToComputeSet(net.testCS, v);

          builder.addEdge(net.stateField, v["state"], false);
          builder.setInitialFieldValue<NonLinearityType>(v["nonLinearityType"],
                                                         nonLinearityType);
          builder.setFieldSize(v["activationIn"], kernelSize * kernelSize);
          builder.setFieldSize(v["activationOut"], layersPerVertex);
          builder.setFieldSize(v["zOut"], layersPerVertex);
          builder.addEdge(net.fwd[i * xDim + j]["indexOut"],
                          v["indexIn"],
                          true);

          builder.addEdge(v["activationOut"],
                          gv["activationIn"][layer],
                          false);

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
    }

    unsigned numWeights =
      (kernelSize * kernelSize * net.prevLayers) * numLayers;

    unsigned paramsPerVertex =
      (kernelSize * kernelSize * net.prevLayers) * layersPerVertex +
      layersPerVertex;

    std::cout << "   -- Added convolutional layer:\n"
              << "        Size: " << kernelSize << "x" << kernelSize << "\n"
              << "        Stride: " << stride << "\n"
              << "        Padding: " << padding << "\n"
              << "        Input: " << xDim << "x" << yDim
                     <<   "x" << net.prevLayers << "\n"
              << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << numLayers << "\n"
              << "        Params: " << numWeights + numLayers << "\n"
              << "        Params per vertex: " << paramsPerVertex << "\n";

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
