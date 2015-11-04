#ifndef _conv_layer_hpp_
#define _conv_layer_hpp_
#include "Net.hpp"

class ConvLayer : public HiddenLayer {
public:
  unsigned kernelSize;
  unsigned stride;
  unsigned padding;
  unsigned numInputGroups;
  unsigned numKernels;
  NonLinearityType nonLinearityType;
  NormalizationType normalizationType;
  // Split the parameter holding vertices in ~20k chunks
  unsigned idealParamsPerVertex = 20000/4;

  ConvLayer(unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned numInputGroups,
            unsigned numKernels,
            NonLinearityType nonLinearityType,
            NormalizationType normalizationType) :
    kernelSize(kernelSize),
    stride(stride),
    padding(padding),
    numInputGroups(numInputGroups),
    numKernels(numKernels),
    nonLinearityType(nonLinearityType),
    normalizationType(normalizationType) { }

  virtual bool requiresLayeredInput() {return true;}
  virtual bool providesLayeredOutput() {return true;}

  void addForward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    unsigned xDim = net.xDim;
    unsigned yDim = net.yDim;

    // TODO: These shouldn't be assert, the network should force this to be
    // the case
    assert(net.prevChunks >= numInputGroups &&
           net.prevChunks % numInputGroups == 0);

    unsigned prevLayersPerChunk = net.prevLayers / net.prevChunks;
    unsigned kernelsPerChunk =
      idealParamsPerVertex /
          ((kernelSize * kernelSize * prevLayersPerChunk) + 1);

    if (kernelsPerChunk == 0) {
      kernelsPerChunk = 1;
    }

    while (numKernels % kernelsPerChunk != 0) {
      kernelsPerChunk--;
    }
    unsigned normPadSize = 2;

    unsigned numChunks = numKernels / kernelsPerChunk;

    if (numChunks < numInputGroups)
      numChunks = numInputGroups;

    while (numChunks % numInputGroups != 0) {
      numChunks++;
    }

    unsigned inputChunksPerGroup = net.prevChunks / numInputGroups;

    VertexRef padDataVertex = builder.addVertex("ConvPaddingVertex");
    builder.setFieldSize(padDataVertex["activationOut"], net.prevLayers);

    VertexRef normPad = builder.addVertex("ConvPaddingVertex");
    builder.setFieldSize(normPad["activationOut"],
                         normPadSize * inputChunksPerGroup);

    unsigned xDimOut = (xDim + padding - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim + padding - kernelSize) / stride + 1;

    std::vector<VertexRef> fwd;
      //     VertexRef gv;
    std::vector<VertexRef> vs(xDimOut * yDimOut);

    for (unsigned i = 0; i < xDimOut * yDimOut * numChunks; ++i) {
      VertexRef v = builder.addVertex("ConvReductionVertex");
      fwd.push_back(v);
      builder.addToComputeSet(net.trainCS, v);
      builder.addToComputeSet(net.testCS, v);
    }


    for (unsigned chunk = 0; chunk < numChunks; ++chunk) {
      VertexRef bv = builder.addVertex("ConvBiasFwdOnlyVertex");
      builder.addToComputeSet(net.trainCS, bv);
      builder.addToComputeSet(net.testCS, bv);
      builder.setFieldSize(bv["bias"], kernelsPerChunk);
      builder.setFieldSize(bv["topBias"], normPadSize);
      builder.setFieldSize(bv["bottomBias"], normPadSize);

      unsigned group = chunk / (numChunks / numInputGroups);

      unsigned firstPrevChunk = group * inputChunksPerGroup;

      for (unsigned prevChunk = firstPrevChunk;
           prevChunk < firstPrevChunk + inputChunksPerGroup;
           prevChunk++) {
      VertexRef wv = builder.addVertex("ConvWeightsFwdOnlyVertex");
      builder.addToComputeSet(net.trainCS, wv);
      builder.addToComputeSet(net.testCS, wv);
      builder.setFieldSize(wv["weights"], kernelSize * kernelSize *
                                          prevLayersPerChunk * kernelsPerChunk);

      for (unsigned i = 0; i <= yDim + padding - kernelSize; i += stride) {
        for (unsigned j = 0; j <= xDim + padding - kernelSize; j += stride) {
          unsigned outIndex = (i/stride) * xDimOut + (j/stride);

          VertexRef gv = fwd[chunk * xDimOut * yDimOut + outIndex];;
          if (prevChunk == firstPrevChunk) {
            builder.setInitialFieldValue<NonLinearityType>(
                  gv["nonLinearityType"],
                  nonLinearityType);
            builder.setInitialFieldValue<NormalizationType>(
                  gv["normalizationType"],
                  normalizationType);
            builder.setFieldSize(gv["zIn"], inputChunksPerGroup);
            builder.setFieldSize(gv["activationOut"], kernelsPerChunk);
            builder.addEdge(net.fwd[0]["indexOut"], gv["indexIn"], true);
            builder.addEdge(net.stateField, gv["state"], false);
            builder.addEdge(bv["bias"], gv["bias"], false);
            builder.addEdge(bv["topBias"], gv["bottomBias"], false);
            builder.addEdge(bv["bottomBias"], gv["topBias"], false);
            builder.setFieldSize(gv["bottomIn"], inputChunksPerGroup);
            builder.setFieldSize(gv["topIn"], inputChunksPerGroup);
            builder.setFieldSize(gv["top"], normPadSize);
            builder.setFieldSize(gv["bottom"], normPadSize);

            if (chunk == numChunks - 1) {
              for (unsigned k = 0; k < inputChunksPerGroup; ++k) {
                builder.addEdge(normPad["activationOut"],
                                gv["topIn"][k],
                                false);
              }
            }

            if (chunk == 0) {
              for (unsigned k = 0; k < inputChunksPerGroup; ++k) {
                builder.addEdge(normPad["activationOut"],
                                gv["bottomIn"][k],
                                false);
              }
            }
          }

          VertexRef v;
          v = builder.addVertex("ConvLayerFwdVertex");
          vs[outIndex] = v;
          builder.addEdge(v["zOut"],
                          gv["zIn"][prevChunk - firstPrevChunk],
                          false);

          if (chunk != 0) {
            VertexRef prevGV = fwd[(chunk-1) * xDimOut * yDimOut + outIndex];
            builder.addEdge(v["bottom"],
                            prevGV["topIn"][prevChunk - firstPrevChunk],
                            false);
          }

          if (chunk != numChunks - 1) {
            VertexRef nextGV = fwd[(chunk+1) * xDimOut * yDimOut + outIndex];
            builder.addEdge(v["top"],
                            nextGV["bottomIn"][prevChunk - firstPrevChunk],
                            false);
          }

          builder.setFieldSize(v["zOut"], kernelsPerChunk);
          builder.setFieldSize(v["top"], normPadSize);
          builder.setFieldSize(v["bottom"], normPadSize);
          builder.addToComputeSet(net.trainCS, v);
          builder.addToComputeSet(net.testCS, v);

          builder.addEdge(net.stateField, v["state"], false);
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
                unsigned offset = prevChunk * xDim * yDim;
                src = net.fwd[offset + y * xDim + x]["activationOut"];
              }
              builder.addEdge(src,
                              v["activationIn"][aIndex++],
                              true);
            }
          }
          builder.addEdge(wv["weights"], v["weights"], false);
        }
      }
    }
    }
    unsigned numWeights =
      (kernelSize * kernelSize * net.prevLayers) * numKernels;

    unsigned paramsPerChunk =
      (kernelSize * kernelSize * prevLayersPerChunk) * kernelsPerChunk +
      kernelsPerChunk;

    std::cout << "   -- Added convolutional layer:\n"
              << "        Size: " << kernelSize << "x" << kernelSize << "\n"
              << "        Stride: " << stride << "\n"
              << "        Padding: " << padding << "\n"
              << "        Input: " << xDim << "x" << yDim
                     <<   "x" << net.prevLayers << "\n"
              << "        Input groups: " << numInputGroups << "\n"
              << "        Output: " << xDimOut << "x" << yDimOut
                     <<   "x" << numKernels << "\n"
              << "        Params: " << numWeights + numKernels << "\n"
              << "        Num input chunks: " << net.prevChunks << "\n"
              << "        Num output chunks: " << numChunks << "\n"
              << "        Params per chunk: " << paramsPerChunk << "\n";

    net.xDim = xDimOut;
    net.yDim = yDimOut;
    net.prevLayers = numKernels;
    assert(fwd.size() == net.xDim * net.yDim * numChunks);
    net.prevNonLinearityType = nonLinearityType;
    net.fwd = fwd;
    net.prevChunks = numChunks;
  }

  void addBackward(Net &net)  {
    GraphBuilder &builder = *net.graphBuilder;
    //TODO
  }
};



#endif // _conv_layer_hpp_
