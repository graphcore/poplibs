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
    Graph &graph = *net.graph;
    unsigned xDim = net.xDim;
    unsigned yDim = net.yDim;
    bool normalize = normalizationType != NORMALIZATION_NONE;
    bool pipelineReduction = true;


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

    VertexRef padDataVertex = graph.addVertex("ConvPaddingVertex");
    graph.setFieldSize(padDataVertex["activationOut"], net.prevLayers);

    VertexRef normPad;
    if (normalize) {
      normPad = graph.addVertex("ConvPaddingVertex");
      graph.setFieldSize(normPad["activationOut"],
                         normPadSize * inputChunksPerGroup);
    }

    unsigned xDimOut = (xDim + padding - kernelSize) / stride + 1;
    unsigned yDimOut = (yDim + padding - kernelSize) / stride + 1;

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


    std::vector<VertexRef> fwd;
      //     VertexRef gv;
    std::vector<VertexRef> vs(xDimOut * yDimOut);

    for (unsigned i = 0; i < xDimOut * yDimOut * numChunks; ++i) {
      VertexRef v;
      if (normalize)
        v = graph.addVertex("ConvReductionNormVertex");
      else
        v = graph.addVertex("ConvReductionVertex");
      fwd.push_back(v);
      graph.addToComputeSet(net.trainCS, v);
      graph.addToComputeSet(net.testCS, v);
    }


    for (unsigned chunk = 0; chunk < numChunks; ++chunk) {
      VertexRef bv = graph.addVertex("ConvBiasFwdOnlyVertex");
      //graph.addToComputeSet(net.trainCS, bv);
      //graph.addToComputeSet(net.testCS, bv);
      graph.setFieldSize(bv["bias"], kernelsPerChunk);
      graph.setFieldSize(bv["topBias"], normPadSize);
      graph.setFieldSize(bv["bottomBias"], normPadSize);

      unsigned group = chunk / (numChunks / numInputGroups);

      unsigned firstPrevChunk = group * inputChunksPerGroup;

      for (unsigned prevChunk = firstPrevChunk;
           prevChunk < firstPrevChunk + inputChunksPerGroup;
           prevChunk++) {
      VertexRef wv = graph.addVertex("ConvWeightsFwdOnlyVertex");
      //graph.addToComputeSet(net.trainCS, wv);
      //graph.addToComputeSet(net.testCS, wv);
      graph.setFieldSize(wv["weights"], kernelSize * kernelSize *
                                          prevLayersPerChunk * kernelsPerChunk);

      for (unsigned i = 0; i <= yDim + padding - kernelSize; i += stride) {
        for (unsigned j = 0; j <= xDim + padding - kernelSize; j += stride) {
          unsigned outIndex = (i/stride) * xDimOut + (j/stride);
          VertexRef v;
          v = graph.addVertex("ConvLayerFwdVertex");
          VertexRef gv = fwd[chunk * xDimOut * yDimOut + outIndex];;
          if (prevChunk == firstPrevChunk) {
            graph.setInitialFieldValue<NonLinearityType>(
                  gv["nonLinearityType"],
                  nonLinearityType);
            graph.setFieldSize(gv["zIn"], inputChunksPerGroup);
            graph.setFieldSize(gv["activationOut"], kernelsPerChunk);
            graph.addEdge(v["indexOut"], gv["indexIn"], pipelineReduction);
            graph.addEdge(net.stateField, gv["state"], false);
            graph.addEdge(bv["bias"], gv["bias"], false);
            if (normalize) {
              graph.setInitialFieldValue<NormalizationType>(
                gv["normalizationType"],
                normalizationType);
              graph.addEdge(bv["topBias"], gv["bottomBias"], false);
              graph.addEdge(bv["bottomBias"], gv["topBias"], false);
              graph.setFieldSize(gv["bottomIn"], inputChunksPerGroup);
              graph.setFieldSize(gv["topIn"], inputChunksPerGroup);
              graph.setFieldSize(gv["top"], normPadSize);
              graph.setFieldSize(gv["bottom"], normPadSize);

              if (chunk == numChunks - 1) {
                for (unsigned k = 0; k < inputChunksPerGroup; ++k) {
                  graph.addEdge(normPad["activationOut"],
                                gv["topIn"][k],
                                false);
                }
              }

              if (chunk == 0) {
                for (unsigned k = 0; k < inputChunksPerGroup; ++k) {
                  graph.addEdge(normPad["activationOut"],
                                gv["bottomIn"][k],
                                false);
                }
              }
            }

          }

          vs[outIndex] = v;
          graph.addEdge(v["zOut"],
                        gv["zIn"][prevChunk - firstPrevChunk],
                        pipelineReduction);

          if (normalize) {
            if (chunk != 0) {
              VertexRef prevGV = fwd[(chunk-1) * xDimOut * yDimOut + outIndex];
              graph.addEdge(v["bottom"],
                            prevGV["topIn"][prevChunk - firstPrevChunk],
                            pipelineReduction);
            }

            if (chunk != numChunks - 1) {
              VertexRef nextGV = fwd[(chunk+1) * xDimOut * yDimOut + outIndex];
              graph.addEdge(v["top"],
                            nextGV["bottomIn"][prevChunk - firstPrevChunk],
                            pipelineReduction);
            }

          }

          graph.setFieldSize(v["top"], normPadSize);
          graph.setFieldSize(v["bottom"], normPadSize);
          graph.setFieldSize(v["zOut"], kernelsPerChunk);
          graph.addToComputeSet(net.trainCS, v);
          graph.addToComputeSet(net.testCS, v);

          graph.addEdge(net.stateField, v["state"], false);
          graph.setFieldSize(v["activationIn"], kernelSize * kernelSize);
          graph.addEdge(net.fwd[i * xDim + j]["indexOut"],
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
              graph.addEdge(src,
                            v["activationIn"][aIndex++],
                            true);
            }
          }
          graph.addEdge(wv["weights"], v["weights"], false);
        }
      }
    }
    }


    net.xDim = xDimOut;
    net.yDim = yDimOut;
    net.prevLayers = numKernels;
    assert(fwd.size() == net.xDim * net.yDim * numChunks);
    net.prevNonLinearityType = nonLinearityType;
    net.fwd = fwd;
    net.prevChunks = numChunks;
  }

  void addBackward(Net &net)  {
    Graph &graph = *net.graph;
    //TODO
  }
};



#endif // _conv_layer_hpp_
