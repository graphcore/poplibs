#include "ConvLayer.hpp"
#include "PerformanceEstimation.hpp"

namespace {
  struct ConvolutionParams {
    unsigned kernelSize;
    unsigned stride;
    unsigned inputDepth;
    unsigned inputWidth;
    unsigned inputHeight;
    unsigned padding;
    unsigned outputDepth;
    unsigned getOutputWidth() const {
      return (inputWidth + padding - kernelSize) / stride + 1;
    }
    unsigned getOutputHeight() const {
      return (inputHeight + padding - kernelSize) / stride + 1;
    }
    ConvolutionParams(unsigned kernelSize,
                      unsigned stride,
                      unsigned inputDepth,
                      unsigned inputWidth,
                      unsigned inputHeight,
                      unsigned padding,
                      unsigned outputDepth) :
      kernelSize(kernelSize),
      stride(stride),
      inputDepth(inputDepth),
      inputWidth(inputWidth),
      inputHeight(inputHeight),
      padding(padding),
      outputDepth(outputDepth) {}
  };
}

static unsigned
getMaxInputRangeSize(unsigned outputRangeSize, unsigned stride,
                     unsigned kernelSize, unsigned numPartitions,
                     unsigned padding) {
  auto inputRangeSize = outputRangeSize * stride + kernelSize - 1;
  // If the number of partitions is small the input range is guaranteed
  // to contain padding.
  switch (numPartitions) {
  default: return inputRangeSize;
  case 1: return inputRangeSize - padding;
  case 2: return inputRangeSize - padding / 2;
  }
}

static unsigned
estimateExchangeCost(bool isFloat, const ConvolutionParams &params,
                     const ConvLayerPartition &partition) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroupAxis = partition.tilesPerInZGroupAxis;
  const auto inChansPerGroup = partition.inChansPerGroup;

  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileOutDepth =
      (params.outputDepth + tilesPerZ - 1) / tilesPerZ;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;
  const auto tileInDepth = tileNumInGroups * inChansPerGroup;
  const auto tileInWidth =
      getMaxInputRangeSize(tileOutWidth, params.stride, params.kernelSize,
                           tilesPerX, params.padding);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.stride, params.kernelSize,
                           tilesPerY, params.padding);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      params.kernelSize * params.kernelSize * tileOutDepth * tileInDepth;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileOutDepth;
  const auto numberOfPartialSums = numberOfOutputElements;
  const auto elementSize = isFloat ? 4 : 2;
  const auto inputElementsBytes = numberOfInputElements * elementSize;
  const auto weightBytes = numberOfWeights * elementSize;
  const auto partialSumBytes = numberOfPartialSums * 4;
  const auto numCycles = (inputElementsBytes + 3) / 4 +
                         (weightBytes + 3) / 4 +
                         (partialSumBytes + 3) / 4;
  return numCycles;
}

static unsigned
estimateVertexCycles(bool isFloat, const ConvolutionParams &params,
                     const ConvLayerPartition &partition) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerInZGroupAxis = partition.tilesPerInZGroupAxis;
  const auto inChansPerGroup = partition.inChansPerGroup;

  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto numInRows = params.kernelSize * tileNumInGroups;

  return getConvPartialCycleEstimate(isFloat, inChansPerGroup, params.stride,
                                     params.kernelSize, numInRows,
                                     tileOutWidth);
}

static unsigned
estimateComputeCost(unsigned numWorkerContexts, bool isFloat,
                    const ConvolutionParams &params,
                    const ConvLayerPartition &partition) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;

  const auto outHeight = params.getOutputHeight();
  const auto outDepth = params.outputDepth;

  const auto tileY = (outHeight + tilesPerY - 1) / tilesPerY;
  const auto tileZ = (outDepth + tilesPerZ - 1) / tilesPerZ;

  const auto tileVertices = tileY * tileZ;
  const auto vertexRuntime = estimateVertexCycles(isFloat, params, partition);
  auto verticesPerWorker = (tileVertices + numWorkerContexts - 1) /
                           numWorkerContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numWorkerContexts;
  return computeCycles;
}

static unsigned
estimatePartitionCost(unsigned numWorkerContexts, bool isFloat,
                      const ConvolutionParams &params,
                      const ConvLayerPartition &partition) {
  return estimateExchangeCost(isFloat, params, partition) +
         estimateComputeCost(numWorkerContexts, isFloat, params, partition);
}

static ConvLayerPartition
choosePartition(unsigned numWorkerContexts,
                bool isFloat,
                unsigned inChansPerGroup,
                const ConvolutionParams &params,
                unsigned numTiles) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  ConvLayerPartition bestPartition;
  if (params.inputDepth % inChansPerGroup != 0) {
    // TODO handle this case.
    std::abort();
  }
  // If tilesPerY is greater than one we end up splitting across the y axis of
  // the output volume. The input elements required to compute output elements
  // on one side of the split will overlap with the input elements required for
  // the otherside of the split, increasing communication.
  // An alternative strategy would be to split across the y axis of
  // the input volume. Now there is no overlap in input elements read by each
  // tile, but nx1 convolutions for rows near the boundary must be summed
  // with nx1 convolutions for rows the other side the boundary. This results
  // to the communication for more partial sums.
  // Assuming a stide of 1, the alterative strategy reads
  // inputsChannelsPerTile * (filterSize - 1) fewer input rows per tile pair
  // but it needs to sends (outputChannelsPerTile * (filterSize - 1) / 2) extra
  // rows of partial sum per tile pair.
  // TODO investigate the alternative strategy outlined above.
  const auto maxTilesPerX = std::min(params.getOutputWidth(), numTiles);
  for (unsigned tilesPerX = 1; tilesPerX < maxTilesPerX; ++tilesPerX) {
    const auto maxTilesPerY = std::min(params.getOutputHeight(),
                                       numTiles / tilesPerX);
    for (unsigned tilesPerY = 1; tilesPerY < maxTilesPerY; ++tilesPerY) {
      const auto maxTilesPerZ =
          std::min(params.outputDepth, numTiles / (tilesPerX * tilesPerY));
      for (unsigned tilesPerZ = 1; tilesPerZ < maxTilesPerZ; ++tilesPerZ) {
        const auto tilesPerInZ =
            std::min(params.inputDepth / inChansPerGroup,
                     numTiles / (tilesPerX * tilesPerY * tilesPerZ));
        ConvLayerPartition candidate(tilesPerX, tilesPerY, tilesPerZ,
                                     tilesPerInZ, inChansPerGroup);
        auto candidateCost =
            estimatePartitionCost(numWorkerContexts, isFloat, params,
                                  candidate);
        if (candidateCost < bestCost) {
          bestPartition = candidate;
          bestCost = candidateCost;
        }
      }
    }
  }
  return bestPartition;
}

ConvLayerImpl::ConvLayerImpl(Net &net,
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
  layerName = "Conv" + std::to_string(kernelSize) + "x" +
              std::to_string(kernelSize);
}

static std::pair<unsigned, unsigned>
getInputRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
              unsigned inputSize) {
  const auto inputCentre = outputIndex * stride;
  const auto distanceFromCentre = (kernelSize - 1) / 2;
  const auto begin =
      inputCentre > distanceFromCentre ? inputCentre - distanceFromCentre : 0;
  const auto end = std::min(inputCentre + distanceFromCentre + 1, inputSize);
  return { begin, end };
}

static std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned inputSize) {
  assert(outputRange.first <= outputRange.second);
  return {
    getInputRange(outputRange.first, stride, kernelSize, inputSize).first,
    getInputRange(outputRange.second, stride, kernelSize, inputSize).second
  };
}

static std::pair<unsigned, unsigned>
getWeightRange(unsigned outputIndex, unsigned stride, unsigned kernelSize,
               unsigned inputSize) {
  const auto inputCentre = outputIndex * stride;
  const auto distanceFromCentre = (kernelSize - 1) / 2;
  unsigned inputBegin, inputEnd;
  std::tie(inputBegin, inputEnd) = getInputRange(outputIndex, stride,
                                                 kernelSize, inputSize);
  const auto weightBegin = inputBegin + distanceFromCentre - inputCentre;
  const auto weightEnd = inputEnd + distanceFromCentre - inputCentre;
  return { weightBegin, weightEnd };
}

std::uint64_t ConvLayerImpl::getNumberOfFlops() {
  std::uint64_t numFlops = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               inDimY);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 inDimX);
      const auto width = inXEnd - inXBegin;
      numFlops += 2 * width * height * outNumChans * inNumChans;
    }
  }
  return numFlops;
}

double ConvLayerImpl::getPerfectCycleCount() {
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  if (getDType() == "float") {
    // Can execute 2 f32 MACs per cycle.
    return static_cast<double>(getNumberOfFlops()) / (2 * 2 * numTiles);
  }
  assert(getDType() == "short");
  if (stride != 1) {
    // Can execute 4 f16 MACs per cycle.
    return static_cast<double>(getNumberOfFlops()) / (4 * 2 * numTiles);
  }
  // Can execute 12 f32 MACs per cycle for convolutions with a stride of 1.
  return static_cast<double>(getNumberOfFlops()) / (12 * 2 * numTiles);
}

void ConvLayerImpl::describe(std::ostream &out) {
  unsigned numParams = weights.numElements() + biases.numElements();
  out << "   -- Convolutional layer:\n"
      << "        Size: " << kernelSize << "x" << kernelSize << "\n"
      << "        Stride: " << stride << "\n"
      << "        Padding: " << padding << "\n"
      << "        Input: " << inDimX << "x" << inDimY
                  <<   "x" << inNumChans << "\n"
      << "        Output: " << outDimX << "x" << outDimY
                   <<   "x" << outNumChans << "\n"
      << "        Params: " << numParams << "\n"
      << "        FLOPs: " << getNumberOfFlops() << "\n";
}

size_t ConvLayerImpl::getNumChannelGroupsIn(size_t xPrev, size_t yPrev,
                                            size_t zPrev) const {
  unsigned inChansPerGroup;
  if (getDType() != "float" && stride == 1 && zPrev % 4 == 0) {
    // If doing the convolution by channel is preferred then try
    // and target the special convolution instructions
    // which require writing back to a 4-element vector.
    inChansPerGroup = 4;
  } else {
    const bool isFloat = getDType() == "float";
    const auto numWorkerContexts = getWorkerContextsPerTile();
    const auto numTiles = getNumIPUs() * getTilesPerIPU();
    unsigned bestCost = std::numeric_limits<unsigned>::max();
    ConvolutionParams params(kernelSize, stride, zPrev, xPrev,
                             yPrev, padding, outNumChans);
    for (unsigned i = 1; i <= zPrev; ++i) {
      if (zPrev % i != 0)
        continue;
      const auto candidate =
        choosePartition(numWorkerContexts, isFloat, i,
                        params, numTiles);
      const auto candidateCost =
          estimatePartitionCost(numWorkerContexts, isFloat, params,
                                candidate);
      if (candidateCost < bestCost) {
        inChansPerGroup = candidate.inChansPerGroup;
        bestCost = candidateCost;
      }
    }
  }
  return zPrev / inChansPerGroup;
}

void ConvLayerImpl::
init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  bool isFloat = dType == "float";
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations();
  inNumChanGroups = in.dim(0);
  inDimY = in.dim(1);
  inDimX = in.dim(2);
  size_t inChansPerGroup = in.dim(3);
  inNumChans = inChansPerGroup * inNumChanGroups;
  outDimX = (inDimX + padding - kernelSize) / stride + 1;
  outDimY = (inDimY + padding - kernelSize) / stride + 1;
  partition =
      choosePartition(this->getWorkerContextsPerTile(), isFloat,
                      inChansPerGroup,
                      ConvolutionParams(kernelSize, stride, inNumChans, inDimX,
                                        inDimY, padding, outNumChans),
                      getNumIPUs() * getTilesPerIPU());
  Layer *next = getNextLayer();
  outNumChanGroups = next->getNumChannelGroupsIn(inDimX, inDimY, outNumChans);
  size_t outChansPerGroup;
  if (outNumChanGroups) {
    outChansPerGroup = outNumChans / outNumChanGroups;
  } else {
    outChansPerGroup = isFloat ? 1 : 2;
    outNumChanGroups = outNumChans / outChansPerGroup;
  }
  assert(outNumChanGroups * outChansPerGroup == outNumChans);
  // Each ConvComplete vertex writes outChansPerGroup output channels. Because
  // sub-word access is not atomic we must ensure output channels are grouped
  // in multiples of two.
  assert(isFloat || outChansPerGroup % 2 == 0);
  z = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                              outChansPerGroup});
  activations = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                        outChansPerGroup});
  weights = graph.addTensor(dType, {inNumChanGroups,
                                    outNumChans,
                                    kernelSize,
                                    kernelSize,
                                    inChansPerGroup});
  biases = graph.addTensor(dType, {outNumChans});
  mapTensor(z, mapping);
  mapTensor(activations, mapping);
  mapTensor(biases, mapping);
}

void
ConvLayerImpl::forwardTile(Graph &graph,
                           IPUModelEngineBuilder::TileMapping *mapping,
                           unsigned tile,
                           unsigned outXBegin, unsigned outXEnd,
                           unsigned outYBegin, unsigned outYEnd,
                           unsigned outZBegin, unsigned outZEnd,
                           unsigned inZGroupBegin, unsigned inZGroupEnd,
                           ComputeSet cs,
                           const Tensor &out) {
  Layer *prev = getPrevLayer();
  Tensor in = prev->getFwdActivations();
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto inZGroups = inZGroupEnd - inZGroupBegin;
  for (unsigned z = outZBegin; z != outZEnd; ++z) {
    for (unsigned y = outYBegin; y != outYEnd; ++y) {
      unsigned inYBegin, inYEnd, inXBegin, inXEnd;
      std::tie(inYBegin, inYEnd) =
          getInputRange(y, stride, kernelSize, inDimY);
      std::tie(inXBegin, inXEnd) =
          getInputRange({outXBegin, outXEnd}, stride, kernelSize, inDimX);
      // Create a window into previous layer.
      const auto width = inXEnd - inXBegin;
      const auto height = inYEnd - inYBegin;
      Tensor inWindow =
          in.slice(
            {inZGroupBegin, inYBegin, inXBegin, 0},
            {inZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
          ).reshape({height * inZGroups,
                     width * inChansPerGroup});
      // Get weights that match the window.
      unsigned weightYBegin, weightYEnd;
      std::tie(weightYBegin, weightYEnd) =
          getWeightRange(y, stride, kernelSize, inDimY);
      assert(weightYEnd - weightYBegin == height);
      Tensor w =
          weights.slice(
            {inZGroupBegin, z, weightYBegin, 0, 0},
            {inZGroupEnd, z + 1 , weightYEnd, kernelSize, inChansPerGroup}
          ).reshape({height * inZGroups,
                     inChansPerGroup * kernelSize});
      Tensor outWindow = out[z][y].slice(outXBegin, outXEnd);
      // Add the vertex.
      auto v = graph.addVertex(cs, "ConvPartial",
          { {"in", inWindow },
            {"weights", w },
            {"out", outWindow },
          });
      const auto padding =
          inXBegin + (kernelSize - 1) / 2 - outXBegin * stride;
      graph.setInitialValue(v["stride"], stride);
      graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
      graph.setInitialValue(v["padding"], padding);
      // Map the vertex and output.
      if (mapping) {
        mapping->setMapping(v, tile);
        mapping->setMapping(outWindow, tile);
      }
    }
  }
}

Program ConvLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  auto prog = Sequence();
  const auto inChansPerGroup = partition.inChansPerGroup;
  Layer *prev = getPrevLayer();
  const auto dType = getDType();
  Tensor in = prev->getFwdActivations();
  unsigned outChansPerVertex = dType == "float" ? 1 : 2;
  assert(outNumChans % outChansPerVertex == 0);
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;

  assert(inNumChans % inChansPerGroup == 0);
  const auto numInZGroups = inNumChans / inChansPerGroup;
  Tensor partials = graph.addTensor("float",
                                    {tilesPerInZGroup,
                                     outNumChans,
                                     outDimY,
                                     outDimX});
  ComputeSet fwdCS = graph.createComputeSet(layerName + ".fwd");
  prog.add(Execute(fwdCS));
  for (unsigned i = 0; i != tilesPerInZGroup; ++i) {
    const auto inZGroupBegin = (i * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((i + 1) * numInZGroups) / tilesPerInZGroup;
    for (unsigned j = 0; j != tilesPerZ; ++j) {
      const auto outZBegin = (j * outNumChans) / tilesPerZ;
      const auto outZEnd = ((j + 1) * outNumChans) / tilesPerZ;
      for (unsigned k = 0; k != tilesPerY; ++k) {
        const auto outYBegin = (k * outDimY) / tilesPerY;
        const auto outYEnd = ((k + 1) * outDimY) / tilesPerY;
        for (unsigned l = 0; l != tilesPerX; ++l) {
          const auto tile =
              l + tilesPerX * (k + tilesPerY * (j + tilesPerZ * i));
          const auto outXBegin = (l * outDimX) / tilesPerX;
          const auto outXEnd = ((l + 1) * outDimX) / tilesPerX;
          forwardTile(graph, mapping,
                      tile, outXBegin, outXEnd, outYBegin, outYEnd, outZBegin,
                      outZEnd, inZGroupBegin, inZGroupEnd, fwdCS,
                      partials[i]);
        }
      }
    }
  }
  if (mapping) {
    for (unsigned i = 0; i != tilesPerInZGroup; ++i) {
      const auto inZGroupBegin = (i * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((i + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned j = 0; j != tilesPerZ; ++j) {
        const auto outZBegin = (j * outNumChans) / tilesPerZ;
        const auto outZEnd = ((j + 1) * outNumChans) / tilesPerZ;
        // Weights that are shared by tiles within this loop body.
        const auto sharedWeights =
            weights.slice({inZGroupBegin, outZBegin, 0, 0, 0},
                          {inZGroupEnd, outZEnd, kernelSize, kernelSize,
                           inChansPerGroup}).flatten();
        const auto numSharedWeights = sharedWeights.numElements();
        // Spread the weights equally across the tiles that read them.
        for (unsigned k = 0; k != tilesPerY * tilesPerX; ++k) {
          const auto tile =
              k + tilesPerY * tilesPerX * (j + tilesPerZ * i);
          const auto sharedWeightBegin =
              (k * numSharedWeights) / (tilesPerY * tilesPerX);
          const auto sharedWeightEnd =
              ((k + 1) * numSharedWeights) / (tilesPerY * tilesPerX);
          const auto tileWeights =
              sharedWeights.slice(sharedWeightBegin, sharedWeightEnd);
          mapping->setMapping(tileWeights, tile);
        }
      }
    }
  }
  Tensor reduced;
  if (tilesPerInZGroup == 1) {
    reduced = partials[0];
  } else {
    // Accumulate the partial sums.
    reduced = graph.addTensor("float", {outNumChans, outDimY, outDimX});
    ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
    const auto numTiles = getNumIPUs() * getTilesPerIPU();
    for (unsigned z = 0; z != outNumChans; ++z) {
      for (unsigned y = 0; y != outDimY; ++y) {
        Tensor in =
            partials.slice({0, z, y, 0},
                           {tilesPerInZGroup, z + 1, y + 1, outDimX}
            ).reshape({tilesPerInZGroup, outDimX});
        Tensor out = reduced[z][y];
        const auto v = graph.addVertex(reduceCS, "ConvReduce",
                                       {{"out", out},
                                       {"partials", in}});
        if (mapping) {
          const auto tile =
              (numTiles * (outDimY * z + y)) / (outDimY * outNumChans);
          mapping->setMapping(v, tile);
          mapping->setMapping(out, tile);
        }
      }
    }
    prog.add(Execute(reduceCS));
  }

  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  ComputeSet completionCS =
     graph.createComputeSet(layerName + ".fwd.complete");
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      biases.reshape({outNumChanGroups, outChansPerGroup});
  for (unsigned outChanGroup = 0; outChanGroup != outNumChanGroups;
       ++outChanGroup) {
    for (unsigned y = 0; y != outDimY; ++y) {
      for (unsigned x = 0; x != outDimX; ++x) {
        Tensor actOut = activations[outChanGroup][y][x];
        Tensor biasSlice = biasesByChanGroup[outChanGroup];
        Tensor reducedByChanGroup =
            reduced.reshape({outNumChanGroups, outChansPerGroup, outDimY,
                             outDimX});
        Tensor in =
            reducedByChanGroup.slice(
              {outChanGroup, 0, y, x},
              {outChanGroup + 1, outChansPerGroup, y + 1, x + 1}
            ).reshape({outChansPerGroup, 1});
        auto v = graph.addVertex(completionCS, "ConvComplete",
                                 {{ "in", in },
                                  { "bias", biasSlice },
                                  { "out", actOut} });
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      }
    }
  }
  mapComputeSet(graph, completionCS, mapping);
  prog.add(Execute(completionCS));
  return prog;
}
