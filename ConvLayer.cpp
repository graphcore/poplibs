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
  const auto partialChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.outputDepth + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileOutDepth = tileNumOutGroups * partialChansPerGroup;
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
  const auto outChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto numInRows = params.kernelSize * tileNumInGroups;

  return getConvPartialCycleEstimate(isFloat, inChansPerGroup, params.stride,
                                     params.kernelSize, numInRows,
                                     tileOutWidth, outChansPerGroup);
}

static unsigned
estimateComputeCost(unsigned numWorkerContexts, bool isFloat,
                    const ConvolutionParams &params,
                    const ConvLayerPartition &partition) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto outChansPerGroup = partition.partialChansPerGroup;

  const auto outHeight = params.getOutputHeight();
  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileY = (outHeight + tilesPerY - 1) / tilesPerY;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  const auto tileVertices = tileY * tileNumOutGroups;
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

static bool
canUseConvolutionInstruction(bool isFloat, unsigned stride, unsigned kernelSize,
                             unsigned inChansPerGroup,
                             unsigned partialChansPerGroup) {
  return isFloat && stride == 1 && kernelSize == 1 && inChansPerGroup == 16 &&
         partialChansPerGroup == 4;
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
  std::vector<unsigned> partialChansPerGroupCandidates = { 1 };
  if (canUseConvolutionInstruction(isFloat, params.stride, params.kernelSize,
                                   inChansPerGroup, 4)) {
    partialChansPerGroupCandidates.push_back(4);
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
  for (unsigned tilesPerX = 1; tilesPerX <= maxTilesPerX; ++tilesPerX) {
    const auto maxTilesPerY = std::min(params.getOutputHeight(),
                                       numTiles / tilesPerX);
    for (unsigned tilesPerY = 1; tilesPerY <= maxTilesPerY; ++tilesPerY) {
      const auto maxTilesPerZ =
          std::min(params.outputDepth, numTiles / (tilesPerX * tilesPerY));
      for (unsigned tilesPerZ = 1; tilesPerZ <= maxTilesPerZ; ++tilesPerZ) {
        const auto tilesPerInZ =
            std::min(params.inputDepth / inChansPerGroup,
                     numTiles / (tilesPerX * tilesPerY * tilesPerZ));
        for (const auto partialChansPerGroup : partialChansPerGroupCandidates) {
          ConvLayerPartition candidate(tilesPerX, tilesPerY, tilesPerZ,
                                       tilesPerInZ, inChansPerGroup,
                                       partialChansPerGroup);
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
  }
  return bestPartition;
}

std::map<ConvImplSpec, ConvLayerImpl *> ConvLayerImpl::implMap;

ConvLayerImpl::ConvLayerImpl(Net &net,
                             int index,
                             unsigned kernelSize,
                             unsigned stride,
                             unsigned padding,
                             unsigned numChannels,
                             NonLinearityType nonLinearityType,
                             NormalizationType normalizationType,
                             unsigned resIndex,
                             enum ResidualMethod resMethod) :
  Layer(net, index),
  kernelSize(kernelSize),
  stride(stride),
  padding(padding),
  outNumChans(numChannels),
  nonLinearityType(nonLinearityType),
  normalizationType(normalizationType),
  createdForwardProg(false),
  resIndex(resIndex),
  resMethod(resMethod) {
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
  return {begin, end};
}

static std::pair<unsigned, unsigned>
getInputRange(std::pair<unsigned, unsigned> outputRange, unsigned stride,
              unsigned kernelSize, unsigned inputSize) {
  assert(outputRange.first <= outputRange.second);
  if (outputRange.first == outputRange.second) {
    return {0, 0};
  }
  const auto begin =
      getInputRange(outputRange.first, stride, kernelSize, inputSize).first;
  const auto end =
      getInputRange(outputRange.second - 1, stride, kernelSize,
                    inputSize).second;
  return {begin, end};
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

std::uint64_t ConvLayerImpl::getNumberOfMACs() {
  std::uint64_t numMACs = 0;
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
      numMACs += width * height * outNumChans * inNumChans;
    }
  }
  return numMACs;
}

std::uint64_t ConvLayerImpl::getNumberOfAdds() {
  if (!resIndex)
    return 0;

  // An addition is required to add in the residual information
  return outNumChans * outDimX * outDimY;
}


std::uint64_t ConvLayerImpl::getNumberOfFlops() {
  return 2 * getNumberOfMACs() + getNumberOfAdds();
}

double ConvLayerImpl::getPerfectCycleCount() {
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  if (getDType() == "float") {
    // Can execute 2 f32 MACs per cycle.
    auto macCycles =
       static_cast<double>(getNumberOfMACs()) / (2 * numTiles);
    // Can execute 2 f32 ADDs per cycle.
    auto addCycles =
       static_cast<double>(getNumberOfAdds()) / (2 * numTiles);
    return macCycles + addCycles;
  }
  assert(getDType() == "short");
  auto macsPerCycles = useConvolutionInstruction() ? 16 : 4;
  auto macCycles = static_cast<double>(getNumberOfMACs()) /
                   (macsPerCycles * numTiles);

  // Can execute 4 f16 ADDs per cycle.
  auto addCycles = static_cast<double>(getNumberOfAdds()) / (4 * numTiles);
  return macCycles + addCycles;
}

void ConvLayerImpl::describe(std::ostream &out) {
  unsigned numParams = weights.numElements() + biases.numElements();
  if (resIndex)
    out << "   -- Convolutional layer (residual):\n";
  else
    out << "   -- Convolutional layer:\n";
  out << "        Size: " << kernelSize << "x" << kernelSize << "\n"
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
  return zPrev / inChansPerGroup;
}

void ConvLayerImpl::
init(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  bool isFloat = dType == "float";
  Layer *prev = getPrevLayer();
  Tensor prevOut = prev->getFwdActivations();
  inNumChanGroups = prevOut.dim(0);
  inDimY = prevOut.dim(1);
  inDimX = prevOut.dim(2);
  size_t inChansPerGroup = prevOut.dim(3);
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
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  if (useConvolutionInstruction()) {
    weights = graph.addTensor(dType, {partialNumChanGroups,
                                      inNumChanGroups,
                                      kernelSize,
                                      kernelSize,
                                      partialChansPerGroup,
                                      inChansPerGroup});
  } else {
    assert(partialChansPerGroup == 1);
    weights = graph.addTensor(dType, {inNumChanGroups,
                                      outNumChans,
                                      kernelSize,
                                      kernelSize,
                                      inChansPerGroup});
  }
  biases = graph.addTensor(dType, {outNumChans});
  mapTensor(biases, mapping);

  unsigned resDimX = 0, resDimY = 0, resNumChans = 0, resNumChanGroups = 0,
           resChansPerGroup;
  if (resIndex) {
    resLayer = this;
    for (unsigned i = 0; i < resIndex; ++i)
      resLayer = resLayer->getPrevLayer();
    auto act = resLayer->getFwdActivations();
    resDimY = act.dim(1);
    resDimX = act.dim(2);
    if (resDimX < outDimX || resDimY < outDimY) {
      throw net_creation_error("Residual layers must use previous layers "
                               "with X and Y dimensions that are larger"
                               "than the current layer's output.");
    }
    resStrideX = resDimX / outDimX;
    resStrideY = resDimY / outDimY;
    resNumChanGroups = act.dim(0);
    resChansPerGroup = act.dim(3);
    resNumChans = resNumChanGroups * resChansPerGroup;
  }

  auto implSpec =
    ConvImplSpec(inNumChans, inNumChanGroups,
                 inDimX, inDimY,
                 outNumChans, outNumChanGroups,
                 outDimX, outDimY,
                 resNumChans, resNumChanGroups,
                 resDimX, resDimY,
                 kernelSize, stride, padding);



  auto emplaceResult = implMap.emplace(implSpec, this);
  if (!emplaceResult.second) {
    // Matching implementation already exists
    reuseImpl = emplaceResult.first->second;
    return;
  }

  in = graph.addTensor(dType, {prevOut.dim(0), prevOut.dim(1),
                               prevOut.dim(2), prevOut.dim(3)});
  z = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                              outChansPerGroup});
  activations = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                        outChansPerGroup});
  if (useConvolutionInstruction()) {
    weightsIn = graph.addTensor(dType, {partialNumChanGroups,
                                        inNumChanGroups,
                                        kernelSize,
                                        kernelSize,
                                        partialChansPerGroup,
                                        inChansPerGroup});
  } else {
    assert(partialChansPerGroup == 1);
    weightsIn = graph.addTensor(dType, {inNumChanGroups,
                                        outNumChans,
                                        kernelSize,
                                        kernelSize,
                                        inChansPerGroup});
  }
  biasesIn = graph.addTensor(dType, {outNumChans});
  mapTensor(z, mapping);
  mapTensor(biasesIn, mapping);
  if (resIndex) {
    resIn = graph.addTensor(dType, {resNumChanGroups,
                                    resDimY, resDimY,
                                    resChansPerGroup});
  }
}

void ConvLayerImpl::
addResidualCalc(Graph &graph,
                ComputeSet cs,
                IPUModelEngineBuilder::TileMapping *mapping) {
  assert(resLayer);
  auto resNumChanGroups = resIn.dim(0);
  auto resChansPerGroup = resIn.dim(3);
  auto resNumChans = resNumChanGroups * resChansPerGroup;
  if (resMethod != RESIDUAL_WEIGHTED_CONV &&
      resNumChans == outNumChans &&
      resNumChanGroups == outNumChanGroups) {
    // We can directly add the output of the previous layer to this
    // layer's output.
    residual = resIn;
    return;
  }
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  size_t resOutNumChanGroups =
      (resNumChans + outChansPerGroup - 1) / outChansPerGroup;
  size_t resOutNumChans = resOutNumChanGroups * outChansPerGroup;
  residual = graph.addTensor(getDType(), {resOutNumChanGroups, outDimY, outDimX,
                                          outChansPerGroup});
  mapTensor(residual, mapping);

  switch (resMethod) {
  case RESIDUAL_PAD:
    for (unsigned outChanGroup = 0;
         outChanGroup < resOutNumChanGroups;
         ++outChanGroup) {
      for (unsigned y = 0; y < outDimY; ++y) {
        for (unsigned x = 0; x < outDimX; ++x) {
          auto chansPerVertex = getDTypeSize() == 2 ? 2 : 1;
          assert(outChansPerGroup % chansPerVertex == 0);
          assert(resChansPerGroup % chansPerVertex == 0);
          for (unsigned outChanGroupElement = 0;
               outChanGroupElement < outChansPerGroup;
               outChanGroupElement += chansPerVertex) {
            Tensor out = residual[outChanGroup][y][x]
              .slice(outChanGroupElement,
                     outChanGroupElement + chansPerVertex);
            auto outChan = outChanGroup * outChansPerGroup +
              outChanGroupElement;
            if (outChan >= resNumChans) {
              auto v = graph.addVertex(cs, "Zero", {{"out",out}});
              continue;
            }
            auto resChanGroup = outChan / resChansPerGroup;
            auto resChanGroupElement = outChan % resChansPerGroup;
            assert(resChanGroup < resNumChanGroups);
            assert(resChanGroupElement < resChansPerGroup);
            assert(y * resStrideX < resIn.dim(1));
            assert(x * resStrideY < resIn.dim(2));
            Tensor in = resIn[resChanGroup][y * resStrideY][x * resStrideX]
              .slice(resChanGroupElement,
                     resChanGroupElement + chansPerVertex);
            auto v = graph.addVertex(cs, "CopyResidual",
                                     {{"in", in}, {"out",out}});
          }
        }
      }
    }
    break;
  case RESIDUAL_WEIGHTED_CONV:
  case RESIDUAL_WEIGHTED_CONV_IF_SIZES_DIFFER:
    assert(0 && "Weighted calculation of residual input not implemented");
    break;
  default:
    assert(0 && "Unknown residual calculation method");
  }
  // This compute set may have more added with a specific mapping later. Here,
  // we map the current vertices of the compute set using the mapComputeSet
  // helper.
  mapComputeSet(graph, cs, mapping);
  resStrideX = resStrideY = 1;
}

bool ConvLayerImpl::useConvolutionInstruction() const {
  const bool isFloat = getDType() == "float";
  return canUseConvolutionInstruction(isFloat, stride, kernelSize,
                                      partition.partialChansPerGroup,
                                      partition.inChansPerGroup);
}

void
ConvLayerImpl::forwardTile(Graph &graph,
                           IPUModelEngineBuilder::TileMapping *mapping,
                           unsigned tile,
                           unsigned outXBegin, unsigned outXEnd,
                           unsigned outYBegin, unsigned outYEnd,
                           unsigned outZGroupBegin, unsigned outZGroupEnd,
                           unsigned inZGroupBegin, unsigned inZGroupEnd,
                           ComputeSet cs,
                           const Tensor &out) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto inZGroups = inZGroupEnd - inZGroupBegin;
  for (unsigned zg = outZGroupBegin; zg != outZGroupEnd; ++zg) {
    for (unsigned y = outYBegin; y != outYEnd; ++y) {
      unsigned inYBegin, inYEnd, inXBegin, inXEnd;
      std::tie(inYBegin, inYEnd) =
          getInputRange(y, stride, kernelSize, inDimY);
      std::tie(inXBegin, inXEnd) =
          getInputRange({outXBegin, outXEnd}, stride, kernelSize, inDimX);
      // Window into previous layer.
      const auto width = inXEnd - inXBegin;
      const auto height = inYEnd - inYBegin;
      // Weights that match the window.
      unsigned weightYBegin, weightYEnd;
      std::tie(weightYBegin, weightYEnd) =
          getWeightRange(y, stride, kernelSize, inDimY);
      if (useConvolutionInstruction()) {
        Tensor inWindow =
            in.slice(
              {inZGroupBegin, inYBegin, inXBegin, 0},
              {inZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
            ).reshape({height * inZGroups,
                       width * inChansPerGroup});
        assert(weightYEnd - weightYBegin == height);
        Tensor w =
            weightsIn[zg].slice(
              {inZGroupBegin, weightYBegin, 0, 0, 0},
              {inZGroupEnd, weightYEnd, kernelSize, outChansPerGroup,
               inChansPerGroup}
            ).flatten();
        Tensor outWindow =
            out[zg][y].slice(
              outXBegin,
              outXEnd
            ).reshape({height, width * outChansPerGroup});
        // Add the vertex.
        auto v = graph.addVertex(cs, "ConvPartial1x1",
            { {"in", inWindow },
              {"weights", w },
              {"out", outWindow },
            });
        // Map the vertex and output.
        if (mapping) {
          mapping->setMapping(v, tile);
          mapping->setMapping(outWindow, tile);
        }
      } else {
        assert(outChansPerGroup == 1);
        const auto z = zg;
        Tensor inWindow =
            in.slice(
              {inZGroupBegin, inYBegin, inXBegin, 0},
              {inZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
            ).reshape({height * inZGroups,
                       width * inChansPerGroup});
        Tensor w =
            weightsIn.slice(
              {inZGroupBegin, z, weightYBegin, 0, 0},
              {inZGroupEnd, z + 1, weightYEnd, kernelSize, inChansPerGroup}
            ).reshape({height * inZGroups,
                       inChansPerGroup * kernelSize});
        Tensor outWindow = out[z][y].slice(outXBegin, outXEnd).flatten();
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
}

/// Convert a set of indices over the different dimensions of the partition
/// into a tile number.
static unsigned
linearizeTileIndices(unsigned izg, unsigned ox, unsigned oy,
                     unsigned ozg,
                     const ConvLayerPartition &partition,
                     bool isMultiIPU) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;

  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  if (isMultiIPU)
    return izg + tilesPerInZGroup *
             (ox + tilesPerX *
               (oy + tilesPerY *
                 ozg));
  // For single IPU systems this order appears to give the best results.
  // TODO understand why this is. Intuitively I'd expect the an ordering
  // that matches the input tensor, i.e. (izg, iy, ix, iz) to result in
  // less exchange.
  return ox + tilesPerX *
           (oy + tilesPerY *
             (ozg + tilesPerZ *
               izg));
}

void ConvLayerImpl::mapActivations(Graph &graph,
                                   IPUModelEngineBuilder::TileMapping *mapping,
                                   Tensor act) {
  if (!mapping)
    return;
  const auto numActivations = act.numElements();
  const auto chansPerGroup = act.dim(3);
  const auto numGroups = numActivations / chansPerGroup;
  const auto numTiles = getTilesPerIPU() * getNumIPUs();
  auto actByGroup = act.reshape({numGroups, chansPerGroup});
  // Spread groups of activations evenly across the tiles.
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto groupBegin = (tile * numGroups) / numTiles;
    const auto groupEnd = ((tile + 1) * numGroups) / numTiles;
    mapping->setMapping(actByGroup.slice(groupBegin, groupEnd), tile);
  }
}

void ConvLayerImpl::mapWeights(Graph &graph,
                               IPUModelEngineBuilder::TileMapping *mapping,
                               Tensor w) {
  if (useConvolutionInstruction()) {
    // TODO
    mapTensor(w, mapping);
    return;
  }
  const auto isMultiIPU = getNumIPUs() > 1;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  assert(partition.partialChansPerGroup == 1);

  if (mapping) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned oz = 0; oz != tilesPerZ; ++oz) {
        const auto outZBegin = (oz * outNumChans) / tilesPerZ;
        const auto outZEnd = ((oz + 1) * outNumChans) / tilesPerZ;
        // Weights that are shared by tiles within this loop body.
        const auto sharedWeights =
            w.slice({inZGroupBegin, outZBegin, 0, 0, 0},
                    {inZGroupEnd, outZEnd, kernelSize, kernelSize,
                     inChansPerGroup}).flatten();
        const auto numSharedWeights = sharedWeights.numElements();
        // Spread the weights equally across the tiles that read them.
        for (unsigned oy = 0; oy != tilesPerY; ++oy) {
          for (unsigned ox = 0; ox != tilesPerX; ++ox) {
            const auto iw = ox + tilesPerX * oy;
            const auto sharedWeightBegin =
                (iw * numSharedWeights) / (tilesPerY * tilesPerX);
            const auto sharedWeightEnd =
                ((iw + 1) * numSharedWeights) / (tilesPerY * tilesPerX);
            const auto tileWeights =
                sharedWeights.slice(sharedWeightBegin, sharedWeightEnd);
            const auto tile = linearizeTileIndices(izg, ox, oy, oz, partition,
                                                   isMultiIPU);
            mapping->setMapping(tileWeights, tile);
          }
        }
      }
    }
  }
}

void ConvLayerImpl::
createFwdProg(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  assert(!createdForwardProg);
  const auto isMultiIPU = getNumIPUs() > 1;

  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;

  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;

  assert(inNumChans % inChansPerGroup == 0);
  const auto numInZGroups = inNumChans / inChansPerGroup;
  Tensor partials = graph.addTensor("float",
                                    {tilesPerInZGroup,
                                     partialNumChanGroups,
                                     outDimY,
                                     outDimX,
                                     partialChansPerGroup});
  ComputeSet fwdCS = graph.createComputeSet(layerName + ".fwd");
  forwardProg.add(Execute(fwdCS));
  for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
    const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
    const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
    for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
      const auto outZGroupBegin = (ozg * partialNumChanGroups) / tilesPerZ;
      const auto outZGroupEnd = ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
      for (unsigned oy = 0; oy != tilesPerY; ++oy) {
        const auto outYBegin = (oy * outDimY) / tilesPerY;
        const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
        for (unsigned ox = 0; ox != tilesPerX; ++ox) {
          const auto outXBegin = (ox * outDimX) / tilesPerX;
          const auto outXEnd = ((ox + 1) * outDimX) / tilesPerX;
          const auto tile = linearizeTileIndices(izg, ox, oy, ozg, partition,
                                                 isMultiIPU);
          forwardTile(graph, mapping,
                      tile, outXBegin, outXEnd, outYBegin, outYEnd,
                      outZGroupBegin, outZGroupEnd, inZGroupBegin, inZGroupEnd,
                      fwdCS, partials[izg]);
        }
      }
    }
  }
  mapWeights(graph, mapping, weightsIn);
  Tensor reduced;
  ComputeSet reduceCS = graph.createComputeSet(layerName + ".fwd.reduce");
  bool executeReduceCS = false;
  if (resLayer) {
    addResidualCalc(graph, reduceCS, mapping);
    executeReduceCS = true;
  }
  if (tilesPerInZGroup == 1) {
    reduced = partials[0];
  } else {
    // Accumulate the partial sums.
    const auto numTiles = getNumIPUs() * getTilesPerIPU();
    reduced = graph.addTensor("float", {partialNumChanGroups, outDimY, outDimX,
                                        partialChansPerGroup});
    if (partialChansPerGroup == 1) {
      size_t outChansPerGroup = outNumChans / outNumChanGroups;
      const auto numGroups = activations.numElements() / outChansPerGroup;
      for (unsigned tile = 0; tile != numTiles; ++tile) {
        const auto groupBegin = (tile * numGroups) / numTiles;
        const auto groupEnd = ((tile + 1) * numGroups) / numTiles;
        if (groupBegin == groupEnd)
          continue;
        const auto rowBegin = groupBegin / outDimX;
        const auto rowEnd = (groupEnd + outDimX - 1) / outDimX;
        for (unsigned row = rowBegin; row != rowEnd; ++row) {
          const auto xBegin = row == rowBegin ? groupBegin - row * outDimX : 0;
          const auto xEnd = row + 1 == rowEnd ? groupEnd - row * outDimX :
                                                outDimX;
          assert(xBegin != xEnd);
          const auto outChanGroup = row / outDimY;
          const auto y = row % outDimY;
          for (unsigned groupIndex = 0; groupIndex != outChansPerGroup;
               ++groupIndex) {
            const auto z = outChanGroup * outChansPerGroup + groupIndex;
            Tensor in =
                partials.slice({0, z, y, xBegin, 0},
                               {tilesPerInZGroup, z + 1, y + 1, xEnd, 1}
                ).reshape({tilesPerInZGroup, xEnd - xBegin});
            Tensor out = reduced[z][y].slice(xBegin, xEnd).flatten();
            const auto v = graph.addVertex(reduceCS, "ConvReduce",
                                           {{"out", out},
                                           {"partials", in}});
            if (mapping) {
              mapping->setMapping(v, tile);
              mapping->setMapping(out, tile);
            }
          }
        }
      }
    } else {
      for (unsigned z = 0; z != partialNumChanGroups; ++z) {
        for (unsigned y = 0; y != outDimY; ++y) {
          Tensor in =
              partials.slice({0, z, y, 0, 0},
                             {tilesPerInZGroup, z + 1, y + 1, outDimX,
                              partialChansPerGroup}
              ).reshape({tilesPerInZGroup, outDimX * partialChansPerGroup});
          Tensor out = reduced[z][y].flatten();
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
    }
    executeReduceCS = true;
  }
  if (executeReduceCS) {
    forwardProg.add(Execute(reduceCS));
  }

  // Apply the non linearity and write back results in the layout desired by
  // the next layer. Each vertex handles outChansPerGroup output elements.
  // TODO: This step could be merged with the reduction step above.
  ComputeSet completionCS =
     graph.createComputeSet(layerName + ".fwd.complete");
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      biasesIn.reshape({outNumChanGroups, outChansPerGroup});
  for (unsigned outChanGroup = 0; outChanGroup != outNumChanGroups;
       ++outChanGroup) {
    for (unsigned y = 0; y != outDimY; ++y) {
      for (unsigned x = 0; x != outDimX; ++x) {
        Tensor actOut = activations[outChanGroup][y][x];
        Tensor biasSlice = biasesByChanGroup[outChanGroup];
        Tensor reducedChans = reduced.slice(
          {0, y, x, 0},
          {partialNumChanGroups, y + 1, x + 1, partialChansPerGroup}
        ).flatten();
        Tensor reducedByChanGroup =
            reducedChans.reshape({outNumChanGroups, outChansPerGroup});
        Tensor in =
            reducedByChanGroup[outChanGroup].reshape({outChansPerGroup, 1});
        auto resOutChanGroups = resLayer ? residual.dim(0) : 0;
        bool needsResidual = resLayer && outChanGroup < resOutChanGroups;
        std::string vertexType =
            needsResidual ? "ConvCompleteRes" : "ConvComplete";
        auto v = graph.addVertex(completionCS, vertexType,
                                 {{ "in", in },
                                  { "bias", biasSlice },
                                  { "out", actOut} });
        graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
        if (needsResidual) {
          // If the residual is taken directly from the previous layer (
          // as opposed to being zero-padded or converted), then striding over
          // the X,Y plane may still be needed (in this case resStride will not
          // be 1).
          Tensor res = residual[outChanGroup][y * resStrideY][x * resStrideX];
          graph.connect(res, v["res"]);
        }
      }
    }
  }
  mapComputeSet(graph, completionCS, mapping);
  mapActivations(graph, mapping, activations);
  forwardProg.add(Execute(completionCS));
  createdForwardProg = true;
}

Program ConvLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  Layer *prev = getPrevLayer();
  auto impl = reuseImpl ? reuseImpl : this;
  auto prog = Sequence();
  impl->mapActivations(graph, mapping, impl->getInputTensor());
  if (resLayer)
    impl->mapActivations(graph, mapping, impl->getInputResidual());
  prog.add(Copy(impl->getInputTensor(), prev->getFwdActivations()));
  impl->mapWeights(graph, mapping, weights);
  prog.add(Copy(impl->getInputWeights(), weights));
  prog.add(Copy(impl->getInputBiases(), biases));
  if (resLayer) {
    prog.add(Copy(impl->getInputResidual(),
                  resLayer->getFwdActivations()));
  }
  prog.add(impl->getOrCreateFwdProg(graph, mapping));
  return prog;
}

