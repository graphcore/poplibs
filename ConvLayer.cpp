#include "ConvLayer.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexTemplates.hpp"
#include "ConvUtil.hpp"
#include "gcd.hpp"

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
      return (inputWidth + (padding * 2) - kernelSize) / stride + 1;
    }
    unsigned getOutputHeight() const {
      return (inputHeight + (padding * 2) - kernelSize) / stride + 1;
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
                     unsigned kernelSize, unsigned padding,
                     unsigned numPartitions,
                     unsigned inputSize, bool contiguousAccess) {
  if (outputRangeSize == 0)
    return 0;
  unsigned inputRangeSize;
  // If the number of partitions is small the input range is guaranteed
  // to contain padding.
  switch (numPartitions) {
  case 1:
  case 2:
    {
      auto inputRange = getInputRange({0, outputRangeSize}, stride,
                                      kernelSize, padding,
                                      inputSize);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    inputRangeSize = (outputRangeSize - 1) * stride + 1 + (kernelSize - 1);
    break;
  }
  if (!contiguousAccess && kernelSize == 1 && stride > 1) {
    inputRangeSize = (inputRangeSize - 1) / stride + 1;
  }
  return inputRangeSize;
}

static unsigned
estimateExchangeCost(IPUModelEngineBuilder &engineBuilder,
                     bool floatActivations, const ConvolutionParams &params,
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
                           params.padding,
                           tilesPerX, params.inputWidth, true);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.stride, params.kernelSize,
                           params.padding,
                           tilesPerY, params.inputWidth, false);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      params.kernelSize * params.kernelSize * tileOutDepth * tileInDepth;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileOutDepth;
  const auto numberOfPartialSums = numberOfOutputElements;
  const auto activationSize = floatActivations ? 4 : 2;
  const auto inputElementsBytes = numberOfInputElements * activationSize;
  const auto weightBytes = numberOfWeights * activationSize;
  const auto partialSize = partition.floatPartials ? 4 : 2;
  const auto partialSumBytes = numberOfPartialSums * partialSize;
  const auto exchangeBytesPerCycle = engineBuilder.getIPUExchangeBandwidth();
  const auto numCycles =
      (inputElementsBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (weightBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return numCycles;
}

static unsigned getNumConvUnits(bool floatPartial,
                                const IPUMachineInfo &machineInfo) {
  return floatPartial ? machineInfo.fp32AccumConvUnitsPerTile :
                        machineInfo.fp16AccumConvUnitsPerTile;
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned stride,
                             unsigned inChansPerGroup,
                             unsigned partialChansPerGroup,
                             const IPUMachineInfo &machineInfo) {
  if (floatActivations || stride >= (1 << 4) ||
      inChansPerGroup != machineInfo.getInputChannelsPerConvUnit())
    return false;
  return partialChansPerGroup == getNumConvUnits(floatPartials, machineInfo);
}

static std::uint64_t
getConvPartialCycleEstimate(bool floatActivations, bool floatPartials,
                            unsigned inChansPerGroup,
                            unsigned stride, unsigned kernelWidth,
                            unsigned inputGroupsPerOutput,
                            unsigned outputHeight,
                            unsigned outputWidth,
                            unsigned outChansPerGroup,
                            const IPUMachineInfo &machineInfo,
                            bool useSupervisorVertices)
{
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   stride, inChansPerGroup,
                                   outChansPerGroup,
                                   machineInfo)) {
    return getConvPartial1x1CycleEstimate(
          kernelWidth, inputGroupsPerOutput, outputHeight, outputWidth,
          machineInfo.convUnitPipelineDepth,
          getNumConvUnits(floatPartials, machineInfo), useSupervisorVertices);
  }
  assert(!useSupervisorVertices);
  return getConvPartialByDotProductCycleEstimate(floatActivations,
                                                 inChansPerGroup,
                                                 kernelWidth,
                                                 inputGroupsPerOutput,
                                                 outputHeight,
                                                 outputWidth,
                                                 outChansPerGroup,
                                                 machineInfo.dataPathWidth);
}

static unsigned
estimateVertexCycles(bool floatActivations,
                     const ConvolutionParams &params,
                     const ConvLayerPartition &partition,
                     const IPUMachineInfo &machineInfo,
                     bool useSupervisorVertices) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerInZGroupAxis = partition.tilesPerInZGroupAxis;
  const auto verticesPerTilePerY = partition.verticesPerTilePerYAxis;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto outRowsPerVertex =
      (tileOutHeight + verticesPerTilePerY - 1) / verticesPerTilePerY;
  const auto inputGroupsPerOutput = params.kernelSize * tileNumInGroups;
  return getConvPartialCycleEstimate(floatActivations, partition.floatPartials,
                                     inChansPerGroup, params.stride,
                                     params.kernelSize, inputGroupsPerOutput,
                                     outRowsPerVertex, tileOutWidth,
                                     outChansPerGroup,
                                     machineInfo,
                                     useSupervisorVertices);
}

static unsigned
estimateFwdComputeCost(IPUModelEngineBuilder &engineBuilder,
                       bool floatActivations,
                       const ConvolutionParams &params,
                       const ConvLayerPartition &partition,
                       const IPUMachineInfo &machineInfo) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto outChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  const auto verticesPerTilePerY =
      std::min(tileOutHeight, partition.verticesPerTilePerYAxis);
  const auto tileVertices = verticesPerTilePerY * tileNumOutGroups;
  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  bool useSupervisorVertices = false;
  unsigned numContexts = engineBuilder.getNumWorkerContexts();
  if (machineInfo.sharedConvWeights &&
      canUseConvolutionInstruction(floatActivations, partition.floatPartials,
                                   params.stride,
                                   partition.inChansPerGroup,
                                   partition.partialChansPerGroup,
                                   machineInfo)) {
    useSupervisorVertices = true;
    numContexts = 1;
  }
  const auto vertexRuntime = estimateVertexCycles(floatActivations, params,
                                                  partition,
                                                  machineInfo,
                                                  useSupervisorVertices);
  auto verticesPerWorker = (tileVertices + numContexts - 1) /
                           numContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numContexts;
  return computeCycles;
}

static unsigned
estimateReduceComputeCost(IPUModelEngineBuilder &engineBuilder,
                          const ConvolutionParams &params,
                          const ConvLayerPartition &partition,
                          const IPUMachineInfo &machineInfo) {
  if (partition.tilesPerInZGroupAxis == 1)
    return 0;
  const auto numTiles = engineBuilder.getNumTiles();
  const auto numOutputs = params.getOutputHeight() * params.getOutputWidth() *
                          params.outputDepth;
  const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
  const auto numPartialSumsPerTile = numOutputsPerTile *
                                     partition.tilesPerInZGroupAxis;
  const auto vectorWidth =
      partition.floatPartials ? machineInfo.getFloatVectorWidth() :
                                machineInfo.getHalfVectorWidth();
  const auto numCycles = (numPartialSumsPerTile + vectorWidth - 1) /
                          vectorWidth;
  return numCycles;
}

static unsigned
estimateComputeCost(IPUModelEngineBuilder &engineBuilder,
                    bool floatActivations, const ConvolutionParams &params,
                    const ConvLayerPartition &partition,
                    const IPUMachineInfo &machineInfo) {
  return estimateFwdComputeCost(engineBuilder, floatActivations, params,
                                partition, machineInfo) +
         estimateReduceComputeCost(engineBuilder, params,
                                   partition, machineInfo);

}

static unsigned
estimatePartitionCostBounded(IPUModelEngineBuilder &engineBuilder,
                             bool floatActivations,
                             const ConvolutionParams &params,
                             const ConvLayerPartition &partition,
                             const IPUMachineInfo &machineInfo,
                             unsigned maxBound) {
  auto cost = estimateExchangeCost(engineBuilder,
                                   floatActivations, params, partition);
  if (cost > maxBound)
    return maxBound;
  cost += estimateComputeCost(engineBuilder, floatActivations, params,
                              partition, machineInfo);
  return std::min(cost, maxBound);
}

static unsigned
estimatePartitionCost(IPUModelEngineBuilder &engineBuilder,
                      bool floatActivations,
                      const ConvolutionParams &params,
                      const ConvLayerPartition &partition,
                      const IPUMachineInfo &machineInfo) {
  return estimatePartitionCostBounded(engineBuilder,
                                      floatActivations, params, partition,
                                      machineInfo,
                                      std::numeric_limits<unsigned>::max());
}

static ConvLayerPartition
choosePartition(IPUModelEngineBuilder &engineBuilder,
                bool floatActivations,
                unsigned inChansPerGroup,
                const ConvolutionParams &params,
                const IPUMachineInfo &machineInfo) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  ConvLayerPartition bestPartition;
  if (params.inputDepth % inChansPerGroup != 0) {
    // TODO handle this case.
    std::abort();
  }
  std::vector<unsigned> partialChansPerGroupCandidates;
  if (machineInfo.fp16AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, false,
                                   params.stride, inChansPerGroup,
                                   machineInfo.fp16AccumConvUnitsPerTile,
                                   machineInfo)) {
    partialChansPerGroupCandidates.push_back(
      machineInfo.fp16AccumConvUnitsPerTile
    );
  }
  if (machineInfo.fp32AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, true,
                                   params.stride, inChansPerGroup,
                                   machineInfo.fp32AccumConvUnitsPerTile,
                                   machineInfo)) {
    partialChansPerGroupCandidates.push_back(
      machineInfo.fp32AccumConvUnitsPerTile
    );
  }
  partialChansPerGroupCandidates.push_back(1);
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
  const auto numTiles = engineBuilder.getNumTiles();
  for (const auto partialChansPerGroup : partialChansPerGroupCandidates) {
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
          auto maxVerticesPerTilePerY =
              (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
          auto minVerticesPerTilePerY = 1;
          if (partialChansPerGroup == machineInfo.fp16AccumConvUnitsPerTile ||
              partialChansPerGroup == machineInfo.fp32AccumConvUnitsPerTile) {
            if (machineInfo.sharedConvWeights) {
              // All workers are utilized in each single supervisor vertex so
              // there is no reason to use more than the minimum number of
              // vertices.
              maxVerticesPerTilePerY = 1;
            }
          } else {
            // The ConvPartial vertex that doesn't use the convolution
            // instruction always computes a single output row.
            minVerticesPerTilePerY = maxVerticesPerTilePerY;
          }
          for (unsigned verticesPerTilePerY = minVerticesPerTilePerY;
               verticesPerTilePerY <= maxVerticesPerTilePerY;
               ++verticesPerTilePerY) {
            bool floatPartials =
                !canUseConvolutionInstruction(
                  floatActivations, false, params.stride, inChansPerGroup,
                  partialChansPerGroup,
                  machineInfo
                );
            ConvLayerPartition candidate(tilesPerX, tilesPerY, tilesPerZ,
                                         verticesPerTilePerY, tilesPerInZ,
                                         inChansPerGroup, partialChansPerGroup,
                                         floatPartials);
            auto candidateCost =
                estimatePartitionCostBounded(engineBuilder, floatActivations,
                                             params, candidate, machineInfo,
                                             bestCost);
            if (candidateCost < bestCost) {
              bestPartition = candidate;
              bestCost = candidateCost;
            }
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
  resMethod(resMethod),
  reuseLayerImplGraphs(net.options.reuseLayerImplGraphs) {
  layerName = "Conv" + std::to_string(kernelSize) + "x" +
              std::to_string(kernelSize);
}

bool ConvLayerImpl::targetSharedConvWeights() const {
  return getNetOptions().ipuMachineInfo.sharedConvWeights;
}

std::uint64_t ConvLayerImpl::getNumberOfMACs() {
  std::uint64_t numMACs = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               padding, inDimY);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, inDimX);
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
  const auto &machineInfo = getNetOptions().ipuMachineInfo;
  if (getDType() == "float") {
    const auto floatVectorWidth = machineInfo.getFloatVectorWidth();
    auto macCycles =
       static_cast<double>(getNumberOfMACs()) / (floatVectorWidth * numTiles);
    auto addCycles =
       static_cast<double>(getNumberOfAdds()) / (floatVectorWidth * numTiles);
    return macCycles + addCycles;
  }
  assert(getDType() == "half");
  const auto convUnitsPerTile =
      std::max(machineInfo.fp16AccumConvUnitsPerTile,
               machineInfo.fp32AccumConvUnitsPerTile);
  const auto halfVectorWidth = machineInfo.getHalfVectorWidth();
  auto macsPerCycle =
      useConvolutionInstruction() ? convUnitsPerTile * halfVectorWidth :
                                    halfVectorWidth;
  auto macCycles = static_cast<double>(getNumberOfMACs()) /
                   (macsPerCycle * numTiles);

  auto addCycles = static_cast<double>(getNumberOfAdds()) /
                   (halfVectorWidth * numTiles);
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
  unsigned inChansPerGroup = zPrev;
  const bool floatActivations = getDType() == "float";
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  ConvolutionParams params(kernelSize, stride, zPrev, xPrev,
                           yPrev, padding, outNumChans);
  for (unsigned i = 1; i <= zPrev; ++i) {
    if (zPrev % i != 0)
      continue;
    if (!floatActivations && i % 2 != 0)
      continue;
    const auto candidate =
      choosePartition(getIPUModelEngineBuilder(), floatActivations, i, params,
                      getNetOptions().ipuMachineInfo);
    const auto candidateCost =
        estimatePartitionCost(getIPUModelEngineBuilder(), floatActivations,
                              params, candidate,
                              getNetOptions().ipuMachineInfo);
    if (candidateCost < bestCost) {
      inChansPerGroup = candidate.inChansPerGroup;
      bestCost = candidateCost;
    }
  }
  return zPrev / inChansPerGroup;
}

void ConvLayerImpl::
init(Graph &graph, std::mt19937 &randomEngine,
     IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dType = getDType();
  bool floatActivations = dType == "float";
  Layer *prev = getPrevLayer();
  Tensor prevOut = prev->getFwdActivations();
  inNumChanGroups = prevOut.dim(0);
  inDimY = prevOut.dim(1);
  inDimX = prevOut.dim(2);
  size_t inChansPerGroup = prevOut.dim(3);
  inNumChans = inChansPerGroup * inNumChanGroups;
  outDimX = (inDimX + (padding * 2) - kernelSize) / stride + 1;
  outDimY = (inDimY + (padding * 2) - kernelSize) / stride + 1;
  partition =
      choosePartition(getIPUModelEngineBuilder(), floatActivations,
                      inChansPerGroup,
                      ConvolutionParams(kernelSize, stride, inNumChans, inDimX,
                                        inDimY, padding, outNumChans),
                      getNetOptions().ipuMachineInfo);
  Layer *next = getNextLayer();
  outNumChanGroups = next->getNumChannelGroupsIn(inDimX, inDimY, outNumChans);
  size_t outChansPerGroup;
  if (outNumChanGroups) {
    outChansPerGroup = outNumChans / outNumChanGroups;
  } else {
    outChansPerGroup = floatActivations ? 1 : 2;
    outNumChanGroups = outNumChans / outChansPerGroup;
  }
  assert(outNumChanGroups * outChansPerGroup == outNumChans);
  // Each ConvComplete vertex writes outChansPerGroup output channels. Because
  // sub-word access is not atomic we must ensure output channels are grouped
  // in multiples of two.
  assert(floatActivations || outChansPerGroup % 2 == 0);
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  weights = graph.addTensor(dType, {partialNumChanGroups,
                                    inNumChanGroups,
                                    kernelSize,
                                    kernelSize,
                                    partialChansPerGroup,
                                    inChansPerGroup}, makeLayerName("weights"));
                              
  fwdActivations = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                           outChansPerGroup},
                                   makeLayerName("fwdActivations"));
  mapActivations(fwdActivations, mapping);
  const auto activationsMapping = computeActivationsMapping(fwdActivations);
  biases = graph.addTensor(dType, {outNumChans}, makeLayerName("biases"));
  mapBiases(biases, activationsMapping, mapping);
  fwdZ = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                 outChansPerGroup});
  mapActivations(fwdZ, mapping);
  if (getNetType() == TrainingNet) {
    zDeltas = graph.addTensor(dType, fwdZ.dims());
    deltas = graph.addTensor(dType, prev->getFwdActivations().dims());
  }

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

  // Initialize weights using "xavier" weight filler that scales
  // variance based on number of inputs to a neuron.
  hWeights = createRandomWeightInitializers(weights, 0, 1.0f / kernelSize,
                                            randomEngine);
  hBiases = createRandomWeightInitializers(biases, 0, 1.0f / kernelSize,
                                           randomEngine);

  auto implSpec =
    ConvImplSpec(inNumChans, inNumChanGroups,
                 inDimX, inDimY,
                 outNumChans, outNumChanGroups,
                 outDimX, outDimY,
                 resNumChans, resNumChanGroups,
                 resDimX, resDimY,
                 kernelSize, stride, padding);



  if (reuseLayerImplGraphs) {
    auto emplaceResult = implMap.emplace(implSpec, this);
    if (!emplaceResult.second) {
      // Matching implementation already exists
      reuseImpl = emplaceResult.first->second;
      return;
    }
  }

  in = graph.addTensor(dType, {prevOut.dim(0), prevOut.dim(1),
                               prevOut.dim(2), prevOut.dim(3)}, 
                       makeLayerName("in"));
  mapActivations(in, mapping);

  z = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                              outChansPerGroup}, makeLayerName("z"));
  activations = graph.addTensor(dType, {outNumChanGroups, outDimY, outDimX,
                                        outChansPerGroup}, 
                                makeLayerName("activations"));
  weightsIn = graph.addTensor(dType, {partialNumChanGroups,
                                      inNumChanGroups,
                                      kernelSize,
                                      kernelSize,
                                      partialChansPerGroup,
                                      inChansPerGroup}, 
                              makeLayerName("weightsIn"));
  biasesIn = graph.addTensor(dType, {outNumChans}, makeLayerName("biasesIn"));
  mapTensor(z, mapping);
  mapBiases(biasesIn, activationsMapping, mapping);
  if (resIndex) {
    resIn = graph.addTensor(dType, {resNumChanGroups,
                                    resDimY, resDimY,
                                    resChansPerGroup}, 
                            makeLayerName("resIn"));
    mapActivations(resIn, mapping);
  }
}

void ConvLayerImpl::
addResidualCalc(Graph &graph,
                ComputeSet cs,
                IPUModelEngineBuilder::TileMapping *mapping) {
  const auto dataPathWidth = getNetOptions().ipuMachineInfo.dataPathWidth;
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
                                          outChansPerGroup}, 
                             makeLayerName("residual"));
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
              auto v = graph.addVertex(cs, templateVertex("Zero", getDType()),
                                       {{"out",out}});
              graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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
            auto v = graph.addVertex(cs,
                                     templateVertex("CopyResidual", getDType()),
                                     {{"in", in}, {"out",out}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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
  const bool floatActivations = getDType() == "float";
  return canUseConvolutionInstruction(
    floatActivations, partition.floatPartials, stride,
    partition.inChansPerGroup, partition.partialChansPerGroup,
    getNetOptions().ipuMachineInfo
  );
}

void ConvLayerImpl::
createConvPartial1x1InOutVertex(Graph &graph,
                                IPUModelEngineBuilder::TileMapping *mapping,
                                unsigned tile,
                                unsigned outXBegin, unsigned outXEnd,
                                unsigned outYBegin, unsigned outYEnd,
                                unsigned outZGroup,
                                unsigned inZGroupBegin, unsigned inZGroupEnd,
                                const std::string &partialType,
                                ComputeSet fwdCS,
                                const Tensor &out) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = getNetOptions().ipuMachineInfo.dataPathWidth;
  const auto contextsPerVertex =
      targetSharedConvWeights() ? getWorkerContextsPerTile() : 1;
  const char *baseClass =
      targetSharedConvWeights() ? "poplar::SupervisorVertex" :
                                  "poplar::Vertex";

  // Add the vertex.
  auto v =
      graph.addVertex(fwdCS,
                      templateVertex("ConvPartial1x1InOut", baseClass,
                                     partialType));
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  if (mapping) {
    mapping->setMapping(v, tile);
  }
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  for (unsigned wy = 0; wy != kernelSize; ++wy) {
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange({outYBegin, outYEnd}, stride, kernelSize,
                       padding, inDimY, wy);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSize; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange({outXBegin, outXEnd}, stride, kernelSize,
                         padding, inDimX, wx);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;
      std::vector<std::vector<PartialRow>> workerPartition =
          partitionConvPartialByWorker(convOutHeight, convOutWidth,
                                       contextsPerVertex);
      assert(workerPartition.size() == contextsPerVertex);
      for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
        Tensor w =
            weightsIn[outZGroup][izg][wy][wx].flatten();
        graph.connect(v["weights"][numWeights], w);
        for (unsigned i = 0; i != contextsPerVertex; ++i) {
          graph.setInitialValue(
            v["weightReuseCount"][numWeights * contextsPerVertex + i],
            static_cast<std::uint32_t>(workerPartition[i].size())
          );
          for (const auto &partialRow : workerPartition[i]) {
            const auto workerOutY = convOutYBegin + partialRow.rowNumber;
            const auto workerOutXBegin = convOutXBegin + partialRow.begin;
            const auto workerOutXEnd = convOutXBegin + partialRow.end;
            const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
            const auto workerInY =
              getInputIndex(workerOutY, stride, kernelSize,
                            padding, inDimY, wy);
            assert(workerInY != ~0U);
            unsigned workerInXBegin, workerInXEnd;
            std::tie(workerInXBegin, workerInXEnd) =
                getInputRange({workerOutXBegin, workerOutXEnd}, stride,
                              kernelSize, padding, inDimX, wx);
            const auto workerInWidth = workerInXEnd - workerInXBegin;
            Tensor inWindow =
                in[izg][workerInY].slice(
                  {workerInXBegin, 0},
                  {workerInXEnd, inChansPerGroup}
                ).reshape({workerInWidth * inChansPerGroup});
            Tensor outWindow =
                out[outZGroup][workerOutY].slice(
                  {workerOutXBegin, 0},
                  {workerOutXEnd, outChansPerGroup}
                ).reshape({workerOutWidth * outChansPerGroup});
            if (mapping) {
              mapping->setMapping(outWindow, tile);
            }
            graph.connect(v["in"][numConvolutions], inWindow);
            graph.connect(v["out"][numConvolutions], outWindow);
            ++numConvolutions;
          }
        }
        ++numWeights;
      }
    }
  }
  graph.setFieldSize(v["in"], numConvolutions);
  graph.setFieldSize(v["out"], numConvolutions);
  graph.setFieldSize(v["weights"], numWeights);
  graph.setFieldSize(v["weightReuseCount"], numWeights * contextsPerVertex);
}


void ConvLayerImpl::
forwardTile(Graph &graph,
            IPUModelEngineBuilder::TileMapping *mapping,
            unsigned tile,
            unsigned tileOutXBegin, unsigned tileOutXEnd,
            unsigned tileOutYBegin, unsigned tileOutYEnd,
            unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
            unsigned tileInZGroupBegin, unsigned tileInZGroupEnd,
            ComputeSet zeroCS,
            ComputeSet fwdCS,
            const Tensor &out) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto dataPathWidth = getNetOptions().ipuMachineInfo.dataPathWidth;
  const auto tileOutHeight = tileOutYEnd - tileOutYBegin;
  const auto tileOutWidth = tileOutXEnd - tileOutXBegin;
  const auto verticesPerY = partition.verticesPerTilePerYAxis;

  if (useConvolutionInstruction() && kernelSize == 1) {
    const auto inZGroups = tileInZGroupEnd - tileInZGroupBegin;
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        const auto outHeight = outYEnd - outYBegin;
        if (outHeight == 0)
          continue;
        unsigned inYBegin, inYEnd, inXBegin, inXEnd;
        std::tie(inYBegin, inYEnd) =
            getInputRange({outYBegin, outYEnd}, stride, kernelSize,
                          padding, inDimY);
        std::tie(inXBegin, inXEnd) =
            getInputRange({tileOutXBegin, tileOutXEnd}, stride,
                          kernelSize, padding, inDimX);
        // Window into previous layer.
        const auto inWidth = inXEnd - inXBegin;
        const auto inHeight = inYEnd - inYBegin;
        Tensor inWindow =
            in.slice(
              {tileInZGroupBegin, inYBegin, inXBegin, 0},
              {tileInZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
            ).reshape({inHeight * inZGroups,
                       inWidth * inChansPerGroup});
        Tensor w =
            weightsIn[ozg].slice(
              {tileInZGroupBegin, 0, 0, 0, 0},
              {tileInZGroupEnd, 1, 1, outChansPerGroup, inChansPerGroup}
            ).flatten();
        Tensor outWindow =
            out[ozg].slice(
              {outYBegin, tileOutXBegin, 0},
              {outYEnd, tileOutXEnd, outChansPerGroup}
            ).reshape({outHeight, tileOutWidth * outChansPerGroup});
        if (stride == 1 && tileOutWidth == outDimX && inWidth == inDimX) {
          // If input rows are contiguous we can flatten the x and y dimensions,
          // reducing the number of in edge pointers.
          inWindow =
              inWindow.reshape({inZGroups, inHeight * inWidth *
                                inChansPerGroup});
          outWindow =
              outWindow.reshape({1, outHeight * tileOutWidth *
                                 outChansPerGroup});
        }
        // Add the vertex.
        const char *baseClass =
            targetSharedConvWeights() ? "poplar::SupervisorVertex" :
                                        "poplar::Vertex";
        auto v = graph.addVertex(
          fwdCS,
          templateVertex("ConvPartial1x1Out", baseClass, getPartialType()),
          {{"weights", w}, {"out", outWindow}}
        );
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
        graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
        if (stride == 1) {
          graph.connect(v["in"], inWindow);
        } else {
          for (unsigned i = 0; i != inZGroups; ++i) {
            for (unsigned j = 0; j != outHeight; ++j) {
              graph.connect(v["in"][j + i * outHeight],
                           inWindow[stride * j + i * inHeight]);
            }
          }
          graph.setFieldSize(v["in"], inZGroups * outHeight);
        }
        // Map the vertex and output.
        if (mapping) {
          mapping->setMapping(v, tile);
          mapping->setMapping(outWindow, tile);
        }
      }
    }
  } else if (useConvolutionInstruction()) {
    // Zero the partial sums.
    Tensor tileOut =
        out.slice(
          {tileOutZGroupBegin, tileOutYBegin, tileOutXBegin, 0},
          {tileOutZGroupEnd, tileOutYEnd, tileOutXEnd, outChansPerGroup}
        );
    const auto outZGroups = tileOutZGroupEnd - tileOutZGroupBegin;
    Tensor tileOutFlattened =
        tileOut.reshape({outZGroups * tileOutHeight,
                         tileOutWidth * outChansPerGroup});
    const auto workersPerTile = getWorkerContextsPerTile();
    const auto tileOutRows = tileOutFlattened.dim(0);
    const auto maxRowsPerWorker =
        (tileOutRows + workersPerTile - 1) / workersPerTile;
    // Choose the number of vertices such that each vertices is reponsible for
    // at most maxRowsPerWorker groups.
    const auto verticesToCreate =
        (tileOutRows + maxRowsPerWorker - 1) / maxRowsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto beginRow = (vertex * tileOutRows) / verticesToCreate;
      const auto endRow = ((vertex + 1) * tileOutRows) / verticesToCreate;
      if (beginRow == endRow)
        continue;
      auto zv = graph.addVertex(
        zeroCS, templateVertex("Zero2D", getPartialType()),
        {{"out", tileOutFlattened.slice(beginRow, endRow)}}
      );
      graph.setInitialValue(zv["dataPathWidth"], dataPathWidth);
      if (mapping) {
        mapping->setMapping(zv, tile);
      }
    }
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        const auto outHeight = outYEnd - outYBegin;
        if (outHeight == 0)
          continue;
        createConvPartial1x1InOutVertex(graph, mapping, tile,
                                        tileOutXBegin, tileOutXEnd,
                                        outYBegin, outYEnd,
                                        ozg,
                                        tileInZGroupBegin, tileInZGroupEnd,
                                        getPartialType(),
                                        fwdCS, out);
      }
    }
  } else {
    const auto inZGroups = tileInZGroupEnd - tileInZGroupBegin;
    for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
      assert(outChansPerGroup == 1);
      const auto z = ozg;
      for (unsigned vy = 0; vy != verticesPerY; ++vy) {
        const auto outYBegin =
            tileOutYBegin + (vy * tileOutHeight) / verticesPerY;
        const auto outYEnd =
            tileOutYBegin + ((vy + 1) * tileOutHeight) / verticesPerY;
        if (outYBegin == outYEnd)
          continue;
        assert(outYEnd - outYBegin == 1);
        const auto y = outYBegin;
        unsigned inYBegin, inYEnd, inXBegin, inXEnd;
        std::tie(inYBegin, inYEnd) =
          getInputRange(y, stride, kernelSize, padding, inDimY);
        std::tie(inXBegin, inXEnd) =
            getInputRange({tileOutXBegin, tileOutXEnd}, stride, kernelSize,
                          padding, inDimX);
        // Window into previous layer.
        const auto inWidth = inXEnd - inXBegin;
        const auto inHeight = inYEnd - inYBegin;
        // Weights that match the window.
        unsigned weightYBegin, weightYEnd;
        std::tie(weightYBegin, weightYEnd) =
          getKernelRange(y, stride, kernelSize, padding, inDimY);
        Tensor inWindow =
            in.slice(
              {tileInZGroupBegin, inYBegin, inXBegin, 0},
              {tileInZGroupEnd, inYEnd, inXEnd, inChansPerGroup}
            ).reshape({inHeight * inZGroups,
                       inWidth * inChansPerGroup});
        Tensor w =
            weightsIn[z].slice(
              {tileInZGroupBegin, weightYBegin, 0, 0, 0},
              {tileInZGroupEnd, weightYEnd, kernelSize, 1, inChansPerGroup}
            ).reshape({inHeight * inZGroups,
                       inChansPerGroup * kernelSize});
        Tensor outWindow = out[z][y].slice(tileOutXBegin, tileOutXEnd).flatten();
        // Add the vertex.
        auto v = graph.addVertex(fwdCS,
                                 templateVertex("ConvPartial", getDType(),
                                                getPartialType()),
            { {"in", inWindow },
              {"weights", w },
              {"out", outWindow },
            });
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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

void ConvLayerImpl::mapBiases(Tensor b,
                              const std::vector<unsigned> &activationsMapping,
                              IPUModelEngineBuilder::TileMapping *mapping) {
  if (!mapping)
    return;
  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  size_t outChansPerGroup = outNumChans / outNumChanGroups;
  Tensor biasesByChanGroup =
      b.reshape({outNumChanGroups, outChansPerGroup});
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = activationsMapping[tile];
    const auto tileActivationsEnd = activationsMapping[tile + 1];
    assert(tileActivationsBegin % outChansPerGroup == 0);
    assert(tileActivationsEnd % outChansPerGroup == 0);
    const auto tileGroupBegin = tileActivationsBegin / outChansPerGroup;
    const auto tileGroupEnd = tileActivationsEnd / outChansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    const auto minOutChanGroup = tileGroupBegin / (outDimX * outDimY);
    const auto maxOutChanGroup = (tileGroupEnd - 1) / (outDimX * outDimY);
    Tensor biasSlice = biasesByChanGroup.slice(minOutChanGroup,
                                               maxOutChanGroup + 1);
    mapping->setMapping(biasSlice, tile);
  }
}

void ConvLayerImpl::mapWeights(Graph &graph,
                               IPUModelEngineBuilder::TileMapping *mapping,
                               Tensor w) {
  const auto isMultiIPU = getNumIPUs() > 1;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto numInZGroups = inNumChans / inChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;

  if (mapping) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      const auto numInZGroups = inZGroupEnd - inZGroupBegin;
      for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
        const auto outZGroupBegin =
            (ozg * partialNumChanGroups) / tilesPerZ;
        const auto outZGroupEnd =
            ((ozg + 1) * partialNumChanGroups) / tilesPerZ;
        const auto numOutZGroups = outZGroupEnd - outZGroupBegin;
        // Group weights that are accessed contiguously by tiles within this
        // loop body.
        Tensor sharedWeights;
        if (useConvolutionInstruction()) {
          if (kernelSize == 1) {
            sharedWeights =
                w.slice(
                  {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
                  {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
                   partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups,
                           numInZGroups * partialChansPerGroup *
                           inChansPerGroup});
          } else {
            sharedWeights =
                w.slice(
                  {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
                  {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
                   partialChansPerGroup, inChansPerGroup}
                ).reshape({numOutZGroups * numInZGroups * kernelSize *
                           kernelSize,
                           partialChansPerGroup * inChansPerGroup});
          }
        } else {
          sharedWeights =
              w.slice(
                {outZGroupBegin, inZGroupBegin, 0, 0, 0, 0},
                {outZGroupEnd, inZGroupEnd, kernelSize, kernelSize,
                 1, inChansPerGroup}
              ).reshape({numInZGroups * numOutZGroups * kernelSize,
                         kernelSize * inChansPerGroup});
        }
        const auto numSharedWeightGroups = sharedWeights.dim(0);
        // Spread groups of weights equally across the tiles that read them.
        for (unsigned oy = 0; oy != tilesPerY; ++oy) {
          for (unsigned ox = 0; ox != tilesPerX; ++ox) {
            const auto iw = ox + tilesPerX * oy;
            const auto sharedWeightGroupBegin =
                (iw * numSharedWeightGroups) / (tilesPerY * tilesPerX);
            const auto sharedWeightGroupEnd =
                ((iw + 1) * numSharedWeightGroups) / (tilesPerY * tilesPerX);
            const auto tileWeights =
                sharedWeights.slice(sharedWeightGroupBegin,
                                    sharedWeightGroupEnd);
            const auto tile = linearizeTileIndices(izg, ox, oy, ozg, partition,
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

  const auto dataPathWidth = getNetOptions().ipuMachineInfo.dataPathWidth;
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
  Tensor partials = graph.addTensor(getPartialType(),
                                    {tilesPerInZGroup,
                                     partialNumChanGroups,
                                     outDimY,
                                     outDimX,
                                     partialChansPerGroup}, 
                                    makeLayerName("partials"));
  ComputeSet zeroCS;
  if (useConvolutionInstruction() && kernelSize != 1) {
    zeroCS = graph.createComputeSet(layerName + ".zero");
    forwardProg.add(Execute(zeroCS));
  }
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
                      zeroCS, fwdCS, partials[izg]);
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
  auto activationsMapping = computeActivationsMapping(activations);
  if (tilesPerInZGroup == 1) {
    reduced = partials[0];
  } else {
    // Accumulate the partial sums.
    const auto numTiles = getNumIPUs() * getTilesPerIPU();
    reduced = graph.addTensor(getPartialType(),
                              {partialNumChanGroups, outDimY, outDimX,
                               partialChansPerGroup}, 
                              makeLayerName("reduced"));
    size_t outChansPerGroup = outNumChans / outNumChanGroups;
    if (outChansPerGroup % partialChansPerGroup == 0) {
      const auto partialGroupsPerOutGroup =
          outChansPerGroup / partialChansPerGroup;
      for (unsigned tile = 0; tile != numTiles; ++tile) {
        const auto activationsBegin = activationsMapping[tile];
        const auto activationsEnd = activationsMapping[tile + 1];
        assert(activationsBegin % outChansPerGroup == 0);
        assert(activationsEnd % outChansPerGroup == 0);
        const auto groupBegin = activationsBegin / outChansPerGroup;
        const auto groupEnd = activationsEnd / outChansPerGroup;
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
          for (unsigned i = 0; i != partialGroupsPerOutGroup; ++i) {
            const auto zg = outChanGroup * partialGroupsPerOutGroup + i;
            Tensor in =
                partials.slice({0, zg, y, xBegin, 0},
                               {tilesPerInZGroup, zg + 1, y + 1, xEnd,
                                partialChansPerGroup}
                ).reshape({tilesPerInZGroup,
                           (xEnd - xBegin) * partialChansPerGroup});
            Tensor out = reduced[zg][y].slice(xBegin, xEnd).flatten();
            const auto v =
                graph.addVertex(reduceCS, templateVertex("ConvReduce",
                                                         getPartialType()),
                                {{"out", out}, {"partials", in}});
            graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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
          const auto v =
              graph.addVertex(reduceCS, templateVertex("ConvReduce",
                                                       getPartialType()),
                              {{"out", out}, {"partials", in}});
          graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
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

  const auto numTiles = getNumIPUs() * getTilesPerIPU();
  const auto workersPerTile = getWorkerContextsPerTile();
  const auto partialChanChunkSize =
      gcd<unsigned>(outChansPerGroup, partialChansPerGroup);
  const auto resOutChanGroups = resLayer ? residual.dim(0) : 0;

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileActivationsBegin = activationsMapping[tile];
    const auto tileActivationsEnd = activationsMapping[tile + 1];
    assert(tileActivationsBegin % outChansPerGroup == 0);
    assert(tileActivationsEnd % outChansPerGroup == 0);
    const auto tileGroupBegin = tileActivationsBegin / outChansPerGroup;
    const auto tileGroupEnd = tileActivationsEnd / outChansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    const auto maxGroupsPerWorker =
        (tileNumGroups + workersPerTile - 1) / workersPerTile;
    // Choose the number of vertices such that each vertices is reponsible for
    // at most maxGroupsPerWorker groups.
    const auto verticesToCreate =
        (tileNumGroups + maxGroupsPerWorker - 1) / maxGroupsPerWorker;
    for (unsigned vertex = 0; vertex != verticesToCreate; ++vertex) {
      const auto groupBegin =
          (vertex * tileNumGroups) / verticesToCreate + tileGroupBegin;
      const auto groupEnd =
          ((vertex + 1) * tileNumGroups) / verticesToCreate + tileGroupBegin;
      if (groupBegin == groupEnd)
        continue;
      // Create a vertex for this worker to process a number of output channel
      // groups.
      const auto numGroups = groupEnd - groupBegin;
      auto v = graph.addVertex(completionCS,
                               templateVertex("ConvComplete", getPartialType(),
                                              getDType()));
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      if (mapping)
        mapping->setMapping(v, tile);

      // Add the biases and a vector that tells the vertex how many output
      // groups to process for each bias.
      auto minOutChanGroup = groupBegin / (outDimX * outDimY);
      auto maxOutChanGroup = (groupEnd - 1) / (outDimX * outDimY);
      Tensor biasSlice = biasesByChanGroup.slice(minOutChanGroup,
                                                 maxOutChanGroup + 1);
      graph.connect(v["bias"], biasSlice);
      graph.setFieldSize(v["outputChanGroupsPerBias"],
                         maxOutChanGroup - minOutChanGroup + 1);
      for (auto outChanGroup = minOutChanGroup;
           outChanGroup <= maxOutChanGroup;
           ++outChanGroup) {
        auto gBegin = std::max(groupBegin, outChanGroup * outDimY * outDimX);
        auto gEnd = std::min(groupEnd, (outChanGroup+1) * outDimY * outDimX);
        unsigned outputsPerBias = gEnd - gBegin;
        auto i = outChanGroup - minOutChanGroup;
        graph.setInitialValue(v["outputChanGroupsPerBias"][i],
                              outputsPerBias);
      }

      // Connect the output channel groups and inputs from the partial sums.
      graph.setFieldSize(v["out"], numGroups);
      graph.setFieldSize(v["z"], numGroups);
      graph.setFieldSize(v["in"],
                         numGroups * outChansPerGroup / partialChanChunkSize);
      unsigned numIn = 0;
      unsigned numResUsed = 0;
      for (auto group = groupBegin; group != groupEnd; ++group) {
        auto outChanGroup = group / (outDimX * outDimY);
        auto y = group % (outDimX * outDimY) / outDimX;
        auto x = group % outDimX;
        auto out = activations[outChanGroup][y][x];
        auto zz = z[outChanGroup][y][x];
        graph.connect(v["out"][group - groupBegin], out);
        graph.connect(v["z"][group - groupBegin], zz);
        Tensor reducedChans = reduced.slice(
           {0, y, x, 0},
           {partialNumChanGroups, y + 1, x + 1, partialChansPerGroup}
        ).flatten();
        Tensor reducedByChanGroup =
            reducedChans.reshape({outNumChanGroups,
                                  outChansPerGroup / partialChanChunkSize,
                                  partialChanChunkSize});
        Tensor in = reducedByChanGroup[outChanGroup];
        for (unsigned i = 0; i < in.dim(0); ++i) {
          graph.connect(in[i], v["in"][numIn++]);
        }
        if (resLayer && outChanGroup < resOutChanGroups) {
          // If the residual is taken directly from the previous layer (
          // as opposed to being zero-padded or converted), then striding over
          // the X,Y plane may still be needed (in this case resStride will not
          // be 1).
          Tensor res = residual[outChanGroup][y * resStrideY][x * resStrideX];
          graph.connect(res, v["res"][numResUsed++]);
        }
      }
      graph.setFieldSize(v["res"], numResUsed);
    }
  }
  mapActivations(activations, mapping);
  mapActivations(z, mapping);
  forwardProg.add(Execute(completionCS));
  createdForwardProg = true;
}

Program ConvLayerImpl::
forward(Graph &graph, IPUModelEngineBuilder::TileMapping *mapping)  {
  if (kernelSize > inDimX || kernelSize > inDimY) {
    // We don't support kernelsize greater than the x/y dimensions.
    std::abort();
  }
  Layer *prev = getPrevLayer();
  auto impl = reuseImpl ? reuseImpl : this;
  auto prog = Sequence();
  prog.add(Copy(impl->getInputTensor(), prev->getFwdActivations()));
  impl->mapWeights(graph, mapping, weights);
  prog.add(Copy(impl->getInputWeights(), weights));
  prog.add(Copy(impl->getInputBiases(), biases));
  if (resLayer) {
    prog.add(Copy(impl->getInputResidual(),
                  resLayer->getFwdActivations()));
  }
  prog.add(impl->getOrCreateFwdProg(graph, mapping));
  prog.add(Copy(fwdActivations, impl->getOutputTensor()));
  prog.add(Copy(fwdZ, impl->getOutputZ()));
  return prog;
}

Program ConvLayerImpl::
backward(Graph &graph) {
  auto bwdNonLinearityCS =
      graph.createComputeSet(layerName + ".bwd.nonLinearity");
  auto deltasIn = getNextLayer()->getBwdDeltas();
  const auto partialChansPerGroup = partition.partialChansPerGroup;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = outNumChans / outNumChanGroups;
  assert(deltasIn.dim(0) == outNumChanGroups);
  assert(deltasIn.dim(1) == outDimY);
  assert(deltasIn.dim(2) == outDimX);
  assert(deltasIn.dim(3) == outChansPerGroup);

  auto v = graph.addVertex(bwdNonLinearityCS,
                           templateVertex("NonLinearityBwd",
                                          getDType()),
                           {{"deltasIn", deltasIn.flatten()},
                            {"z", z.flatten()},
                            {"deltasOut", zDeltas.flatten()},
                           });
  graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
  auto partials = graph.addTensor(getPartialType(),
                                  {outNumChanGroups,
                                   inNumChans,
                                   kernelSize,
                                   kernelSize,
                                   inDimY, inDimX});
  auto zeroCS = graph.createComputeSet(layerName + ".bwd.zero");
  graph.addVertex(zeroCS, templateVertex("Zero", getPartialType()),
                  {{"out",partials.flatten()}});
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  for (unsigned outGroup = 0; outGroup < outNumChanGroups; ++outGroup) {
    for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
      for (unsigned wy = 0; wy < kernelSize; ++wy) {
        for (unsigned wx = 0; wx < kernelSize; ++wx) {
          unsigned convOutYBegin, convOutYEnd;
          std::tie(convOutYBegin, convOutYEnd) =
              getOutputRange({0, outDimY}, stride, kernelSize,
                             padding, inDimY, wy);
          const auto convOutHeight = convOutYEnd - convOutYBegin;
          if (convOutHeight == 0) {
            std::abort();
            continue;
          }
          unsigned convOutXBegin, convOutXEnd;
          std::tie(convOutXBegin, convOutXEnd) =
              getOutputRange({0, outDimX}, stride, kernelSize,
                             padding, inDimX, wx);
          const auto convOutWidth = convOutXEnd - convOutXBegin;
          if (convOutWidth == 0)
            continue;
          unsigned convInYBegin, convInYEnd;
          std::tie(convInYBegin, convInYEnd) =
              getInputRange({0, outDimY}, stride, kernelSize,
                            padding, inDimY,
                            wy);
          const auto convInHeight = convInYEnd - convInYBegin;
          if (convInHeight == 0)
            continue;
          unsigned convInXBegin, convInXEnd;
          std::tie(convInXBegin, convInXEnd) =
              getInputRange({0, outDimX}, stride, kernelSize, padding,
                                inDimX, wx);
          const auto convInWidth = convInXEnd - convInXBegin;
          if (convInWidth == 0)
            continue;
          auto out =
              partials[outGroup][inChan][wy][wx]
                  .slice({convInYBegin, convInXBegin},
                         {convInYEnd, convInXEnd})
                  .reshape({convInHeight, convInWidth});
          auto in = zDeltas[outGroup]
                         .slice({convOutYBegin, convOutXBegin, 0},
                                {convOutYEnd, convOutXEnd, outChansPerGroup})
                         .reshape({convOutHeight, convOutWidth * outChansPerGroup});
          auto zz = fwdZ[outGroup]
                         .slice({convOutYBegin, convOutXBegin, 0},
                                {convOutYEnd, convOutXEnd, outChansPerGroup})
                         .reshape({convOutHeight, convOutWidth * outChansPerGroup});
          auto v = graph.addVertex(bwdCS,
                                   templateVertex("ConvBwd",
                                                  getDType(),
                                                  getPartialType()),
                                   {{"in", in},
                                    {"out", out}});
          graph.setFieldSize(v["weights"], outChansPerGroup);
          for (unsigned i = 0; i < outChansPerGroup; ++i) {
            auto outChan = outGroup * outChansPerGroup + i;
            Tensor w;
            w = weights[outChan / partialChansPerGroup]
                       [inChan / inChansPerGroup]
                       [wy]
                       [wx]
                       [outChan % partialChansPerGroup]
                       [inChan % inChansPerGroup];
            graph.connect(v["weights"][i], w);
          }
        }
      }
    }
  }
  auto reduced = graph.addTensor(getPartialType(),
                                 {inNumChans, inDimY, inDimX});
  auto reduceCS = graph.createComputeSet(layerName + ".bwd.reduce");
  for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
    auto p = partials.slice({0, inChan, 0, 0, 0, 0},
                            {outNumChanGroups, inChan + 1, kernelSize, kernelSize, inDimX, inDimY})
                     .reshape({outNumChanGroups * kernelSize * kernelSize,
                               inDimX * inDimY});
    graph.addVertex(reduceCS, templateVertex("ConvReduce", getPartialType()),
                    {{"out", reduced[inChan].flatten()},
                     {"partials", p}});
  }

  auto completeCS = graph.createComputeSet(layerName + ".bwd.complete");
  for (unsigned inChanGroup = 0; inChanGroup < inNumChanGroups; ++inChanGroup) {
    for (unsigned y = 0; y < inDimY; ++y) {
      for (unsigned x = 0; x < inDimX; ++x) {
        auto inChanBegin = inChanGroup * inChansPerGroup;
        auto inChanEnd = (inChanGroup + 1) * inChansPerGroup;
        auto in = reduced.slice({inChanBegin, y, x},
                                {inChanEnd, y+1, x+1})
                         .flatten();
        graph.addVertex(completeCS,
                        templateVertex("ConvCompleteBwd",
                                       getPartialType(), getDType()),
                        {{"out", deltas[inChanGroup][y][x].flatten()},
                         {"in", in}});
      }
    }
  }

  return Sequence(Execute(bwdNonLinearityCS), Execute(zeroCS), Execute(bwdCS),
                  Execute(reduceCS), Execute(completeCS));
}

Program ConvLayerImpl::
weightUpdate(Graph &graph) {
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = outNumChans / outNumChanGroups;
  const auto partialChansPerGroup = partition.partialChansPerGroup;

  auto deltasIn = getNextLayer()->getBwdDeltas();
  auto wPartials = graph.addTensor(getDType(),
                                   {outNumChans, outDimY, outDimX,
                                    kernelSize, kernelSize, inNumChans});
  auto zeroCS = graph.createComputeSet(layerName + ".weight_update.zero");
  graph.addVertex(zeroCS, templateVertex("Zero", getDType()),
                  {{"out",wPartials.flatten()}});
  auto partialCS = graph.createComputeSet(layerName + ".weight_update.partial");
  Layer *prev = getPrevLayer();
  auto act = prev->getFwdActivations();
  for (unsigned outChanGroup = 0; outChanGroup < outNumChanGroups; ++outChanGroup) {
    for (unsigned y = 0; y < outDimY; ++y) {
      for (unsigned x = 0; x < outDimX; ++x) {
        for (unsigned outChanInGroup = 0; outChanInGroup < outChansPerGroup; ++outChanInGroup) {
          for (unsigned wy = 0; wy < kernelSize; ++wy) {
            for (unsigned wx = 0; wx < kernelSize; ++wx) {
              auto inX = getInputIndex(x, stride, kernelSize,
                                       padding, inDimX, wx);
              if (inX == ~0U)
                continue;
              auto inY = getInputIndex(y, stride, kernelSize,
                                       padding, inDimY, wy);
              if (inY == ~0U)
                continue;
              auto outChan = outChanGroup * outChansPerGroup + outChanInGroup;
              auto w = wPartials[outChan][y][x][wy][wx].flatten();
              auto d = zDeltas[outChanGroup][y][x][outChanInGroup];
              auto ii = act.slice({0, inY, inX, 0},
                                  {inNumChanGroups, inY + 1, inX + 1, inChansPerGroup}).flatten();
              auto v = graph.addVertex(partialCS,
                                       templateVertex("ConvPartialWeightUpdate", getDType()),
                                       {{"d", d},
                                        {"in", ii},
                                        {"weightUpdates", w}});
            }
          }
        }
      }
    }
  }
  auto reduceCS = graph.createComputeSet(layerName + ".weight_update.reduce");

  for (unsigned inChan = 0; inChan < inNumChans; ++inChan) {
    for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
      for (unsigned wy = 0; wy < kernelSize; ++wy) {
        for (unsigned wx = 0; wx < kernelSize; ++wx) {
          auto w = weights[outChan / partialChansPerGroup]
                          [inChan / inChansPerGroup]
                          [wy][wx]
                          [outChan % partialChansPerGroup]
                          [inChan % inChansPerGroup];
          auto in =
              wPartials[outChan].slice({0, 0, wy, wx, inChan},
                                       {outDimY, outDimX,
                                        wy + 1, wx + 1,
                                        inChan + 1})
                                .flatten();
          auto v = graph.addVertex(reduceCS,
                                   templateVertex("ConvWeightUpdateReduce", getDType()),
                                   {{"weight", w}, {"partials", in}});
          graph.setInitialValue(v["eta"],
                                getLearningRate());
        }
      }
    }
  }

  for (unsigned outChan = 0; outChan < outNumChans; ++outChan) {
    const auto outChanGroup = outChan / outChansPerGroup;
    const auto outChanInGroup = outChan % outChansPerGroup;
    auto in = zDeltas.slice({outChanGroup, 0, 0, outChanInGroup},
                            {outChanGroup + 1, outDimY, outDimX,
                             outChanInGroup + 1}).flatten();
    auto v = graph.addVertex(reduceCS,
                             templateVertex("ConvBiasUpdate", getDType()),
                             {{"bias", biases[outChan]}, {"deltas", in}});
    graph.setInitialValue(v["eta"],
                          getLearningRate());
  }


  return Sequence(Execute(zeroCS), Execute(partialCS), Execute(reduceCS));
}

