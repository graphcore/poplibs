#include "popnn/ConvPlan.hpp"

#include "popnn/Convolution.hpp"
#include "poplar/Graph.hpp"
#include "ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "PerformanceEstimation.hpp"


static unsigned getNumConvUnits(bool floatPartial,
                                const poplar::DeviceInfo &deviceInfo) {
  return floatPartial ? deviceInfo.fp32AccumConvUnitsPerTile :
                        deviceInfo.fp16AccumConvUnitsPerTile;
}

namespace {
struct ConvVertexType {
  bool useConvInstruction;
  bool floatPartials;
  unsigned partialChansPerGroup;
  ConvVertexType(const poplar::DeviceInfo &deviceInfo,
                 bool useConvInstruction, bool floatPartials) :
    useConvInstruction(useConvInstruction), floatPartials(floatPartials) {
    if (!useConvInstruction) {
      partialChansPerGroup = 1;
    } else {
      partialChansPerGroup = getNumConvUnits(floatPartials, deviceInfo);
    }
  }
  ConvVertexType(bool useConvInstruction, bool floatPartials,
                 unsigned partialChansPerGroup) :
    useConvInstruction(useConvInstruction), floatPartials(floatPartials),
    partialChansPerGroup(partialChansPerGroup) {}
};
}

namespace conv {

enum class Phase {
  FORWARD,
  BACKWARD,
  WEIGHTUPDATE
};

struct ConvolutionParams {
  unsigned kernelSize;
  unsigned stride;
  unsigned inputDepth;
  unsigned inputWidth;
  unsigned inputHeight;
  unsigned padding;
  unsigned outputDepth;
  unsigned batchSize;
  unsigned getOutputWidth(Phase phase) const {
    if (phase == Phase::BACKWARD)
      return (inputWidth * stride + kernelSize - 1) - (padding * 2);
    else
      return inputWidth + padding * 2 < kernelSize
        ? 0
        : (inputWidth + (padding * 2) - kernelSize) / stride + 1;
  }
  unsigned getOutputHeight(Phase phase) const {
    if (phase == Phase::BACKWARD)
      return (inputHeight * stride + kernelSize - 1) - (padding * 2);
    else
      return inputHeight + padding * 2 < kernelSize
        ? 0
        : (inputHeight + (padding * 2) - kernelSize) / stride + 1;
  }
  ConvolutionParams(unsigned kernelSize,
                    unsigned stride,
                    unsigned inputDepth,
                    unsigned inputWidth,
                    unsigned inputHeight,
                    unsigned padding,
                    unsigned outputDepth,
                    unsigned batchSize) :
    kernelSize(kernelSize),
    stride(stride),
    inputDepth(inputDepth),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    padding(padding),
    outputDepth(outputDepth),
    batchSize(batchSize) {}
};

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned stride,
                             unsigned inChansPerGroup,
                             const poplar::DeviceInfo &deviceInfo) {
  if (floatActivations) {
    if (!deviceInfo.convInstructionsFloat) {
      return false;
    }
    if (!floatPartials) {
      return false;
    }
  }
  if (deviceInfo.getWeightsPerConvUnit(floatActivations) %
      inChansPerGroup != 0) {
    return false;
  }
  // Check we can use aligned loads.
  if ((inChansPerGroup * (floatActivations ? 32 : 16)) %
      deviceInfo.dataPathWidth != 0) {
    return false;
  }
  if (stride >= (1 << 4))
    return false;
  return true;
}

static unsigned
getMaxInputRangeSize(unsigned outputRangeSize, unsigned stride,
                     unsigned kernelSize, unsigned padding,
                     unsigned numPartitions,
                     unsigned inputSize, bool contiguousAccess,
                     Phase phase) {
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
                                      inputSize, phase != Phase::BACKWARD);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    if (phase != Phase::BACKWARD) {
      inputRangeSize = (outputRangeSize - 1) * stride + 1 + (kernelSize - 1);
    } else {
      inputRangeSize = (outputRangeSize - kernelSize ) / stride + 1;
    }
    break;
  }
  if (phase == Phase::FORWARD &&
      !contiguousAccess && kernelSize == 1 && stride > 1) {
    inputRangeSize = (inputRangeSize - 1) / stride + 1;
  }
  return inputRangeSize;
}

static std::uint64_t
getConvPartialnx1CycleEstimate(unsigned passesPerOutput,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               bool useSupervisorVertices,
                               unsigned outputStride,
                               unsigned numInputPointers)
{
  if (useSupervisorVertices) {
    std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
    for (unsigned i = 0; i != passesPerOutput; ++i) {
      const auto numWorkerContexts = 6;
      std::vector<std::vector<PartialRow>> partition =
          partitionConvPartialByWorker(outputHeight, outputWidth,
                                       numWorkerContexts, outputStride);
      convSizesByWeightAndWorker.emplace_back();
      convSizesByWeightAndWorker.back().reserve(partition.size());
      for (const auto &entry : partition) {
        convSizesByWeightAndWorker.back().emplace_back();
        convSizesByWeightAndWorker.back().back().reserve(entry.size());
        for (const auto &partialRow : entry) {
          auto convSize = (partialRow.end - partialRow.begin) / outputStride;
          convSizesByWeightAndWorker.back().back().push_back(convSize);
        }
      }
    }
    return getConvPartialnx1SupervisorCycleEstimate(convSizesByWeightAndWorker,
                                                    convUnitPipelineDepth,
                                                    numConvUnitsPerTile,
                                                    numInputPointers);
  }
  std::vector<std::vector<unsigned>> convSizesByWeight;
  for (unsigned i = 0; i != passesPerOutput; ++i) {
    convSizesByWeight.emplace_back();
    for (unsigned j = 0; j != outputHeight; ++j) {
      convSizesByWeight.back().push_back(outputWidth);
    }
  }
  return getConvPartialnx1CycleWorkerEstimate(convSizesByWeight,
                                              convUnitPipelineDepth,
                                              numConvUnitsPerTile,
                                              numInputPointers);
}

static unsigned
estimateExchangeCost(const poplar::DeviceInfo &deviceInfo,
                     bool floatActivations, const ConvolutionParams &params,
                     const Partition &partition,
                     Phase phase) {
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroupAxis = partition.tilesPerInZGroupAxis;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto partialChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutWidth =
      (params.getOutputWidth(phase) + tilesPerX - 1) / tilesPerX;
  const auto tileOutHeight =
      (params.getOutputHeight(phase) + tilesPerY - 1) / tilesPerY;
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
                           tilesPerX, params.inputWidth, true, phase);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.stride, params.kernelSize,
                           params.padding,
                           tilesPerY, params.inputWidth, false, phase);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      params.kernelSize * params.kernelSize * tileOutDepth * tileInDepth;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileOutDepth;

  const auto activationSize = floatActivations ? 4 : 2;
  auto inputElementsBytes = numberOfInputElements * activationSize;
  auto weightBytes = numberOfWeights * activationSize;
  unsigned partialSumBytes;
  const auto partialSize = partition.floatPartials ? 4 : 2;
  if (phase == Phase::WEIGHTUPDATE) {
    const auto deltaElementBytes = numberOfOutputElements * activationSize;
    inputElementsBytes += deltaElementBytes;
    partialSumBytes = numberOfWeights * partialSize;
    weightBytes = 0;
  } else {
    const auto numberOfPartialSums = numberOfOutputElements;
    partialSumBytes = numberOfPartialSums * partialSize;
  }

  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  const auto numCycles =
      (inputElementsBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (weightBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return numCycles;
}

static unsigned
estimateVertexCycles(bool floatActivations,
                     const ConvolutionParams &params,
                     const Partition &partition,
                     const poplar::DeviceInfo &deviceInfo,
                     bool useSupervisorVertices,
                     Phase phase) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerX = partition.tilesPerXAxis;
  const auto tilesPerInZGroupAxis = partition.tilesPerInZGroupAxis;
  const auto verticesPerTilePerY = partition.verticesPerTilePerYAxis;
  const auto inChansPerGroup = partition.inChansPerGroup;
  const auto outChansPerGroup = partition.partialChansPerGroup;

  const auto tileOutHeight =
      (params.getOutputHeight(phase) + tilesPerY - 1) / tilesPerY;
  const auto tileOutWidth =
      (params.getOutputWidth(phase) + tilesPerX - 1) / tilesPerX;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto outRowsPerVertex =
      (tileOutHeight + verticesPerTilePerY - 1) / verticesPerTilePerY;
  const auto outputStride = phase != Phase::BACKWARD ? 1 : params.stride;
  const auto inputStride = phase != Phase::BACKWARD ? params.stride : 1;
  if (phase == Phase::WEIGHTUPDATE) {
    auto vectorWidth = deviceInfo.dataPathWidth / (floatActivations ? 32 : 16);
    return getWeightGradCalcCycles(tileOutHeight, tileOutHeight * inputStride,
                                   tileOutWidth, tileOutWidth * inputStride,
                                   outChansPerGroup, inChansPerGroup,
                                   inputStride, params.kernelSize,
                                   0, 0, vectorWidth);
  }
  if (partition.useConvolutionInstructions) {
    assert(deviceInfo.getWeightsPerConvUnit(floatActivations) %
           inChansPerGroup == 0);
    const auto convUnitWeightHeight =
        deviceInfo.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
    const auto passesPerFilter =
        params.kernelSize *
        (params.kernelSize + convUnitWeightHeight - 1) / convUnitWeightHeight;
    const auto passesPerOutput = passesPerFilter * tileNumInGroups;
    return getConvPartialnx1CycleEstimate(
          passesPerOutput, outRowsPerVertex,
          tileOutWidth, deviceInfo.convUnitPipelineDepth,
          getNumConvUnits(partition.floatPartials, deviceInfo),
          useSupervisorVertices,
          outputStride,
          convUnitWeightHeight);
  }
  assert(!useSupervisorVertices);
  return outRowsPerVertex * outChansPerGroup *
         getConvPartialByDotProductCycleEstimate(
           floatActivations, inChansPerGroup, params.kernelSize,
           params.kernelSize * tileNumInGroups, tileOutWidth,
           deviceInfo.dataPathWidth, outputStride
         );
}

static unsigned
estimatePartialCalcComputeCost(const poplar::DeviceInfo &deviceInfo,
                               bool floatActivations,
                               const ConvolutionParams &params,
                               const Partition &partition,
                               Phase phase) {
  const auto tilesPerY = partition.tilesPerYAxis;
  const auto tilesPerZ = partition.tilesPerZAxis;
  const auto tilesPerInZGroup = partition.tilesPerInZGroupAxis;
  const auto outChansPerGroup = partition.partialChansPerGroup;
  const auto inChansPerGroup = partition.inChansPerGroup;

  const auto tileOutHeight =
      (params.getOutputHeight(phase) + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroup - 1) / tilesPerInZGroup;

  const auto verticesPerTilePerY =
      std::min(tileOutHeight, partition.verticesPerTilePerYAxis);
  const auto tileVertices =
      phase == Phase::WEIGHTUPDATE ? tileNumOutGroups * tileNumInGroups
                                   :  verticesPerTilePerY * tileNumOutGroups;
  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  bool useSupervisorVertices = false;
  unsigned numContexts = deviceInfo.numWorkerContexts;
  if (deviceInfo.sharedConvWeights &&
      partition.useConvolutionInstructions) {
    useSupervisorVertices = true;
    numContexts = 1;
  }
  const auto vertexRuntime = estimateVertexCycles(floatActivations, params,
                                                  partition,
                                                  deviceInfo,
                                                  useSupervisorVertices,
                                                  phase);
  auto verticesPerWorker = (tileVertices + numContexts - 1) /
                           numContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numContexts;
  return computeCycles;
}

static unsigned
estimateReduceComputeCost(const poplar::DeviceInfo &deviceInfo,
                          const ConvolutionParams &params,
                          const Partition &partition,
                          Phase phase) {
  if (partition.tilesPerInZGroupAxis == 1)
    return 0;
  const auto numTiles = deviceInfo.getNumTiles() / params.batchSize;
  const auto numOutputs = params.getOutputHeight(phase) *
                          params.getOutputWidth(phase) *
                          params.outputDepth;
  const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
  const auto numPartialSumsPerTile = numOutputsPerTile *
                                     partition.tilesPerInZGroupAxis;
  const auto vectorWidth =
      partition.floatPartials ? deviceInfo.getFloatVectorWidth() :
                                deviceInfo.getHalfVectorWidth();
  const auto numCycles = (numPartialSumsPerTile + vectorWidth - 1) /
                          vectorWidth;
  return numCycles;
}

static unsigned
estimateComputeCost(const poplar::DeviceInfo &deviceInfo,
                    bool floatActivations, const ConvolutionParams &params,
                    const Partition &partition,
                    Phase phase) {
  return estimatePartialCalcComputeCost(deviceInfo, floatActivations, params,
                                     partition, phase) +
         estimateReduceComputeCost(deviceInfo, params,
                                   partition, phase);

}

static unsigned
estimatePartitionCostBounded(const poplar::DeviceInfo &deviceInfo,
                             bool floatActivations,
                             const ConvolutionParams &params,
                             const Partition &partition,
                             unsigned maxBound,
                             Phase phase) {
  auto cost = estimateExchangeCost(deviceInfo,
                                   floatActivations, params, partition,
                                   phase);
  if (cost > maxBound)
    return maxBound;
  cost += estimateComputeCost(deviceInfo, floatActivations, params,
                              partition, phase);
  return std::min(cost, maxBound);
}

static unsigned
estimatePartitionCost(const poplar::DeviceInfo &deviceInfo,
                      bool floatActivations,
                      const ConvolutionParams &params,
                      const Partition &partition,
                      Phase phase) {
  return estimatePartitionCostBounded(deviceInfo,
                                      floatActivations, params, partition,
                                      std::numeric_limits<unsigned>::max(),
                                      phase);
}

static std::pair<Partition, unsigned>
choosePartition(const poplar::DeviceInfo &deviceInfo,
                bool floatActivations,
                bool preferConvInstructions,
                unsigned inChansPerGroup,
                const ConvVertexType &convVertexType,
                const ConvolutionParams &params,
                Phase phase) {
  if (convVertexType.useConvInstruction &&
      phase == Phase::WEIGHTUPDATE) {
    assert(params.kernelSize == 1);
    assert(params.stride == 1);
    assert(params.padding == 0);
    assert(params.batchSize == 1);
    // The weight update can be implemented as a convolution with a different
    // axis of accumulation.
    // weight update field: fwd out channels.
    // weight update in chans: flattened fwd field.
    // weight update out chans: fwd in channels.
    // See the implementation of the Convolution layer for more details.
    // Partition the weight update phase by populating ConvolutionParams
    // struct with the dimensions of this convolution and performing
    // a recursive call using these parameters.
    const auto fieldGroupSize =
        deviceInfo.getWeightsPerConvUnit(floatActivations);
    const auto fieldSize = params.getOutputHeight(phase) *
                           params.getOutputWidth(phase);
    const auto paddedFieldSize =
        ((fieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
    auto newParams = ConvolutionParams(
                       params.kernelSize,
                       params.stride,
                       paddedFieldSize,
                       params.outputDepth,
                       1,
                       0 /*padding*/,
                       params.inputDepth,
                       params.batchSize);
    return choosePartition(deviceInfo, floatActivations, false,
                           deviceInfo.getWeightsPerConvUnit(floatActivations),
                           convVertexType, newParams, Phase::FORWARD);
  }
  const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition bestPartition;
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
  const auto numTiles = deviceInfo.getNumTiles() / params.batchSize;
  const auto maxTilesPerX = std::min(params.getOutputWidth(phase), numTiles);
  for (unsigned tilesPerX = 1; tilesPerX <= maxTilesPerX; ++tilesPerX) {
    const auto maxTilesPerY = std::min(params.getOutputHeight(phase),
                                       numTiles / tilesPerX);
    for (unsigned tilesPerY = 1; tilesPerY <= maxTilesPerY; ++tilesPerY) {
      const auto maxTilesPerZ =
          std::min(params.outputDepth / partialChansPerGroup,
                   numTiles / (tilesPerX * tilesPerY));
      for (unsigned tilesPerZ = 1; tilesPerZ <= maxTilesPerZ; ++tilesPerZ) {
        const auto tilesPerInZ =
            std::min(params.inputDepth / inChansPerGroup,
                     numTiles / (tilesPerX * tilesPerY * tilesPerZ));
        auto maxVerticesPerTilePerY =
            (params.getOutputHeight(phase) + tilesPerY - 1) / tilesPerY;
        auto minVerticesPerTilePerY = 1;
        if (convVertexType.useConvInstruction) {
          if (deviceInfo.sharedConvWeights) {
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
          Partition candidate(tilesPerX, tilesPerY, tilesPerZ,
                              verticesPerTilePerY, tilesPerInZ,
                              inChansPerGroup, partialChansPerGroup,
                              convVertexType.floatPartials,
                              convVertexType.useConvInstruction);

          auto candidateCost =
              estimatePartitionCostBounded(deviceInfo, floatActivations,
                                           params, candidate,
                                           bestCost, phase);
          if (preferConvInstructions &&
              !convVertexType.useConvInstruction)
            candidateCost *= 100000;
          if (candidateCost < bestCost) {
            bestPartition = candidate;
            bestCost = candidateCost;
          }
        }
      }
    }
  }
  return {bestPartition, bestCost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            unsigned inChansPerGroup,
                            const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (deviceInfo.fp16AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, false,
                                   params.stride, inChansPerGroup,
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(deviceInfo, true, false);
  }
  if (deviceInfo.fp32AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, true,
                                   params.stride, inChansPerGroup,
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(deviceInfo, true, true);
  }
  convVertexTypeCandidates.emplace_back(deviceInfo, false, true);
  return convVertexTypeCandidates;
}

static std::vector<ConvVertexType>
getWeightUpdateVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                                    bool floatActivations,
                                    unsigned deltasChansPerGroup,
                                    const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (params.stride == 1 && params.kernelSize == 1 && params.padding == 0) {
    if (deviceInfo.fp16AccumConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(deviceInfo, true, false);
    }
    if (deviceInfo.fp32AccumConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(deviceInfo, true, true);
    }
  }
  convVertexTypeCandidates.emplace_back(false, floatActivations,
                                        deltasChansPerGroup);
  return convVertexTypeCandidates;
}

static std::pair<Partition, unsigned>
choosePartition(const poplar::DeviceInfo &deviceInfo,
                bool floatActivations,
                bool preferConvInstructions,
                unsigned inChansPerGroup,
                const ConvolutionParams &params,
                Phase phase) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition bestPartition;
  if (params.inputDepth % inChansPerGroup != 0) {
    // TODO handle this case.
    std::abort();
  }
  const auto convVertexTypeCandidates =
      getConvVertexTypeCandidates(deviceInfo, floatActivations, inChansPerGroup,
                                  params);
  for (const auto &convVertexType : convVertexTypeCandidates) {
    Partition candidate;
    unsigned candidateCost;
    std::tie(candidate, candidateCost) =
        choosePartition(deviceInfo, floatActivations, preferConvInstructions,
                        inChansPerGroup, convVertexType, params, phase);
    if (candidateCost < bestCost) {
      bestPartition = candidate;
      bestCost = candidateCost;
    }
  }
  return {bestPartition, bestCost};
}

std::vector<unsigned>
getInChansPerGroupCandidates(const ConvolutionParams &params,
                             bool floatActivations) {
  std::vector<unsigned> candidates;
  for (unsigned i = 1; i <= params.inputDepth; ++i) {
    if (params.inputDepth % i != 0)
      continue;
    if (!floatActivations && i % 2 != 0)
      continue;
    candidates.push_back(i);
  }
  if (candidates.empty())
    candidates.push_back(params.inputDepth);
  return candidates;
}

static std::pair<Partition, unsigned>
choosePartition(const poplar::DeviceInfo &deviceInfo, bool floatActivations,
                bool preferConvInstructions,
                const ConvolutionParams &params, Phase phase) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition best;
  auto inChansPerGroupCandidates =
      getInChansPerGroupCandidates(params, floatActivations);
  for (auto inChansPerGroup : inChansPerGroupCandidates) {
    Partition candidate;
    unsigned candidateCost;
    std::tie(candidate, candidateCost) =
        choosePartition(deviceInfo, floatActivations,
                        preferConvInstructions, inChansPerGroup, params,
                        phase);
    assert(candidate.inChansPerGroup == inChansPerGroup);
    if (candidateCost < bestCost) {
      best = candidate;
      bestCost = candidateCost;
    }
  }
  return {best, bestCost};
}

ConvPlan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSize, unsigned stride, unsigned padding,
                    unsigned numChannels, unsigned batchSize,
                    std::string dType,
                    const poplar::Graph &graph, bool forwardOnly) {
  validateLayerParams(inDimY, inDimX, inNumChans, kernelSize, stride, padding,
                      numChannels, dType);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  bool preferConvInstructions =
      graph.getDevice().getDeviceType() == poplar::DeviceType::CPU;
  ConvPlan plan;
  if (kernelSize == 1 && stride == 1 && padding == 0) {
    plan.flattenXY = true;
    inDimX = inDimX * inDimY;
    inDimY = 1;
  } else {
    plan.flattenXY = false;
  }
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  ConvolutionParams fwdParams(kernelSize, stride, inNumChans, inDimX, inDimY,
                              padding, numChannels, batchSize);
  ConvolutionParams bwdParams(kernelSize, stride, numChannels, outDimX,
                              outDimY, padding, inNumChans, batchSize);
  const bool floatActivations = dType == "float";
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  auto inChansPerGroupCandidates =
      getInChansPerGroupCandidates(fwdParams, floatActivations);
  if (!forwardOnly) {
    plan.bwdPartition = choosePartition(deviceInfo, floatActivations,
                                        preferConvInstructions,
                                        bwdParams, Phase::BACKWARD).first;
  }
  for (auto inChansPerGroup : inChansPerGroupCandidates) {
    const auto convVertexTypeCandidates =
        getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                    inChansPerGroup, fwdParams);
    for (const auto &convVertexType : convVertexTypeCandidates) {
      Partition fwdCandidate;
      unsigned fwdCandidateCost;
      std::tie(fwdCandidate, fwdCandidateCost) =
          choosePartition(deviceInfo, floatActivations, preferConvInstructions,
                          inChansPerGroup, convVertexType,
                          fwdParams, Phase::FORWARD);
      Partition bestwuCandidate;
      unsigned bestwuCandidateCost = std::numeric_limits<unsigned>::max();
      if (!forwardOnly) {
        const auto wuVertexTypeCandidates =
            getWeightUpdateVertexTypeCandidates(
              deviceInfo, floatActivations, convVertexType.partialChansPerGroup,
              fwdParams);
        Partition wuCandidate;
        unsigned wuCandidateCost = std::numeric_limits<unsigned>::max();
        for (const auto &wuVertexType : wuVertexTypeCandidates) {
          std::tie(wuCandidate, wuCandidateCost) =
              choosePartition(deviceInfo, floatActivations,
                              preferConvInstructions, inChansPerGroup,
                              wuVertexType, fwdParams,
                              Phase::WEIGHTUPDATE);
          if (wuCandidateCost < bestwuCandidateCost) {
            bestwuCandidate = wuCandidate;
            bestwuCandidateCost = wuCandidateCost;
          }
        }
      }
      unsigned cost = fwdCandidateCost + bestwuCandidateCost;
      if (cost < bestCost) {
        bestCost = cost;
        plan.fwdPartition = fwdCandidate;
        if (!forwardOnly)
          plan.wuPartition = bestwuCandidate;
      }
    }
  }
  return plan;
}

}
