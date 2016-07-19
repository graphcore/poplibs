#include "ConvPlan.hpp"
#include "Convolution.hpp"
#include "ConvUtil.hpp"
#include "PerformanceEstimation.hpp"

namespace conv {

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

static unsigned getNumConvUnits(bool floatPartial,
                                const DeviceInfo &deviceInfo) {
  return floatPartial ? deviceInfo.fp32AccumConvUnitsPerTile :
                        deviceInfo.fp16AccumConvUnitsPerTile;
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned stride,
                             unsigned inChansPerGroup,
                             unsigned partialChansPerGroup,
                             const DeviceInfo &deviceInfo) {
  if (floatActivations || stride >= (1 << 4) ||
      inChansPerGroup != deviceInfo.getInputChannelsPerConvUnit())
    return false;
  return partialChansPerGroup == getNumConvUnits(floatPartials, deviceInfo);
}

static unsigned
getMaxInputRangeSize(unsigned outputRangeSize, unsigned stride,
                     unsigned kernelSize, unsigned padding,
                     unsigned numPartitions,
                     unsigned inputSize, bool contiguousAccess,
                     bool forward) {
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
                                      inputSize, forward);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    if (forward) {
      inputRangeSize = (outputRangeSize - 1) * stride + 1 + (kernelSize - 1);
    } else {
      inputRangeSize = (outputRangeSize - kernelSize ) / stride + 1;
    }
    break;
  }
  if (forward && !contiguousAccess && kernelSize == 1 && stride > 1) {
    inputRangeSize = (inputRangeSize - 1) / stride + 1;
  }
  return inputRangeSize;
}

static std::uint64_t
getConvPartial1x1CycleEstimate(unsigned kernelWidth,
                               unsigned inputGroupsPerOutput,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               bool useSupervisorVertices,
                               unsigned outputStride)
{
  if (useSupervisorVertices) {
    std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
    for (unsigned i = 0; i != inputGroupsPerOutput * kernelWidth; ++i) {
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
    return getConvPartial1x1SupervisorCycleEstimate(convSizesByWeightAndWorker,
                                                    convUnitPipelineDepth,
                                                    numConvUnitsPerTile);
  }
  std::vector<std::vector<unsigned>> convSizesByWeight;
  for (unsigned i = 0; i != inputGroupsPerOutput * kernelWidth; ++i) {
    convSizesByWeight.emplace_back();
    for (unsigned j = 0; j != outputHeight; ++j) {
      convSizesByWeight.back().push_back(outputWidth);
    }
  }
  return getConvPartial1x1CycleWorkerEstimate(convSizesByWeight,
                                              convUnitPipelineDepth,
                                              numConvUnitsPerTile);
}


static std::uint64_t
getConvPartialCycleEstimate(bool floatActivations, bool floatPartials,
                            unsigned inChansPerGroup,
                            unsigned inputStride, unsigned outputStride,
                            unsigned kernelWidth,
                            unsigned inputGroupsPerOutput,
                            unsigned outputHeight,
                            unsigned outputWidth,
                            unsigned outChansPerGroup,
                            const DeviceInfo &deviceInfo,
                            bool useSupervisorVertices)
{
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   inputStride, inChansPerGroup,
                                   outChansPerGroup,
                                   deviceInfo)) {
    return getConvPartial1x1CycleEstimate(
          kernelWidth, inputGroupsPerOutput, outputHeight, outputWidth,
          deviceInfo.convUnitPipelineDepth,
          getNumConvUnits(floatPartials, deviceInfo), useSupervisorVertices,
          outputStride);
  }
  assert(!useSupervisorVertices);
  return getConvPartialByDotProductCycleEstimate(floatActivations,
                                                 inChansPerGroup,
                                                 kernelWidth,
                                                 inputGroupsPerOutput,
                                                 outputHeight,
                                                 outputWidth,
                                                 outChansPerGroup,
                                                 deviceInfo.dataPathWidth,
                                                 outputStride);
}

static unsigned
estimateExchangeCost(const DeviceInfo &deviceInfo,
                     bool floatActivations, const ConvolutionParams &params,
                     const Partition &partition,
                     bool forward) {
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
                           tilesPerX, params.inputWidth, true, forward);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.stride, params.kernelSize,
                           params.padding,
                           tilesPerY, params.inputWidth, false, forward);
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
  const auto exchangeBytesPerCycle = deviceInfo.getIPUExchangeBandwidth();
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
                     const DeviceInfo &deviceInfo,
                     bool useSupervisorVertices,
                     bool forward) {
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
  const auto inputStride = forward ? params.stride : 1;
  const auto outputStride = forward ? 1 : params.stride;
  return getConvPartialCycleEstimate(floatActivations, partition.floatPartials,
                                     inChansPerGroup, inputStride,
                                     outputStride,
                                     params.kernelSize, inputGroupsPerOutput,
                                     outRowsPerVertex, tileOutWidth,
                                     outChansPerGroup,
                                     deviceInfo,
                                     useSupervisorVertices);
}

static unsigned
estimateConvolveComputeCost(const DeviceInfo &deviceInfo,
                            bool floatActivations,
                            const ConvolutionParams &params,
                            const Partition &partition,
                            bool forward) {
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
  unsigned numContexts = deviceInfo.getNumWorkerContexts();
  if (deviceInfo.sharedConvWeights &&
      canUseConvolutionInstruction(floatActivations, partition.floatPartials,
                                   params.stride,
                                   partition.inChansPerGroup,
                                   partition.partialChansPerGroup,
                                   deviceInfo)) {
    useSupervisorVertices = true;
    numContexts = 1;
  }
  const auto vertexRuntime = estimateVertexCycles(floatActivations, params,
                                                  partition,
                                                  deviceInfo,
                                                  useSupervisorVertices,
                                                  forward);
  auto verticesPerWorker = (tileVertices + numContexts - 1) /
                           numContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numContexts;
  return computeCycles;
}

static unsigned
estimateReduceComputeCost(const DeviceInfo &deviceInfo,
                          const ConvolutionParams &params,
                          const Partition &partition) {
  if (partition.tilesPerInZGroupAxis == 1)
    return 0;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numOutputs = params.getOutputHeight() * params.getOutputWidth() *
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
estimateComputeCost(const DeviceInfo &deviceInfo,
                    bool floatActivations, const ConvolutionParams &params,
                    const Partition &partition,
                    bool forward) {
  return estimateConvolveComputeCost(deviceInfo, floatActivations, params,
                                     partition, forward) +
         estimateReduceComputeCost(deviceInfo, params,
                                   partition);

}

static unsigned
estimatePartitionCostBounded(const DeviceInfo &deviceInfo,
                             bool floatActivations,
                             const ConvolutionParams &params,
                             const Partition &partition,
                             unsigned maxBound,
                             bool forward) {
  auto cost = estimateExchangeCost(deviceInfo,
                                   floatActivations, params, partition,
                                   forward);
  if (cost > maxBound)
    return maxBound;
  cost += estimateComputeCost(deviceInfo, floatActivations, params,
                              partition, forward);
  return std::min(cost, maxBound);
}

static unsigned
estimatePartitionCost(const DeviceInfo &deviceInfo,
                      bool floatActivations,
                      const ConvolutionParams &params,
                      const Partition &partition,
                      bool forward) {
  return estimatePartitionCostBounded(deviceInfo,
                                      floatActivations, params, partition,
                                      std::numeric_limits<unsigned>::max(),
                                      forward);
}

static Partition
choosePartition(const DeviceInfo &deviceInfo,
                bool floatActivations,
                unsigned inChansPerGroup,
                const ConvolutionParams &params,
                bool forward) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition bestPartition;
  if (params.inputDepth % inChansPerGroup != 0) {
    // TODO handle this case.
    std::abort();
  }
  std::vector<unsigned> partialChansPerGroupCandidates;
  if (deviceInfo.fp16AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, false,
                                   params.stride, inChansPerGroup,
                                   deviceInfo.fp16AccumConvUnitsPerTile,
                                   deviceInfo)) {
    partialChansPerGroupCandidates.push_back(
      deviceInfo.fp16AccumConvUnitsPerTile
    );
  }
  if (deviceInfo.fp32AccumConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, true,
                                   params.stride, inChansPerGroup,
                                   deviceInfo.fp32AccumConvUnitsPerTile,
                                   deviceInfo)) {
    partialChansPerGroupCandidates.push_back(
      deviceInfo.fp32AccumConvUnitsPerTile
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
  const auto numTiles = deviceInfo.getNumTiles();
  for (const auto partialChansPerGroup : partialChansPerGroupCandidates) {
    const auto maxTilesPerX = std::min(params.getOutputWidth(), numTiles);
    for (unsigned tilesPerX = 1; tilesPerX <= maxTilesPerX; ++tilesPerX) {
      const auto maxTilesPerY = std::min(params.getOutputHeight(),
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
              (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
          auto minVerticesPerTilePerY = 1;
          if (partialChansPerGroup == deviceInfo.fp16AccumConvUnitsPerTile ||
              partialChansPerGroup == deviceInfo.fp32AccumConvUnitsPerTile) {
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
            bool floatPartials =
                !canUseConvolutionInstruction(
                  floatActivations, false, params.stride, inChansPerGroup,
                  partialChansPerGroup,
                  deviceInfo
                );
            Partition candidate(tilesPerX, tilesPerY, tilesPerZ,
                                verticesPerTilePerY, tilesPerInZ,
                                inChansPerGroup, partialChansPerGroup,
                                floatPartials);
            auto candidateCost =
                estimatePartitionCostBounded(deviceInfo, floatActivations,
                                             params, candidate,
                                             bestCost, forward);
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

static Partition
choosePartition(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                unsigned inNumChanGroups,
                unsigned kernelSize, unsigned stride, unsigned padding,
                unsigned outNumChans,
                std::string dType,
                const DeviceInfo &deviceInfo,
                bool forward) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  const bool floatActivations = dType == "float";
  ConvolutionParams params(kernelSize, stride, inNumChans, inDimX, inDimY,
                           padding, outNumChans);
  return choosePartition(deviceInfo, floatActivations,
                         inNumChans / inNumChanGroups,
                         params, forward);
}

static Partition
choosePartition(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                unsigned kernelSize, unsigned stride, unsigned padding,
                unsigned outNumChans,
                std::string dType,
                const DeviceInfo &deviceInfo, bool forward) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX, kernelSize,
                                            stride, padding);
  const bool floatActivations = dType == "float";
  ConvolutionParams params(kernelSize, stride, inNumChans, inDimX, inDimY,
                           padding, outNumChans);
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition best;
  for (unsigned i = 1; i <= inNumChans; ++i) {
    if (inNumChans % i != 0)
      continue;
    if (!floatActivations && i % 2 != 0)
      continue;

    const auto candidate = choosePartition(inDimY, inDimX, inNumChans,
                                           inNumChans / i,
                                           kernelSize, stride, padding,
                                           outNumChans, dType, deviceInfo,
                                           forward);
    assert(candidate.inChansPerGroup == i);
    const auto candidateCost =
        estimatePartitionCost(deviceInfo, floatActivations, params,
                              candidate, forward);
    if (candidateCost < bestCost) {
      best = candidate;
      bestCost = candidateCost;
    }
  }
  if (bestCost == std::numeric_limits<unsigned>::max()) {
    return choosePartition(inDimY, inDimX, inNumChans, 1,
                           kernelSize, stride, padding,
                           outNumChans, dType, deviceInfo,
                           forward);
  }
  return best;
}


ConvPlan createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSize, unsigned stride, unsigned padding,
                    unsigned numChannels, std::string dType,
                    const DeviceInfo &deviceInfo) {
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
  plan.fwdPartition = choosePartition(inDimY, inDimX, inNumChans,
                                      kernelSize, stride, padding,
                                      numChannels, dType,
                                      deviceInfo, true);
  plan.bwdPartition = choosePartition(outDimY, outDimX, numChannels,
                                      kernelSize, stride, padding,
                                      inNumChans, dType,
                                      deviceInfo, false);
  const bool floatActivations = dType == "float";
  plan.fwdPartition.useConvolutionInstructions =
      canUseConvolutionInstruction(floatActivations,
                                   plan.fwdPartition.floatPartials,
                                   stride, plan.fwdPartition.inChansPerGroup,
                                   plan.fwdPartition.partialChansPerGroup,
                                   deviceInfo);
  plan.bwdPartition.useConvolutionInstructions =
      canUseConvolutionInstruction(floatActivations,
                                   plan.bwdPartition.floatPartials,
                                   stride, plan.bwdPartition.inChansPerGroup,
                                   plan.bwdPartition.partialChansPerGroup,
                                   deviceInfo);
  return plan;
}

}
