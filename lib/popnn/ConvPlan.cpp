#include "popnn/ConvPlan.hpp"
#include "popnn/Convolution.hpp"
#include "popnn/exceptions.hpp"
#include "poplar/Graph.hpp"
#include "ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "PerformanceEstimation.hpp"
#include <map>
#include <tuple>

// A simple function to memoize other functions. Any recursive calls
// with the function are non memoized
template <typename Ret, typename... Args>
class Memo {
  std::map<std::tuple<Args...>, Ret> table;
  Ret (*fn)(Args...);
 public:
  Memo(Ret (*fn)(Args...)) : fn(fn) {}
  Ret operator()(Args... args) {
    const auto key = std::make_tuple(args...);
    const auto match = table.find(key);
    if(match == table.end()) {
      auto result = fn(args...);
      table[key] = result;
      return result;
    } else {
      return match->second;
    }
  }
  void clearTable() {
    table.clear();
  }
};

template <typename Ret, typename... Args>
static Memo<Ret, Args...> memoize(Ret (*fn)(Args...)) {
  return Memo<Ret, Args...>(fn);
}

static unsigned getNumConvUnits(bool floatActivations,
                                bool floatPartial,
                                const poplar::DeviceInfo &deviceInfo) {
  if (floatActivations) {
    return deviceInfo.fp32InFp32OutConvUnitsPerTile;
  } else {
    return floatPartial ? deviceInfo.fp16InFp32OutConvUnitsPerTile :
                          deviceInfo.fp16InFp16OutConvUnitsPerTile;
  }
}

namespace {
struct ConvVertexType {
  bool useConvInstruction;
  bool floatActivations;
  bool floatPartials;
  unsigned partialChansPerGroup;
  ConvVertexType(const poplar::DeviceInfo &deviceInfo,
                 bool useConvInstruction, bool floatActivations,
                 bool floatPartials) :
    useConvInstruction(useConvInstruction),
    floatActivations(floatActivations),
    floatPartials(floatPartials) {
    if (!useConvInstruction) {
      partialChansPerGroup = 1;
    } else {
      partialChansPerGroup = getNumConvUnits(floatActivations,
                                             floatPartials, deviceInfo);
    }
  }
  ConvVertexType(bool useConvInstruction, bool floatActivations,
                 bool floatPartials,
                 unsigned partialChansPerGroup) :
    useConvInstruction(useConvInstruction),
    floatActivations(floatActivations),
    floatPartials(floatPartials),
    partialChansPerGroup(partialChansPerGroup) {}
};
}

namespace conv {

static std::uint64_t
getConvPartialnx1CycleEstimate(unsigned passesPerOutput,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               bool useSupervisorVertices,
                               unsigned outputStride,
                               unsigned numInputPointers,
                               PlannerCache *cache);

class PlannerCache {
public:
  decltype(memoize(partitionConvPartialByWorker))
    mPartitionConvPartialByWorker;
  decltype(memoize(getConvPartialnx1CycleEstimate))
    mGetConvPartialnx1CycleEstimate;
  PlannerCache() :
    mPartitionConvPartialByWorker(memoize(partitionConvPartialByWorker)),
    mGetConvPartialnx1CycleEstimate(
      memoize(getConvPartialnx1CycleEstimate)
    ) {}
};

Planner::Planner() {
  cache = std::unique_ptr<PlannerCache>(new PlannerCache());
}

Planner::~Planner() = default;

enum class Phase {
  FORWARD,
  BACKWARD,
  WEIGHTUPDATE
};

struct ConvolutionParams {
  unsigned kernelSizeY;
  unsigned kernelSizeX;
  unsigned strideY;
  unsigned strideX;
  unsigned inputDepth;
  unsigned inputWidth;
  unsigned inputHeight;
  unsigned paddingY;
  unsigned paddingX;
  unsigned outputDepth;
  unsigned batchSize;
  unsigned getOutputWidth(Phase phase) const {
    if (phase == Phase::BACKWARD)
      return (inputWidth * strideX + kernelSizeX - 1) - (paddingX * 2);
    else
      return inputWidth + paddingX * 2 < kernelSizeX
        ? 0
        : (inputWidth + (paddingX * 2) - kernelSizeX) / strideX + 1;
  }
  unsigned getOutputHeight(Phase phase) const {
    if (phase == Phase::BACKWARD)
      return (inputHeight * strideY + kernelSizeY - 1) - (paddingY * 2);
    else
      return inputHeight + paddingY * 2 < kernelSizeY
        ? 0
        : (inputHeight + (paddingY * 2) - kernelSizeY) / strideY + 1;
  }
  ConvolutionParams(unsigned kernelSizeY,
                    unsigned kernelSizeX,
                    unsigned strideY,
                    unsigned strideX,
                    unsigned inputDepth,
                    unsigned inputWidth,
                    unsigned inputHeight,
                    unsigned paddingY,
                    unsigned paddingX,
                    unsigned outputDepth,
                    unsigned batchSize) :
    kernelSizeY(kernelSizeY),
    kernelSizeX(kernelSizeX),
    strideY(strideY),
    strideX(strideX),
    inputDepth(inputDepth),
    inputWidth(inputWidth),
    inputHeight(inputHeight),
    paddingY(paddingY),
    paddingX(paddingX),
    outputDepth(outputDepth),
    batchSize(batchSize) {}
};

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned strideY, unsigned strideX,
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
  if (strideX >= (1 << 4))
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
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               bool useSupervisorVertices,
                               unsigned outputStride,
                               unsigned numInputPointers,
                               PlannerCache * cache)
{
  if (useSupervisorVertices) {
    std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
    for (unsigned i = 0; i != passesPerOutput; ++i) {
      const auto numWorkerContexts = 6;
      std::vector<std::vector<PartialRow>> partition =
          cache->mPartitionConvPartialByWorker(outputHeight, outputWidth,
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
    return getConvPartialnx1SupervisorCycleEstimate(
                  convSizesByWeightAndWorker,
                  convUnitPipelineDepth,
                  numConvUnitsPerTile,
                  convUnitCoeffLoadBytesPerCycle,
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
                                              convUnitCoeffLoadBytesPerCycle,
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
      getMaxInputRangeSize(tileOutWidth, params.strideX, params.kernelSizeX,
                           params.paddingX,
                           tilesPerX, params.inputWidth, true, phase);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.strideY, params.kernelSizeY,
                           params.paddingY,
                           tilesPerY, params.inputWidth, false, phase);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      params.kernelSizeY * params.kernelSizeX * tileOutDepth * tileInDepth;
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

  const auto tilesPerSuperTile = deviceInfo.tilesPerSuperTile;
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;

  const auto inputElementBytesPerCycle =
      (tilesPerZ % tilesPerSuperTile) == 0 ? exchangeBytesPerCycle :
                                             exchangeBytesPerCycle *
                                             tilesPerSuperTile;
  const auto numCycles =
      (inputElementsBytes + inputElementBytesPerCycle - 1) /
      inputElementBytesPerCycle +
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
                     Phase phase,
                     PlannerCache *cache) {
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
  const auto outputStrideY = phase != Phase::BACKWARD ? 1 : params.strideY;
  const auto inputStrideY = phase != Phase::BACKWARD ? params.strideY : 1;
  const auto inputStrideX = phase != Phase::BACKWARD ? params.strideX : 1;

  if (phase == Phase::WEIGHTUPDATE) {
    auto vectorWidth = deviceInfo.dataPathWidth / (floatActivations ? 32 : 16);
    return getWeightGradCalcCycles(tileOutHeight, tileOutHeight * inputStrideY,
                                   tileOutWidth, tileOutWidth * inputStrideX,
                                   outChansPerGroup, inChansPerGroup,
                                   inputStrideY, inputStrideX,
                                   params.kernelSizeY,
                                   params.kernelSizeX,
                                   0, 0, vectorWidth);
  }
  if (partition.useConvolutionInstructions) {
    assert(deviceInfo.getWeightsPerConvUnit(floatActivations) %
           inChansPerGroup == 0);
    const auto convUnitWeightHeight =
        deviceInfo.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
    const auto passesPerFilter =
        params.kernelSizeX *
        (params.kernelSizeY + convUnitWeightHeight - 1) / convUnitWeightHeight;
    const auto passesPerOutput = passesPerFilter * tileNumInGroups;
    return cache->mGetConvPartialnx1CycleEstimate(
          passesPerOutput, outRowsPerVertex,
          tileOutWidth, deviceInfo.convUnitPipelineDepth,
          getNumConvUnits(floatActivations,
                          partition.floatPartials, deviceInfo),
          deviceInfo.convUnitCoeffLoadBytesPerCycle,
          useSupervisorVertices,
          outputStrideY,
          convUnitWeightHeight,
          cache);
  }
  assert(!useSupervisorVertices);
  return outRowsPerVertex * outChansPerGroup *
         getConvPartialByDotProductCycleEstimate(
           floatActivations, inChansPerGroup, params.kernelSizeY,
           params.kernelSizeX * tileNumInGroups, tileOutWidth,
           deviceInfo.dataPathWidth, outputStrideY
         );
}

static unsigned
estimatePartialCalcComputeCost(const poplar::DeviceInfo &deviceInfo,
                               bool floatActivations,
                               const ConvolutionParams &params,
                               const Partition &partition,
                               Phase phase,
                               PlannerCache *cache) {
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
                                                  phase,
                                                  cache);
  auto verticesPerWorker = (tileVertices + numContexts - 1) /
                           numContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numContexts;
  return computeCycles;
}

static unsigned calcNumUsableTiles(
    unsigned numTiles,
    unsigned numBatchGroups) {
  const auto batchElemsPerTile = (numBatchGroups + numTiles - 1) / numTiles;
  const auto batchSubGroups =
      (numBatchGroups + batchElemsPerTile - 1) / batchElemsPerTile;

  return numTiles / batchSubGroups;
}

static unsigned
estimateReduceComputeCost(const poplar::DeviceInfo &deviceInfo,
                          const ConvolutionParams &params,
                          const Partition &partition,
                          unsigned numBatchGroups,
                          Phase phase) {
  if (partition.tilesPerInZGroupAxis == 1)
    return 0;

  /* The reduction is actually done on tiles in which the output
   * activations reside. Thus the output height here may be different
   * from the one the tensor uses in the reduction. The numOutputsPerTile
   * below however is approximately the same except for any rounding
   */
  const auto numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(),
                                           numBatchGroups);
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
                    unsigned numBatchGroups,
                    Phase phase,
                    PlannerCache *cache) {
  return estimatePartialCalcComputeCost(deviceInfo, floatActivations, params,
                                        partition, phase, cache) +
         estimateReduceComputeCost(deviceInfo, params,
                                   partition, numBatchGroups, phase);

}

static unsigned
estimatePartitionCostBounded(const poplar::DeviceInfo &deviceInfo,
                             bool floatActivations,
                             const ConvolutionParams &params,
                             const Partition &partition,
                             unsigned maxBound,
                             unsigned numBatchGroups,
                             Phase phase,
                             PlannerCache *cache) {
  auto cost = estimateExchangeCost(deviceInfo,
                                   floatActivations, params, partition,
                                   phase);
  if (cost > maxBound)
    return maxBound;
  cost += estimateComputeCost(deviceInfo, floatActivations, params,
                              partition, numBatchGroups, phase, cache);
  return std::min(cost, maxBound);
}

static std::pair<Partition, unsigned>
choosePartition(const poplar::DeviceInfo &deviceInfo,
                bool floatActivations,
                bool preferConvInstructions,
                unsigned inChansPerGroup,
                const ConvVertexType &convVertexType,
                const ConvolutionParams &params,
                Phase phase,
                const unsigned numBatchGroups,
                PlannerCache *cache) {
  if (convVertexType.useConvInstruction &&
      phase == Phase::WEIGHTUPDATE) {
    // The weight update can be implemented as a convolution with a different
    // axis of accumulation.
    // weight update field: fwd out channels.
    // weight update in chans: flattened fwd field.
    // weight update out chans: fwd in channels * kernel elements
    // See the implementation of the Convolution layer for more details.
    // Partition the weight update phase by populating ConvolutionParams
    // struct with the dimensions of this convolution and performing
    // a recursive call using these parameters.
    const auto fieldGroupSize =
        deviceInfo.getWeightsPerConvUnit(floatActivations);
    const auto fieldSize = params.getOutputHeight(phase) *
                           params.getOutputWidth(phase) *
                           params.batchSize;
    const auto paddedFieldSize =
        ((fieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
    const auto numKernelElements = params.kernelSizeY * params.kernelSizeX;
    const auto outputSize =  params.inputDepth * numKernelElements;
    const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
    const auto paddedOutputSize =
        ((outputSize + partialChansPerGroup - 1) / partialChansPerGroup)
            * partialChansPerGroup;
    auto newParams = ConvolutionParams(
                       1 /* kernelSizeY */,
	          				   1 /* kernelSizeX */,
                       params.strideY,
                       params.strideX,
                       paddedFieldSize,
                       params.outputDepth,
                       1,
                       0 /*paddingY*/,
					             0 /*paddingX*/,
                       paddedOutputSize,
                       1);
    return choosePartition(deviceInfo, floatActivations, false,
                           deviceInfo.getWeightsPerConvUnit(floatActivations),
                           convVertexType, newParams, Phase::FORWARD,
                           1, cache);
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
  const auto numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(),
                                           numBatchGroups);

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
                                           bestCost, numBatchGroups,
                                           phase, cache);
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
  bestPartition.batchesPerGroup = params.batchSize / numBatchGroups;
  bestPartition.numBatchGroups = numBatchGroups;
  return {bestPartition, bestCost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            bool floatPartials,
                            unsigned inChansPerGroup,
                            const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;

  if (deviceInfo.fp16InFp16OutConvUnitsPerTile > 0 &&
      !floatPartials &&
      canUseConvolutionInstruction(floatActivations, false,
                                   params.strideY, params.strideX,
                                   inChansPerGroup, deviceInfo)) {

    convVertexTypeCandidates.emplace_back(deviceInfo, true, false, false);
  } else {

    if (deviceInfo.fp16InFp32OutConvUnitsPerTile > 0 &&
        canUseConvolutionInstruction(floatActivations, true,
                                     params.strideY, params.strideX,
                                     inChansPerGroup, deviceInfo)) {
      convVertexTypeCandidates.emplace_back(deviceInfo, true, false, true);
    }
  }

  if (deviceInfo.fp32InFp32OutConvUnitsPerTile > 0 &&
      canUseConvolutionInstruction(floatActivations, true,
                                   params.strideY, params.strideX,
                                   inChansPerGroup,
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(deviceInfo, true, true, true);
  }
  convVertexTypeCandidates.emplace_back(deviceInfo, false, floatActivations,
                                        true);
  return convVertexTypeCandidates;
}

static std::vector<ConvVertexType>
getWeightUpdateVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                                    bool floatActivations,
                                    bool floatPartials,
                                    unsigned deltasChansPerGroup,
                                    const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (!floatActivations) {
    if (!floatPartials && deviceInfo.fp16InFp16OutConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(deviceInfo, true, false, false);
    } else {
      if (deviceInfo.fp16InFp32OutConvUnitsPerTile > 0) {
        convVertexTypeCandidates.emplace_back(deviceInfo, true, false, true);
      }
	}
  }

  if (deviceInfo.fp32InFp32OutConvUnitsPerTile > 0) {
    convVertexTypeCandidates.emplace_back(deviceInfo, true, true, true);
  }
  convVertexTypeCandidates.emplace_back(false, floatActivations, floatPartials,
                                        deltasChansPerGroup);
  return convVertexTypeCandidates;
}

static std::pair<Partition, unsigned>
choosePartition(const poplar::DeviceInfo &deviceInfo,
                bool floatActivations,
                bool floatPartials,
                bool preferConvInstructions,
                unsigned inChansPerGroup,
                const ConvolutionParams &params,
                Phase phase,
                unsigned numBatchGroups,
                PlannerCache *cache) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition bestPartition;
  if (params.inputDepth % inChansPerGroup != 0) {
    throw popnn::popnn_error("Input depths that are not a multiple of the "
                             "channel grouping are not supported");
  }
  const auto convVertexTypeCandidates =
      getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                  floatPartials, inChansPerGroup,
                                  params);
  for (const auto &convVertexType : convVertexTypeCandidates) {
    Partition candidate;
    unsigned candidateCost;
    std::tie(candidate, candidateCost) =
        choosePartition(deviceInfo, floatActivations, preferConvInstructions,
                        inChansPerGroup, convVertexType, params, phase,
                        numBatchGroups, cache);
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
                bool floatPartials,
                bool preferConvInstructions,
                const ConvolutionParams &params, Phase phase,
                unsigned numBatchGroups,
                PlannerCache *cache) {
  unsigned bestCost = std::numeric_limits<unsigned>::max();
  Partition best;
  auto inChansPerGroupCandidates =
      getInChansPerGroupCandidates(params, floatActivations);
  for (auto inChansPerGroup : inChansPerGroupCandidates) {
    Partition candidate;
    unsigned candidateCost;
    std::tie(candidate, candidateCost) =
        choosePartition(deviceInfo, floatActivations, floatPartials,
                        preferConvInstructions, inChansPerGroup, params,
                        phase, numBatchGroups, cache);
    assert(candidate.inChansPerGroup == inChansPerGroup);
    if (candidateCost < bestCost) {
      best = candidate;
      bestCost = candidateCost;
    }
  }
  return {best, bestCost};
}

ConvPlan
Planner::createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX,
                    unsigned numChannels, unsigned batchSize,
                    std::string dType,
                    std::string partialsType,
                    const poplar::Graph &graph, bool forwardOnly) {
  validateLayerParams(inDimY, inDimX, inNumChans, kernelSizeY, kernelSizeX,
                      strideY, strideX, paddingY, paddingX,
                      numChannels, dType);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  bool preferConvInstructions =
      graph.getDevice().getDeviceType() == poplar::DeviceType::CPU;
  ConvPlan plan;
  if (kernelSizeY == 1 && kernelSizeX == 1
      && strideY == 1 && strideX == 1
      && paddingY == 0 && paddingX == 0) {
    plan.flattenXY = true;
    inDimX = inDimX * inDimY;
    inDimY = 1;
  } else {
    plan.flattenXY = false;
  }

  const bool floatActivations = dType == "float";
  const bool floatPartials = partialsType == "float";

  if (!forwardOnly) {

    unsigned bwdCandidateCost = std::numeric_limits<unsigned>::max();
    Partition bwdCandidate;

    for (auto batchesPerGroup = 1U; batchesPerGroup <= batchSize;
              ++batchesPerGroup) {

      /* only allow integer division of batches.
       *  i.e. batchSize = batchesPerGroup * numBatchGroups
       */

      if ((batchSize % batchesPerGroup) ||
          (!plan.flattenXY && batchesPerGroup > 1)) {
        continue;
      }

      const unsigned numBatchGroups = batchSize / batchesPerGroup;

      unsigned outDimY, outDimX;
      std::tie(outDimY, outDimX) = getOutputDim(inDimY * batchesPerGroup,
                                                inDimX,
                                                kernelSizeY, kernelSizeX,
                                                strideY, strideX,
                                                paddingY, paddingX);

      ConvolutionParams bwdParams(kernelSizeY, kernelSizeX,
                                  strideY, strideX,
                                  numChannels, outDimX, outDimY,
                                  paddingY, paddingX, inNumChans, batchSize);

      Partition candidate;
      unsigned candidateCost;
      std::tie(candidate, candidateCost) =
          choosePartition(deviceInfo, floatActivations,
                          floatPartials,
                          preferConvInstructions,
                          bwdParams, Phase::BACKWARD,
                          numBatchGroups,
                          cache.get());

      if (candidateCost < bwdCandidateCost) {
        bwdCandidateCost = candidateCost;
        bwdCandidate = candidate;
      }
    }
    plan.bwdPartition = bwdCandidate;
  }


  unsigned bestCost = std::numeric_limits<unsigned>::max();

  for (auto batchesPerGroup = 1U; batchesPerGroup <= batchSize;
            ++batchesPerGroup) {

    /* only allow integer division of batches.
     *  i.e. batchSize = batchesPerGroup * numBatchGroups
     */

    if ((batchSize % batchesPerGroup) ||
        (!plan.flattenXY && batchesPerGroup > 1)) {
      continue;
    }

    const unsigned numBatchGroups = batchSize / batchesPerGroup;

    unsigned outDimY, outDimX;
    std::tie(outDimY, outDimX) = getOutputDim(inDimY * batchesPerGroup,
                                              inDimX,
                                              kernelSizeY, kernelSizeX,
                                              strideY, strideX,
                                              paddingY, paddingX);

    ConvolutionParams fwdParams(kernelSizeY, kernelSizeX, strideY, strideX,
                                inNumChans, inDimX, inDimY * batchesPerGroup,
                                paddingY, paddingX, numChannels, batchSize);

    ConvolutionParams wuParams(kernelSizeY, kernelSizeX, strideY, strideX,
                               inNumChans, inDimX, inDimY,
                               paddingY, paddingX, numChannels, batchSize);


    auto inChansPerGroupCandidates =
          getInChansPerGroupCandidates(fwdParams, floatActivations);


    unsigned fwdWuCandidateCost = std::numeric_limits<unsigned>::max();
    Partition fwdCandidateSelected, wuCandidateSelected;


    for (auto inChansPerGroup : inChansPerGroupCandidates) {
      const auto convVertexTypeCandidates =
          getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                      floatPartials,
                                      inChansPerGroup, fwdParams);
      for (const auto &convVertexType : convVertexTypeCandidates) {
        Partition fwdCandidate;
        unsigned fwdCandidateCost;
        std::tie(fwdCandidate, fwdCandidateCost) =
            choosePartition(deviceInfo, floatActivations,
                            preferConvInstructions,
                            inChansPerGroup, convVertexType,
                            fwdParams, Phase::FORWARD, numBatchGroups,
                            cache.get());

        Partition bestwuCandidate;
        unsigned bestwuCandidateCost = 0;
        if (!forwardOnly) {
          bestwuCandidateCost = std::numeric_limits<unsigned>::max();
          const auto wuVertexTypeCandidates =
              getWeightUpdateVertexTypeCandidates(
                deviceInfo, floatActivations, floatPartials,
                convVertexType.partialChansPerGroup,
                wuParams);
          Partition wuCandidate;
          unsigned wuCandidateCost = std::numeric_limits<unsigned>::max();
          for (const auto &wuVertexType : wuVertexTypeCandidates) {
            std::tie(wuCandidate, wuCandidateCost) =
                choosePartition(deviceInfo, floatActivations,
                                preferConvInstructions, inChansPerGroup,
                                wuVertexType, wuParams,
                                Phase::WEIGHTUPDATE, batchSize, cache.get());
            if (wuCandidateCost < bestwuCandidateCost) {
              bestwuCandidate = wuCandidate;
              bestwuCandidateCost = wuCandidateCost;
            }
          }
        }

        unsigned cost = fwdCandidateCost + bestwuCandidateCost;
        if (cost < fwdWuCandidateCost) {
          fwdWuCandidateCost = cost;
          fwdCandidateSelected = fwdCandidate;
          if (!forwardOnly)
              wuCandidateSelected = bestwuCandidate;
        }
      }
    }

    if (fwdWuCandidateCost < bestCost) {
      bestCost = fwdWuCandidateCost;
      plan.fwdPartition = fwdCandidateSelected;

      if (!forwardOnly) {
        plan.wuPartition = wuCandidateSelected;
      }
    }
  }
  return plan;
}

}
