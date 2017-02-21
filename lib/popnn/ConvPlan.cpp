#include "popnn/ConvPlan.hpp"
#include "popnn/Convolution.hpp"
#include "popnn/exceptions.hpp"
#include "poplar/Graph.hpp"
#include "ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "PerformanceEstimation.hpp"
#include <map>
#include <tuple>
#include <iostream>

using namespace convutil;

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
  ConvVertexType(bool useConvInstruction, bool floatActivations,
                 bool floatPartials) :
    useConvInstruction(useConvInstruction),
    floatActivations(floatActivations),
    floatPartials(floatPartials) {}
};
}

namespace conv {

std::ostream& operator<<(std::ostream &os, const Plan &p)
{
  os << "  Plan: TilesPerAxisXYZ         " << p.tilesPerXAxis << "*"
                                           << p.tilesPerYAxis << "*"
                                           << p.tilesPerZAxis << "="
    << p.tilesPerXAxis * p.tilesPerYAxis * p.tilesPerZAxis << "\n"
    << "        tilesPerInZGroupAxis    " << p.tilesPerInZGroupAxis << "\n"
    << "        verticesPerTilePerYAxis " << p.verticesPerTilePerYAxis << "\n"
    << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
    << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
    << "        batchesPerGroup         " << p.batchesPerGroup << "\n"
    << "        flattenXY               " << p.flattenXY << "\n";
  return os;
}

static std::uint64_t
getConvPartialnx1CycleEstimate(unsigned passesPerOutput,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
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

Planner::Planner(unsigned percentageCyclesExcessForMemOptim) :
    percentageCyclesExcessForMemOptim(percentageCyclesExcessForMemOptim){
  cache = std::unique_ptr<PlannerCache>(new PlannerCache());
}

Planner::~Planner() = default;

struct Cost {
  unsigned cycles;

  /* memory in bytes */
  unsigned memory;

  Cost(unsigned cycles, unsigned memory) : cycles(cycles), memory(memory) {}
  Cost() {}

  Cost operator*=(unsigned a) {
    cycles *= a;
    memory *= a;
    return *this;
  }

  Cost operator+(Cost b) {
    return {cycles + b.cycles, memory + b.cycles};
  }

  Cost operator+=(Cost b) {
    cycles += b.cycles;
    memory += b.memory;
    return *this;
  }

  bool operator==(Cost b) {
    return cycles == b.cycles && memory == b.memory;
  }
};

struct CostBounds {
  friend bool compareCost(Cost a, Cost b, CostBounds bounds);
  CostBounds(unsigned cycles, unsigned memory, bool primaryCheckIsCycles = true)
    :cycles(cycles), memory(memory), primaryCheckIsCycles(primaryCheckIsCycles)
      {}
private:
  unsigned cycles;
  unsigned memory;
  bool primaryCheckIsCycles;
};


bool compareCost(Cost a, Cost b, CostBounds bounds) {
  if (bounds.primaryCheckIsCycles) {
    return a.cycles < bounds.cycles && b.cycles < bounds.cycles ?
                              a.memory < b.memory : a.cycles < b.cycles;
  } else {
    return a.memory < bounds.memory && b.memory < bounds.memory ?
                              a.cycles < b.cycles : a.memory < b.memory;
  }
}

static Cost highestCost(std::numeric_limits<unsigned>::max(),
                        std::numeric_limits<unsigned>::max());

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
  bool isFractional;
  bool isWeightUpdate;
  unsigned getOutputWidth() const {
    if (isFractional)
      return (inputWidth * strideX + kernelSizeX - 1) - (paddingX * 2);
    else
      return inputWidth + paddingX * 2 < kernelSizeX
        ? 0
        : (inputWidth + (paddingX * 2) - kernelSizeX) / strideX + 1;
  }
  unsigned getOutputHeight() const {
    if (isFractional)
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
                    unsigned batchSize,
                    bool isFractional,
                    bool isWeightUpdate) :
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
    batchSize(batchSize),
    isFractional(isFractional),
    isWeightUpdate(isWeightUpdate) {}
};

static unsigned
getConvUnitsPerTile(const poplar::DeviceInfo &deviceInfo,
                    bool floatActivations, bool floatPartials) {
  if (floatActivations) {
    return floatPartials ? deviceInfo.fp32InFp32OutConvUnitsPerTile : 0;
  }
  return floatPartials ? deviceInfo.fp16InFp32OutConvUnitsPerTile :
                         deviceInfo.fp16InFp16OutConvUnitsPerTile;
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned strideY, unsigned strideX,
                             const poplar::DeviceInfo &deviceInfo) {
  if (getConvUnitsPerTile(deviceInfo, floatActivations, floatPartials) == 0) {
    return false;
  }
  if (floatActivations) {
    if (!deviceInfo.convInstructionsFloat) {
      return false;
    }
    if (!floatPartials) {
      return false;
    }
  }
  if (strideX >= (1 << 4))
    return false;
  return true;
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned strideY, unsigned strideX,
                             unsigned inChansPerGroup,
                             const poplar::DeviceInfo &deviceInfo) {
  if (!canUseConvolutionInstruction(floatActivations, floatPartials,
                                    strideY, strideX,
                                    deviceInfo))
    return false;
  if (deviceInfo.getWeightsPerConvUnit(floatActivations) %
      inChansPerGroup != 0) {
    return false;
  }
  // Check we can use aligned loads.
  if ((inChansPerGroup * (floatActivations ? 32 : 16)) %
      deviceInfo.dataPathWidth != 0) {
    return false;
  }
  return true;
}

static unsigned
getMaxInputRangeSize(unsigned outputRangeSize, unsigned stride,
                     unsigned kernelSize, unsigned padding,
                     unsigned numPartitions,
                     unsigned inputSize, bool contiguousAccess,
                     bool isFractional, bool isWeightUpdate) {
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
                                      inputSize, !isFractional);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    if (!isFractional) {
      inputRangeSize = (outputRangeSize - 1) * stride + 1 + (kernelSize - 1);
    } else {
      inputRangeSize = (outputRangeSize - kernelSize ) / stride + 1;
    }
    break;
  }
  if (!isFractional && !isWeightUpdate &&
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
                               unsigned outputStride,
                               unsigned numInputPointers,
                               PlannerCache * cache)
{
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

static Cost
estimateExchangeCost(const poplar::DeviceInfo &deviceInfo,
                     bool floatActivations, const ConvolutionParams &params,
                     const Plan &plan) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;

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
      getMaxInputRangeSize(tileOutWidth, params.strideX, params.kernelSizeX,
                           params.paddingX,
                           tilesPerX, params.inputWidth, true,
                           params.isFractional, params.isWeightUpdate);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.strideY, params.kernelSizeY,
                           params.paddingY,
                           tilesPerY, params.inputWidth,
                           false, params.isFractional, params.isWeightUpdate);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      params.kernelSizeY * params.kernelSizeX * tileOutDepth * tileInDepth;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileOutDepth;

  const auto activationSize = floatActivations ? 4 : 2;
  auto inputElementsBytes = numberOfInputElements * activationSize;
  auto weightBytes = numberOfWeights * activationSize;
  unsigned partialSumBytes;
  const auto partialSize = plan.floatPartials ? 4 : 2;
  if (params.isWeightUpdate) {
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
      (deviceInfo.supportsSuperTileSendReceive &&
       (tilesPerZ % tilesPerSuperTile) == 0) ? exchangeBytesPerCycle *
                                               tilesPerSuperTile :
                                               exchangeBytesPerCycle;
  const auto numCycles =
      (inputElementsBytes + inputElementBytesPerCycle - 1) /
      inputElementBytesPerCycle +
      (weightBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;
  return {numCycles, 0};
}

static unsigned
estimateConvVertexCycles(bool floatActivations,
                         const ConvolutionParams &params,
                         const Plan &plan,
                         const poplar::DeviceInfo &deviceInfo,
                         PlannerCache *cache) {
  assert(!params.isWeightUpdate);
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto verticesPerTilePerY = plan.verticesPerTilePerYAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;

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
  const auto outputStrideY = params.isFractional ? params.strideY : 1;

  if (plan.useConvolutionInstructions) {
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
                          plan.floatPartials, deviceInfo),
          deviceInfo.convUnitCoeffLoadBytesPerCycle,
          outputStrideY,
          convUnitWeightHeight,
          cache);
  }
  return outRowsPerVertex * outChansPerGroup *
         getConvPartialByDotProductCycleEstimate(
           floatActivations, inChansPerGroup, params.kernelSizeY,
           params.kernelSizeX * tileNumInGroups, tileOutWidth,
           deviceInfo.dataPathWidth, outputStrideY
         );
}

static Cost
estimateWeightUpdatePartialCalcComputeCost(const poplar::DeviceInfo &deviceInfo,
                                           bool floatActivations,
                                           const ConvolutionParams &params,
                                           const Plan &plan,
                                           PlannerCache *cache) {
  assert(params.isWeightUpdate);
  assert(!plan.useConvolutionInstructions);
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto floatPartials = plan.floatPartials;

  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroup - 1) / tilesPerInZGroup;

  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerX = plan.tilesPerXAxis;

  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;

  const auto numWorkerContexts = deviceInfo.numWorkerContexts;
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  unsigned tasks = params.kernelSizeY * params.kernelSizeX *
                   tileNumOutGroups * tileNumInGroups;
  unsigned maxTasksPerVertex =
      (tasks + numWorkerContexts - 1) / numWorkerContexts;
  std::vector<std::vector<unsigned>>
      shape(maxTasksPerVertex,
            std::vector<unsigned>(tileOutHeight, tileOutWidth));
  const auto vertexCycles =
      getWeightGradAopCycles(floatActivations, floatPartials, dataPathWidth,
                             inChansPerGroup, outChansPerGroup, shape);
  unsigned totalCycles = vertexCycles * numWorkerContexts;
  return {totalCycles, 0};
}


static unsigned estimatePartialCalcMemory(
                     bool floatActivations,
                     const ConvolutionParams &params,
                     const Plan &plan,
                     const poplar::DeviceInfo &deviceInfo) {

  unsigned vertexFields = 0;
  unsigned edgePtrsPerVertex = 0;
  unsigned tensorMemory = 0;

  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;


  const auto tilesUsedPerBatchGroup = tilesPerX
                                      * tilesPerY
                                      * tilesPerInZGroup
                                      * tilesPerZ;
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroup - 1) / tilesPerInZGroup;
  const auto verticesPerTilePerY =
      std::min(tileOutHeight, plan.verticesPerTilePerYAxis);
  const auto tileVertices = verticesPerTilePerY * tileNumOutGroups;

  bool useConvPartial1x1OutVertex = false;
  auto convUnitWeightHeight = 1U;
  if (plan.useConvolutionInstructions) {
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(floatActivations);
    assert(weightsPerConvUnit % inChansPerGroup == 0);
    convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
    if (convUnitWeightHeight != 1) {
      vertexFields += 1;
    }
  }

  useConvPartial1x1OutVertex =
      params.kernelSizeX == 1 && params.kernelSizeY == 1
      && (params.strideX == 1 && params.strideY == 1);

  if (useConvPartial1x1OutVertex) {
    vertexFields += 4;
    const auto numConvolutions = tileNumInGroups * tileOutHeight;
    edgePtrsPerVertex += 1 + 2 * numConvolutions;
  } else if (plan.useConvolutionInstructions) {
    vertexFields += 4;
    const unsigned numWeights =
          tileNumInGroups * params.kernelSizeX
          * ((params.kernelSizeY + convUnitWeightHeight - 1)
             /convUnitWeightHeight);
    const unsigned numConvolutions = numWeights * tileOutHeight;

    edgePtrsPerVertex += 2 * numWeights * convUnitWeightHeight
                         + numConvolutions
                         + numConvolutions * convUnitWeightHeight;
  } else {
    vertexFields += 6;
    edgePtrsPerVertex += 1 + 2 * tileNumInGroups * tileOutHeight;
  }

  tensorMemory += plan.tilesPerInZGroupAxis
                  * params.getOutputHeight()
                  * params.getOutputWidth()
                  * params.outputDepth
                  * (plan.floatPartials ? 4 : 2);

  const auto bytesPerVertexElem = 4;
  assert(params.batchSize % plan.batchesPerGroup == 0);
  const auto numBatchGroups = params.batchSize / plan.batchesPerGroup;
  const auto vertexMemory = tilesUsedPerBatchGroup * numBatchGroups
                            * tileVertices * vertexFields
                            * bytesPerVertexElem;

  const auto bytesPerEdgePtr = deviceInfo.inEdgePointerBytes;
  const auto edgePtrsMemory = tilesUsedPerBatchGroup * numBatchGroups
                              * edgePtrsPerVertex * tileVertices
                              * bytesPerEdgePtr;

  auto totalMemory = vertexMemory + edgePtrsMemory;
  return totalMemory;
}

static Cost
estimatePartialCalcComputeCost(const poplar::DeviceInfo &deviceInfo,
                               bool floatActivations,
                               const ConvolutionParams &params,
                               const Plan &plan,
                               PlannerCache *cache) {
  if (params.isWeightUpdate) {
    return
      estimateWeightUpdatePartialCalcComputeCost(deviceInfo, floatActivations,
                                                 params, plan, cache);
  }
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;

  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  const auto verticesPerTilePerY =
      std::min(tileOutHeight, plan.verticesPerTilePerYAxis);
  const auto tileVertices = verticesPerTilePerY * tileNumOutGroups;
  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = deviceInfo.numWorkerContexts;
  if (plan.useConvolutionInstructions) {
    numContexts = 1;
  }
  const auto vertexRuntime = estimateConvVertexCycles(floatActivations, params,
                                                      plan,
                                                      deviceInfo,
                                                      cache);
  auto verticesPerWorker = (tileVertices + numContexts - 1) /
                           numContexts;
  auto computeCycles = vertexRuntime * verticesPerWorker * numContexts;

  auto memoryEstimate = estimatePartialCalcMemory(
                              floatActivations,
                              params, plan,
                              deviceInfo);
  return {computeCycles, memoryEstimate};
}

static unsigned calcNumUsableTiles(
    unsigned numTiles,
    unsigned numBatchGroups) {
  const auto batchElemsPerTile = (numBatchGroups + numTiles - 1) / numTiles;
  const auto batchSubGroups =
      (numBatchGroups + batchElemsPerTile - 1) / batchElemsPerTile;

  return numTiles / batchSubGroups;
}


static unsigned estimateReduceMemory(const poplar::DeviceInfo &deviceInfo,
                                     const ConvolutionParams &params,
                                     const Plan &plan,
                                     unsigned numTiles) {
  unsigned vertexFields = 0;
  unsigned edgePtrsPerVertex = 0;


  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto batchesPerGroup = plan.batchesPerGroup;

  const auto numOutGroups =
    (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileOutHeight =
    (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;

  const auto tileNumOutGroups =
     (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  assert(params.batchSize % batchesPerGroup == 0);
  const auto numBatchGroups = params.batchSize / batchesPerGroup;

  vertexFields += 2;

  /* this is an estimate */
  edgePtrsPerVertex += tilesPerInZGroup * tileNumOutGroups * tileOutHeight
                       + tileNumOutGroups * tileOutHeight;


  const auto bytesPerVertexElem = 4;
  const auto vertexMemory = numTiles
                            * numBatchGroups
                            * vertexFields
                            * bytesPerVertexElem;


  /* A better estimate of reduction edge pointers would require the output
   * channel group mapping. We assume here that all X elements are
   * contiguous and require only one edge pointer
   */
  const auto bytesPerEdgePtr = deviceInfo.inEdgePointerBytes;
  const auto edgePtrsMemory = numTiles
                              * numBatchGroups
                              * edgePtrsPerVertex
                              * bytesPerEdgePtr;

  const auto totalMemory = vertexMemory + edgePtrsMemory;
  return totalMemory;
}


static Cost
estimateReduceComputeCost(const poplar::DeviceInfo &deviceInfo,
                          const ConvolutionParams &params,
                          const Plan &plan) {
  if (plan.tilesPerInZGroupAxis == 1)
    return {0, 0};

  /* The reduction is actually done on tiles in which the output
   * activations reside. Thus the output height here may be different
   * from the one the tensor uses in the reduction. The numOutputsPerTile
   * below however is approximately the same except for any rounding
   */
  unsigned numPartialSumsPerTile;
  unsigned numTiles;
  if (params.isWeightUpdate) {
    assert(plan.batchesPerGroup == 1);
    numTiles = deviceInfo.getNumTiles();
    const auto numOutputs = params.outputDepth * params.inputDepth *
                            params.kernelSizeY * params.kernelSizeX;
    const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
    numPartialSumsPerTile = numOutputsPerTile * plan.tilesPerYAxis *
                            plan.tilesPerXAxis * params.batchSize;
  } else {
    const auto numBatchGroups = params.batchSize / plan.batchesPerGroup;
    numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(), numBatchGroups);
    const auto numOutputs = params.getOutputHeight() *
                            params.getOutputWidth() *
                            params.outputDepth;
    const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
    numPartialSumsPerTile = numOutputsPerTile * plan.tilesPerInZGroupAxis;
  }
  const auto vectorWidth =
      plan.floatPartials ? deviceInfo.getFloatVectorWidth() :
                                deviceInfo.getHalfVectorWidth();
  const auto numCycles = (numPartialSumsPerTile + vectorWidth - 1) /
                          vectorWidth;
  const auto memory = estimateReduceMemory(deviceInfo,
                                           params,
                                           plan,
                                           numTiles);
  return {numCycles, memory};
}

static Cost
estimateComputeCost(const poplar::DeviceInfo &deviceInfo,
                    bool floatActivations, const ConvolutionParams &params,
                    const Plan &plan,
                    PlannerCache *cache) {
  return estimatePartialCalcComputeCost(deviceInfo, floatActivations, params,
                                        plan, cache) +
         estimateReduceComputeCost(deviceInfo, params,
                                   plan);

}

static Cost
estimatePlanCostBounded(const poplar::DeviceInfo &deviceInfo,
                             bool floatActivations,
                             const ConvolutionParams &params,
                             const Plan &plan,
                             Cost maxCost,
                             CostBounds costBounds,
                             PlannerCache *cache) {
  auto cost = estimateExchangeCost(deviceInfo,
                                   floatActivations, params, plan);
  if (!compareCost(cost, maxCost, costBounds))
    return maxCost;
  cost += estimateComputeCost(deviceInfo, floatActivations, params,
                              plan, cache);
  return compareCost(cost, maxCost, costBounds) ? cost : maxCost;
}

static Cost
estimateWeightUpdateByAmpReorderCost(const poplar::DeviceInfo &deviceInfo,
                                     bool floatActivations,
                                     const ConvolutionParams &reorderedParams) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numActs = reorderedParams.batchSize *
                       reorderedParams.inputDepth *
                       reorderedParams.inputHeight *
                       reorderedParams.inputWidth;
  const auto numDeltas = reorderedParams.kernelSizeY *
                         reorderedParams.kernelSizeX *
                         reorderedParams.inputDepth *
                         reorderedParams.outputDepth;
  const auto numWeightDeltas = reorderedParams.batchSize *
                               reorderedParams.outputDepth *
                               reorderedParams.getOutputHeight() *
                               reorderedParams.getOutputWidth();
  const auto bytesPerElement = floatActivations ? 4U : 2U;
  const auto reorderBytesPerCycle =
      std::min(deviceInfo.memcpyBytesPerCycle, bytesPerElement);
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  const auto reorderBytesPre = (numActs + numDeltas) * bytesPerElement;
  const auto reorderBytesPost = numWeightDeltas * bytesPerElement;
  const auto reorderBytesTilePre = (reorderBytesPre + numTiles - 1) / numTiles;
  const auto reorderBytesTilePost = (reorderBytesPost + numTiles - 1) /
                                    numTiles;
  unsigned cycles = 0;
  cycles += (reorderBytesTilePre + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += (reorderBytesTilePre + reorderBytesPerCycle - 1) /
            reorderBytesPerCycle;
  cycles += (reorderBytesTilePost + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += (reorderBytesTilePost + reorderBytesPerCycle - 1) /
            reorderBytesPerCycle;
  return {cycles, 0};
}

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           unsigned inChansPerGroup,
           unsigned partialChansPerGroup,
           const ConvVertexType &convVertexType,
           const ConvolutionParams &params,
           const unsigned batchesPerGroup,
           const CostBounds costBounds,
           PlannerCache *cache) {
  const auto floatActivations = convVertexType.floatActivations;
  if (convVertexType.useConvInstruction &&
      params.isWeightUpdate) {
    assert(inChansPerGroup ==
           deviceInfo.getWeightsPerConvUnit(floatActivations));
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
    const auto fieldSize = params.getOutputHeight() *
                           params.getOutputWidth() *
                           params.batchSize;
    const auto paddedFieldSize =
        ((fieldSize + fieldGroupSize - 1) / fieldGroupSize) * fieldGroupSize;
    const auto numKernelElements = params.kernelSizeY * params.kernelSizeX;
    const auto outputSize =  params.inputDepth * numKernelElements;
    const auto paddedOutputSize =
        ((outputSize + partialChansPerGroup - 1) / partialChansPerGroup)
            * partialChansPerGroup;
    auto newParams = ConvolutionParams(
                       1 /* kernelSizeY */,
                       1 /* kernelSizeX */,
                       1,
                       1,
                       paddedFieldSize,
                       params.outputDepth,
                       1,
                       0 /*paddingY*/,
                       0 /*paddingX*/,
                       paddedOutputSize,
                       1, false, false);
    Plan plan;
    Cost cost;
    std::tie(plan, cost) = choosePlan(deviceInfo,
                                      inChansPerGroup,
                                      partialChansPerGroup,
                                      convVertexType, newParams,
                                      1, costBounds, cache);
    cost += estimateWeightUpdateByAmpReorderCost(deviceInfo, floatActivations,
                                                 newParams);
    return {plan, cost};
  }
  Cost bestCost = highestCost;
  Plan bestPlan;
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
  const auto numBatchGroups = params.batchSize / batchesPerGroup;
  const auto numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(),
                                           numBatchGroups);

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
        if (convVertexType.useConvInstruction) {
          // All workers are utilized in each single supervisor vertex so
          // there is no reason to use more than the minimum number of
          // vertices.
          maxVerticesPerTilePerY = 1;
        } else {
          // The ConvPartial vertex that doesn't use the convolution
          // instruction always computes a single output row.
          minVerticesPerTilePerY = maxVerticesPerTilePerY;
        }
        for (unsigned verticesPerTilePerY = minVerticesPerTilePerY;
             verticesPerTilePerY <= maxVerticesPerTilePerY;
             ++verticesPerTilePerY) {
          Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                         verticesPerTilePerY, tilesPerInZ,
                         inChansPerGroup, partialChansPerGroup,
                         batchesPerGroup,
                         convVertexType.floatPartials,
                         convVertexType.useConvInstruction);

          auto candidateCost =
              estimatePlanCostBounded(deviceInfo, floatActivations, params,
                                      candidate, bestCost, costBounds, cache);
          if (compareCost(candidateCost, bestCost, costBounds)) {
            bestPlan = candidate;
            bestCost = candidateCost;
          }
        }
      }
    }
  }
  return {bestPlan, bestCost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            bool floatPartials,
                            const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   params.strideY, params.strideX,
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(true, floatActivations,
                                          floatPartials);
  } else if (!floatActivations && !floatPartials &&
             canUseConvolutionInstruction(false, true,
                                          params.strideY, params.strideX,
                                          deviceInfo)) {
    convVertexTypeCandidates.emplace_back(true, false, true);
  }
  convVertexTypeCandidates.emplace_back(false, floatActivations, floatPartials);
  return convVertexTypeCandidates;
}

static std::vector<ConvVertexType>
getWeightUpdateVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                                    bool floatActivations,
                                    bool floatPartials,
                                    const ConvolutionParams &params,
                                    const PlanControl &planControl) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (!planControl.forceAOPForWU) {
    if (getConvUnitsPerTile(deviceInfo, floatActivations, floatPartials) > 0) {
      convVertexTypeCandidates.emplace_back(true, floatActivations,
                                            floatPartials);
    } else if (!floatActivations && !floatPartials &&
               deviceInfo.fp16InFp32OutConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(true, false, true);
    }
  }
  convVertexTypeCandidates.emplace_back(false, floatActivations, floatPartials);
  return convVertexTypeCandidates;
}

static std::vector<unsigned>
getInChansPerGroupCandidates(const ConvolutionParams &params,
                             const ConvVertexType &convVertexType,
                             const poplar::DeviceInfo &deviceInfo) {
  std::vector<unsigned> candidates;
  for (unsigned i = 1; i <= params.inputDepth; ++i) {
    if (params.inputDepth % i != 0)
      continue;
    if (!convVertexType.floatActivations && i % 2 != 0)
      continue;
    if (convVertexType.useConvInstruction &&
        !canUseConvolutionInstruction(convVertexType.floatActivations,
                                      convVertexType.floatPartials,
                                      params.strideY, params.strideX,
                                      i, deviceInfo))
      continue;
    candidates.push_back(i);
  }
  if (!convVertexType.useConvInstruction && candidates.empty())
    candidates.push_back(params.inputDepth);
  return candidates;
}

static Cost
estimateWeightUpdateByAopReorderCost(const poplar::DeviceInfo &deviceInfo,
                                     unsigned tensorOutChansPerGroup,
                                     unsigned tensorWeightOutChansPerGroup,
                                     bool floatActivations,
                                     const ConvolutionParams &params,
                                     unsigned partialChansPerGroup) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto bytesPerElement = floatActivations ? 4U : 2U;
  const auto reorderBytesPerCycle =
      std::min(deviceInfo.memcpyBytesPerCycle, bytesPerElement);
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  auto reorderBytesPre = 0;
  if (partialChansPerGroup != tensorOutChansPerGroup) {
    // Reorder deltas.
    const auto numDeltas = params.batchSize *
                                 params.outputDepth *
                                 params.getOutputHeight() *
                                 params.getOutputWidth();
    reorderBytesPre += numDeltas * bytesPerElement;
  }
  auto reorderBytesPost = 0;
  if (partialChansPerGroup != tensorWeightOutChansPerGroup) {
    // Reorder weight deltas.
    const auto numWeightDeltas = params.kernelSizeY *
                                 params.kernelSizeX *
                                 params.inputDepth *
                                 params.outputDepth;
    reorderBytesPost += numWeightDeltas * bytesPerElement;
  }
  const auto reorderBytesTilePre = (reorderBytesPre + numTiles - 1) / numTiles;
  const auto reorderBytesTilePost = (reorderBytesPost + numTiles - 1) /
                                    numTiles;
  unsigned cycles = 0;
  cycles += (reorderBytesTilePre + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += (reorderBytesTilePre + reorderBytesPerCycle - 1) /
            reorderBytesPerCycle;
  cycles += (reorderBytesTilePost + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  cycles += (reorderBytesTilePost + reorderBytesPerCycle - 1) /
            reorderBytesPerCycle;
  return {cycles, 0};
}

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           const ConvVertexType &convVertexType,
           unsigned tensorInChansPerGroup,
           unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           const ConvolutionParams &params,
           unsigned batchesPerGroup,
           const CostBounds costBounds,
           PlannerCache *cache) {
  Plan best;
  Cost bestCost = highestCost;
  if (params.isWeightUpdate && !convVertexType.useConvInstruction) {
    // Use the existing channel grouping to avoid the need to regroup the
    // activations.
    assert(tensorInChansPerGroup != 0);
    assert(tensorOutChansPerGroup != 0);
    assert(tensorWeightOutChansPerGroup != 0);
    const auto inChansPerGroup = tensorInChansPerGroup;
    std::vector<unsigned> partialChansPerGroupCandidates = {
      tensorOutChansPerGroup,
      tensorWeightOutChansPerGroup
    };
    for (auto partialChansPerGroup : partialChansPerGroupCandidates) {
      Plan candidate;
      Cost candidateCost;
      std::tie(candidate, candidateCost) =
          choosePlan(deviceInfo, inChansPerGroup, partialChansPerGroup,
                     convVertexType, params, batchesPerGroup, costBounds,
                     cache);
      candidateCost +=
          estimateWeightUpdateByAopReorderCost(deviceInfo,
                                               tensorOutChansPerGroup,
                                               tensorWeightOutChansPerGroup,
                                               convVertexType.floatActivations,
                                               params,
                                               partialChansPerGroup);
     if (compareCost(candidateCost, bestCost, costBounds)) {
        best = candidate;
        bestCost = candidateCost;
      }
    }
    return {best, bestCost};
  }
  std::vector<unsigned> inChansPerGroupCandidates;
  if (params.isWeightUpdate) {
    assert(convVertexType.useConvInstruction);
    inChansPerGroupCandidates.push_back(
      deviceInfo.getWeightsPerConvUnit(convVertexType.floatActivations)
    );
  } else {
    assert(tensorInChansPerGroup == 0);
    assert(tensorOutChansPerGroup == 0);
    assert(tensorWeightOutChansPerGroup == 0);
    inChansPerGroupCandidates =
        getInChansPerGroupCandidates(params, convVertexType, deviceInfo);
  }
  unsigned partialChansPerGroup;
  if (convVertexType.useConvInstruction) {
    partialChansPerGroup = getNumConvUnits(convVertexType.floatActivations,
                                           convVertexType.floatPartials,
                                           deviceInfo);
  } else {
    assert(!params.isWeightUpdate);
    partialChansPerGroup = 1;
  }
  for (auto inChansPerGroup : inChansPerGroupCandidates) {
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, inChansPerGroup, partialChansPerGroup,
                   convVertexType, params, batchesPerGroup, costBounds, cache);
    if (compareCost(candidateCost, bestCost, costBounds)) {
      best = candidate;
      bestCost = candidateCost;
    }
  }
  return {best, bestCost};
}

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo, bool floatActivations,
           bool floatPartials,
           bool preferConvInstructions,
           unsigned tensorInChansPerGroup,
           unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           const ConvolutionParams &params,
           unsigned batchesPerGroup,
           CostBounds costBounds,
           PlannerCache *cache,
           const PlanControl &planControl) {
  Cost bestCost = highestCost;
  Plan bestPlan;
  const auto convVertexTypeCandidates =
      params.isWeightUpdate
        ? getWeightUpdateVertexTypeCandidates(deviceInfo, floatActivations,
                                              floatPartials, params,
                                              planControl)
        : getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                      floatPartials, params);
  for (const auto &convVertexType : convVertexTypeCandidates) {
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, convVertexType, tensorInChansPerGroup,
                   tensorOutChansPerGroup, tensorWeightOutChansPerGroup,
                   params, batchesPerGroup, costBounds, cache);
    if (candidateCost == highestCost)
      continue;
    if (preferConvInstructions &&
        !convVertexType.useConvInstruction)
      candidateCost *= 100000;
    if (compareCost(candidateCost, bestCost, costBounds)) {
      bestPlan = candidate;
      bestCost = candidateCost;
    }
  }
  return {bestPlan, bestCost};
}

static std::pair<Plan, Cost>
createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
           unsigned tensorInChansPerGroup, unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           unsigned kernelSizeY, unsigned kernelSizeX,
           unsigned strideY, unsigned strideX,
           unsigned paddingY, unsigned paddingX,
           unsigned numChannels, unsigned batchSize,
           std::string dType,
           std::string partialsType, bool isFractional,
           bool isWeightUpdate,
           const PlanControl &planControl,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlannerCache *cache) {
  validateLayerParams(inDimY, inDimX, inNumChans, kernelSizeY, kernelSizeX,
                      strideY, strideX, paddingY, paddingX,
                      numChannels, dType);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  bool preferConvInstructions =
      graph.getDevice().getDeviceType() == poplar::DeviceType::CPU;
  bool flattenXY;
  if (kernelSizeY == 1 && kernelSizeX == 1
      && strideY == 1 && strideX == 1
      && paddingY == 0 && paddingX == 0) {
    flattenXY = true;
    inDimX = inDimX * inDimY;
    inDimY = 1;
  } else {
    flattenXY = false;
  }

  const bool floatActivations = dType == "float";
  const bool floatPartials = partialsType == "float";
  Cost bestCandidateCost = highestCost;
  Plan bestCandidate;
  for (auto batchesPerGroup = 1U; batchesPerGroup <= batchSize;
       ++batchesPerGroup) {

    /* only allow integer division of batches.
     *  i.e. batchSize = batchesPerGroup * numBatchGroups
     *
     * Weight Update doesn't use batch grouping
     */

    if ((batchSize % batchesPerGroup) ||
        ((!flattenXY || isWeightUpdate) && batchesPerGroup > 1)) {
      continue;
    }

    ConvolutionParams params(kernelSizeY, kernelSizeX, strideY, strideX,
                             inNumChans, inDimX, inDimY * batchesPerGroup,
                             paddingY, paddingX, numChannels, batchSize,
                             isFractional, isWeightUpdate);
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, floatActivations,
                   floatPartials,
                   preferConvInstructions,
                   tensorInChansPerGroup,
                   tensorOutChansPerGroup,
                   tensorWeightOutChansPerGroup,
                   params,
                   batchesPerGroup,
                   costBounds,
                   cache,
                   planControl);
    if (compareCost(candidateCost, bestCandidateCost, costBounds)) {
      bestCandidateCost = candidateCost;
      bestCandidate = candidate;
    }
  }
  bestCandidate.flattenXY = flattenXY;
  return {bestCandidate, bestCandidateCost};
}

Plan Planner::
createPlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
           unsigned kernelSizeY, unsigned kernelSizeX,
           unsigned strideY, unsigned strideX,
           unsigned paddingY, unsigned paddingX,
           unsigned numChannels, unsigned batchSize,
           std::string dType, std::string partialsType,
           bool isFractional,
           const poplar::Graph &graph,
           const conv::PlanControl &planControl) {
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);

  std::tie(plan, cost) = conv::createPlan(inDimY, inDimX, inNumChans, 0, 0, 0,
                                          kernelSizeY, kernelSizeX,
                                          strideY, strideX, paddingY, paddingX,
                                          numChannels, batchSize, dType,
                                          partialsType,
                                          isFractional, false, planControl,
                                          costBounds, graph,
                                          cache.get());
  if (percentageCyclesExcessForMemOptim) {
    /* Set new bounds based on previous search */
    const double newCyclesBound =
        static_cast<double>(cost.cycles)
        * (100.0 + percentageCyclesExcessForMemOptim) / 100.0;

    CostBounds newCostBounds(static_cast<unsigned>(newCyclesBound), 0);

    std::tie(plan, cost) = conv::createPlan(inDimY, inDimX, inNumChans, 0, 0, 0,
                                            kernelSizeY, kernelSizeX,
                                            strideY, strideX,
                                            paddingY, paddingX,
                                            numChannels, batchSize, dType,
                                            partialsType,
                                            isFractional, false, planControl,
                                            newCostBounds, graph,
                                            cache.get());

  }
  return plan;
}

Plan Planner::
createWeightUpdatePlan(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                       unsigned actChansPerGroup, unsigned deltasChansPerGroup,
                       unsigned weightOutChansPerGroup,
                       unsigned kernelSizeY, unsigned kernelSizeX,
                       unsigned strideY, unsigned strideX,
                       unsigned paddingY, unsigned paddingX,
                       unsigned numChannels, unsigned batchSize,
                       std::string dType, std::string partialsType,
                       bool isFractional,
                       const poplar::Graph &graph,
                       const PlanControl &planControl) {
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);

  std::tie(plan, cost) = conv::createPlan(inDimY, inDimX, inNumChans,
                                          actChansPerGroup, deltasChansPerGroup,
                                          weightOutChansPerGroup,
                                          kernelSizeY, kernelSizeX,
                                          strideY, strideX, paddingY, paddingX,
                                          numChannels, batchSize, dType,
                                          partialsType, isFractional,
                                          true, planControl,
                                          costBounds, graph, cache.get());
  return plan;
}

}
