#include "ConvPlan.hpp"
#include "popconv/Convolution.hpp"
#include "popstd/exceptions.hpp"
#include "poplar/Graph.hpp"
#include "popconv/ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "util/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "util/Compiler.hpp"
#include <cassert>
#include <map>
#include <tuple>
#include <iostream>
#include <popsolver/Model.hpp>

namespace popconv {

const char *asString(const WeightUpdateMethod &method) {
  switch (method) {
  case WeightUpdateMethod::AOP: return "aop";
  case WeightUpdateMethod::AMP: return "amp";
  case WeightUpdateMethod::AUTO: return "auto";
  }
  POPLIB_UNREACHABLE();
}

std::ostream &
operator<<(std::ostream &os, const WeightUpdateMethod &method) {
  return os << asString(method);
}

std::istream &operator>>(std::istream &is, WeightUpdateMethod &method) {
  std::string token;
  is >> token;
  if (token == "aop")
    method = WeightUpdateMethod::AOP;
  else if (token == "amp")
    method = WeightUpdateMethod::AMP;
  else if (token == "auto")
    method = WeightUpdateMethod::AUTO;
  else
    throw popstd::poplib_error(
      "Unknown weight update method <" + token + ">");
  return is;

}

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

std::ostream& operator<<(std::ostream &os, const Plan &p)
{
  os << "  Plan: TilesPerAxisXYZ         " << p.tilesPerXAxis << "*"
                                           << p.tilesPerYAxis << "*"
                                           << p.tilesPerZAxis << "="
    << p.tilesPerXAxis * p.tilesPerYAxis * p.tilesPerZAxis << "\n"
    << "        tilesPerKernelYAxis     " << p.tilesPerKernelYAxis << "\n"
    << "        tilesPerInZGroupAxis    " << p.tilesPerInZGroupAxis << "\n"
    << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
    << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
    << "        batchesPerGroup         " << p.batchesPerGroup << "\n"
    << "        useConvInstructions     " << p.useConvolutionInstructions
    << "\n"
    << "        flattenXY               " << p.flattenXY << "\n"
    << "        deltasAsCoefficents     "
    << (p.ampWUMethod == Plan::DELTAS_AS_COEFFICENTS) << "\n";
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
                               unsigned numInputPointers);

static unsigned
estimateConvPartialHorizontalMacCycles(unsigned tileNumInGroups,
                                       unsigned numOutRows,
                                       unsigned tileOutWidth,
                                       unsigned outputStrideX,
                                       unsigned tileKernelHeight,
                                       unsigned tileKernelWidth,
                                       unsigned numWorkers,
                                       bool floatActivations,
                                       unsigned inChansPerGroup,
                                       unsigned dataPathWidth);

static unsigned
estimateConvPartialHorizontalMacCycles(unsigned tileNumInGroups,
                                       unsigned numOutRows,
                                       unsigned tileOutWidth,
                                       unsigned outputStrideX,
                                       unsigned tileKernelHeight,
                                       unsigned tileKernelWidth,
                                       unsigned numWorkers,
                                       bool floatActivations,
                                       unsigned inChansPerGroup,
                                       unsigned dataPathWidth);

class PlanningCacheImpl {
public:
  decltype(memoize(getConvPartialnx1CycleEstimate))
    mGetConvPartialnx1CycleEstimate;
  decltype(memoize(estimateConvPartialHorizontalMacCycles))
    mEstimateConvPartialHorizontalMacCycles;
  PlanningCacheImpl() :
    mGetConvPartialnx1CycleEstimate(
      memoize(getConvPartialnx1CycleEstimate)
    ),
    mEstimateConvPartialHorizontalMacCycles(
      memoize(estimateConvPartialHorizontalMacCycles)
    ) {}
  struct Key {
    ConvParams convParams;
    ConvOptions options;
    bool isWeightUpdate;
    unsigned actChansPerGroup;
    unsigned deltaChansPerGroup;
    Key(ConvParams params,
        ConvOptions options,
        bool isWeightUpdate,
        unsigned actChansPerGroup,
        unsigned deltaChansPerGroup) :
      convParams(std::move(params)),
      options(std::move(options)),
      isWeightUpdate(isWeightUpdate),
      actChansPerGroup(isWeightUpdate ? actChansPerGroup : 0),
      deltaChansPerGroup(isWeightUpdate ? deltaChansPerGroup : 0) {}
    bool operator<(const Key &other) const {
      return std::tie(convParams, isWeightUpdate, options) <
               std::tie(other.convParams, other.isWeightUpdate, other.options);
    }
  };
  std::map<Key, std::unique_ptr<Plan>> plans;
};

PlanningCache::PlanningCache() {
  impl = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl());
}

PlanningCache::~PlanningCache() = default;

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
  unsigned cycles;
  unsigned memory;
  bool primaryCheckIsCycles;
};


bool compareCost(Cost a, Cost b, CostBounds bounds) {
  bool aCyclesOutOfBounds = a.cycles >= bounds.cycles;
  bool bCyclesOutOfBounds = b.cycles >= bounds.cycles;
  bool aMemoryOutOfBounds = a.memory >= bounds.memory;
  bool bMemoryOutOfBounds = b.memory >= bounds.memory;
  if (bounds.primaryCheckIsCycles) {
    return std::tie(aCyclesOutOfBounds, aMemoryOutOfBounds, a.cycles,
                    a.memory) <
           std::tie(bCyclesOutOfBounds, bMemoryOutOfBounds, b.cycles,
                    b.memory);
  }
  return std::tie(aMemoryOutOfBounds, aCyclesOutOfBounds, a.memory, a.cycles) <
         std::tie(bMemoryOutOfBounds, bCyclesOutOfBounds, b.memory, b.cycles);
}

static Cost highestCost(std::numeric_limits<unsigned>::max(),
                        std::numeric_limits<unsigned>::max());

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
getMaxInputRangeSize(unsigned outputRangeSize, unsigned dim,
                     const ConvParams &params,
                     unsigned tileKernelSize,
                     unsigned numPartitions,
                     bool contiguousAccess,
                     bool isWeightUpdate)  {
  if (outputRangeSize == 0)
    return 0;
  unsigned inputRangeSize;
  const auto kernelSize = params.kernelShape[dim];
  const auto stride = params.stride[dim];
  // If the number of partitions is small the input range is guaranteed
  // to contain padding.
  switch (numPartitions) {
  case 1:
  case 2:
    {
      auto inputRange = getInputRange(dim, {0, outputRangeSize},
                                      {kernelSize - tileKernelSize,
                                       kernelSize},
                                      params);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    if (!params.isFractional) {
      inputRangeSize = (outputRangeSize - 1) * stride + 1 +
                       (tileKernelSize - 1);
    } else {
      inputRangeSize = (outputRangeSize - tileKernelSize) / stride + 1;
    }
    break;
  }
  if (!params.isFractional && !isWeightUpdate &&
      !contiguousAccess && tileKernelSize == 1 && stride > 1) {
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
                               unsigned numInputPointers)
{
  unsigned numInputEdges = 0;
  unsigned numOutputEdges = 0;
  const auto numWorkerContexts = 6;
  std::vector<std::vector<PartialRow>> partition =
      partitionConvPartialByWorker(outputHeight, outputWidth,
                                   numWorkerContexts, outputStride);
  std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
  for (unsigned i = 0; i != passesPerOutput; ++i) {
    convSizesByWeightAndWorker.emplace_back();
    convSizesByWeightAndWorker.back().reserve(partition.size());
    for (const auto &entry : partition) {
      convSizesByWeightAndWorker.back().emplace_back();
      convSizesByWeightAndWorker.back().back().reserve(entry.size());
      numInputEdges += numInputPointers * entry.size();
      numOutputEdges += entry.size();
      for (const auto &partialRow : entry) {
        auto convSize = (partialRow.end - partialRow.begin) / outputStride;
        convSizesByWeightAndWorker.back().back().push_back(convSize);
      }
    }
  }

  auto numEdges = convSizesByWeightAndWorker.size()
                  + numInputEdges
                  + numOutputEdges;
  return getConvPartialnx1SupervisorCycleEstimate(
                convSizesByWeightAndWorker,
                convUnitPipelineDepth,
                numConvUnitsPerTile,
                convUnitCoeffLoadBytesPerCycle,
                numInputPointers,
                useDeltaEdgesForConvPartials(numEdges));
}

static unsigned
estimateExchangeCycles(const poplar::DeviceInfo &deviceInfo,
                       bool floatActivations,
                       const ConvParams &params, bool isWeightUpdate,
                       const Plan &plan) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto kernelSizeX = params.kernelShape[1];
  const auto tileKernelHeight = (kernelSizeX + tilesPerKernelYAxis - 1) /
                                tilesPerKernelYAxis;
  const auto tileKernelWidth =kernelSizeX;
  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto outputDepth = params.kernelShape[2];
  const auto numOutGroups =
      (outputDepth + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileOutDepth = tileNumOutGroups * partialChansPerGroup;
  const auto inputDepth = params.inputShape[3];
  const auto numInGroups =
      (inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;
  const auto tileInDepth = tileNumInGroups * inChansPerGroup;
  const auto tileInWidth =
      getMaxInputRangeSize(tileOutWidth, 1, params,
                           tileKernelWidth, tilesPerX,
                           true, isWeightUpdate);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, 0, params,
                           tileKernelHeight, tilesPerY,
                           false, isWeightUpdate);
  const auto numberOfInputElements = tileInWidth * tileInHeight * tileInDepth;
  const auto numberOfWeights =
      tileKernelHeight * tileKernelWidth * tileOutDepth * tileInDepth;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileOutDepth;

  const auto activationSize = floatActivations ? 4 : 2;
  auto inputElementsBytes = numberOfInputElements * activationSize;
  auto weightBytes = numberOfWeights * activationSize;
  unsigned partialSumBytes;
  const auto partialSize = plan.floatPartials ? 4 : 2;
  if (isWeightUpdate) {
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
  return numCycles;
}

static unsigned
estimateWeightUpdatePartialCalcCycles(const poplar::DeviceInfo &deviceInfo,
                                      bool floatActivations,
                                      const ConvParams &params,
                                      const Plan &plan) {
  assert(params.isWeightUpdate);
  assert(!plan.useConvolutionInstructions);
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto floatPartials = plan.floatPartials;
  const auto outputDepth = params.kernelShape[2];
  const auto inputDepth = params.inputShape[3];
  const auto numOutGroups =
      (outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;

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
  const auto tileKernelHeight =
      (params.kernelShape[0] + tilesPerKernelYAxis - 1) / tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelShape[1];
  unsigned tasks = tileKernelHeight * tileKernelWidth *
                   tileNumOutGroups * tileNumInGroups;
  unsigned maxTasksPerVertex =
      (tasks + numWorkerContexts - 1) / numWorkerContexts;
  std::vector<std::vector<unsigned>>
      shape(maxTasksPerVertex,
            std::vector<unsigned>(tileOutHeight, tileOutWidth));
  /* AOP edge type selection */
  const auto numEdges = maxTasksPerVertex * (2 * tileOutHeight + 1);
  const auto useDeltasForEdges = useDeltaEdgesForWeightGradAop(numEdges);
  const auto numAopAccumulators =
              floatActivations ? deviceInfo.fp32NumAopAccumulators :
                                 deviceInfo.fp16NumAopAccumulators;
  const auto vertexCycles =
      getWeightGradAopCycles(floatActivations, floatPartials, dataPathWidth,
                             inChansPerGroup, outChansPerGroup, shape,
                             useDeltasForEdges, numAopAccumulators);
  unsigned totalCycles = vertexCycles * numWorkerContexts;
  return totalCycles;
}

static unsigned
estimateWeightUpdatePartialCalcMemory(const poplar::DeviceInfo &deviceInfo,
                                      bool floatActivations,
                                      const ConvParams &params,
                                      const Plan &plan) {
  return 0;
}

static unsigned
estimatePartialCalcMemory(const poplar::DeviceInfo &deviceInfo,
                          bool floatActivations,
                          const ConvParams &params, bool isWeightUpdate,
                          const Plan &plan) {
  if (isWeightUpdate) {
    return estimateWeightUpdatePartialCalcMemory(deviceInfo, floatActivations,
                                                 params, plan);
  }

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
      (params.getOutputDepth() + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (params.getInputDepth() + (inChansPerGroup - 1)) / inChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroup - 1) / tilesPerInZGroup;
  // TODO the plan no longer uses verticesPerTilePerY -
  // estimatePartialCalcMemory() needs updating to reflect this.
  const auto verticesPerTilePerY = 1;
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
      params.kernelShape[0] == 1 && params.kernelShape[1] == 1
      && (params.stride[0] == 1 && params.stride[1] == 1);

  if (useConvPartial1x1OutVertex) {
    vertexFields += 4;
    const auto numConvolutions = tileNumInGroups * tileOutHeight;
    edgePtrsPerVertex += 1 + 2 * numConvolutions;
  } else if (plan.useConvolutionInstructions) {
    vertexFields += 4;
    const unsigned numWeights =
          tileNumInGroups * params.kernelShape[1]
          * ((params.kernelShape[0] + convUnitWeightHeight - 1)
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
                  * params.getOutputDepth()
                  * (plan.floatPartials ? 4 : 2);

  const auto bytesPerVertexElem = 4;
  assert(params.getBatchSize() % plan.batchesPerGroup == 0);
  const auto numBatchGroups = params.getBatchSize() / plan.batchesPerGroup;
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

static unsigned
estimateConvPartialHorizontalMacCycles(unsigned tileNumInGroups,
                                       unsigned numOutRows,
                                       unsigned tileOutWidth,
                                       unsigned outputStrideX,
                                       unsigned tileKernelHeight,
                                       unsigned tileKernelWidth,
                                       unsigned numWorkers,
                                       bool floatActivations,
                                       unsigned inChansPerGroup,
                                       unsigned dataPathWidth) {
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numOutRows);
  unsigned numPartRows = numOutRows * rowSplitFactor;
  const auto maxPartRows = (numPartRows + numWorkers - 1) / numWorkers;
  const auto workerWholeRows = maxPartRows / rowSplitFactor;
  const auto workerPartRows = maxPartRows % rowSplitFactor;
  std::vector<unsigned> convSizes;
  const auto wholeRowConvSize =
      (tileOutWidth + outputStrideX - 1) / outputStrideX;
  convSizes.resize(workerWholeRows * tileKernelWidth * tileKernelHeight *
                   tileNumInGroups, wholeRowConvSize);
  if (workerPartRows > 0) {
    convSizes.resize(convSizes.size() + tileKernelWidth * tileKernelHeight *
                     tileNumInGroups,
                     (wholeRowConvSize * workerPartRows) / rowSplitFactor);
  }
  return getConvPartialHorizontalMacCycleEstimate(
    floatActivations,
    inChansPerGroup,
    convSizes,
    dataPathWidth
  );
}

static unsigned
estimatePartialCalcCycles(const poplar::DeviceInfo &deviceInfo,
                          bool floatActivations,
                          const ConvParams &params, bool isWeightUpdate,
                          const Plan &plan,
                          PlanningCacheImpl *cache) {
  if (isWeightUpdate) {
    return
      estimateWeightUpdatePartialCalcCycles(deviceInfo, floatActivations,
                                            params, plan);
  }
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;

  const auto tileOutWidth =
      (params.getOutputWidth() + tilesPerX - 1) / tilesPerX;
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto numOutGroups =
      (params.getOutputDepth() + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = deviceInfo.numWorkerContexts;
  if (plan.useConvolutionInstructions) {
    numContexts = 1;
  }
  const auto numInGroups =
      (params.getInputDepth() + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto outputStrideY = params.isFractional ? params.stride[0] : 1;

  const auto tileKernelHeight =
      (params.kernelShape[0] + tilesPerKernelYAxis - 1) /
          tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelShape[1];
  unsigned computeCycles;
  if (plan.useConvolutionInstructions) {
    assert(deviceInfo.getWeightsPerConvUnit(floatActivations) %
           inChansPerGroup == 0);
    const auto convUnitWeightHeight =
        deviceInfo.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
    const auto passesPerFilter =
        tileKernelWidth *
        (tileKernelHeight + convUnitWeightHeight - 1) / convUnitWeightHeight;
    const auto numConvUnits = getNumConvUnits(floatActivations,
                                              plan.floatPartials,
                                              deviceInfo);
    assert(outChansPerGroup % numConvUnits == 0);
    const auto passesPerOutputGroup = outChansPerGroup / numConvUnits;
    const auto passesPerOutput = passesPerFilter * passesPerOutputGroup *
                                 tileNumInGroups;
    computeCycles =
        tileNumOutGroups *
        cache->mGetConvPartialnx1CycleEstimate(
          passesPerOutput, tileOutHeight, tileOutWidth,
          deviceInfo.convUnitPipelineDepth,
          getNumConvUnits(floatActivations, plan.floatPartials, deviceInfo),
          deviceInfo.convUnitCoeffLoadBytesPerCycle, outputStrideY,
          convUnitWeightHeight);
  } else {
    const auto outputStrideX = params.isFractional ? params.stride[1]
                                                       : 1;
    const auto outputStrideY = params.isFractional ? params.stride[0]
                                                       : 1;
    const auto numOutRows = tileNumOutGroups *
                            (tileOutHeight + outputStrideY - 1) / outputStrideY;
    auto vertexRuntime =
        cache->mEstimateConvPartialHorizontalMacCycles(
          tileNumInGroups,
          numOutRows,
          tileOutWidth,
          outputStrideX,
          tileKernelHeight,
          tileKernelWidth,
          deviceInfo.numWorkerContexts,
          floatActivations,
          inChansPerGroup,
          deviceInfo.dataPathWidth);
    computeCycles = vertexRuntime * numContexts;
  }
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

static unsigned estimateReduceMemory(const poplar::DeviceInfo &deviceInfo,
                                     const ConvParams &params,
                                     bool isWeightUpdate,
                                     const Plan &plan) {
  unsigned vertexFields = 0;
  unsigned edgePtrsPerVertex = 0;

  unsigned numTiles;
  if (isWeightUpdate) {
    assert(plan.batchesPerGroup == 1);
    numTiles = deviceInfo.getNumTiles();
  } else {
    const auto numBatchGroups =
        params.getBatchSize() / plan.batchesPerGroup;
    numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(), numBatchGroups);
  }

  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto batchesPerGroup = plan.batchesPerGroup;

  const auto numOutGroups =
    (params.getOutputDepth() + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileOutHeight =
    (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;

  const auto tileNumOutGroups =
     (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  assert(params.getBatchSize() % batchesPerGroup == 0);
  const auto numBatchGroups = params.getBatchSize() / batchesPerGroup;

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


static unsigned
estimateReduceCycles(const poplar::DeviceInfo &deviceInfo,
                          const ConvParams &params, bool isWeightUpdate,
                          const Plan &plan) {
  if (plan.tilesPerInZGroupAxis == 1 &&
      plan.tilesPerKernelYAxis == 1)
    return 0;

  /* The reduction is actually done on tiles in which the output
   * activations reside. Thus the output height here may be different
   * from the one the tensor uses in the reduction. The numOutputsPerTile
   * below however is approximately the same except for any rounding
   */
  unsigned numPartialSumsPerTile;
  unsigned numTiles;
  if (isWeightUpdate) {
    assert(plan.batchesPerGroup == 1);
    numTiles = deviceInfo.getNumTiles();
    const auto numOutputs = params.getOutputDepth() *
                            params.getInputDepth() *
                            params.kernelShape[0] *
                            params.kernelShape[1];
    const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
    numPartialSumsPerTile = numOutputsPerTile * plan.tilesPerYAxis *
                            plan.tilesPerXAxis * params.getBatchSize();
  } else {
    const auto numBatchGroups =
        params.getBatchSize() / plan.batchesPerGroup;
    numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(), numBatchGroups);
    const auto numOutputs = params.getOutputHeight() *
                            params.getOutputWidth() *
                            params.getOutputDepth();
    const auto numOutputsPerTile = (numOutputs + numTiles - 1) / numTiles;
    numPartialSumsPerTile = numOutputsPerTile * plan.tilesPerInZGroupAxis *
                            plan.tilesPerKernelYAxis;
  }
  const auto vectorWidth =
      plan.floatPartials ? deviceInfo.getFloatVectorWidth() :
                                deviceInfo.getHalfVectorWidth();
  const auto numCycles = (numPartialSumsPerTile + vectorWidth - 1) /
                          vectorWidth;
  return numCycles;
}

static Cost
estimateWeightUpdateByAmpReorderCost(const poplar::DeviceInfo &deviceInfo,
                                     bool floatActivations,
                                     const ConvParams &reorderedParams,
                                     const Plan &plan) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numInElements = reorderedParams.getBatchSize() *
                             reorderedParams.getInputDepth() *
                             reorderedParams.getInputHeight() *
                             reorderedParams.getInputWidth();
  const auto numOutElements = reorderedParams.getBatchSize() *
                              reorderedParams.getOutputDepth() *
                              reorderedParams.getOutputHeight() *
                              reorderedParams.getOutputWidth();
  const auto numCoefficients = reorderedParams.kernelShape[0] *
                               reorderedParams.kernelShape[1] *
                               reorderedParams.getInputDepth() *
                               reorderedParams.getOutputDepth();
  const auto bytesPerElement = floatActivations ? 4U : 2U;
  const auto reorderBytesPerCycle =
      std::min(deviceInfo.memcpyBytesPerCycle, bytesPerElement);
  unsigned weightDeltaReorderBytesPerCycle;
  switch (plan.ampWUMethod) {
  case Plan::ACTIVATIONS_AS_COEFFICENTS:
    weightDeltaReorderBytesPerCycle =
        std::min(deviceInfo.memcpyBytesPerCycle,
                 plan.partialChansPerGroup * bytesPerElement);
    break;
  case Plan::DELTAS_AS_COEFFICENTS:
    weightDeltaReorderBytesPerCycle = reorderBytesPerCycle;
    break;
  }
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  const auto reorderBytesPre =
      (numInElements + numCoefficients) * bytesPerElement;
  const auto reorderBytesPost = numOutElements * bytesPerElement;
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
  cycles += (reorderBytesTilePost + weightDeltaReorderBytesPerCycle - 1) /
            weightDeltaReorderBytesPerCycle;
  return {cycles, 0};
}

ConvParams
weightUpdateByAmpTransformParams(const ConvParams &params,
                                 const poplar::DeviceInfo &deviceInfo,
                                 const Plan &plan) {
  bool floatActivations = params.dType == "float";
  ConvParams newParams;
  unsigned expandedFieldWidth;
  unsigned expandedActivationsHeight;
  unsigned expandedDeltasHeight;
  unsigned expandedActivationsPaddingYLower;
  unsigned expandedActivationsPaddingYUpper;
  unsigned expandedInputDepth;
  unsigned expandedDeltasUpsampleFactorY;
  if (plan.flattenXY) {
    expandedFieldWidth =
       params.getBatchSize() * params.getOutputHeight() *
                                   params.getOutputWidth();
    expandedActivationsHeight = 1;
    expandedDeltasHeight = 1;
    expandedActivationsPaddingYLower = 0;
    expandedActivationsPaddingYUpper = 0;
    expandedInputDepth =
        params.getInputDepth() * params.kernelShape[0] *
                                 params.kernelShape[1];
    expandedDeltasUpsampleFactorY = 1;
  } else {
    expandedFieldWidth = params.getBatchSize() *
                         params.getOutputWidth();
    expandedActivationsHeight = params.getInputHeight();
    expandedDeltasHeight = params.getOutputHeight();
    expandedActivationsPaddingYLower = params.paddingLower[0];
    expandedActivationsPaddingYUpper = params.paddingUpper[0];
    expandedInputDepth =
        params.getInputDepth() * params.kernelShape[1];
    expandedDeltasUpsampleFactorY = params.stride[0];
  }
  const auto fieldGroupSize =
      deviceInfo.getWeightsPerConvUnit(floatActivations);
  const auto paddedFieldWidth =
      ((expandedFieldWidth + fieldGroupSize - 1) / fieldGroupSize) *
      fieldGroupSize;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  switch (plan.ampWUMethod) {
  case Plan::DELTAS_AS_COEFFICENTS:
    {

      // weight update x-axis: fwd in chans
      // weight update y-axis: fwd y-axis
      // weight update in chans: fwd x-axis
      // weight update out chans: fwd out chans
      const auto paddedOutputDepth =
          ((params.getOutputDepth() + partialChansPerGroup - 1) /
           partialChansPerGroup) * partialChansPerGroup;
      newParams = ConvParams(params.dType,
                    {1, expandedActivationsHeight,  expandedInputDepth,
                     paddedFieldWidth}, /* inputShape */
                    {expandedDeltasHeight, 1,
                     paddedOutputDepth,
                     paddedFieldWidth }, /* kernelShape */
                    {1, 1}, /* stride */
                    {expandedActivationsPaddingYLower, 0},
                    {expandedActivationsPaddingYUpper, 0},
                    false);
    }
    break;
  case Plan::ACTIVATIONS_AS_COEFFICENTS:
    {
      // weight update x-axis: fwd out chans
      // weight update y-axis: fwd y-axis
      // weight update in chans: fwd x-axis
      // weight update out chans: fwd in chans
      const auto isFractional = expandedDeltasUpsampleFactorY > 1;
      const auto paddedExpandedInputDepth =
          ((expandedInputDepth + partialChansPerGroup - 1) /
           partialChansPerGroup) * partialChansPerGroup;
      newParams = ConvParams(params.dType,
                    {1, expandedDeltasHeight,
                     params.getOutputDepth(),
                     paddedFieldWidth}, // inputShape
                    {expandedActivationsHeight +
                     expandedActivationsPaddingYLower +
                     expandedActivationsPaddingYUpper,
                     1,
                     paddedExpandedInputDepth,
                     paddedFieldWidth}, // kernelShape
                     {expandedDeltasUpsampleFactorY, 1}, // stride,
                     {0, 0}, // paddingLower
                     {0, 0}, // paddingUpper,
                     isFractional);
    }
    break;
  }
  return newParams;
}

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           unsigned inChansPerGroup,
           unsigned partialChansPerGroup,
           const ConvVertexType &convVertexType,
           const ConvParams &params, bool isWeightUpdate,
           const unsigned batchesPerGroup,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  const auto floatActivations = convVertexType.floatActivations;
  if (convVertexType.useConvInstruction && isWeightUpdate) {
    assert(inChansPerGroup ==
           deviceInfo.getWeightsPerConvUnit(floatActivations));
    // Implementing weight update directly as a convolution is typically
    // inefficient since the height and width of the output is small (the size
    // of the kernel). Instead we rearrange the activations and deltas so the
    // the amp unit can accumulate across the x-axis instead of over channels.
    // We choose a partition for the weight update phase by populating
    // ConvolutionParams struct with the dimensions of the transformed
    // convolution and call ourselves recursively with these parameters.
    Cost bestCost = highestCost;
    Plan bestPlan;
    // To ensure the activations we load in the inner loop are contiguous and
    // aligned we expand the x-axis by taking the activations that are
    // multiplied by each column of the weights and turning them into
    // different input channels. If flattenXY is true we also expand the
    // y-axis in the same way. Expanding the y axis may be desirable if it
    // means less padding is required when the field size is rounded up to
    // a multiple of the weights per convolutional unit.
    for (bool flattenXY : {false, true}) {
      for (Plan::AmpWUMethod method : {Plan::DELTAS_AS_COEFFICENTS,
                                       Plan::ACTIVATIONS_AS_COEFFICENTS}) {
        Plan plan;
        plan.flattenXY = flattenXY;
        plan.ampWUMethod = method;
        plan.partialChansPerGroup = partialChansPerGroup;
        // There is currently no support for dilated convolutions.
        // TODO add support for this.
        if (!plan.flattenXY && params.stride[0] != 1) {
          continue;
        }
        auto newParams =
            weightUpdateByAmpTransformParams(params, deviceInfo, plan);
        Cost cost;
        std::tie(plan, cost) = choosePlan(deviceInfo,
                                          inChansPerGroup,
                                          partialChansPerGroup,
                                          convVertexType, newParams, false,
                                          1, costBounds, cache,
                                          options);
        plan.flattenXY = flattenXY;
        plan.ampWUMethod = method;
        plan.partialChansPerGroup = partialChansPerGroup;
        cost += estimateWeightUpdateByAmpReorderCost(deviceInfo,
                                                     floatActivations,
                                                     newParams,
                                                     plan);
        if (compareCost(cost, bestCost, costBounds)) {
          bestPlan = plan;
          bestCost = cost;
        }
      }
    }
    return {bestPlan, bestCost};
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
  using namespace popsolver;
  Model m;
  const auto numBatchGroups = params.getBatchSize() / batchesPerGroup;
  const auto numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(),
                                           numBatchGroups);
  const auto tilesPerX = m.addVariable(1, params.getOutputWidth());
  const auto tilesPerY = m.addVariable(1, params.getOutputHeight());
  const auto outChanGroups =
      (params.getOutputDepth() + partialChansPerGroup - 1) /
           partialChansPerGroup;
  const auto tilesPerZ = m.addVariable(1, outChanGroups);
  const auto tilesPerKernelY = m.addVariable(1, params.kernelShape[0]);
  const auto inChanGroups = (params.getInputDepth() + inChansPerGroup - 1) /
                             inChansPerGroup;
  const auto tilesPerInZ = m.addVariable(1, inChanGroups);
  const auto usedTiles = m.product({tilesPerX, tilesPerY, tilesPerZ,
                                    tilesPerKernelY, tilesPerInZ});
  m.lessOrEqual(usedTiles, numTiles);
  const auto exchangeCycles =
      m.call({tilesPerX, tilesPerY, tilesPerZ, tilesPerKernelY, tilesPerInZ},
             [&](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerZ = values[2];
    const auto tilesPerKernelY = values[3];
    const auto tilesPerInZ = values[4];
    Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ,
                   inChansPerGroup, partialChansPerGroup,
                   batchesPerGroup,
                   convVertexType.floatPartials,
                   convVertexType.useConvInstruction);
    return estimateExchangeCycles(deviceInfo, floatActivations, params,
                                  isWeightUpdate,
                                  candidate);
  });
  const auto partialCalcCycles =
      m.call({tilesPerX, tilesPerY, tilesPerZ, tilesPerKernelY, tilesPerInZ},
             [&](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerZ = values[2];
    const auto tilesPerKernelY = values[3];
    const auto tilesPerInZ = values[4];
    Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ,
                   inChansPerGroup, partialChansPerGroup,
                   batchesPerGroup,
                   convVertexType.floatPartials,
                   convVertexType.useConvInstruction);
    return estimatePartialCalcCycles(deviceInfo, floatActivations, params,
                                     isWeightUpdate, candidate, cache);
  });
  const auto reduceCycles =
      m.call({tilesPerX, tilesPerY, tilesPerZ, tilesPerKernelY, tilesPerInZ},
             [&](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerZ = values[2];
    const auto tilesPerKernelY = values[3];
    const auto tilesPerInZ = values[4];
    Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ,
                   inChansPerGroup, partialChansPerGroup,
                   batchesPerGroup,
                   convVertexType.floatPartials,
                   convVertexType.useConvInstruction);
    return estimateReduceCycles(deviceInfo, params, isWeightUpdate, candidate);
  });
  const auto cycles = m.sum({exchangeCycles, partialCalcCycles, reduceCycles});
  const auto partialCalcMemory =
      m.call({tilesPerX, tilesPerY, tilesPerZ, tilesPerKernelY, tilesPerInZ},
             [&](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerZ = values[2];
    const auto tilesPerKernelY = values[3];
    const auto tilesPerInZ = values[4];
    Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ,
                   inChansPerGroup, partialChansPerGroup,
                   batchesPerGroup,
                   convVertexType.floatPartials,
                   convVertexType.useConvInstruction);
    return estimatePartialCalcMemory(deviceInfo, floatActivations, params,
                                     isWeightUpdate, candidate);
  });
  const auto reduceMemory =
      m.call({tilesPerX, tilesPerY, tilesPerZ, tilesPerKernelY, tilesPerInZ},
             [&](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerZ = values[2];
    const auto tilesPerKernelY = values[3];
    const auto tilesPerInZ = values[4];
    Plan candidate(tilesPerX, tilesPerY, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ,
                   inChansPerGroup, partialChansPerGroup,
                   batchesPerGroup,
                   convVertexType.floatPartials,
                   convVertexType.useConvInstruction);
    return estimateReduceMemory(deviceInfo, params, isWeightUpdate, candidate);
  });
  const auto memory = m.sum({partialCalcMemory, reduceMemory});
  if (costBounds.cycles > 0) {
    m.lessOrEqual(cycles, costBounds.cycles);
  }
  if (costBounds.memory > 0) {
    m.lessOrEqual(memory, costBounds.memory);
  }
  Solution s;
  try {
    if (costBounds.primaryCheckIsCycles) {
      s = m.minimize({cycles, memory});
    } else {
      s = m.minimize({memory, cycles});
    }
  } catch (NoSolution) {
    return {Plan(), highestCost};
  }
  Plan bestPlan(s[tilesPerX], s[tilesPerY], s[tilesPerZ],
                s[tilesPerKernelY], s[tilesPerInZ],
                inChansPerGroup, partialChansPerGroup,
                batchesPerGroup,
                convVertexType.floatPartials,
                convVertexType.useConvInstruction);
  Cost bestCost = {s[cycles], s[memory]};
  return {bestPlan, bestCost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            bool floatPartials,
                            const ConvParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  // We limit the use of the convolution instruction to cases where the number
  // of output channels is a multiple of the output channel grouping that would
  // be used.
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   params.stride[0], params.stride[1],
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(true, floatActivations,
                                          floatPartials);
  } else if (!floatActivations && !floatPartials &&
             canUseConvolutionInstruction(false, true,
                                          params.stride[0], params.stride[1],
                                          deviceInfo) &&
             params.getOutputDepth() %
             deviceInfo.fp16InFp32OutConvUnitsPerTile == 0) {
    convVertexTypeCandidates.emplace_back(true, false, true);
  }
  convVertexTypeCandidates.emplace_back(false, floatActivations, floatPartials);
  return convVertexTypeCandidates;
}

static std::vector<ConvVertexType>
getWeightUpdateVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                                    bool floatActivations,
                                    bool floatPartials,
                                    const ConvOptions &options) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (options.weightUpdateMethod == WeightUpdateMethod::AMP ||
      options.weightUpdateMethod == WeightUpdateMethod::AUTO) {
    if (getConvUnitsPerTile(deviceInfo, floatActivations, floatPartials) > 0) {
      convVertexTypeCandidates.emplace_back(true, floatActivations,
                                            floatPartials);
    } else if (!floatActivations && !floatPartials &&
               deviceInfo.fp16InFp32OutConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(true, false, true);
    }
  }
  if (options.weightUpdateMethod == WeightUpdateMethod::AOP ||
      options.weightUpdateMethod == WeightUpdateMethod::AUTO) {
    convVertexTypeCandidates.emplace_back(false, floatActivations,
                                          floatPartials);
  }
  return convVertexTypeCandidates;
}

static std::vector<unsigned>
getInChansPerGroupCandidates(const ConvParams &params,
                             const ConvVertexType &convVertexType,
                             const poplar::DeviceInfo &deviceInfo) {
  std::vector<unsigned> candidates;
  for (unsigned i = 1; i <= params.getInputDepth(); ++i) {
    if (params.getInputDepth() % i != 0)
      continue;
    if (!convVertexType.floatActivations && i % 2 != 0)
      continue;
    if (convVertexType.useConvInstruction &&
        !canUseConvolutionInstruction(convVertexType.floatActivations,
                                      convVertexType.floatPartials,
                                      params.stride[0], params.stride[1],
                                      i, deviceInfo))
      continue;
    candidates.push_back(i);
  }
  if (candidates.empty()) {
    if (convVertexType.useConvInstruction) {
      // Drop the requirement that the input channel grouping must divide
      // the number of input channels. This causes the input to be zero padded
      // before the convolution.
      // TODO Currently we only consider input channel groupings that need
      // padding if we didn't find an input channel grouping that divides the
      // number of channels exactly. Ideally we would always consider all
      // input channel groupings and pick the one with the lowest cost.
      // We would need to check whether the cost model is sufficiently accurate
      // before making this change.
      for (unsigned i = 1; i <= params.getInputDepth(); ++i) {
        if (!convVertexType.floatActivations && i % 2 != 0)
          continue;
        if (convVertexType.useConvInstruction &&
            !canUseConvolutionInstruction(convVertexType.floatActivations,
                                          convVertexType.floatPartials,
                                          params.stride[0], params.stride[1],
                                          i, deviceInfo))
          continue;
        candidates.push_back(i);
      }
    } else {
      candidates.push_back(params.getInputDepth());
    }
  }
  return candidates;
}

static Cost
estimateWeightUpdateByAopReorderCost(const poplar::DeviceInfo &deviceInfo,
                                     unsigned tensorOutChansPerGroup,
                                     unsigned tensorWeightOutChansPerGroup,
                                     bool floatActivations,
                                     const ConvParams &params,
                                     unsigned partialChansPerGroup) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto bytesPerElement = floatActivations ? 4U : 2U;
  const auto reorderBytesPerCycle =
      std::min(deviceInfo.memcpyBytesPerCycle, bytesPerElement);
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  auto reorderBytesPre = 0;
  if (partialChansPerGroup != tensorOutChansPerGroup) {
    // Reorder deltas.
    const auto numDeltas = params.getBatchSize() *
                           params.getOutputDepth() *
                           params.getOutputHeight() *
                           params.getOutputWidth();
    reorderBytesPre += numDeltas * bytesPerElement;
  }
  auto reorderBytesPost = 0;
  if (partialChansPerGroup != tensorWeightOutChansPerGroup) {
    // Reorder weight deltas.
    const auto numWeightDeltas = params.kernelShape[0] *
                                 params.kernelShape[1] *
                                 params.getInputDepth() *
                                 params.getOutputDepth();
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
           const ConvParams &params, bool isWeightUpdate,
           unsigned batchesPerGroup,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  Plan best;
  Cost bestCost = highestCost;
  if (isWeightUpdate && !convVertexType.useConvInstruction) {
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
                     convVertexType, params, isWeightUpdate,
                     batchesPerGroup, costBounds,
                     cache, options);
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
  if (isWeightUpdate) {
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
                   convVertexType, params, isWeightUpdate, batchesPerGroup,
                   costBounds, cache, options);
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
           unsigned tensorInChansPerGroup,
           unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           const ConvParams &params, bool isWeightUpdate,
           unsigned batchesPerGroup,
           CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  Cost bestCost = highestCost;
  Plan bestPlan;
  const auto convVertexTypeCandidates =
      isWeightUpdate
        ? getWeightUpdateVertexTypeCandidates(deviceInfo, floatActivations,
                                              floatPartials, options)
        : getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                      floatPartials, params);
  for (const auto &convVertexType : convVertexTypeCandidates) {
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, convVertexType, tensorInChansPerGroup,
                   tensorOutChansPerGroup, tensorWeightOutChansPerGroup,
                   params, isWeightUpdate, batchesPerGroup, costBounds, cache,
                   options);
    if (candidateCost == highestCost)
      continue;
    if (compareCost(candidateCost, bestCost, costBounds)) {
      bestPlan = candidate;
      bestCost = candidateCost;
    }
  }
  return {bestPlan, bestCost};
}

static std::pair<Plan, Cost>
createPlan(ConvParams params,
           unsigned tensorInChansPerGroup, unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           std::string partialsType,
           bool isWeightUpdate,
           const ConvOptions &options,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlanningCacheImpl *cache) {
  validateLayerParams(params);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  bool flattenXY;
  if (params.kernelShape[0] == 1 && params.kernelShape[1] == 1
      && params.stride[0] == 1 && params.stride[1] == 1
      && params.paddingLower[0] == 0 && params.paddingLower[1] == 0
      && params.paddingUpper[0] == 0 && params.paddingUpper[1] == 0) {
    flattenXY = true;
    params.inputShape[2] = params.inputShape[1] * params.inputShape[2];
    params.inputShape[1] = 1;
  } else {
    flattenXY = false;
  }

  const bool floatActivations = params.dType == "float";
  const bool floatPartials = partialsType == "float";
  Cost bestCandidateCost = highestCost;
  Plan bestCandidate;
  for (auto batchesPerGroup = 1U; batchesPerGroup <= params.getBatchSize();
       ++batchesPerGroup) {

    /* only allow integer division of batches.
     *  i.e. batchSize = batchesPerGroup * numBatchGroups
     *
     * Weight Update doesn't use batch grouping
     */

    if ((params.getBatchSize() % batchesPerGroup) ||
        ((!flattenXY || isWeightUpdate) && batchesPerGroup > 1)) {
      continue;
    }

    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, floatActivations,
                   floatPartials,
                   tensorInChansPerGroup,
                   tensorOutChansPerGroup,
                   tensorWeightOutChansPerGroup,
                   params, isWeightUpdate,
                   batchesPerGroup,
                   costBounds,
                   cache,
                   options);
    if (compareCost(candidateCost, bestCandidateCost, costBounds)) {
      bestCandidateCost = candidateCost;
      bestCandidate = candidate;
    }
  }
  if (flattenXY)
    bestCandidate.flattenXY = true;
  return {bestCandidate, bestCandidateCost};
}

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options) {
  assert (weightsShape.size() == 3);
  assert (stride.size() == 2);
  assert (paddingLower.size() == 2);
  assert (paddingUpper.size() == 2);
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  const auto partialsType = options.partialsType;
  auto cache = options.cache ? options.cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cache = tempCache.get();
  }
  PlanningCacheImpl::Key key(params, options, false, 0, 0);
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto match = plans.find(key);
    if (match != plans.end())
      return *match->second;
  }
  if (options.useWinograd) {
    if (options.winogradPatchSize != 4 ||
        params.stride[0] != 1 || params.stride[1] != 1 ||
        params.kernelShape[0] != 3 || params.kernelShape[1] != 3) {
      throw popstd::poplib_error("Attempt to force winograd convolution for "
                               "invalid parameters");

    }
    plan.useWinograd = true;
    plan.winogradPatchSize = options.winogradPatchSize;
    return plan;
  }

  std::tie(plan, cost) = popconv::createPlan(params,
                                             0, 0, 0,
                                             partialsType,
                                             false, options,
                                             costBounds, graph,
                                             cache);
  if (options.percentageCyclesExcessForMemOptim) {
    /* Set new bounds based on previous search */
    const double newCyclesBound =
        static_cast<double>(cost.cycles)
        * (100.0 + options.percentageCyclesExcessForMemOptim) / 100.0;

    CostBounds newCostBounds(static_cast<unsigned>(newCyclesBound), 0, false);

    std::tie(plan, cost) = popconv::createPlan(params,
                                               0, 0, 0,
                                               partialsType,
                                               false,
                                               options,
                                               newCostBounds, graph,
                                               cache);
  }
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(plan)));
    auto res = plans.emplace(std::make_pair(key, std::move(pPlan)));
    return *res.first->second;
  }
  return plan;
}

Plan getWeightUpdatePlan(const poplar::Graph &graph,
                         const poplar::Tensor &activations,
                         const poplar::Tensor &deltas,
                         const ConvParams &params,
                         ConvOptions options) {
  assert (params.kernelShape.size() == 3);
  assert (params.stride.size() == 2);
  assert (params.paddingLower.size() == 2);
  assert (params.paddingUpper.size() == 2);
  if (options.noLHSRearrangement) {
    auto plan = getPlan(graph, params, options);
    plan.useConvolutionInstructions = false;
    return plan;
  }
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  const auto partialsType = options.partialsType;
  const auto fwdPlan = getPlan(graph, params, options);
  const auto actChansPerGroup = fwdPlan.inChansPerGroup;
  const auto deltasChansPerGroup = deltas.dim(4);
  auto cache = options.cache ? options.cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cache = tempCache.get();
  }
  std::tie(plan, cost) = popconv::createPlan(params,
                                             actChansPerGroup,
                                             deltasChansPerGroup,
                                             fwdPlan.partialChansPerGroup,
                                             partialsType, true, options,
                                             costBounds, graph, cache);
  return plan;
}

} // end namespace conv
