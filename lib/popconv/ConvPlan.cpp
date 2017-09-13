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
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <popsolver/Model.hpp>
#include "util/print.hpp"

namespace popconv {

std::uint64_t getNumberOfMACs(const ConvParams &params) {
  std::uint64_t numMACs = params.getNumConvGroups() *
                          params.getBatchSize() *
                          params.getOutputDepthPerConvGroup() *
                          params.getInputDepthPerConvGroup();
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    unsigned nonZeroInputs = 0;
    for (unsigned x = 0; x < params.getOutputSize(dim); ++x) {
      for (unsigned k = 0; k < params.kernelShape[dim]; ++k) {
        if (getInputIndex(dim, x, k, params) != ~0U) {
          ++nonZeroInputs;
        }
      }
    }
    numMACs *= nonZeroInputs;
  }
  return numMACs;
}

const char *asString(const WeightUpdateMethod &method) {
  switch (method) {
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
  if (token == "amp")
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
  std::map<std::tuple<typename std::remove_reference<Args>::type...>,
           Ret> table;
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
  Plan::Method method;
  bool floatActivations;
  bool floatPartials;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  ConvVertexType(Plan::Method method, bool floatActivations,
                 bool floatPartials, unsigned inChansPerGroup,
                 unsigned partialChansPerGroup) :
    method(method),
    floatActivations(floatActivations),
    floatPartials(floatPartials),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup) {}
};

static const char *asString(Plan::Method m) {
  switch (m) {
  case Plan::Method::AMP: return "AMP";
  case Plan::Method::MAC: return "MAC";
  case Plan::Method::OUTER_PRODUCT: return "OUTER_PRODUCT";
  }
}

std::ostream& operator<<(std::ostream &os, Plan::Method m) {
  os << asString(m);
  return os;
}

std::ostream& operator<<(std::ostream &os, const Plan &p)
{
  os << "  Plan: tilesPerXAxis           " << p.tilesPerXAxis << "\n"
     << "        tilesPerYAxis           " << p.tilesPerYAxis << "\n"
     << "        tilesPerBatchAxis       " << p.tilesPerBatchAxis << "\n"
     << "        tilesPerZAxis           " << p.tilesPerZAxis << "\n"
     << "        tilesPerKernelYAxis     " << p.tilesPerKernelYAxis << "\n"
     << "        tilesPerInZGroupAxis    " << p.tilesPerInZGroupAxis << "\n"
     << "        tilesPerConvGroups      " << p.tilesPerConvGroups << "\n"
     << "        xAxisGrainSize          " << p.xAxisGrainSize << "\n"
     << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n"
     << "        expandDims              ";
  printContainer(p.expandDims, os);
  os << "\n"
     << "        outChanFlattenDims      ";
  printContainer(p.outChanFlattenDims, os);
  os << "\n"
     << "        flattenDims             ";
  printContainer(p.flattenDims, os);
  os << "\n";
  return os;
}

static std::uint64_t
getConvPartialnx1CycleEstimate(unsigned passesPerOutput,
                               unsigned batchElements,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               const std::vector<unsigned> &inputDilation,
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
                             const poplar::DeviceInfo &deviceInfo) {
  if (getConvUnitsPerTile(deviceInfo, floatActivations, floatPartials) == 0) {
    return false;
  }
  if (floatActivations) {
    if (!floatPartials) {
      return false;
    }
  }
  return true;
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             unsigned inChansPerGroup,
                             const poplar::DeviceInfo &deviceInfo) {
  if (!canUseConvolutionInstruction(floatActivations, floatPartials,
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
                     bool contiguousAccess)  {
  if (outputRangeSize == 0)
    return 0;
  unsigned inputRangeSize;
  const auto kernelSize = params.kernelShape[dim];
  const auto stride = params.stride[dim];
  const auto inputDilation = params.inputDilation[dim];
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
    {
    const auto preDownSampleOutputSize = (outputRangeSize - 1) * stride + 1;
    const auto dilatedInputSize = preDownSampleOutputSize + tileKernelSize - 1;
    inputRangeSize = (dilatedInputSize - 1) / inputDilation + 1;
    }
    break;
  }
  if (inputDilation == 1 && !contiguousAccess && tileKernelSize == 1 &&
      stride > 1) {
    inputRangeSize = (inputRangeSize - 1) / stride + 1;
  }
  return inputRangeSize;
}

static std::uint64_t
getConvPartialnx1CycleEstimate(unsigned passesPerOutput,
                               unsigned batchElements,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned convUnitPipelineDepth,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               const std::vector<unsigned> &inputDilation,
                               unsigned numInputPointers)
{
  unsigned numInputEdges = 0;
  unsigned numOutputEdges = 0;
  const auto numWorkerContexts = 6;
  std::vector<std::vector<PartialRow>> partition =
      partitionConvPartialByWorker(batchElements, outputHeight, outputWidth,
                                   numWorkerContexts, inputDilation);
  std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
  convSizesByWeightAndWorker.emplace_back();
  convSizesByWeightAndWorker.back().reserve(partition.size());
  for (const auto &entry : partition) {
    convSizesByWeightAndWorker.back().emplace_back();
    convSizesByWeightAndWorker.back().back().reserve(entry.size());
    numInputEdges += numInputPointers * entry.size();
    numOutputEdges += entry.size();
    for (const auto &partialRow : entry) {
      auto convSize = (partialRow.xEnd - partialRow.xBegin) /
                      inputDilation[1];
      convSizesByWeightAndWorker.back().back().push_back(convSize);
    }
  }

  auto numEdges = (convSizesByWeightAndWorker.size()
                   + numInputEdges
                   + numOutputEdges) * passesPerOutput;
  return getConvPartialnx1SupervisorCycleEstimate(
                convSizesByWeightAndWorker,
                passesPerOutput,
                convUnitPipelineDepth,
                numConvUnitsPerTile,
                convUnitCoeffLoadBytesPerCycle,
                numInputPointers,
                useDeltaEdgesForConvPartials(numEdges));
}

static unsigned
getMaxTileOutWidth(const ConvParams &params, const Plan &plan) {
  const auto outWidth = params.getOutputWidth();
  const auto grainSize = plan.xAxisGrainSize;
  const auto numGrains = (outWidth + grainSize - 1) / grainSize;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tileNumGrains = (numGrains + tilesPerX - 1) / tilesPerX;
  const auto tileOutWidth = std::min(outWidth, tileNumGrains * grainSize);
  return tileOutWidth;
}

static unsigned
estimateExchangeCycles(const poplar::DeviceInfo &deviceInfo,
                       bool floatActivations,
                       const ConvParams &params,
                       const Plan &plan) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerBatch = plan.tilesPerBatchAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto tilesPerConvGroups = plan.tilesPerConvGroups;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto kernelSizeY = params.kernelShape[0];
  const auto kernelSizeX = params.kernelShape[1];
  const auto tileKernelHeight = (kernelSizeY + tilesPerKernelYAxis - 1) /
                                tilesPerKernelYAxis;
  const auto tileKernelWidth = kernelSizeX;
  const auto tileOutWidth = getMaxTileOutWidth(params, plan);
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileBatchElements =
      (params.getBatchSize() + tilesPerBatch - 1) / tilesPerBatch;
  const auto outputDepth = params.getOutputDepthPerConvGroup();
  const auto numOutGroups =
      (outputDepth + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileOutDepth = tileNumOutGroups * partialChansPerGroup;
  const auto inputDepth = params.getInputDepthPerConvGroup();
  const auto numInGroups =
      (inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;
  const auto tileInDepth = tileNumInGroups * inChansPerGroup;
  const auto tileInWidth =
      getMaxInputRangeSize(tileOutWidth, 1, params,
                           tileKernelWidth, tilesPerX,
                           true);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, 0, params,
                           tileKernelHeight, tilesPerY,
                           false);
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + tilesPerConvGroups - 1) / tilesPerConvGroups;
  const auto numberOfInputElements = tileInWidth * tileInHeight *
                                     tileBatchElements * tileInDepth *
                                     tileNumGroupedConv;
  const auto numberOfWeights =
      tileKernelHeight * tileKernelWidth * tileOutDepth * tileInDepth *
      tileNumGroupedConv;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileBatchElements * tileOutDepth *
      tileNumGroupedConv;
  const auto activationSize = floatActivations ? 4 : 2;
  auto inputElementsBytes = numberOfInputElements * activationSize;
  auto weightBytes = numberOfWeights * activationSize;
  const auto partialSize = plan.floatPartials ? 4 : 2;
  const auto numberOfPartialSums = numberOfOutputElements;
  const auto partialSumBytes = numberOfPartialSums * partialSize;

  const auto tilesPerSuperTile = deviceInfo.tilesPerSuperTile;
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;

  const auto inputElementBytesPerCycle =
      (deviceInfo.supportsSuperTileSendReceive &&
       plan.linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD &&
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
  const auto wholeRowConvSize =
      (tileOutWidth + outputStrideX - 1) / outputStrideX;
  unsigned convCount =
      workerWholeRows * tileKernelWidth * tileKernelHeight * tileNumInGroups;
  unsigned totalConvSize = convCount * wholeRowConvSize;
  if (workerPartRows > 0) {
    auto pConv = tileKernelWidth * tileKernelHeight * tileNumInGroups;
    convCount += pConv;
    totalConvSize +=
        pConv * ((wholeRowConvSize * workerPartRows + rowSplitFactor - 1) /
                 rowSplitFactor);
  }
  return getConvPartialHorizontalMacCycleEstimate(
    floatActivations,
    inChansPerGroup,
    convCount, totalConvSize,
    dataPathWidth
  );
}

static bool zeroPartials(const ConvParams &params, const Plan &plan) {
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto kernelSizeY = params.kernelShape[0];
  const auto kernelSizeX = params.kernelShape[1];
  const auto tileKernelHeight =
      (kernelSizeY + tilesPerKernelYAxis - 1) /
          tilesPerKernelYAxis;
  if (kernelSizeX != 1 || tileKernelHeight != 1) {
    return true;
  }
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    if (params.stride[dim] != 1 ||
        params.inputDilation[dim] != 1 ||
        params.kernelDilation[dim] != 1 ||
        params.inputPaddingLower[dim] != 0 ||
        params.inputPaddingUpper[dim] != 0 ||
        params.kernelPaddingLower[dim] != 0 ||
        params.kernelPaddingUpper[dim] != 0) {
      return true;
    }
  }
  return false;
}

static unsigned
estimateZeroCycles(const poplar::DeviceInfo &deviceInfo,
                   const ConvParams &params,
                   const Plan &plan) {
  if (!zeroPartials(params, plan)) {
    return 0;
  }
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerBatch = plan.tilesPerBatchAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerConvGroups = plan.tilesPerConvGroups;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto tileOutWidth = getMaxTileOutWidth(params, plan);
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileBatchElements =
      (params.getBatchSize() + tilesPerBatch - 1) / tilesPerBatch;
  const auto outputDepth = params.getOutputDepthPerConvGroup();
  const auto numOutGroups =
      (outputDepth + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileOutDepth = tileNumOutGroups * partialChansPerGroup;
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + tilesPerConvGroups - 1) / tilesPerConvGroups;
  const auto numberOfOutputElements =
      tileOutWidth * tileOutHeight * tileBatchElements * tileOutDepth *
      tileNumGroupedConv;
  const auto vectorWidth =
      plan.floatPartials ? deviceInfo.getFloatVectorWidth() :
                           deviceInfo.getHalfVectorWidth();
  const auto numCycles = (numberOfOutputElements + vectorWidth - 1) /
                         vectorWidth;
  return numCycles;
}

static unsigned
estimatePartialCalcCycles(const poplar::DeviceInfo &deviceInfo,
                          bool floatActivations,
                          const ConvParams &params,
                          const Plan &plan,
                          PlanningCacheImpl *cache) {
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerBatch = plan.tilesPerBatchAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto tilesPerConvGroups = plan.tilesPerConvGroups;

  const auto tileOutWidth = getMaxTileOutWidth(params, plan);
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileBatchElements =
      (params.getBatchSize() + tilesPerBatch - 1) / tilesPerBatch;
  const auto numOutGroups =
      (params.getOutputDepthPerConvGroup() + (outChansPerGroup - 1))
      / outChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + tilesPerConvGroups - 1) / tilesPerConvGroups;

  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = deviceInfo.numWorkerContexts;
  if (plan.method == Plan::Method::AMP) {
    numContexts = 1;
  }
  const auto numInGroups =
      (params.getInputDepthPerConvGroup() + (inChansPerGroup - 1))
      / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto tileKernelHeight =
      (params.kernelShape[0] + tilesPerKernelYAxis - 1) /
          tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelShape[1];
  unsigned computeCycles;
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    {
      assert(deviceInfo.getWeightsPerConvUnit(floatActivations) %
             inChansPerGroup == 0);
      const auto convUnitWeightHeight =
          deviceInfo.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
      const auto passesPerFilter =
          tileKernelWidth *
          ((tileKernelHeight + convUnitWeightHeight - 1) /
           convUnitWeightHeight);
      const auto numConvUnits = getNumConvUnits(floatActivations,
                                                plan.floatPartials,
                                                deviceInfo);
      assert(outChansPerGroup % numConvUnits == 0);
      const auto passesPerOutputGroup = outChansPerGroup / numConvUnits;
      const auto passesPerOutput = passesPerFilter * passesPerOutputGroup *
                                   tileNumInGroups;
      computeCycles =
          tileNumOutGroups * tileNumGroupedConv *
          cache->mGetConvPartialnx1CycleEstimate(
            passesPerOutput, tileBatchElements, tileOutHeight, tileOutWidth,
            deviceInfo.convUnitPipelineDepth,
            getNumConvUnits(floatActivations, plan.floatPartials, deviceInfo),
            deviceInfo.convUnitCoeffLoadBytesPerCycle, params.inputDilation,
            convUnitWeightHeight);

    }
    break;
  case Plan::Method::MAC:
    {
      const auto outputStrideX = params.inputDilation[1];
      const auto outputStrideY = params.inputDilation[0];
      const auto numOutRows =
          tileNumOutGroups * tileNumGroupedConv *
          (tileOutHeight + outputStrideY - 1) / outputStrideY *
          tileBatchElements;
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
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      assert(tileOutHeight == 1);
      assert(tileBatchElements == 1);
      assert(tileNumInGroups == 1);
      assert(params.stride[0] == 1);
      assert(params.stride[1] == 1);
      assert(params.inputDilation[0] == 1);
      assert(params.inputDilation[1] == 1);
      const auto workerOutWidth =
          (tileOutWidth + deviceInfo.numWorkerContexts - 1) /
          deviceInfo.numWorkerContexts;
      auto vertexRuntime =
          getOuterProductCycleEstimate(floatActivations, workerOutWidth,
                                       tileNumOutGroups * outChansPerGroup *
                                       tileNumGroupedConv,
                                       outChansPerGroup,
                                       deviceInfo.dataPathWidth);
      computeCycles = vertexRuntime * numContexts;
    }
    break;
  }
  return computeCycles;
}


static unsigned
estimateReduceCycles(const poplar::DeviceInfo &deviceInfo,
                     const ConvParams &params, const Plan &plan) {
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  if (tilesPerKernelYAxis == 1 &&
      tilesPerInZGroupAxis == 1)
    return 0;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerBatch = plan.tilesPerBatchAxis;
  const auto tilesPerConvGroups = plan.tilesPerConvGroups;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto tileOutWidth = getMaxTileOutWidth(params, plan);
  const auto tileOutHeight =
      (params.getOutputHeight() + tilesPerY - 1) / tilesPerY;
  const auto tileBatchElements =
      (params.getBatchSize() + tilesPerBatch - 1) / tilesPerBatch;
  const auto numOutGroups =
      (params.getOutputDepthPerConvGroup() + (outChansPerGroup - 1))
      / outChansPerGroup;
  const auto tileNumConvGroups =
      (params.getNumConvGroups() + tilesPerConvGroups - 1) / tilesPerConvGroups;
  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;
  const auto numOutputs = tileOutWidth *
                          tileOutHeight *
                          tileBatchElements *
                          tileNumOutGroups * outChansPerGroup *
                          tileNumConvGroups;
  // Consider a group of tiles that compute partial sums for the same output
  // volume. The number of partial sums that to be reduced is
  // numOutputs * numTiles. Calculation of the output is spread evenly across
  // the tiles so the number of partial sums each tile must reduce is
  // (numOutputs * numTiles) / numTiles = numOutputs.
  const auto reduceElementsPerTile = numOutputs;

  const auto vectorWidth =
      plan.floatPartials ? deviceInfo.getFloatVectorWidth() :
                                deviceInfo.getHalfVectorWidth();
  const auto numCycles = (reduceElementsPerTile + vectorWidth - 1) /
                         vectorWidth;
  return numCycles;
}

unsigned getMaxMACsPerCyclePerTile(const poplar::DeviceInfo &deviceInfo,
                                   bool floatPartials,
                                   bool floatActivations,
                                   Plan::Method method) {
  const auto vectorWidth =
      floatActivations ? deviceInfo.getFloatVectorWidth() :
                         deviceInfo.getHalfVectorWidth();
  switch (method) {
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT:
    return vectorWidth;
  case Plan::Method::AMP:
    {
      unsigned numConvUnits;
      if (floatActivations) {
        assert(floatPartials);
        numConvUnits = deviceInfo.fp32InFp32OutConvUnitsPerTile;
      } else if (floatPartials) {
        numConvUnits = deviceInfo.fp16InFp32OutConvUnitsPerTile;
      } else {
        numConvUnits = deviceInfo.fp16InFp16OutConvUnitsPerTile;
      }
      return numConvUnits * vectorWidth;
    }
  }
}

static popsolver::Variable
addCycleEstimate(popsolver::Model &m, popsolver::Variable tilesPerX,
                 popsolver::Variable tilesPerY,
                 popsolver::Variable tilesPerBatch,
                 popsolver::Variable tilesPerZ,
                 popsolver::Variable tilesPerKernelY,
                 popsolver::Variable tilesPerInZ,
                 popsolver::Variable tilesPerConvGroups,
                 popsolver::Variable usedTiles,
                 const poplar::DeviceInfo &deviceInfo,
                 const ConvParams &params,
                 unsigned inChansPerGroup,
                 unsigned partialChansPerGroup,
                 unsigned xAxisGrainSize,
                 bool floatPartials,
                 bool floatActivations,
                 Plan::Method method,
                 Plan::LinearizeTileOrder linearizeTileOrder,
                 PlanningCacheImpl *cache) {
  const auto exchangeCycles =
      m.call({tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ, tilesPerKernelY,
              tilesPerInZ, tilesPerConvGroups},
             [=,&deviceInfo](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerBatch = values[2];
    const auto tilesPerZ = values[3];
    const auto tilesPerKernelY = values[4];
    const auto tilesPerInZ = values[5];
    const auto tilesPerConvGroups = values[6];
    Plan candidate(tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ, tilesPerConvGroups,
                   inChansPerGroup, partialChansPerGroup,
                   xAxisGrainSize,
                   floatPartials,
                   method,
                   linearizeTileOrder);
    return estimateExchangeCycles(deviceInfo, floatActivations, params,
                                  candidate);
  });
  const auto zeroCycles =
      m.call({tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ, tilesPerKernelY,
              tilesPerInZ, tilesPerConvGroups},
             [=,&deviceInfo](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerBatch = values[2];
    const auto tilesPerZ = values[3];
    const auto tilesPerKernelY = values[4];
    const auto tilesPerInZ = values[5];
    const auto tilesPerConvGroups = values[6];
    Plan candidate(tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ, tilesPerConvGroups,
                   inChansPerGroup, partialChansPerGroup,
                   xAxisGrainSize,
                   floatPartials,
                   method,
                   linearizeTileOrder);
    return estimateZeroCycles(deviceInfo, params, candidate);
  });
  const auto partialCalcCycles =
      m.call({tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ, tilesPerKernelY,
              tilesPerInZ, tilesPerConvGroups},
             [=,&deviceInfo](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerBatch = values[2];
    const auto tilesPerZ = values[3];
    const auto tilesPerKernelY = values[4];
    const auto tilesPerInZ = values[5];
    const auto tilesPerConvGroups = values[6];
    Plan candidate(tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ, tilesPerConvGroups,
                   inChansPerGroup, partialChansPerGroup,
                   xAxisGrainSize,
                   floatPartials,
                   method,
                   linearizeTileOrder);
    return estimatePartialCalcCycles(deviceInfo, floatActivations, params,
                                     candidate, cache);
  });
  // Add a redunant inequality that relates the cycles required to calculate the
  // partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile =
      getMaxMACsPerCyclePerTile(deviceInfo, floatPartials, floatActivations,
                                method);
  const auto totalMacs = getNumberOfMACs(params);
  m.lessOrEqual(totalMacs / maxMACsPerCyclePerTile,
                m.product({usedTiles, partialCalcCycles}));
  const auto reduceCycles =
      m.call({tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ, tilesPerKernelY,
              tilesPerInZ, tilesPerConvGroups},
             [=,&deviceInfo](const std::vector<unsigned> &values) {
    const auto tilesPerX = values[0];
    const auto tilesPerY = values[1];
    const auto tilesPerBatch = values[2];
    const auto tilesPerZ = values[3];
    const auto tilesPerKernelY = values[4];
    const auto tilesPerInZ = values[5];
    const auto tilesPerConvGroups = values[6];
    Plan candidate(tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                   tilesPerKernelY, tilesPerInZ, tilesPerConvGroups,
                   inChansPerGroup, partialChansPerGroup,
                   xAxisGrainSize,
                   floatPartials,
                   method,
                   linearizeTileOrder);
    return estimateReduceCycles(deviceInfo, params, candidate);
  });
  return m.sum({exchangeCycles, zeroCycles, partialCalcCycles, reduceCycles});
}

static Plan::Method
getFullyConnectedWUMethod(const ConvParams &fwdParams,
                          Plan::Method fwdMethod,
                          unsigned fwdInChansPerGroup) {
  if (fwdParams.getOutputDepthPerConvGroup() == 1) {
    return Plan::Method::OUTER_PRODUCT;
  }
  const auto wuPartialChansPerGroup = fwdInChansPerGroup;
  if (wuPartialChansPerGroup != 1) {
    // ConvPartialHorizontalMacVertex only supports an output grouping of 1.
    // so we must force the use of the convolutional instructions.
    return Plan::Method::AMP;
  }
  return fwdMethod;
}

static Plan::Method
getFullyConnectedBwdMethod(const ConvParams &fwdParams,
                           Plan::Method fwdMethod) {
  return fwdMethod;
}

template <class TransformCostFn>
static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           unsigned xAxisGrainSize,
           const ConvVertexType &convVertexType,
           const ConvParams &params,
           TransformCostFn transformCost,
           Cost bestCost,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  const auto inChansPerGroup = convVertexType.inChansPerGroup;
  const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
  const auto floatActivations = convVertexType.floatActivations;
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
  const auto numTiles = deviceInfo.getNumTiles();
  const unsigned numXGrains =
      (params.getOutputWidth() + xAxisGrainSize - 1) / xAxisGrainSize;
  const auto tilesPerX = m.addVariable(1, numXGrains);
  const auto tilesPerY = m.addVariable(1, params.getOutputHeight());
  const auto tilesPerBatch = m.addVariable(1, params.getBatchSize());
  const auto tilesPerConvGroups = m.addVariable(1, params.getNumConvGroups());
  unsigned maxTilesPerZ;
  if (options.fullyConnectedPass == FullyConnectedPass::FWD) {
    // The joint planning cost function assumes that no exchange is required to
    // rearrange weights between passes. Because of the way we derive the
    // backward and weight update plans from the forward plan this is guaranteed
    // to be the case if each weight is used on exactly one tile in the forward
    // pass. Disallow splitting of fully connected batch (or equivalently the
    // convolutional output channels) across tiles to ensure this holds.
    maxTilesPerZ = 1;
  } else {
    const auto outChanGroups =
        (params.getOutputDepthPerConvGroup() + partialChansPerGroup - 1) /
        partialChansPerGroup;
    maxTilesPerZ = outChanGroups;
  }
  const auto tilesPerZ = m.addVariable(1, maxTilesPerZ);
  const auto tilesPerKernelY = m.addVariable(1, params.kernelShape[0]);
  const auto inChanGroups =
      (params.getInputDepthPerConvGroup() + inChansPerGroup - 1)
      / inChansPerGroup;
  const auto tilesPerInZ = m.addVariable(1, inChanGroups);
  const auto usedTiles = m.product({tilesPerX, tilesPerY, tilesPerBatch,
                                    tilesPerZ, tilesPerKernelY, tilesPerInZ,
                                    tilesPerConvGroups});
  m.lessOrEqual(usedTiles, numTiles);

  auto cycles =
      addCycleEstimate(m, tilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                       tilesPerKernelY, tilesPerInZ, tilesPerConvGroups,
                       usedTiles, deviceInfo, params,
                       inChansPerGroup, partialChansPerGroup,
                       xAxisGrainSize, convVertexType.floatPartials,
                       floatActivations, convVertexType.method,
                       Plan::LinearizeTileOrder::STANDARD, cache);
  if (options.fullyConnectedPass == FullyConnectedPass::FWD) {
    popsolver::Variable bwdCycles;
    auto bwdParams = params;
    std::swap(bwdParams.inputFieldShape[1], bwdParams.inputChannels);
    const auto bwdTilesPerX = tilesPerInZ;
    const auto bwdTilesPerInZ = tilesPerX;
    const auto bwdInChansPerGroup = xAxisGrainSize;
    const auto bwdXAxisGrainSize = inChansPerGroup;
    const auto bwdMethod =
        getFullyConnectedBwdMethod(params,
                                   convVertexType.method);
    bwdCycles =
        addCycleEstimate(m, bwdTilesPerX, tilesPerY, tilesPerBatch, tilesPerZ,
                         tilesPerKernelY, bwdTilesPerInZ, tilesPerConvGroups,
                         usedTiles, deviceInfo, params,
                         bwdInChansPerGroup, partialChansPerGroup,
                         bwdXAxisGrainSize, convVertexType.floatPartials,
                         floatActivations, bwdMethod,
                         Plan::LinearizeTileOrder::FC_BWD_AS_CONV, cache);
    auto wuParams = params;
    std::swap(wuParams.inputChannels, wuParams.outputChannels);
    const auto wuTilesPerZ = tilesPerInZ;
    const auto wuTilesPerInZ = tilesPerZ;
    const auto wuInChansPerGroup = partialChansPerGroup;
    const auto wuPartialChansPerGroup = inChansPerGroup;
    const auto wuMethod =
        getFullyConnectedWUMethod(params,
                                  convVertexType.method,
                                  inChansPerGroup);
    const auto wuCycles =
        addCycleEstimate(m, tilesPerX, tilesPerY, tilesPerBatch, wuTilesPerZ,
                         tilesPerKernelY, wuTilesPerInZ, tilesPerConvGroups,
                         usedTiles, deviceInfo, wuParams,
                         wuInChansPerGroup, wuPartialChansPerGroup,
                         1, convVertexType.floatPartials,
                         floatActivations, wuMethod,
                         Plan::LinearizeTileOrder::FC_WU,
                         cache);
    cycles = m.sum({cycles, bwdCycles, wuCycles});
  }
  auto transformCycles =
      m.call({usedTiles},
             [&](const std::vector<unsigned> &values) {
        return transformCost(values[0]);
      });
  cycles = m.sum({cycles, transformCycles});
  auto cycleBound = bestCost.cycles;
  if (costBounds.cycles > 0) {
    cycleBound = std::min(cycleBound, costBounds.cycles);
  }
  if (cycleBound < std::numeric_limits<unsigned>::max()) {
    m.lessOrEqual(cycles, cycleBound);
  }
  Solution s;
  try {
    assert(costBounds.primaryCheckIsCycles);
    s = m.minimize(cycles);
  } catch (NoSolution) {
    return {Plan(), highestCost};
  }
  Plan plan(s[tilesPerX], s[tilesPerY], s[tilesPerBatch], s[tilesPerZ],
            s[tilesPerKernelY], s[tilesPerInZ], s[tilesPerConvGroups],
            inChansPerGroup, partialChansPerGroup,
            xAxisGrainSize,
            convVertexType.floatPartials,
            convVertexType.method,
            Plan::LinearizeTileOrder::STANDARD);
  // TODO estimate memory usage.
  unsigned memory = 0;
  Cost cost = {s[cycles], memory};
  return {plan, cost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            bool floatPartials,
                            const ConvParams &params,
                            const ConvOptions &options) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (params.getInputDepthPerConvGroup() == 1 &&
      params.getPaddedDilatedKernelSize(0) == 1 &&
      params.getPaddedDilatedKernelSize(1) == 1 &&
      params.getOutputHeight() == 1 &&
      params.getBatchSize() == 1) {
    const auto partialChansPerGroup = floatActivations ?
                                      deviceInfo.getFloatVectorWidth() :
                                      deviceInfo.getHalfVectorWidth();
    convVertexTypeCandidates.emplace_back(Plan::Method::OUTER_PRODUCT,
                                          floatActivations,
                                          floatActivations, 1,
                                          partialChansPerGroup);
  }
  bool ampFloatPartials = floatPartials;
  auto numConvUnits = getNumConvUnits(floatActivations,
                                      ampFloatPartials,
                                      deviceInfo);
  if (numConvUnits == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnits = getNumConvUnits(floatActivations,
                                   ampFloatPartials,
                                   deviceInfo);
  }
  const bool isFullyConnectedFwd =
      options.fullyConnectedPass == FullyConnectedPass::FWD;
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   deviceInfo)) {
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(floatActivations);
    for (unsigned inChansPerGroup = 1; inChansPerGroup <= weightsPerConvUnit;
         ++inChansPerGroup) {
      if (!floatActivations && inChansPerGroup % 2 != 0)
        continue;
      if (!canUseConvolutionInstruction(floatActivations,
                                        floatPartials,
                                        inChansPerGroup, deviceInfo))
        continue;
      if (isFullyConnectedFwd) {
        // The input channels in the forward pass become the output channels of
        // the weight update pass. Make sure it is a multiple of the supported
        // output channels per group.
        if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
          continue;
      }
      // TODO take into account the best grouping for all the phases if
      // options.fullyConnectedFwd is set.
      const auto partialChansPerGroup = numConvUnits;
      convVertexTypeCandidates.emplace_back(Plan::Method::AMP, floatActivations,
                                            ampFloatPartials, inChansPerGroup,
                                            partialChansPerGroup);
    }
  }
  // Constrain the input channel grouping to a mulitple of two if the activation
  // type is half. This ensures that we never need to apply padding when sending
  // activations over the exchange.
  auto grainSize = floatActivations ? 1 : 2;
  const auto roundedInputDepth =
      ((params.getInputDepthPerConvGroup() + grainSize - 1) / grainSize) *
      grainSize;
  unsigned previousInChanGroups = 0;
  for (unsigned inChansPerGroup = grainSize;
       inChansPerGroup <= roundedInputDepth;
       inChansPerGroup += grainSize) {
    unsigned inChanGroups = (roundedInputDepth + inChansPerGroup - 1) /
                            inChansPerGroup;
    if (inChanGroups == previousInChanGroups) {
      // There is no point considering a larger group size if it doesn't
      // decrease the number of groups - the zero padding increases the
      // amount of work per group and we can't use fewer groups per tile.
      continue;
    }
    if (isFullyConnectedFwd) {
      // The input channels in the forward pass become the output channels of
      // the weight update pass. Make sure it is a multiple of the supported
      // output channels per group.
      if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
        continue;
    }
    convVertexTypeCandidates.emplace_back(Plan::Method::MAC, floatActivations,
                                          floatPartials, inChansPerGroup, 1);
    previousInChanGroups = inChanGroups;
  }
  return convVertexTypeCandidates;
}

/// Return whether expanding the specified spatial dimension involves
/// expanding the activations or the weights.
bool expandDimExpandActs(ConvParams &params, unsigned dim) {
  auto paddedInputSize = params.getPaddedDilatedInputSize(dim);
  auto paddedKernelSize = params.getPaddedDilatedKernelSize(dim);
  if (paddedInputSize == paddedKernelSize) {
    // We could legitimately expand either operand. zero padding /
    // input dilation is made explicit for the operand we expand so we are
    // better off expanding the operand with less padding.
    return params.inputFieldShape[dim] > params.kernelShape[dim];
  }
  return paddedInputSize > paddedKernelSize;
}

static void expandDim(ConvParams &params, unsigned dim) {
  if (expandDimExpandActs(params, dim)) {
    params.inputFieldShape[dim] = params.getOutputSize(dim);
    params.inputChannels *= params.kernelShape[dim];
    params.kernelShape[dim] = 1;
  } else {
    params.kernelShape[dim] = params.getOutputSize(dim);
    params.inputChannels *= params.inputFieldShape[dim];
    params.inputFieldShape[dim] = 1;
  }
  params.stride[dim] = 1;
  params.inputDilation[dim] = 1;
  params.inputPaddingLower[dim] = 0;
  params.inputPaddingUpper[dim] = 0;
  params.kernelDilation[dim] = 1;
  params.kernelPaddingLower[dim] = 0;
  params.kernelPaddingUpper[dim] = 0;
}

static unsigned
estimateTransformCycles(const poplar::DeviceInfo &deviceInfo,
                        const ConvParams &params,
                        std::vector<unsigned> expandDims,
                        unsigned inChansPadding,
                        unsigned partialChansPadding,
                        unsigned usedTiles) {
  bool rearrangeInput = !expandDims.empty() || inChansPadding;
  bool rearrangeWeights = !expandDims.empty() || inChansPadding ||
                          partialChansPadding;
  bool rearrangeOutput = partialChansPadding;
  auto expandedParams = params;
  for (const auto dim : expandDims) {
    expandDim(expandedParams, dim);
  }
  expandedParams.inputChannels += inChansPadding;
  expandedParams.outputChannels += partialChansPadding;

  unsigned expandedInputFieldSize = 1;
  unsigned expandedFilterSize = 1;
  unsigned expandedOutputFieldSize = 1;
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    expandedInputFieldSize *= params.inputFieldShape[dim];
    expandedFilterSize *= params.kernelShape[dim];
    expandedOutputFieldSize *= params.getOutputSize(dim);
  }
  unsigned cycles = 0;
  const auto bytesPerElement = expandedParams.dType == "float" ? 4U : 2U;
  const auto expandedInputChannelsPerGroup =
      expandedParams.getInputDepthPerConvGroup();
  unsigned rearrangeElementsPerTile = 0;
  if (rearrangeInput) {
    const auto expandedInputElements =
        expandedInputFieldSize * expandedInputChannelsPerGroup *
        expandedParams.getBatchSize() * expandedParams.getNumConvGroups();
    const auto expandedInputElementsPerTile =
        (expandedInputElements + usedTiles - 1) / usedTiles;
    rearrangeElementsPerTile += expandedInputElementsPerTile;
  }
  if (rearrangeWeights) {
    const auto expandedFilterElements =
        expandedFilterSize * expandedInputChannelsPerGroup *
        expandedParams.getOutputDepthPerConvGroup() *
        expandedParams.getNumConvGroups();
    const auto expandedFilterElementsPerTile =
        (expandedFilterElements + usedTiles - 1) / usedTiles;
    rearrangeElementsPerTile += expandedFilterElementsPerTile;
  }
  if (rearrangeOutput) {
    const auto expandedOutputElements =
        expandedOutputFieldSize * expandedParams.getOutputDepthPerConvGroup() *
        expandedParams.getBatchSize() * expandedParams.getNumConvGroups();
    const auto expandedOutputElementsPerTile =
        (expandedOutputElements + usedTiles - 1) / usedTiles;
    rearrangeElementsPerTile += expandedOutputElementsPerTile;
  }
  const auto expandedBytesPerTile =
      rearrangeElementsPerTile * bytesPerElement;
  // Estimate cost assuming every byte must be exchanged and copied.
  const auto exchangeBytesPerCycle = deviceInfo.exchangeBytesPerCycle;
  cycles += (expandedBytesPerTile + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  // Assume we copy at most one element per cycle.
  const auto reorderBytesPerCycle =
      std::min(deviceInfo.memcpyBytesPerCycle, bytesPerElement);
  cycles += (expandedBytesPerTile + reorderBytesPerCycle - 1) /
            reorderBytesPerCycle;
  // Apply an experimentally determined fudge factor to account for other
  // overheads that aren't modeled.
  double fudgeFactor = 1.5;
  cycles *= fudgeFactor;
  return cycles;
}

static bool expandingDimChangesParams(const ConvParams &params, unsigned dim) {
  auto newParams = params;
  expandDim(newParams, dim);
  return newParams != params;
}

// Given a set return the set of all subsets. The set is specified as a
// vector that is assumed to have no duplicates. The relative order of
// items in each subset returned by this function matches the relative order
// of the items in the set of all items.
template <class T>
static std::vector<std::vector<T>> getPowerSet(const std::vector<T> &items) {
  unsigned numItems = items.size();
  if (numItems >= std::numeric_limits<unsigned>::digits) {
    // Not handled.
    std::abort();
  }
  std::vector<std::vector<T>> subsets;
  // We associate each subset with a number. The nth bit of the number indicates
  // whether the nth item is in the subset. We enumerate all subsets by
  // iterating over all numbers in the range [0, 1 << numItems).
  for (unsigned i = 0; i < (1 << numItems); ++i) {
    subsets.emplace_back();
    for (unsigned item = 0; item != numItems; ++item) {
      if ((i >> item) & 1)
        subsets.back().push_back(items[item]);
    }
  }
  return subsets;
}

static std::vector<std::vector<unsigned>>
getExpandDimsCandidates(const ConvParams &params) {
  std::vector<unsigned> candidateDims;
  for (unsigned i = 0; i != params.getNumFieldDims(); ++i) {
    if (expandingDimChangesParams(params, i)) {
      candidateDims.push_back(i);
    }
  }
  auto subsets = getPowerSet(candidateDims);
  for (auto &subset : subsets) {
    // The subsets returned by getPowerSet have the outermost dimension first
    // but it is more efficient to expand the innermost dimension first.
    std::reverse(subset.begin(), subset.end());
  }
  return subsets;
}

static bool dimCanBeFlattened(const ConvParams &params, unsigned dim) {
  return params.getPaddedDilatedKernelSize(dim) == 1 &&
         params.stride[dim] == 1 &&
         params.inputDilation[dim] == 1 &&
         params.inputPaddingLower[dim] == 0 &&
         params.inputPaddingUpper[dim] == 0;
}

static bool
dimCanBeFlattenedIntoOutChans(const ConvParams &params, unsigned dim) {
  return params.getPaddedDilatedInputSize(dim) == 1 &&
         params.stride[dim] == 1 &&
         params.kernelDilation[dim] == 1 &&
         params.kernelPaddingLower[dim] == 0 &&
         params.kernelPaddingUpper[dim] == 0;
}

static std::pair<Plan, Cost>
createPlan(ConvParams params,
           std::string partialsType,
           const ConvOptions &options,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlanningCacheImpl *cache) {
  validateLayerParams(params);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  Cost bestCost = highestCost;
  Plan bestPlan;
  for (std::vector<unsigned> expandDims :
       getExpandDimsCandidates(params)) {
    auto expandedParams = params;
    for (unsigned dim : expandDims) {
      expandDim(expandedParams, dim);
    }
    std::vector<unsigned> outChanFlattenDims;

    for (unsigned spatialDim = 0;
         spatialDim != expandedParams.getNumFieldDims();
         ++spatialDim) {
      if (dimCanBeFlattenedIntoOutChans(expandedParams, spatialDim) &&
          expandedParams.kernelShape[spatialDim] > 1) {
        outChanFlattenDims.push_back(spatialDim);
        expandedParams.outputChannels *= expandedParams.kernelShape[spatialDim];
        expandedParams.kernelShape[spatialDim] = 1;
      }
    }
    // Flatten from the innermost out.
    std::reverse(outChanFlattenDims.begin(), outChanFlattenDims.end());
    std::vector<unsigned> flattenDims;
    flattenDims.push_back(0);
    for (unsigned spatialDim = 0;
         spatialDim != expandedParams.getNumFieldDims();
         ++spatialDim) {
      if (dimCanBeFlattened(expandedParams, spatialDim)) {
        flattenDims.push_back(spatialDim + 1);
      }
    }
    if (flattenDims.size() > 1) {
      const auto innermostFlattenableDim = flattenDims.back();
      assert(innermostFlattenableDim > 0);
      for (auto it = std::next(flattenDims.rbegin()),
           end = flattenDims.rend(); it != end; ++it) {
        const auto fromDimIndex = *it;
        auto &fromDimSize =
            fromDimIndex ? expandedParams.inputFieldShape[fromDimIndex - 1] :
                           expandedParams.batchSize;
        expandedParams.inputFieldShape[innermostFlattenableDim - 1] *=
            fromDimSize;
        fromDimSize = 1;
      }
    } else {
      flattenDims.clear();
    }
    const bool floatActivations = params.dType == "float";
    const bool floatPartials = partialsType == "float";
    const auto convVertexTypeCandidates =
        getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                    floatPartials, params, options);
    for (const auto &convVertexType : convVertexTypeCandidates) {
      auto paddedParams = expandedParams;
      const auto inChansPerGroup = convVertexType.inChansPerGroup;
      const auto inChans = expandedParams.getInputDepthPerConvGroup();
      paddedParams.inputChannels =
          ((inChans + inChansPerGroup - 1) / inChansPerGroup) *
          inChansPerGroup;
      unsigned inChansPadding = paddedParams.inputChannels - inChans;
      const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
      const auto partialChans = expandedParams.getOutputDepthPerConvGroup();
      paddedParams.outputChannels =
          ((partialChans + partialChansPerGroup - 1) / partialChansPerGroup) *
          partialChansPerGroup;
      unsigned partialChansPadding = paddedParams.outputChannels - partialChans;
      unsigned xAxisGrainSize = 1;
      if (options.fullyConnectedPass == FullyConnectedPass::FWD) {
        // The xAxisGrainSize becomes the inChansPerGroup in the backward pass.
        // For now assume the same grouping in both passes.
        // TODO search for the optimal grouping in each pass.
        xAxisGrainSize = convVertexType.inChansPerGroup;
      }
      Plan candidate;
      Cost candidateCost;
      auto transformCostFn = [&](unsigned usedTiles) {
        return estimateTransformCycles(graph.getDevice().getDeviceInfo(),
                                       paddedParams, expandDims, inChansPadding,
                                       partialChansPadding, usedTiles);
      };
      std::tie(candidate, candidateCost) =
          choosePlan(deviceInfo, xAxisGrainSize, convVertexType, paddedParams,
                     transformCostFn, bestCost, costBounds, cache, options);
      candidate.expandDims = expandDims;
      candidate.outChanFlattenDims = outChanFlattenDims;
      candidate.flattenDims = flattenDims;
      if (compareCost(candidateCost, bestCost, costBounds)) {
        bestPlan = candidate;
        bestCost = candidateCost;
      }
    }
  }
  return {bestPlan, bestCost};
}

static ConvParams getFullyConnectedFwdParams(const ConvParams &params,
                                             const ConvOptions &options) {
  // Translate back into parameters of the fully connected layer.
  unsigned outputSize, inputSize, batchSize;
  assert(params.getInputHeight() == 1);
  assert(params.stride == std::vector<unsigned>({1U, 1U}));
  assert(params.inputPaddingLower == std::vector<int>({0, 0}));
  assert(params.inputPaddingUpper == std::vector<int>({0, 0}));
  assert(params.kernelShape[0] == 1 && params.kernelShape[1] == 1);
  assert(params.inputDilation[0] == 1 && params.inputDilation[1] == 1);
  switch (options.fullyConnectedPass) {
  default: assert(0 && "Unexpected pass");
  case FullyConnectedPass::BWD:
    inputSize = params.getInputWidth();
    batchSize = params.getOutputDepthPerConvGroup();
    outputSize = params.getInputDepthPerConvGroup();
    break;
  case FullyConnectedPass::WU:
    outputSize = params.getInputWidth();
    batchSize = params.getInputDepthPerConvGroup();
    inputSize = params.getOutputDepthPerConvGroup();
    break;
  }
  return ConvParams(params.dType,
                    1,                       // batchSize
                    {1, outputSize},         // inputShape
                    {1, 1},                  // kernelShape
                    inputSize,               // input channels
                    batchSize,               // output channels
                    {1, 1},                  // stride
                    {0, 0},
                    {0, 0},
                    {1, 1},
                    {0, 0},
                    {0, 0},
                    {1, 1},
                    params.getNumConvGroups());
}

static ConvOptions getFullyConnectedFwdOptions(const ConvOptions &options) {
  auto newOptions = options;
  newOptions.fullyConnectedPass = FullyConnectedPass::FWD;
  return newOptions;
}

static Plan getFullyConnectedWUPlan(const poplar::DeviceInfo &deviceInfo,
                                    const ConvParams &fwdParams,
                                    const ConvOptions &fwdOptions,
                                    const Plan &fwdPlan) {
  assert(fwdPlan.method == Plan::Method::AMP ||
         fwdPlan.method == Plan::Method::MAC);
  auto plan = fwdPlan;
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_WU;
  plan.tilesPerInZGroupAxis = fwdPlan.tilesPerZAxis;
  plan.tilesPerZAxis = fwdPlan.tilesPerInZGroupAxis;
  plan.partialChansPerGroup = fwdPlan.inChansPerGroup;
  plan.tilesPerConvGroups = fwdPlan.tilesPerConvGroups;
  plan.method = getFullyConnectedWUMethod(fwdParams, fwdPlan.method,
                                          fwdPlan.inChansPerGroup);
  // TODO make the fwd pass aware that it would be good to use a grouping of
  // 16 if possible.
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  if (plan.method == Plan::Method::AMP &&
      !canUseConvolutionInstruction(fwdParams.dType == "float",
                                    fwdOptions.partialsType == "float",
                                    plan.inChansPerGroup, deviceInfo)) {
    plan.inChansPerGroup =
        deviceInfo.getWeightsPerConvUnit(fwdParams.dType == "float");
  }
  // If the result type is half and all the reduction is done within a single
  // pass of the AMP unit then there is no reason to use a higher precision
  // partial type.
  if (fwdParams.dType != "float" &&
      fwdParams.getOutputDepthPerConvGroup() == plan.inChansPerGroup &&
      deviceInfo.fp16InFp16OutConvUnitsPerTile ==
      deviceInfo.fp16InFp32OutConvUnitsPerTile) {
    plan.floatPartials = false;
  }
  return plan;
}

static Plan getFullyConnectedBwdPlan(const poplar::DeviceInfo &deviceInfo,
                                      const ConvParams &fwdParams,
                                      const ConvOptions &fwdOptions,
                                      const Plan &fwdPlan) {
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdParams, fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  std::swap(plan.tilesPerXAxis, plan.tilesPerInZGroupAxis);
  std::swap(plan.xAxisGrainSize, plan.inChansPerGroup);
  return plan;
}

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  assert (params.kernelShape.size() == 2);
  assert (params.stride.size() == 2);
  assert (params.inputPaddingLower.size() == 2);
  assert (params.inputPaddingUpper.size() == 2);
  if (options.fullyConnectedPass == FullyConnectedPass::WU ||
      options.fullyConnectedPass == FullyConnectedPass::BWD) {
    auto fwdParams = getFullyConnectedFwdParams(params, options);
    auto fwdOptions = getFullyConnectedFwdOptions(options);
    const auto fwdPlan =
        getPlan(graph, fwdParams, fwdOptions);
    if (options.fullyConnectedPass == FullyConnectedPass::WU)
      return getFullyConnectedWUPlan(deviceInfo, fwdParams, fwdOptions,
                                     fwdPlan);
    assert(options.fullyConnectedPass == FullyConnectedPass::BWD);
    return getFullyConnectedBwdPlan(deviceInfo, fwdParams, fwdOptions,
                                    fwdPlan);
  }
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
        params.inputDilation[0] != 1 || params.inputDilation[1] != 1 ||
        params.kernelShape[0] != 3 || params.kernelShape[1] != 3 ||
        params.getNumConvGroups() == 1) {
      throw popstd::poplib_error("Attempt to force winograd convolution for "
                               "invalid parameters");

    }
    plan.useWinograd = true;
    plan.winogradPatchSize = options.winogradPatchSize;
    return plan;
  }

  std::tie(plan, cost) = popconv::createPlan(params,
                                             partialsType,
                                             options,
                                             costBounds, graph,
                                             cache);
  if (options.percentageCyclesExcessForMemOptim) {
    throw popstd::poplib_error("Optimizing for memory is not supported");
  }
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(plan)));
    auto res = plans.emplace(std::make_pair(key, std::move(pPlan)));
    return *res.first->second;
  }
  return plan;
}

} // end namespace conv
