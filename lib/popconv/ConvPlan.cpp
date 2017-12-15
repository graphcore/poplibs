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

static bool equalsZero(unsigned x) {
  return x == 0;
};

static bool equalsOne(unsigned x) {
  return x == 1;
};

std::uint64_t getNumberOfMACs(const ConvParams &params) {
  std::uint64_t numMACs = params.getNumConvGroups() *
                          params.getBatchSize() *
                          params.getNumOutputChansPerConvGroup() *
                          params.getNumInputChansPerConvGroup();
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
                                const poplar::Target &target) {
  if (floatActivations) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  } else {
    return floatPartial ? target.getFp16InFp32OutConvUnitsPerTile() :
                          target.getFp16InFp16OutConvUnitsPerTile();
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
  os << "  Plan: fieldTileSplit          ";
  printContainer(p.fieldTileSplit, os);
  os << "\n"
     << "        batchTileSplit          " << p.batchTileSplit << "\n"
     << "        outChanTileSplit        " << p.outChanTileSplit << "\n"
     << "        kernelTileSplit         ";
  printContainer(p.kernelTileSplit, os);
  os << "\n"
     << "        inChanTileSplit         " << p.inChanTileSplit << "\n"
     << "        convGroupTileSplit      " << p.convGroupTileSplit << "\n"
     << "        fieldAxisGrainSize      ";
  printContainer(p.fieldAxisGrainSize, os);
  os << "\n"
     << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n"
     << "        swapOperands            " << p.swapOperands << "\n"
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
getConvPartialnx1CycleEstimate(unsigned convGroups,
                               unsigned batchElements,
                               const std::vector<unsigned> &outShape,
                               unsigned numInGroups,
                               unsigned numOutGroups,
                               const std::vector<unsigned> &kernelShape,
                               unsigned filterHeight,
                               unsigned inChansPerGroup,
                               unsigned convUnitInputLoadElemsPerCycle,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               unsigned numWorkerContexts,
                               bool floatWeights,
                               const std::vector<unsigned> &inputDilation,
                               const std::vector<unsigned> &stride)
{
  uint64_t cycles = 0;
  auto kernelElements = std::accumulate(kernelShape.begin(),
                                        kernelShape.end(), 1UL,
                                        std::multiplies<std::size_t>());
  std::vector<std::vector<PartialRow>> partition =
      partitionConvPartialByWorker(batchElements, outShape,
                                   numWorkerContexts, inputDilation,
                                   stride);

  unsigned numEdges = numInGroups + numOutGroups + numInGroups * numOutGroups;
  // use conv nx1 vertex
  std::vector<std::vector<std::vector<unsigned>>> workList;
  unsigned positionsOuter =
      (kernelShape[0] + filterHeight - 1) / filterHeight;
  unsigned numKernelPositions =
      (positionsOuter * kernelElements / kernelShape[0]);
  const auto outStrideX = inputDilation.back() / gcd(inputDilation.back(),
                                                     stride.back());
  for (unsigned context = 0; context < numWorkerContexts; ++context) {
    workList.emplace_back();
    for (auto k = 0U; k != numKernelPositions; ++k) {
      workList.back().emplace_back();
      for (const auto &partialRow : partition[context]) {
        const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
        auto numFieldPos = (workerOutWidth + outStrideX - 1) / outStrideX;
        if (numFieldPos) {
          workList.back().back().push_back(numFieldPos);
        }
      }
    }
  }
  cycles = getConvPartialnx1SupervisorCycleEstimate(
                      workList, convGroups, numOutGroups, numInGroups,
                      numKernelPositions, filterHeight,
                      inChansPerGroup, convUnitInputLoadElemsPerCycle,
                      numConvUnitsPerTile,
                      convUnitCoeffLoadBytesPerCycle,
                      numWorkerContexts, floatWeights,
                      useDeltaEdgesForConvPartials(numEdges));
  return cycles;
}

static std::uint64_t
getConvPartial1x1CycleEstimate(unsigned convGroups,
                               unsigned batchElements,
                               const std::vector<unsigned> &outShape,
                               unsigned numInGroups,
                               unsigned numOutGroups,
                               unsigned inChansPerGroup,
                               unsigned convUnitInputLoadElemsPerCycle,
                               unsigned numConvUnitsPerTile,
                               unsigned convUnitCoeffLoadBytesPerCycle,
                               unsigned numWorkerContexts,
                               bool floatWeights,
                               const std::vector<unsigned> &inputDilation,
                               const std::vector<unsigned> &stride)
{
  assert(inputDilation == stride);
  uint64_t cycles = 0;
  std::vector<std::vector<PartialRow>> partition =
      partitionConvPartialByWorker(batchElements, outShape,
                                   numWorkerContexts, inputDilation,
                                   stride);

  unsigned numEdges = numInGroups + numOutGroups + numInGroups * numOutGroups;
  // use conv 1x1 vertex
  std::vector<std::vector<unsigned>> worklist(numWorkerContexts);
  for (unsigned context = 0; context != numWorkerContexts; ++context) {
    for (const auto &partialRow : partition[context]) {
      const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
      if (workerOutWidth == 0)
        continue;
      worklist[context].push_back(workerOutWidth);
    }
  }
  cycles += getConvPartial1x1SupervisorCycleEstimate(
                      worklist, convGroups, numInGroups, numOutGroups,
                      convUnitInputLoadElemsPerCycle,
                      numConvUnitsPerTile, convUnitCoeffLoadBytesPerCycle,
                      numWorkerContexts, floatWeights,
                      useDeltaEdgesForConvPartials(numEdges));
  return cycles;
}

static std::uint64_t
estimateConvPartialHorizontalMacCycles(unsigned tileNumConvGroups,
                                       unsigned tileNumInGroups,
                                       unsigned tileNumOutGroups,
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
  decltype(memoize(getConvPartial1x1CycleEstimate))
    mGetConvPartial1x1CycleEstimate;
  decltype(memoize(getConvPartialnx1CycleEstimate))
    mGetConvPartialnx1CycleEstimate;
  decltype(memoize(estimateConvPartialHorizontalMacCycles))
    mEstimateConvPartialHorizontalMacCycles;
  PlanningCacheImpl() :
    mGetConvPartial1x1CycleEstimate(
      memoize(getConvPartial1x1CycleEstimate)
    ),
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
getConvUnitsPerTile(const poplar::Target &target,
                    bool floatActivations, bool floatPartials) {
  if (floatActivations) {
    return floatPartials ? target.getFp32InFp32OutConvUnitsPerTile() : 0;
  }
  return floatPartials ? target.getFp16InFp32OutConvUnitsPerTile() :
                         target.getFp16InFp16OutConvUnitsPerTile();
}

static bool
canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
                             const poplar::Target &target) {
  if (getConvUnitsPerTile(target, floatActivations, floatPartials) == 0) {
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
                             const poplar::Target &target) {
  if (!canUseConvolutionInstruction(floatActivations, floatPartials,
                                    target))
    return false;
  if (target.getWeightsPerConvUnit(floatActivations) %
      inChansPerGroup != 0) {
    return false;
  }
  // Check we can use aligned loads.
  if ((inChansPerGroup * (floatActivations ? 32 : 16)) %
      target.getDataPathWidth() != 0) {
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

static unsigned
getMaxTileOutSize(const ConvParams &params, const Plan &plan, unsigned dim) {
  const auto outSize = params.getOutputSize(dim);
  const auto grainSize = plan.fieldAxisGrainSize[dim];
  const auto numGrains = (outSize + grainSize - 1) / grainSize;
  const auto dimTileSplit = plan.fieldTileSplit[dim];
  const auto tileNumGrains = (numGrains + dimTileSplit - 1) / dimTileSplit;
  const auto tileOutSize = std::min(outSize, tileNumGrains * grainSize);
  return tileOutSize;
}

static unsigned
estimateExchangeCycles(const poplar::Target &target,
                       bool floatActivations,
                       const ConvParams &params,
                       const Plan &plan) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto batchTileSplit = plan.batchTileSplit;
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto inChanTileSplit = plan.inChanTileSplit;
  const auto convGroupTileSplit = plan.convGroupTileSplit;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto tileBatchElements =
      (params.getBatchSize() + batchTileSplit - 1) / batchTileSplit;
  const auto numOutChans = params.getNumOutputChansPerConvGroup();
  const auto numOutGroups =
      (numOutChans + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + outChanTileSplit - 1) / outChanTileSplit;
  const auto tileNumOutChans = tileNumOutGroups * partialChansPerGroup;
  const auto numInChans = params.getNumInputChansPerConvGroup();
  const auto numInGroups =
      (numInChans + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + inChanTileSplit - 1) / inChanTileSplit;
  const auto tileNumInChans = tileNumInGroups * inChansPerGroup;
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + convGroupTileSplit - 1) / convGroupTileSplit;
  auto numberOfInputElements = tileBatchElements * tileNumInChans *
                               tileNumGroupedConv;
  auto numberOfWeights = tileNumOutChans * tileNumInChans * tileNumGroupedConv;
  auto numberOfOutputElements = tileBatchElements * tileNumOutChans *
                                tileNumGroupedConv;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto tileKernelSize =
        (params.kernelShape[dim] + plan.kernelTileSplit[dim] - 1) /
        plan.kernelTileSplit[dim];
    const auto tileOutSize = getMaxTileOutSize(params, plan, dim);
    bool contiguousAccess = dim == numFieldDims - 1;
    const auto tileInSize =
        getMaxInputRangeSize(tileOutSize, dim, params,
                             tileKernelSize, plan.fieldTileSplit[dim],
                             contiguousAccess);
    numberOfInputElements *= tileInSize;
    numberOfWeights *= tileKernelSize;
    numberOfOutputElements *= tileOutSize;
  }
  const auto activationSize = floatActivations ? 4 : 2;
  auto inputElementsBytes = numberOfInputElements * activationSize;
  auto weightBytes = numberOfWeights * activationSize;
  const auto partialSize = plan.floatPartials ? 4 : 2;
  const auto numberOfPartialSums = numberOfOutputElements;
  const auto partialSumBytes = numberOfPartialSums * partialSize;

  const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  const auto inputElementBytesPerCycle =
      (target.supportsExchangeBusSharing() &&
       plan.linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD &&
       (outChanTileSplit % tilesPerSuperTile) == 0) ? exchangeBytesPerCycle *
                                                      tilesPerSuperTile :
                                                      exchangeBytesPerCycle;
  const auto numCycles =
      (inputElementsBytes + inputElementBytesPerCycle - 1) /
      inputElementBytesPerCycle +
      (weightBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle +
      (partialSumBytes + exchangeBytesPerCycle - 1) / exchangeBytesPerCycle;

  return numCycles;
}

static std::uint64_t
estimateConvPartialHorizontalMacCycles(unsigned tileNumConvGroups,
                                       unsigned tileNumInGroups,
                                       unsigned tileNumOutGroups,
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
  const auto workerPartRows = maxPartRows % rowSplitFactor ;
  const auto wholeRowConvSize =
      (tileOutWidth + outputStrideX - 1) / outputStrideX;
  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  workerPartitions.emplace_back();
  const auto kernelSize = tileKernelWidth * tileKernelHeight;
  for (auto k = 0U; k != kernelSize; ++k) {
    workerPartitions.back().emplace_back();
    if (wholeRowConvSize) {
      for (unsigned r = 0; r != workerWholeRows; ++r) {
        workerPartitions.back().back().push_back(wholeRowConvSize);
      }
      if (workerPartRows) {
        auto convSize =
          workerPartRows *
          (wholeRowConvSize +  rowSplitFactor - 1) / rowSplitFactor;
        workerPartitions.back().back().push_back(convSize);
      }
    }
  }

  return getConvPartialHorizontalMacSupervisorCycleEstimate(
    workerPartitions,
    tileNumConvGroups,
    tileNumInGroups,
    tileNumOutGroups,
    kernelSize,
    inChansPerGroup,
    dataPathWidth,
    numWorkers,
    floatActivations);
}

static bool zeroPartials(const ConvParams &params, const Plan &plan) {
  if (plan.method == Plan::Method::MAC) {
    return true;
  }
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto tileKernelSize =
        (params.kernelShape[dim] + plan.kernelTileSplit[dim] - 1) /
        plan.kernelTileSplit[dim];
    if (tileKernelSize != 1)
      return true;
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
estimateZeroCycles(const poplar::Target &target,
                   const ConvParams &params,
                   const Plan &plan) {
  if (!zeroPartials(params, plan)) {
    return 0;
  }
  const auto batchTileSplit = plan.batchTileSplit;
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto convGroupTileSplit = plan.convGroupTileSplit;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto tileBatchElements =
      (params.getBatchSize() + batchTileSplit - 1) / batchTileSplit;
  const auto numOutChans = params.getNumOutputChansPerConvGroup();
  const auto numOutGroups =
      (numOutChans + (partialChansPerGroup - 1)) / partialChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + outChanTileSplit - 1) / outChanTileSplit;
  const auto tileNumOutChans = tileNumOutGroups * partialChansPerGroup;
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + convGroupTileSplit - 1) / convGroupTileSplit;
  auto numberOfOutputElements = tileBatchElements * tileNumOutChans *
                                tileNumGroupedConv;
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto tileOutSize = getMaxTileOutSize(params, plan, dim);
    numberOfOutputElements *= tileOutSize;
  }
  const auto vectorWidth =
      plan.floatPartials ? target.getFloatVectorWidth() :
                           target.getHalfVectorWidth();
  const auto numCycles = (numberOfOutputElements + vectorWidth - 1) /
                         vectorWidth;
  return numCycles;
}

static bool
canUseConvPartial1x1Vertex(const ConvParams &params,
                           unsigned convUnitWeightHeight,
                           const std::vector<unsigned> &tileKernelShape) {
  if (convUnitWeightHeight != 1)
    return false;
  if (params.inputDilation != params.stride)
    return false;
  auto tileKernelElements = std::accumulate(tileKernelShape.begin(),
                                        tileKernelShape.end(), 1UL,
                                        std::multiplies<std::size_t>());
  if (tileKernelElements != 1)
    return false;
  // We can only use the 1x1 vertex if every output value is written. It may be
  // the case every output value is written on some tiles but not others - we
  // return false in this case since we are interested in the worse case
  // and we assume the nx1 vertex is always slower.
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    std::pair<unsigned, unsigned> outputRange = {
      0,
      params.getOutputSize(dim)
    };
    for (unsigned k = 0; k != params.kernelShape[dim]; ++k) {
      const auto writtenOutputRange =
          getOutputRange(dim, outputRange, k, params);
      if (writtenOutputRange != outputRange) {
        return false;
      }
    }
  }
  return true;
}

static unsigned
estimatePartialCalcCycles(const poplar::Target &target,
                          bool floatActivations,
                          const ConvParams &params,
                          const Plan &plan,
                          PlanningCacheImpl *cache) {
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto batchTileSplit = plan.batchTileSplit;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto inChanTileSplit = plan.inChanTileSplit;
  const auto convGroupTileSplit = plan.convGroupTileSplit;
  const auto tileBatchElements =
      (params.getBatchSize() + batchTileSplit - 1) / batchTileSplit;
  const auto numOutGroups =
      (params.getNumOutputChansPerConvGroup() + (outChansPerGroup - 1))
      / outChansPerGroup;
  const auto tileNumOutGroups =
      (numOutGroups + outChanTileSplit - 1) / outChanTileSplit;
  const auto tileNumGroupedConv =
      (params.getNumConvGroups() + convGroupTileSplit - 1) / convGroupTileSplit;

  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = target.getNumWorkerContexts();
  if (plan.method == Plan::Method::AMP) {
    numContexts = 1;
  }
  const auto numInGroups =
      (params.getNumInputChansPerConvGroup() + (inChansPerGroup - 1))
      / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + inChanTileSplit - 1) / inChanTileSplit;

  const auto numFieldDims = params.getNumFieldDims();

  unsigned tileOutElements = 1;
  unsigned tileKernelFieldElements = 1;
  std::vector<unsigned> tileOutShape, tileKernelShape;
  tileOutShape.reserve(numFieldDims);
  tileKernelShape.reserve(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto tileOutSize = getMaxTileOutSize(params, plan, dim);
    tileOutShape.push_back(tileOutSize);
    tileOutElements *= tileOutSize;
    const auto tileKernelSize =
        (params.kernelShape[dim] + plan.kernelTileSplit[dim] - 1) /
        plan.kernelTileSplit[dim];
    tileKernelShape.push_back(tileKernelSize);
    tileKernelFieldElements *= tileKernelSize;
  }
  const auto tileKernelWidth =
      (params.kernelShape[numFieldDims - 1] +
       plan.kernelTileSplit[numFieldDims - 1] - 1) /
      plan.kernelTileSplit[numFieldDims - 1];
  const auto tileOutWidth = getMaxTileOutSize(params, plan, numFieldDims - 1);
  unsigned computeCycles;
  switch (plan.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    {
      assert(target.getWeightsPerConvUnit(floatActivations) %
             inChansPerGroup == 0);
      const auto convUnitWeightHeight =
          target.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
      const auto numConvUnits = getNumConvUnits(floatActivations,
                                                plan.floatPartials,
                                                target);
      assert(outChansPerGroup % numConvUnits == 0);
      const auto convUnitInputLoadElemsPerCycle =
          target.getConvUnitInputLoadElemsPerCycle(floatActivations);
      if (canUseConvPartial1x1Vertex(params, convUnitWeightHeight,
                                     tileKernelShape)) {
        computeCycles =
            cache->mGetConvPartial1x1CycleEstimate(
              tileNumGroupedConv, tileBatchElements, tileOutShape,
              tileNumInGroups, tileNumOutGroups, inChansPerGroup,
              convUnitInputLoadElemsPerCycle, numConvUnits,
              target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(),
              floatActivations, params.inputDilation, params.stride);
      } else {
        computeCycles =
            cache->mGetConvPartialnx1CycleEstimate(
              tileNumGroupedConv, tileBatchElements, tileOutShape,
              tileNumInGroups, tileNumOutGroups,
              tileKernelShape, convUnitWeightHeight, inChansPerGroup,
              convUnitInputLoadElemsPerCycle, numConvUnits,
              target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(),
              floatActivations, params.inputDilation, params.stride);
      }
    }
    break;
  case Plan::Method::MAC:
    {
      const auto outputStrideX = params.inputDilation.back();
      unsigned numActiveOutRows = tileBatchElements;
      for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
        const auto dimActiveRows =
            (tileOutShape[dim] + params.inputDilation[dim] - 1) /
            params.inputDilation[dim];
        numActiveOutRows *= dimActiveRows;
      }
      computeCycles =
          cache->mEstimateConvPartialHorizontalMacCycles(
            tileNumGroupedConv,
            tileNumInGroups,
            tileNumOutGroups,
            numActiveOutRows,
            tileOutWidth,
            outputStrideX,
            tileKernelFieldElements / tileKernelWidth,
            tileKernelWidth,
            target.getNumWorkerContexts(),
            floatActivations,
            inChansPerGroup,
            target.getDataPathWidth());
    }
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      assert(tileOutElements == tileOutWidth);
      assert(tileBatchElements == 1);
      assert(tileNumInGroups == 1);
      assert(std::all_of(params.stride.begin(),
                         params.stride.end(), equalsOne));
      assert(std::all_of(params.inputDilation.begin(),
                         params.inputDilation.end(), equalsOne));
      const auto workerOutWidth =
          (tileOutWidth + target.getNumWorkerContexts() - 1) /
          target.getNumWorkerContexts();
      auto vertexRuntime =
          getOuterProductCycleEstimate(floatActivations, workerOutWidth,
                                       tileNumOutGroups * outChansPerGroup *
                                       tileNumGroupedConv,
                                       outChansPerGroup,
                                       target.getDataPathWidth());
      computeCycles = vertexRuntime * numContexts;
    }
    break;
  }
  return computeCycles;
}


static unsigned
estimateReduceCycles(const poplar::Target &target,
                     const ConvParams &params, const Plan &plan) {
  const auto inChanTileSplit = plan.inChanTileSplit;
  if (inChanTileSplit == 1 &&
      std::all_of(plan.kernelTileSplit.begin(),
                  plan.kernelTileSplit.end(), equalsOne))
    return 0;
  const auto outChanTileSplit = plan.outChanTileSplit;
  const auto batchTileSplit = plan.batchTileSplit;
  const auto convGroupTileSplit = plan.convGroupTileSplit;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto tileBatchElements =
      (params.getBatchSize() + batchTileSplit - 1) / batchTileSplit;
  const auto numOutGroups =
      (params.getNumOutputChansPerConvGroup() + (outChansPerGroup - 1))
      / outChansPerGroup;
  const auto tileNumConvGroups =
      (params.getNumConvGroups() + convGroupTileSplit - 1) / convGroupTileSplit;
  const auto tileNumOutGroups =
      (numOutGroups + outChanTileSplit - 1) / outChanTileSplit;
  auto numOutputs = tileBatchElements * tileNumOutGroups * outChansPerGroup *
                    tileNumConvGroups;
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    numOutputs *= getMaxTileOutSize(params, plan, dim);
  }
  // Consider a group of tiles that compute partial sums for the same output
  // volume. The number of partial sums that to be reduced is
  // numOutputs * numTiles. Calculation of the output is spread evenly across
  // the tiles so the number of partial sums each tile must reduce is
  // (numOutputs * numTiles) / numTiles = numOutputs.
  const auto reduceElementsPerTile = numOutputs;

  const auto vectorWidth =
      plan.floatPartials ? target.getFloatVectorWidth() :
                                target.getHalfVectorWidth();
  const auto numCycles = (reduceElementsPerTile + vectorWidth - 1) /
                         vectorWidth;
  return numCycles;
}

unsigned getMaxMACsPerCyclePerTile(const poplar::Target &target,
                                   bool floatPartials,
                                   bool floatActivations,
                                   Plan::Method method) {
  const auto vectorWidth =
      floatActivations ? target.getFloatVectorWidth() :
                         target.getHalfVectorWidth();
  switch (method) {
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT:
    return vectorWidth;
  case Plan::Method::AMP:
    {
      unsigned numConvUnits;
      if (floatActivations) {
        assert(floatPartials);
        numConvUnits = target.getFp32InFp32OutConvUnitsPerTile();
      } else if (floatPartials) {
        numConvUnits = target.getFp16InFp32OutConvUnitsPerTile();
      } else {
        numConvUnits = target.getFp16InFp16OutConvUnitsPerTile();
      }
      return numConvUnits * vectorWidth;
    }
  }
}

static popsolver::Variable
addCycleEstimate(popsolver::Model &m,
                 const std::vector<popsolver::Variable> &fieldTileSplit,
                 const std::vector<popsolver::Variable> &kernelTileSplit,
                 popsolver::Variable batchTileSplit,
                 popsolver::Variable outChanTileSplit,
                 popsolver::Variable inChanTileSplit,
                 popsolver::Variable convGroupTileSplit,
                 popsolver::Variable usedTiles,
                 const poplar::Target &target,
                 const ConvParams &params,
                 unsigned inChansPerGroup,
                 unsigned partialChansPerGroup,
                 const std::vector<unsigned> &fieldGrainSize,
                 bool floatPartials,
                 bool floatActivations,
                 Plan::Method method,
                 Plan::LinearizeTileOrder linearizeTileOrder,
                 PlanningCacheImpl *cache) {
  const auto numFieldDims = fieldTileSplit.size();
  std::vector<popsolver::Variable> variables = {
    {batchTileSplit, outChanTileSplit, inChanTileSplit, convGroupTileSplit}
  };
  variables.insert(variables.end(), fieldTileSplit.begin(),
                   fieldTileSplit.end());
  variables.insert(variables.end(), kernelTileSplit.begin(),
                   kernelTileSplit.end());
  const auto exchangeCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    const auto batchTileSplit = values[0];
    const auto outChanTileSplit = values[1];
    const auto inChanTileSplit = values[2];
    const auto convGroupTileSplit = values[3];
    std::vector<unsigned> fieldTileSplit(values.begin() + 4,
                                         values.begin() + 4 + numFieldDims);
    std::vector<unsigned> kernelTileSplit(values.begin() + 4 + numFieldDims,
                                          values.begin() + 4 +
                                          2 * numFieldDims);
    Plan candidate(fieldTileSplit, batchTileSplit, outChanTileSplit,
                   kernelTileSplit, inChanTileSplit, convGroupTileSplit,
                   inChansPerGroup, partialChansPerGroup, fieldGrainSize,
                   floatPartials, method, linearizeTileOrder);
    return estimateExchangeCycles(target, floatActivations, params,
                                  candidate);
  });
  const auto zeroCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    const auto batchTileSplit = values[0];
    const auto outChanTileSplit = values[1];
    const auto inChanTileSplit = values[2];
    const auto convGroupTileSplit = values[3];
    std::vector<unsigned> fieldTileSplit(values.begin() + 4,
                                         values.begin() + 4 + numFieldDims);
    std::vector<unsigned> kernelTileSplit(values.begin() + 4 + numFieldDims,
                                          values.begin() + 4 +
                                          2 * numFieldDims);
    Plan candidate(fieldTileSplit, batchTileSplit, outChanTileSplit,
                   kernelTileSplit, inChanTileSplit, convGroupTileSplit,
                   inChansPerGroup, partialChansPerGroup, fieldGrainSize,
                   floatPartials, method, linearizeTileOrder);
    return estimateZeroCycles(target, params, candidate);
  });
  const auto partialCalcCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    const auto batchTileSplit = values[0];
    const auto outChanTileSplit = values[1];
    const auto inChanTileSplit = values[2];
    const auto convGroupTileSplit = values[3];
    std::vector<unsigned> fieldTileSplit(values.begin() + 4,
                                         values.begin() + 4 + numFieldDims);
    std::vector<unsigned> kernelTileSplit(values.begin() + 4 + numFieldDims,
                                          values.begin() + 4 +
                                          2 * numFieldDims);
    Plan candidate(fieldTileSplit, batchTileSplit, outChanTileSplit,
                   kernelTileSplit, inChanTileSplit, convGroupTileSplit,
                   inChansPerGroup, partialChansPerGroup, fieldGrainSize,
                   floatPartials, method, linearizeTileOrder);
    return estimatePartialCalcCycles(target, floatActivations, params,
                                     candidate, cache);
  });
  // Add a redunant inequality that relates the cycles required to calculate the
  // partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile =
      getMaxMACsPerCyclePerTile(target, floatPartials, floatActivations,
                                method);
  const auto totalMacs = getNumberOfMACs(params);
  m.lessOrEqual(totalMacs / maxMACsPerCyclePerTile,
                m.product({usedTiles, partialCalcCycles}));
  const auto reduceCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    const auto batchTileSplit = values[0];
    const auto outChanTileSplit = values[1];
    const auto inChanTileSplit = values[2];
    const auto convGroupTileSplit = values[3];
    std::vector<unsigned> fieldTileSplit(values.begin() + 4,
                                         values.begin() + 4 + numFieldDims);
    std::vector<unsigned> kernelTileSplit(values.begin() + 4 + numFieldDims,
                                          values.begin() + 4 +
                                          2 * numFieldDims);
    Plan candidate(fieldTileSplit, batchTileSplit, outChanTileSplit,
                   kernelTileSplit, inChanTileSplit, convGroupTileSplit,
                   inChansPerGroup, partialChansPerGroup, fieldGrainSize,
                   floatPartials, method, linearizeTileOrder);
    return estimateReduceCycles(target, params, candidate);
  });
  return m.sum({exchangeCycles, zeroCycles, partialCalcCycles, reduceCycles});
}

static Plan::Method
getFullyConnectedWUMethod(const ConvParams &fwdParams,
                          Plan::Method fwdMethod,
                          unsigned fwdOutChansPerGroups,
                          unsigned fwdInChansPerGroup) {
  const auto wuInChansPerGroup = fwdOutChansPerGroups;

  // Avoid outer product method if the padded input channels per group are not
  // 1. This is because the current implementation of createOuterProductVertex
  // only supports channel grouping of 1.
  if (fwdParams.getNumOutputChansPerConvGroup() == 1 &&
      wuInChansPerGroup == 1) {
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
choosePlan(const poplar::Target &target,
           const std::vector<unsigned> &fieldGrainSize,
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
  // If yTileSplit is greater than one we end up splitting across the y axis of
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
  const auto numTiles = target.getNumTiles();
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<Variable> fieldTileSplit;
  std::vector<Variable> kernelTileSplit;
  fieldTileSplit.reserve(numFieldDims);
  kernelTileSplit.reserve(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const unsigned numGrains =
        (params.getOutputSize(dim) + fieldGrainSize[dim] - 1) /
        fieldGrainSize[dim];
    fieldTileSplit.push_back(m.addVariable(1, numGrains));
    // Currenlty the implementation doesn't support splitting the innermost
    // kernel dimension. TODO lift this restriction.
    kernelTileSplit.push_back(m.addVariable(1, dim == numFieldDims - 1 ?
                                               1 : params.kernelShape[dim]));
  }
  const auto batchTileSplit = m.addVariable(1, params.getBatchSize());
  const auto convGroupTileSplit = m.addVariable(1, params.getNumConvGroups());
  unsigned maxTilesPerZ;
  if (options.pass == Pass::FC_TRAINING_FWD) {
    // The joint planning cost function assumes that no exchange is required to
    // rearrange weights between passes. Because of the way we derive the
    // backward and weight update plans from the forward plan this is guaranteed
    // to be the case if each weight is used on exactly one tile in the forward
    // pass. Disallow splitting of fully connected batch (or equivalently the
    // convolutional output channels) across tiles to ensure this holds.
    maxTilesPerZ = 1;
  } else {
    const auto outChanGroups =
        (params.getNumOutputChansPerConvGroup() + partialChansPerGroup - 1) /
        partialChansPerGroup;
    maxTilesPerZ = outChanGroups;
  }
  const auto outChanTileSplit = m.addVariable(1, maxTilesPerZ);
  const auto inChanGroups =
      (params.getNumInputChansPerConvGroup() + inChansPerGroup - 1)
      / inChansPerGroup;
  const auto inChanTileSplit = m.addVariable(1, inChanGroups);
  std::vector<Variable> splits = {
    batchTileSplit, outChanTileSplit, inChanTileSplit, convGroupTileSplit
  };
  splits.insert(splits.end(), fieldTileSplit.begin(), fieldTileSplit.end());
  splits.insert(splits.end(), kernelTileSplit.begin(), kernelTileSplit.end());
  const auto usedTiles = m.product(splits);
  m.lessOrEqual(usedTiles, numTiles);

  auto cycles =
      addCycleEstimate(m, fieldTileSplit, kernelTileSplit, batchTileSplit,
                       outChanTileSplit, inChanTileSplit,
                       convGroupTileSplit, usedTiles, target, params,
                       inChansPerGroup, partialChansPerGroup,
                       fieldGrainSize, convVertexType.floatPartials,
                       floatActivations, convVertexType.method,
                       Plan::LinearizeTileOrder::STANDARD, cache);
  if (options.pass == Pass::FC_TRAINING_FWD) {
    popsolver::Variable bwdCycles;
    auto bwdParams = params;
    std::swap(bwdParams.inputFieldShape.back(), bwdParams.inputChannels);
    auto bwdFieldTileSplit = fieldTileSplit;
    bwdFieldTileSplit.back() = inChanTileSplit;
    const auto bwdInChanTileSplit = fieldTileSplit.back();
    const auto bwdInChansPerGroup = fieldGrainSize.back();
    auto bwdFieldGrainSize = fieldGrainSize;
    bwdFieldGrainSize.back() = inChansPerGroup;
    const auto bwdMethod =
        getFullyConnectedBwdMethod(params,
                                   convVertexType.method);
    bwdCycles =
        addCycleEstimate(m, bwdFieldTileSplit, kernelTileSplit, batchTileSplit,
                         outChanTileSplit, bwdInChanTileSplit,
                         convGroupTileSplit,
                         usedTiles, target, params,
                         bwdInChansPerGroup, partialChansPerGroup,
                         bwdFieldGrainSize, convVertexType.floatPartials,
                         floatActivations, bwdMethod,
                         Plan::LinearizeTileOrder::FC_BWD_AS_CONV, cache);
    auto wuParams = params;
    std::swap(wuParams.inputChannels, wuParams.outputChannels);
    const auto wuOutChanTileSplit = inChanTileSplit;
    const auto wuInChanTileSplit = outChanTileSplit;
    const auto wuInChansPerGroup = partialChansPerGroup;
    const auto wuPartialChansPerGroup = inChansPerGroup;
    std::vector<unsigned> wuFieldGrainSize(numFieldDims, 1);
    const auto wuMethod =
        getFullyConnectedWUMethod(params,
                                  convVertexType.method,
                                  partialChansPerGroup,
                                  inChansPerGroup);
    const auto wuCycles =
        addCycleEstimate(m, fieldTileSplit, kernelTileSplit, batchTileSplit,
                         wuOutChanTileSplit,
                         wuInChanTileSplit, convGroupTileSplit,
                         usedTiles, target, wuParams,
                         wuInChansPerGroup, wuPartialChansPerGroup,
                         wuFieldGrainSize, convVertexType.floatPartials,
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
  std::vector<unsigned> fieldTileSplitValues;
  std::vector<unsigned> kernelTileSplitValues;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    fieldTileSplitValues.push_back(s[fieldTileSplit[dim]]);
    kernelTileSplitValues.push_back(s[kernelTileSplit[dim]]);
  }
  Plan plan(fieldTileSplitValues, s[batchTileSplit],
            s[outChanTileSplit], kernelTileSplitValues, s[inChanTileSplit],
            s[convGroupTileSplit],
            inChansPerGroup, partialChansPerGroup,
            fieldGrainSize,
            convVertexType.floatPartials,
            convVertexType.method,
            Plan::LinearizeTileOrder::STANDARD);
  // TODO estimate memory usage.
  unsigned memory = 0;
  Cost cost = {s[cycles], memory};
  return {plan, cost};
}

static bool allKernelDimensionsAreOne(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (params.getPaddedDilatedKernelSize(dim) != 1) {
      return false;
    }
  }
  return true;
}

static bool canUseOuterProductMethod(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
    if (params.getOutputSize(dim) != 1)
      return false;
  }
  return params.getNumInputChansPerConvGroup() == 1 &&
         params.getBatchSize() == 1 &&
         std::all_of(params.stride.begin(),
                     params.stride.end(), equalsOne) &&
         std::all_of(params.inputDilation.begin(),
                     params.inputDilation.end(), equalsOne) &&
         std::all_of(params.flipInput.begin(),
                     params.flipInput.end(), equalsZero) &&
         allKernelDimensionsAreOne(params);
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            bool floatActivations,
                            bool floatPartials,
                            const ConvParams &params,
                            const ConvOptions &options) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (canUseOuterProductMethod(params)) {
    const auto partialChansPerGroup = floatActivations ?
                                      target.getFloatVectorWidth() :
                                      target.getHalfVectorWidth();
    convVertexTypeCandidates.emplace_back(Plan::Method::OUTER_PRODUCT,
                                          floatActivations,
                                          floatActivations, 1,
                                          partialChansPerGroup);
  }
  bool ampFloatPartials = floatPartials;
  auto numConvUnits = getNumConvUnits(floatActivations,
                                      ampFloatPartials,
                                      target);
  if (numConvUnits == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnits = getNumConvUnits(floatActivations,
                                   ampFloatPartials,
                                   target);
  }
  const bool isFullyConnectedFwd =
      options.pass == Pass::FC_TRAINING_FWD;
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   target)) {
    const auto weightsPerConvUnit =
        target.getWeightsPerConvUnit(floatActivations);
    for (unsigned inChansPerGroup = 1; inChansPerGroup <= weightsPerConvUnit;
         ++inChansPerGroup) {
      if (!floatActivations && inChansPerGroup % 2 != 0)
        continue;
      if (!canUseConvolutionInstruction(floatActivations,
                                        floatPartials,
                                        inChansPerGroup, target))
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
  const auto roundedNumInChans =
      ((params.getNumInputChansPerConvGroup() + grainSize - 1) / grainSize) *
      grainSize;
  unsigned previousInChanGroups = 0;
  for (unsigned inChansPerGroup = grainSize;
       inChansPerGroup <= roundedNumInChans;
       inChansPerGroup += grainSize) {
    unsigned inChanGroups = (roundedNumInChans + inChansPerGroup - 1) /
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

static void expandDim(ConvParams &params, unsigned dim) {
  params.inputFieldShape[dim] = params.getOutputSize(dim);
  params.inputChannels *= params.kernelShape[dim];
  params.kernelShape[dim] = 1;
  params.stride[dim] = 1;
  params.inputDilation[dim] = 1;
  params.inputPaddingLower[dim] = 0;
  params.inputPaddingUpper[dim] = 0;
  params.flipInput[dim] = false;
  params.kernelDilation[dim] = 1;
  params.kernelPaddingLower[dim] = 0;
  params.kernelPaddingUpper[dim] = 0;
  params.flipKernel[dim] = false;
}

static unsigned
estimateTransformCycles(const poplar::Target &target,
                        const ConvParams &params,
                        const ConvOptions &options,
                        bool swapOperands,
                        std::vector<unsigned> expandDims,
                        unsigned inChansPadding,
                        unsigned partialChansPadding,
                        unsigned usedTiles) {
  assert(options.pass != Pass::FC_TRAINING_WU &&
         options.pass != Pass::FC_TRAINING_BWD);
  bool isWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool rearrangeInput = isWeightUpdate || !expandDims.empty() || swapOperands ||
                        inChansPadding;
  bool rearrangeWeights = isWeightUpdate || !expandDims.empty() ||
                          swapOperands || inChansPadding || partialChansPadding;
  bool rearrangeOutput = (!isWeightUpdate && swapOperands) ||
                         (isWeightUpdate && !swapOperands) ||
                         partialChansPadding;
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
  const auto bytesPerElement = target.getTypeSize(expandedParams.dType);
  const auto expandedInputChannelsPerGroup =
      expandedParams.getNumInputChansPerConvGroup();
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
        expandedParams.getNumOutputChansPerConvGroup() *
        expandedParams.getNumConvGroups();
    const auto expandedFilterElementsPerTile =
        (expandedFilterElements + usedTiles - 1) / usedTiles;
    rearrangeElementsPerTile += expandedFilterElementsPerTile;
  }
  if (rearrangeOutput) {
    const auto expandedOutputElements =
        expandedOutputFieldSize *
        expandedParams.getNumOutputChansPerConvGroup() *
        expandedParams.getBatchSize() * expandedParams.getNumConvGroups();
    const auto expandedOutputElementsPerTile =
        (expandedOutputElements + usedTiles - 1) / usedTiles;
    rearrangeElementsPerTile += expandedOutputElementsPerTile;
  }
  const auto expandedBytesPerTile =
      rearrangeElementsPerTile * bytesPerElement;
  // Estimate cost assuming every byte must be exchanged and copied.
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  cycles += (expandedBytesPerTile + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  // Assume we copy at most one element per cycle.
  const auto reorderBytesPerCycle =
      std::min<unsigned>(target.getMemcpyBytesPerCycle(), bytesPerElement);
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
  // TODO two dimensions can be flattened if they both have flipInput set to
  // true. To target this we would need to pass information about the two
  // dimensions that are candidates for flattening.
  return params.getPaddedDilatedKernelSize(dim) == 1 &&
         params.stride[dim] == 1 &&
         params.inputDilation[dim] == 1 &&
         params.inputPaddingLower[dim] == 0 &&
         params.inputPaddingUpper[dim] == 0 &&
         !params.flipInput[dim];
}

static bool
dimCanBeFlattenedIntoOutChans(const ConvParams &params, unsigned dim) {
  return params.getPaddedDilatedInputSize(dim) == 1 &&
         params.stride[dim] == 1 &&
         params.kernelDilation[dim] == 1 &&
         params.kernelPaddingLower[dim] == 0 &&
         params.kernelPaddingUpper[dim] == 0 &&
         !params.flipKernel[dim];
}

void
swapOperands(ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto paddedDilatedInputSize = params.getPaddedDilatedInputSize(dim);
    const auto paddedDilatedKernelSize = params.getPaddedDilatedKernelSize(dim);
    std::swap(params.inputFieldShape[dim], params.kernelShape[dim]);
    std::swap(params.inputPaddingLower[dim], params.kernelPaddingLower[dim]);
    std::swap(params.inputPaddingUpper[dim], params.kernelPaddingUpper[dim]);
    std::swap(params.inputDilation[dim], params.kernelDilation[dim]);
    std::swap(params.flipInput[dim], params.flipKernel[dim]);
    params.flipInput[dim] = !params.flipInput[dim];
    params.flipKernel[dim] = !params.flipKernel[dim];
    const auto extraInputPadding =
        paddedDilatedInputSize - paddedDilatedKernelSize;
    params.inputPaddingLower[dim] += extraInputPadding;
    params.inputPaddingUpper[dim] += extraInputPadding;
  }
  std::swap(params.batchSize, params.outputChannels);
  canonicalizeParams(params);
}

static std::vector<bool> getSwapOperandCandidates(const ConvOptions &options) {
  switch (options.pass) {
  case Pass::FC_TRAINING_FWD:
  case Pass::FC_TRAINING_BWD:
  case Pass::FC_TRAINING_WU:
    // The joint planning logic doesn't yet handle swapped operands.
    // TODO lift this restriction.
    return {false};
  default:
    return {false, true};
  }
}

template <class T>
void insertAtFront(std::vector<T> &v, std::size_t n, const T &val) {
  v.insert(v.begin(), n, val);
}

void addExtraDims(ConvParams &params, unsigned extraDims) {
  if (extraDims == 0)
    return;
  insertAtFront(params.inputDilation, extraDims, 1U);
  insertAtFront(params.inputFieldShape, extraDims, std::size_t(1));
  insertAtFront(params.inputPaddingLower, extraDims, 0);
  insertAtFront(params.inputPaddingUpper, extraDims, 0);
  insertAtFront(params.flipInput, extraDims, false);
  insertAtFront(params.kernelDilation, extraDims, 1U);
  insertAtFront(params.kernelPaddingLower, extraDims, 0);
  insertAtFront(params.kernelPaddingUpper, extraDims, 0);
  insertAtFront(params.kernelShape, extraDims, std::size_t(1));
  insertAtFront(params.flipKernel, extraDims, false);
  insertAtFront(params.stride, extraDims, 1U);
}

static std::pair<Plan, Cost>
createPlan(ConvParams params,
           const poplar::Type &partialsType,
           const ConvOptions &options,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlanningCacheImpl *cache) {
  validateLayerParams(params);
  unsigned addedFieldDims = 0;
  auto numFieldDims = params.getNumFieldDims();
  if (numFieldDims < 2) {
    // Various places assume there are at least two dimensions. In particular
    // code related to the nx1ConvPartial vertex has special handling for the
    // outermost dimension and special handling for the innermost dimension
    // and there is an assumption that these two dimensions are distinct.
    addedFieldDims = 2 - numFieldDims;
    addExtraDims(params, addedFieldDims);
    numFieldDims = 2;
  }
  const auto &target = graph.getTarget();
  Cost bestCost = highestCost;
  Plan bestPlan;
  for (bool swapOperands : getSwapOperandCandidates(options)) {
    auto swappedParams = params;
    if (swapOperands) {
      popconv::swapOperands(swappedParams);
    }
    for (std::vector<unsigned> expandDims :
         getExpandDimsCandidates(params)) {
      auto expandedParams = swappedParams;
      for (unsigned dim : expandDims) {
        expandDim(expandedParams, dim);
      }
      std::vector<unsigned> outChanFlattenDims;
      for (unsigned spatialDim = 0; spatialDim != numFieldDims;
           ++spatialDim) {
        if (dimCanBeFlattenedIntoOutChans(expandedParams, spatialDim) &&
            expandedParams.kernelShape[spatialDim] > 1) {
          outChanFlattenDims.push_back(spatialDim);
          expandedParams.outputChannels *=
              expandedParams.kernelShape[spatialDim];
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
      const bool floatActivations = params.dType == poplar::FLOAT;
      const bool floatPartials = partialsType == poplar::FLOAT;
      const auto convVertexTypeCandidates =
          getConvVertexTypeCandidates(target, floatActivations,
                                      floatPartials, expandedParams, options);
      for (const auto &convVertexType : convVertexTypeCandidates) {
        auto paddedParams = expandedParams;
        const auto inChansPerGroup = convVertexType.inChansPerGroup;
        const auto inChans = expandedParams.getNumInputChansPerConvGroup();
        paddedParams.inputChannels =
            ((inChans + inChansPerGroup - 1) / inChansPerGroup) *
            inChansPerGroup;
        unsigned inChansPadding = paddedParams.inputChannels - inChans;
        const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
        const auto partialChans =
            expandedParams.getNumOutputChansPerConvGroup();
        paddedParams.outputChannels =
            ((partialChans + partialChansPerGroup - 1) / partialChansPerGroup) *
            partialChansPerGroup;
        unsigned partialChansPadding = paddedParams.outputChannels -
                                       partialChans;
        std::vector<unsigned> fieldGrainSize(numFieldDims, 1);
        if (options.pass == Pass::FC_TRAINING_FWD) {
          // The innermost grain size becomes the inChansPerGroup in the
          // backward pass. For now assume the same grouping in both passes.
          // TODO search for the optimal grouping in each pass.
          fieldGrainSize.back() = convVertexType.inChansPerGroup;
        }
        Plan candidate;
        Cost candidateCost;
        auto transformCostFn = [&](unsigned usedTiles) {
          return estimateTransformCycles(graph.getTarget(),
                                         paddedParams, options, swapOperands,
                                         expandDims, inChansPadding,
                                         partialChansPadding, usedTiles);
        };
        std::tie(candidate, candidateCost) =
            choosePlan(target, fieldGrainSize, convVertexType,
                       paddedParams, transformCostFn, bestCost, costBounds,
                       cache, options);
        if (candidateCost == highestCost)
          continue;
        candidate.extraFieldDims = addedFieldDims;
        candidate.swapOperands = swapOperands;
        candidate.expandDims = expandDims;
        candidate.outChanFlattenDims = outChanFlattenDims;
        candidate.flattenDims = flattenDims;
        if (compareCost(candidateCost, bestCost, costBounds)) {
          bestPlan = candidate;
          bestCost = candidateCost;
        }
      }
    }
  }
  return {bestPlan, bestCost};
}

static ConvParams getFullyConnectedFwdParams(const ConvParams &params,
                                             const ConvOptions &options) {
  // Translate back into parameters of the fully connected layer.
  unsigned outputSize, inputSize, batchSize;
  assert(params.getNumFieldDims() == 1);
  assert(params.stride[0] == 1);
  assert(params.inputPaddingLower[0] == 0);
  assert(params.inputPaddingUpper[0] == 0);
  assert(params.kernelShape[0] == 1);
  assert(params.inputDilation[0] == 1);
  switch (options.pass) {
  default: assert(0 && "Unexpected pass");
  case Pass::FC_TRAINING_BWD:
    inputSize = params.getInputSize(0);
    batchSize = params.getNumOutputChansPerConvGroup();
    outputSize = params.getNumInputChansPerConvGroup();
    break;
  case Pass::FC_TRAINING_WU:
    outputSize = params.getInputSize(0);
    batchSize = params.getNumInputChansPerConvGroup();
    inputSize = params.getNumOutputChansPerConvGroup();
    break;
  }
  return ConvParams(params.dType,
                    1,                    // batchSize
                    {outputSize},         // inputShape
                    {1},                  // kernelShape
                    inputSize,            // input channels
                    batchSize,            // output channels
                    {1},                  // stride
                    {0},                  // input padding lower
                    {0},                  // input padding upper
                    {1},                  // input dilation
                    {false},              // flip input
                    {0},                  // kernel padding lower
                    {0},                  // kernel padding upper
                    {1},                  // kernel dilation
                    {false},              // flip kernel
                    params.getNumConvGroups());
}

static ConvOptions getFullyConnectedFwdOptions(const ConvOptions &options) {
  auto newOptions = options;
  newOptions.pass = Pass::FC_TRAINING_FWD;
  return newOptions;
}

static Plan getFullyConnectedWUPlan(const poplar::Target &target,
                                    const ConvParams &fwdParams,
                                    const ConvOptions &fwdOptions,
                                    const Plan &fwdPlan) {
  assert(fwdPlan.method == Plan::Method::AMP ||
         fwdPlan.method == Plan::Method::MAC);
  assert(!fwdPlan.swapOperands);
  auto plan = fwdPlan;
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_WU;
  plan.inChanTileSplit = fwdPlan.outChanTileSplit;
  plan.outChanTileSplit = fwdPlan.inChanTileSplit;
  plan.partialChansPerGroup = fwdPlan.inChansPerGroup;

  plan.method = getFullyConnectedWUMethod(fwdParams, fwdPlan.method,
                                          fwdPlan.partialChansPerGroup,
                                          fwdPlan.inChansPerGroup);
  // TODO make the fwd pass aware that it would be good to use a grouping of
  // 16 if possible.
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  if (plan.method == Plan::Method::AMP &&
      !canUseConvolutionInstruction(fwdParams.dType == poplar::FLOAT,
                                    fwdOptions.partialsType == poplar::FLOAT,
                                    plan.inChansPerGroup, target)) {
    plan.inChansPerGroup =
        target.getWeightsPerConvUnit(fwdParams.dType == poplar::FLOAT);
  }
  // If the result type is half and all the reduction is done within a single
  // pass of the AMP unit then there is no reason to use a higher precision
  // partial type.
  if (fwdParams.dType != poplar::FLOAT &&
      fwdParams.getNumOutputChansPerConvGroup() == plan.inChansPerGroup &&
      target.getFp16InFp16OutConvUnitsPerTile() ==
      target.getFp16InFp32OutConvUnitsPerTile()) {
    plan.floatPartials = false;
  }

  // Set the partials type to the output type as there are no reductions
  // required
  if (plan.method == Plan::Method::OUTER_PRODUCT) {
    plan.floatPartials = fwdParams.dType == poplar::FLOAT;
  }
  return plan;
}

static Plan getFullyConnectedBwdPlan(const poplar::Target &target,
                                      const ConvParams &fwdParams,
                                      const ConvOptions &fwdOptions,
                                      const Plan &fwdPlan) {
  assert(!fwdPlan.swapOperands);
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdParams, fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  std::swap(plan.fieldTileSplit.back(), plan.inChanTileSplit);
  std::swap(plan.fieldAxisGrainSize.back(), plan.inChansPerGroup);
  return plan;
}

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             ConvOptions options) {
  const auto &target = graph.getTarget();
  if (options.pass == Pass::FC_TRAINING_WU ||
      options.pass == Pass::FC_TRAINING_BWD) {
    auto fwdParams = getFullyConnectedFwdParams(params, options);
    auto fwdOptions = getFullyConnectedFwdOptions(options);
    const auto fwdPlan =
        getPlan(graph, fwdParams, fwdOptions);
    if (options.pass == Pass::FC_TRAINING_WU)
      return getFullyConnectedWUPlan(target, fwdParams, fwdOptions,
                                     fwdPlan);
    assert(options.pass == Pass::FC_TRAINING_BWD);
    return getFullyConnectedBwdPlan(target, fwdParams, fwdOptions,
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
    assert(params.kernelShape.size() == 2);
    assert(params.stride.size() == 2);
    assert(params.inputPaddingLower.size() == 2);
    assert(params.inputPaddingUpper.size() == 2);
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
