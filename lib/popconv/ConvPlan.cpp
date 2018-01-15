#include "popconv/internal/ConvPlan.hpp"
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
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <popsolver/Model.hpp>
#include "util/print.hpp"

namespace popconv {

namespace {
  struct PartitionVariables {
    std::vector<popsolver::Variable> fieldSplit;
    popsolver::Variable batchSplit;
    popsolver::Variable outChanSplit;
    std::vector<popsolver::Variable> kernelSplit;
    popsolver::Variable inChanSplit;
    popsolver::Variable convGroupSplit;
    std::vector<unsigned> fieldGrainSize;
    unsigned inChanGrainSize;
    unsigned outChanGrainSize;
  };
} // End anonymous namespace

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
  os << "  Plan:";
  const auto numPartitions = p.partitions.size();
  for (unsigned i = 0; i != numPartitions; ++i) {
    os << "        partition #" << i << "\n";
    os << p.partitions[i];
  }
  os << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
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

std::ostream& operator<<(std::ostream &os, const Partition &p) {
  os << "  Partition: fieldSplit          ";
  printContainer(p.fieldSplit, os);
  os << "\n"
     << "             batchSplit          " << p.batchSplit << "\n"
     << "             outChanSplit        " << p.outChanSplit << "\n"
     << "             kernelSplit         ";
  printContainer(p.kernelSplit, os);
  os << "\n"
     << "             inChanSplit         " << p.inChanSplit << "\n"
     << "             convGroupSplit      " << p.convGroupSplit << "\n"
     << "             fieldAxisGrainSize  ";
  printContainer(p.fieldAxisGrainSize, os);
  os << "\n"
     << "             inChanGrainSize     " << p.inChanGrainSize << "\n"
     << "             outChanGrainSize    " << p.outChanGrainSize << "\n";
  return os;
}

static std::uint64_t
getConvPartialnx1InnerLoopCycleEstimate(
    unsigned batchElements,
    const std::vector<unsigned> &outShape,
    const std::vector<unsigned> &kernelShape,
    unsigned filterHeight,
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
  cycles = getConvPartialnx1SupervisorCycleInnerLoopEstimate(
             workList, numKernelPositions, filterHeight,
             convUnitInputLoadElemsPerCycle, numConvUnitsPerTile,
             convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights);
  return cycles;
}


static std::uint64_t
getConvPartial1x1InnerLoopCycleEstimate(
    unsigned batchElements,
    const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride) {
  assert(inputDilation == stride);
  uint64_t cycles = 0;
  std::vector<std::vector<PartialRow>> partition =
      partitionConvPartialByWorker(batchElements, outShape,
                                   numWorkerContexts, inputDilation,
                                   stride);
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
  cycles +=
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(worklist,
                                                        numWorkerContexts);
  return cycles;
}

static std::uint64_t
estimateConvPartialHorizontalMacInnerLoopCycles(unsigned numOutRows,
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
  decltype(memoize(getConvPartial1x1InnerLoopCycleEstimate))
    mGetConvPartial1x1InnerLoopCycleEstimate;
  decltype(memoize(getConvPartialnx1InnerLoopCycleEstimate))
    mGetConvPartialnx1InnerLoopCycleEstimate;
  decltype(memoize(estimateConvPartialHorizontalMacInnerLoopCycles))
    mEstimateConvPartialHorizontalMacInnerLoopCycles;
  PlanningCacheImpl() :
    mGetConvPartial1x1InnerLoopCycleEstimate(
      memoize(getConvPartial1x1InnerLoopCycleEstimate)
    ),
    mGetConvPartialnx1InnerLoopCycleEstimate(
      memoize(getConvPartialnx1InnerLoopCycleEstimate)
    ),
    mEstimateConvPartialHorizontalMacInnerLoopCycles(
      memoize(estimateConvPartialHorizontalMacInnerLoopCycles)
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
                     const ConvParams &params, unsigned tileKernelSize)  {
  if (outputRangeSize == 0)
    return 0;
  if (outputRangeSize == params.getOutputSize(dim) &&
      tileKernelSize == params.kernelShape[dim]) {
    auto inputRange = getInputRange(dim, {0, outputRangeSize}, params);
    return inputRange.second - inputRange.first;
  }
  const auto stride = params.stride[dim];
  const auto inputDilation = params.inputDilation[dim];
  const auto preDownSampleOutputSize = (outputRangeSize - 1) * stride + 1;
  const auto dilatedInputSize = preDownSampleOutputSize + tileKernelSize - 1;
  const auto inputRangeSize = (dilatedInputSize - 1) / inputDilation + 1;
  return inputRangeSize;
}

static unsigned
getMaxSize(unsigned size, unsigned split, unsigned grainSize = 1) {
  const auto numGrains = (size + grainSize - 1) / grainSize;
  const auto tileNumGrains = (numGrains + split - 1) / split;
  return std::min(size, tileNumGrains * grainSize);
}

static unsigned
getMaxTileOutSize(const ConvParams &params, const Plan &plan, unsigned dim) {
  auto size = params.getOutputSize(dim);
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.fieldSplit[dim],
                      partition.fieldAxisGrainSize[dim]);
  }
  return size;
}

static unsigned getMaxTileInChans(const ConvParams &params, const Plan &plan) {
  auto size = params.getNumInputChansPerConvGroup();
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.inChanSplit,
                      partition.inChanGrainSize);
  }
  return size;
}

static unsigned getMaxTileOutChans(const ConvParams &params, const Plan &plan) {
  auto size = params.getNumOutputChansPerConvGroup();
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.outChanSplit,
                      partition.outChanGrainSize);
  }
  return size;
}

static unsigned
getMaxTileKernelSize(const ConvParams &params, const Plan &plan, unsigned dim) {
  auto size = params.kernelShape[dim];
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.kernelSplit[dim]);
  }
  return size;
}

static unsigned
getMaxTileBatchElements(const ConvParams &params, const Plan &plan) {
  auto size = params.getBatchSize();
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.batchSplit);
  }
  return size;
}

static unsigned
getMaxTileConvGroups(const ConvParams &params, const Plan &plan) {
  auto size = params.getNumConvGroups();
  for (const auto &partition : plan.partitions) {
    size = getMaxSize(size, partition.convGroupSplit);
  }
  return size;
}

static unsigned
estimateExchangeCycles(
    const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    bool floatActivations,
    const ConvParams &params,
    const Plan &plan) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto activationSize = floatActivations ? 4 : 2;
  const auto partialSize = plan.floatPartials ? 4 : 2;
  auto batchElements = params.getBatchSize();
  auto numOutChans = params.getNumOutputChansPerConvGroup();
  auto numInChans = params.getNumInputChansPerConvGroup();
  auto numConvGroups = params.getNumConvGroups();
  auto outputFieldSize = params.getOutputFieldShape();
  auto kernelSize = params.kernelShape;
  unsigned numCycles = 0;
  const auto numLevelsOfHierarchy = plan.partitions.size();
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    const auto &partition = plan.partitions[level];
    batchElements = getMaxSize(batchElements, partition.batchSplit);
    numOutChans = getMaxSize(numOutChans, partition.outChanSplit,
                             partition.outChanGrainSize);
    numInChans = getMaxSize(numInChans, partition.inChanSplit,
                            partition.inChanGrainSize);
    numConvGroups = getMaxSize(numConvGroups, partition.convGroupSplit);
    auto numberOfInputElements = batchElements * numInChans *
                                 numConvGroups;
    auto numberOfWeights = numOutChans * numInChans * numConvGroups;
    auto numberOfOutputElements = batchElements * numOutChans *
                                  numConvGroups;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      kernelSize[dim] = getMaxSize(kernelSize[dim],
                                   partition.kernelSplit[dim]);
      outputFieldSize[dim] = getMaxSize(outputFieldSize[dim],
                                        partition.fieldSplit[dim],
                                        partition.fieldAxisGrainSize[dim]);
      const auto tileInSize =
          getMaxInputRangeSize(outputFieldSize[dim], dim, params,
                               kernelSize[dim]);
      numberOfInputElements *= tileInSize;
      numberOfWeights *= kernelSize[dim];
      numberOfOutputElements *= outputFieldSize[dim];
    }
    auto inputElementsBytes = numberOfInputElements * activationSize;
    auto weightBytes = numberOfWeights * activationSize;
    const auto numberOfPartialSums = numberOfOutputElements;
    const auto partialSumBytes = numberOfPartialSums * partialSize;

    const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();
    const auto exchangeBytesPerCycle = perLevelExchangeBytesPerCycle[level];

    const auto inputElementBytesPerCycle =
        (target.supportsExchangeBusSharing() &&
         level + 1 == numLevelsOfHierarchy &&
         plan.linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD &&
         (partition.outChanSplit % tilesPerSuperTile) == 0) ?
              exchangeBytesPerCycle * tilesPerSuperTile :
              exchangeBytesPerCycle;
    numCycles +=
        std::ceil(inputElementsBytes / inputElementBytesPerCycle) +
        std::ceil(weightBytes / exchangeBytesPerCycle) +
        std::ceil(partialSumBytes / exchangeBytesPerCycle);

  }
  return numCycles;
}

static std::uint64_t
estimateConvPartialHorizontalMacInnerLoopCycles(unsigned numOutRows,
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

  return getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
    workerPartitions,
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
    const auto tileKernelSize = getMaxTileKernelSize(params, plan, dim);
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
  const auto tileBatchElements = getMaxTileBatchElements(params, plan);
  const auto tileNumOutChans = getMaxTileOutChans(params, plan);
  const auto tileNumGroupedConv = getMaxTileConvGroups(params, plan);
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
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto tileBatchElements = getMaxTileBatchElements(params, plan);
  const auto tileNumGroupedConv = getMaxTileConvGroups(params, plan);
  const auto tileNumOutChans = getMaxTileOutChans(params, plan);
  const auto tileNumOutGroups =
      (tileNumOutChans + outChansPerGroup - 1) / outChansPerGroup;

  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = target.getNumWorkerContexts();
  if (plan.method == Plan::Method::AMP) {
    numContexts = 1;
  }
  const auto tileNumInChans = getMaxTileInChans(params, plan);
  const auto tileNumInGroups =
      (tileNumInChans + (inChansPerGroup - 1)) / inChansPerGroup;

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
    const auto tileKernelSize = getMaxTileKernelSize(params, plan, dim);
    tileKernelShape.push_back(tileKernelSize);
    tileKernelFieldElements *= tileKernelSize;
  }
  const auto tileKernelWidth =
      getMaxTileKernelSize(params, plan, numFieldDims - 1);
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
        auto innerLoopCycles =
            cache->mGetConvPartial1x1InnerLoopCycleEstimate(
              tileBatchElements, tileOutShape,
              target.getNumWorkerContexts(), params.inputDilation,
              params.stride);
        unsigned numEdges =
            tileNumInGroups + tileNumOutGroups +
            tileNumInGroups * tileNumOutGroups;
        computeCycles =
          getConvPartial1x1SupervisorOuterLoopCycleEstimate(
              innerLoopCycles, tileNumGroupedConv, tileNumInGroups,
              tileNumOutGroups, convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              floatActivations, useDeltaEdgesForConvPartials(numEdges));
      } else {
        auto innerLoopCycles =
            cache->mGetConvPartialnx1InnerLoopCycleEstimate(
              tileBatchElements, tileOutShape, tileKernelShape,
              convUnitWeightHeight, convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(), floatActivations,
              params.inputDilation, params.stride);
        unsigned numEdges =
            tileNumInGroups + tileNumOutGroups +
            tileNumInGroups * tileNumOutGroups;
        computeCycles =
            getConvPartialnx1SupervisorCycleOuterLoopEstimate(
              innerLoopCycles, tileNumGroupedConv, tileNumOutGroups,
              tileNumInGroups, useDeltaEdgesForConvPartials(numEdges));
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
      auto innerLoopCycles =
          cache->mEstimateConvPartialHorizontalMacInnerLoopCycles(
            numActiveOutRows,
            tileOutWidth,
            outputStrideX,
            tileKernelFieldElements / tileKernelWidth,
            tileKernelWidth,
            target.getNumWorkerContexts(),
            floatActivations,
            inChansPerGroup,
            target.getDataPathWidth());
      computeCycles =
          getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
            innerLoopCycles,
            tileNumGroupedConv,
            tileNumInGroups,
            tileNumOutGroups);
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
estimateReduceCycles(const poplar::Target &target, const ConvParams &params,
                     const Plan &plan) {
  const auto tileBatchElements = getMaxTileBatchElements(params, plan);
  const auto tileNumConvGroups = getMaxTileConvGroups(params, plan);
  const auto tileNumOutChans = getMaxTileOutChans(params, plan);
  auto partialsPerTile = tileBatchElements * tileNumOutChans *
                         tileNumConvGroups;
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    partialsPerTile *= getMaxTileOutSize(params, plan, dim);
  }
  unsigned numCycles = 0;
  for (int level = plan.partitions.size() - 1; level >= 0; --level) {
    const auto &partition = plan.partitions[level];
    const auto reductionDepth =
        std::accumulate(partition.kernelSplit.begin(),
                        partition.kernelSplit.end(),
                        partition.inChanSplit,
                        std::multiplies<std::size_t>());
    if (reductionDepth > 1) {
      // Consider a group of tiles that compute partial sums for the same output
      // volume. The number of partial sums that to be reduced is
      // partialsPerTile * numTiles. Calculation of the output is spread evenly
      // across the tiles so the number of partial sums each tile must reduce is
      // (partialsPerTile * numTiles) / numTiles = partialsPerTile.
      const auto reduceElementsPerTile = partialsPerTile;
      const auto vectorWidth =
          plan.floatPartials ? target.getFloatVectorWidth() :
                               target.getHalfVectorWidth();
      numCycles += (reduceElementsPerTile + vectorWidth - 1) / vectorWidth;
      partialsPerTile = (partialsPerTile + reductionDepth - 1) / reductionDepth;
    }
  }
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
                 std::vector<PartitionVariables> &partitionVars,
                 popsolver::Variable usedTiles,
                 const poplar::Target &target,
                 const std::vector<double> &perLevelExchangeBytesPerCycle,
                 const ConvParams &params,
                 unsigned inChansPerGroup,
                 unsigned partialChansPerGroup,
                 bool floatPartials,
                 bool floatActivations,
                 Plan::Method method,
                 Plan::LinearizeTileOrder linearizeTileOrder,
                 PlanningCacheImpl *cache) {
  const auto numLevelsOfHierarchy = partitionVars.size();
  const auto numFieldDims = partitionVars[0].fieldGrainSize.size();
  std::vector<popsolver::Variable> variables;
  for (const auto &vars : partitionVars) {
    variables.push_back(vars.batchSplit);
    variables.push_back(vars.outChanSplit);
    variables.push_back(vars.inChanSplit);
    variables.push_back(vars.convGroupSplit);
    variables.insert(variables.end(), vars.fieldSplit.begin(),
                     vars.fieldSplit.end());
    variables.insert(variables.end(), vars.kernelSplit.begin(),
                     vars.kernelSplit.end());
  };
  auto makePlan = [=](const std::vector<unsigned> &values) {
    std::vector<Partition> partitions;
    partitions.reserve(numLevelsOfHierarchy);
    const auto varsPerLevel = 4 + numFieldDims * 2;
    assert(varsPerLevel * numLevelsOfHierarchy == values.size());
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      const auto batchSplit = values[level * varsPerLevel];
      const auto outChanSplit = values[level * varsPerLevel + 1];
      const auto inChanSplit = values[level * varsPerLevel + 2];
      const auto convGroupSplit = values[level * varsPerLevel + 3];
      std::vector<unsigned> fieldSplit(
        values.begin() + level * varsPerLevel + 4,
        values.begin() + level * varsPerLevel + 4 + numFieldDims
      );
      std::vector<unsigned> kernelSplit(
        values.begin() + level * varsPerLevel + 4 + numFieldDims,
        values.begin() + level * varsPerLevel + 4 + 2 * numFieldDims
      );
      Partition partition(fieldSplit, batchSplit, outChanSplit,
                          kernelSplit, inChanSplit, convGroupSplit,
                          partitionVars[level].fieldGrainSize,
                          partitionVars[level].inChanGrainSize,
                          partitionVars[level].outChanGrainSize);
      partitions.push_back(std::move(partition));
    }
    Plan candidate(std::move(partitions), inChansPerGroup, partialChansPerGroup,
                   floatPartials, method, linearizeTileOrder);
    return candidate;
  };
  const auto exchangeCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    Plan candidate = makePlan(values);
    return estimateExchangeCycles(target, perLevelExchangeBytesPerCycle,
                                  floatActivations, params, candidate);
  });
  const auto zeroCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    Plan candidate = makePlan(values);
    return estimateZeroCycles(target, params, candidate);
  });
  const auto partialCalcCycles =
      m.call(variables,
             [=,&target](const std::vector<unsigned> &values) {
    Plan candidate = makePlan(values);
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
    Plan candidate = makePlan(values);
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

static Partition
makePartition(const popsolver::Solution &s, const PartitionVariables &vars) {
  std::vector<unsigned> fieldSplitValues;
  for (const auto var : vars.fieldSplit) {
    fieldSplitValues.push_back(s[var]);
  }
  std::vector<unsigned> kernelSplitValues;
  for (const auto var : vars.kernelSplit) {
    kernelSplitValues.push_back(s[var]);
  }
  Partition partition(std::move(fieldSplitValues), s[vars.batchSplit],
                      s[vars.outChanSplit], std::move(kernelSplitValues),
                      s[vars.inChanSplit], s[vars.convGroupSplit],
                      vars.fieldGrainSize, vars.inChanGrainSize,
                      vars.outChanGrainSize);
  return partition;
}

template <class TransformCostFn>
static std::pair<Plan, Cost>
choosePlan(const poplar::Target &target,
           const std::vector<unsigned> &hierarchy,
           const std::vector<double> &perLevelExchangeBytesPerCycle,
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
  const auto inChanGrainSize = inChansPerGroup;
  const auto outChanGrainSize = partialChansPerGroup;
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
  const auto numFieldDims = params.getNumFieldDims();
  assert(hierarchy.size() >= 1);
  std::vector<PartitionVariables> partitionVars;
  const auto outChanGrains =
      (params.getNumOutputChansPerConvGroup() + outChanGrainSize - 1) /
      outChanGrainSize;
  const auto inChanGrains =
      (params.getNumInputChansPerConvGroup() + inChanGrainSize - 1)
      / inChanGrainSize;
  std::vector<Variable> maxFieldSplit;
  std::vector<Variable> maxKernelSplit;
  maxFieldSplit.reserve(numFieldDims);
  maxKernelSplit.reserve(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const unsigned numGrains =
        (params.getOutputSize(dim) + fieldGrainSize[dim] - 1) /
        fieldGrainSize[dim];
    maxFieldSplit.push_back(m.addConstant(numGrains));
    // Currenlty the implementation doesn't support splitting the innermost
    // kernel dimension. TODO lift this restriction.
    maxKernelSplit.push_back(m.addConstant(dim == numFieldDims - 1 ? 1 :
                                             params.kernelShape[dim]));
  }
  auto maxBatchSplit = m.addConstant(params.getBatchSize());
  auto maxConvGroupSplit = m.addConstant(params.getNumConvGroups());
  // The joint planning cost function assumes that no exchange is required to
  // rearrange weights between passes. Because of the way we derive the
  // backward and weight update plans from the forward plan this is guaranteed
  // to be the case if each weight is used on exactly one tile in the forward
  // pass. Disallow splitting of fully connected batch (or equivalently the
  // convolutional output channels) across tiles to ensure this holds.
  auto maxOutChanSplit =
      m.addConstant(options.pass == Pass::FC_TRAINING_FWD ? 1 : outChanGrains);
  auto maxInChanSplit = m.addConstant(inChanGrains);
  for (unsigned level = 0; level != hierarchy.size(); ++level) {
    const auto levelMaxSplit = hierarchy[level];
    PartitionVariables p;
    p.fieldSplit.reserve(numFieldDims);
    p.kernelSplit.reserve(numFieldDims);
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      p.fieldSplit.push_back(m.addVariable(1, levelMaxSplit));
      p.kernelSplit.push_back(m.addVariable(1, levelMaxSplit));
      m.lessOrEqual(p.fieldSplit.back(), maxFieldSplit[dim]);
      m.lessOrEqual(p.kernelSplit.back(), maxKernelSplit[dim]);
      if (level + 1 != hierarchy.size()) {
        maxFieldSplit[dim] = m.ceildiv(maxFieldSplit[dim], p.fieldSplit.back());
        maxKernelSplit[dim] =
            m.ceildiv(maxKernelSplit[dim], p.kernelSplit.back());
      }
    }
    p.batchSplit = m.addVariable(1, levelMaxSplit);
    p.convGroupSplit = m.addVariable(1, levelMaxSplit);
    p.outChanSplit = m.addVariable(1, levelMaxSplit);
    p.inChanSplit = m.addVariable(1, levelMaxSplit);
    m.lessOrEqual(p.batchSplit, maxBatchSplit);
    m.lessOrEqual(p.convGroupSplit, maxConvGroupSplit);
    m.lessOrEqual(p.outChanSplit, maxOutChanSplit);
    m.lessOrEqual(p.inChanSplit, maxInChanSplit);
    p.outChanGrainSize = outChanGrainSize;
    p.inChanGrainSize = inChanGrainSize;
    p.fieldGrainSize = fieldGrainSize;
    if (level + 1 != hierarchy.size()) {
      maxBatchSplit = m.ceildiv(maxBatchSplit, p.batchSplit);
      maxConvGroupSplit = m.ceildiv(maxConvGroupSplit, p.convGroupSplit);
      maxOutChanSplit = m.ceildiv(maxOutChanSplit, p.outChanSplit);
      maxInChanSplit = m.ceildiv(maxInChanSplit, p.inChanSplit);
    }
    partitionVars.push_back(std::move(p));
  }

  std::vector<Variable> perLevelSplits;
  for (unsigned level = 0; level != partitionVars.size(); ++level) {
    const auto &p = partitionVars[level];
    std::vector<Variable> splits;
    splits.push_back(p.batchSplit);
    splits.push_back(p.outChanSplit);
    splits.push_back(p.inChanSplit);
    splits.push_back(p.convGroupSplit);
    splits.insert(splits.end(), p.fieldSplit.begin(), p.fieldSplit.end());
    splits.insert(splits.end(), p.kernelSplit.begin(), p.kernelSplit.end());
    const auto levelSplit = m.product(splits);
    m.lessOrEqual(levelSplit, hierarchy[level]);
    perLevelSplits.push_back(levelSplit);
  }
  const auto usedTiles = m.product(perLevelSplits);

  auto cycles =
      addCycleEstimate(m, partitionVars, usedTiles, target,
                       perLevelExchangeBytesPerCycle, params,
                       inChansPerGroup, partialChansPerGroup,
                       convVertexType.floatPartials,
                       floatActivations, convVertexType.method,
                       Plan::LinearizeTileOrder::STANDARD, cache);
  if (options.pass == Pass::FC_TRAINING_FWD) {
    auto bwdParams = params;
    std::swap(bwdParams.inputFieldShape.back(), bwdParams.inputChannels);
    std::vector<PartitionVariables> bwdPartitionVars;
    for (const auto &p : partitionVars) {
      auto bwdP = p;
      bwdP.fieldSplit.back() = p.inChanSplit;
      bwdP.inChanSplit = p.fieldSplit.back();
      bwdP.inChanGrainSize = p.fieldGrainSize.back();
      bwdP.fieldGrainSize.back() = inChansPerGroup;
      bwdPartitionVars.push_back(bwdP);
    }
    const auto bwdInChansPerGroup = bwdPartitionVars.back().inChanGrainSize;
    const auto bwdMethod =
        getFullyConnectedBwdMethod(params,
                                   convVertexType.method);
    const auto bwdCycles =
        addCycleEstimate(m, bwdPartitionVars,
                         usedTiles, target, perLevelExchangeBytesPerCycle,
                         params, bwdInChansPerGroup, partialChansPerGroup,
                         convVertexType.floatPartials,
                         floatActivations, bwdMethod,
                         Plan::LinearizeTileOrder::FC_BWD_AS_CONV, cache);
    auto wuParams = params;
    std::swap(wuParams.inputChannels, wuParams.outputChannels);
    std::vector<PartitionVariables> wuPartitionVars;
    for (const auto &p : partitionVars) {
      auto wuP = p;
      wuP.outChanSplit = p.inChanSplit;
      wuP.inChanSplit = p.outChanSplit;
      wuP.inChanGrainSize = p.outChanGrainSize;
      wuP.outChanGrainSize = p.inChanGrainSize;
      wuP.fieldGrainSize = std::vector<unsigned>(numFieldDims, 1);
      wuPartitionVars.push_back(wuP);
    }
    const auto wuInChansPerGroup = partialChansPerGroup;
    const auto wuPartialChansPerGroup = inChansPerGroup;
    const auto wuMethod =
        getFullyConnectedWUMethod(params,
                                  convVertexType.method,
                                  partialChansPerGroup,
                                  inChansPerGroup);
    const auto wuCycles =
        addCycleEstimate(m, wuPartitionVars,
                         usedTiles, target, perLevelExchangeBytesPerCycle,
                         wuParams, wuInChansPerGroup, wuPartialChansPerGroup,
                         convVertexType.floatPartials,
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
  std::vector<Partition> partitions;
  for (const auto &p : partitionVars) {
    partitions.push_back(makePartition(s, p));
  }
  Plan plan(std::move(partitions),
            inChansPerGroup, partialChansPerGroup,
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
                        const ConvParams &transformedParams,
                        const ConvOptions &options,
                        bool swapOperands,
                        const std::vector<unsigned> &expandDims,
                        const std::vector<unsigned> &outChanFlattenDims,
                        unsigned inChansPadding,
                        unsigned partialChansPadding,
                        unsigned usedTiles) {
  assert(options.pass != Pass::FC_TRAINING_WU &&
         options.pass != Pass::FC_TRAINING_BWD);
  bool isWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool rearrangeInput = isWeightUpdate || !expandDims.empty() || swapOperands ||
                        inChansPadding;
  bool rearrangeWeights = isWeightUpdate || !expandDims.empty() ||
                          !outChanFlattenDims.empty() ||
                          swapOperands || inChansPadding || partialChansPadding;
  bool rearrangeOutput = (!isWeightUpdate && swapOperands) ||
                         (isWeightUpdate && !swapOperands) ||
                         !outChanFlattenDims.empty() ||
                         partialChansPadding;
  auto expandedParams = transformedParams;
  unsigned expandedInputFieldSize = 1;
  unsigned expandedFilterSize = 1;
  unsigned expandedOutputFieldSize = 1;
  for (unsigned dim = 0; dim != expandedParams.getNumFieldDims(); ++dim) {
    expandedInputFieldSize *= expandedParams.inputFieldShape[dim];
    expandedFilterSize *= expandedParams.kernelShape[dim];
    expandedOutputFieldSize *= expandedParams.getOutputSize(dim);
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
    if (!expandingDimChangesParams(params, i)) {
      continue;
    }
    // Don't expand this dimension if the number of non zero kernel entries is
    // larger than the number of non zero input entries as it is unlikely to be
    // profitable. This heuristic cuts down the size of the search space.
    // TODO investigate better heuristics.
    if (params.inputFieldShape[i] < params.kernelShape[i])
      continue;
    candidateDims.push_back(i);
  }
  auto subsets = getPowerSet(candidateDims);
  for (auto &subset : subsets) {
    // The subsets returned by getPowerSet have the outermost dimension first
    // but it is more efficient to expand the innermost dimension first.
    std::reverse(subset.begin(), subset.end());
  }
  return subsets;
}

static std::vector<std::vector<unsigned>>
getOutChanFlattenDimsCandidates(const ConvParams &params) {
  auto swappedParams = params;
  popconv::swapOperands(swappedParams);
  std::vector<unsigned> candidateDims;
  for (unsigned i = 0; i != swappedParams.getNumFieldDims(); ++i) {
    // Don't flatten this dimension into the output channel dimension if it
    // wouldn't increase the number of output channels.
    if (params.getOutputSize(i) == 1)
      continue;
    // Don't flatten this dimension into the output channel dimension if the
    // number of non zero input entries is larger than the number of non zero
    // kernel entries as it is unlikely to be profitable. This heuristic cuts
    // down the size of the search space. TODO investigate better heuristics.
    if (params.inputFieldShape[i] > params.kernelShape[i])
      continue;
    candidateDims.push_back(i);
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
  params = canonicalizeParams(params);
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

static ConvParams
calculateSwappedParams(const ConvParams &params, bool swapOperands) {
  auto swappedParams = params;
  if (swapOperands) {
    popconv::swapOperands(swappedParams);
  }
  return swappedParams;
}

static ConvParams
calculateExpandedParams(const ConvParams &params,
                        const std::vector<unsigned> &expandDims) {
  auto expandedParams = params;
  for (unsigned dim : expandDims) {
    expandDim(expandedParams, dim);
  }
  return expandedParams;
}

static ConvParams
calculateFlattenedParams(const ConvParams &params,
                         const std::vector<unsigned> &outChanFlattenDims,
                         std::vector<unsigned> &flattenDims) {
  auto flattenedParams = params;
  if (!outChanFlattenDims.empty()) {
    popconv::swapOperands(flattenedParams);
    for (unsigned dim : outChanFlattenDims) {
      expandDim(flattenedParams, dim);
      // Flatten into the batch axis (this will become the output channel
      // axis when we swap back).
      flattenedParams.batchSize *= flattenedParams.inputFieldShape[dim];
      flattenedParams.inputFieldShape[dim] = 1;
    }
    popconv::swapOperands(flattenedParams);
  }
  // Flatten from the innermost out.

  flattenDims.push_back(0);
  for (unsigned spatialDim = 0;
       spatialDim != flattenedParams.getNumFieldDims();
       ++spatialDim) {
    if (dimCanBeFlattened(flattenedParams, spatialDim)) {
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
          fromDimIndex ?
            flattenedParams.inputFieldShape[fromDimIndex - 1] :
          flattenedParams.batchSize;
      flattenedParams.inputFieldShape[innermostFlattenableDim - 1] *=
          fromDimSize;
      fromDimSize = 1;
    }
  } else {
    flattenDims.clear();
  }
  return flattenedParams;
}

static ConvParams
calculatePaddedParams(const ConvParams &params, unsigned inChansPerGroup,
                      unsigned partialChansPerGroup,
                      unsigned &inChansPadding, unsigned &partialChansPadding) {
  auto paddedParams = params;
  const auto inChans = params.getNumInputChansPerConvGroup();
  paddedParams.inputChannels =
      ((inChans + inChansPerGroup - 1) / inChansPerGroup) *
      inChansPerGroup;
  inChansPadding = paddedParams.inputChannels - inChans;
  const auto partialChans =
      params.getNumOutputChansPerConvGroup();
  paddedParams.outputChannels =
      ((partialChans + partialChansPerGroup - 1) / partialChansPerGroup) *
      partialChansPerGroup;
  partialChansPadding = paddedParams.outputChannels - partialChans;
  return paddedParams;
}

static std::vector<unsigned>
getTileHierarchy(const poplar::Target &target,
                 std::vector<double> &perLevelExchangeBytesPerCycle) {
  std::vector<unsigned> hierarchy;
  perLevelExchangeBytesPerCycle.clear();
  // TODO query target, see T2125
  const auto clockFrequency = 1.6 * 1000 * 1000 * 1000;
  if (target.getNumIPUs() > 1) {
    hierarchy.push_back(target.getNumIPUs());
    auto ipuExchangeBytesPerCycle =
        static_cast<double>(std::numeric_limits<double>::infinity());
    // Compute the maximum number of bytes per cycle for a traffic pattern
    // where every IPU sends an equal amount of data to every other IPU.
    for (const auto &constraint : target.getGlobalExchangeConstraints()) {
      std::map<unsigned, unsigned> ipuSrcCount;
      std::map<unsigned, unsigned> ipuDstCount;
      for (const auto &flow : constraint.flows) {
        ++ipuSrcCount[flow.src];
        ++ipuDstCount[flow.dst];
      }
      auto secondLess = [](const std::pair<unsigned, unsigned> &a,
                           const std::pair<unsigned, unsigned> &b) {
        return a.second < b.second;
      };
      const auto maxSrcCount =
          std::max_element(ipuSrcCount.begin(), ipuSrcCount.end(),
                           secondLess)->second;
      const auto maxDstCount =
          std::max_element(ipuDstCount.begin(), ipuDstCount.end(),
                           secondLess)->second;
      const auto maxCount = std::max(maxSrcCount, maxDstCount);
      const auto constraintBytesPerCycle =
          (constraint.bandwidth / clockFrequency) / 8;
      ipuExchangeBytesPerCycle = std::min(ipuExchangeBytesPerCycle,
                                          constraintBytesPerCycle / maxCount);
    }
    perLevelExchangeBytesPerCycle.push_back(ipuExchangeBytesPerCycle);
  }
  perLevelExchangeBytesPerCycle.push_back(target.getExchangeBytesPerCycle());
  hierarchy.push_back(target.getTilesPerIPU());
  return hierarchy;
}

std::vector<unsigned> getTileHierarchy(const poplar::Target &target) {
  std::vector<double> dummy;
  return getTileHierarchy(target, dummy);
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
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy = getTileHierarchy(target,
                                          perLevelExchangeBytesPerCycle);
  Cost bestCost = highestCost;
  Plan bestPlan;
  for (bool swapOperands : getSwapOperandCandidates(options)) {
    auto swappedParams = calculateSwappedParams(params, swapOperands);
    for (std::vector<unsigned> expandDims :
         getExpandDimsCandidates(swappedParams)) {
      auto expandedParams = calculateExpandedParams(swappedParams, expandDims);
      for (std::vector<unsigned> outChanFlattenDims :
           getOutChanFlattenDimsCandidates(expandedParams)) {
        std::vector<unsigned> flattenDims;
        auto flattenedParams = calculateFlattenedParams(expandedParams,
                                                        outChanFlattenDims,
                                                        flattenDims);
        const bool floatActivations = params.dType == poplar::FLOAT;
        const bool floatPartials = partialsType == poplar::FLOAT;
        const auto convVertexTypeCandidates =
            getConvVertexTypeCandidates(target, floatActivations,
                                        floatPartials, flattenedParams,
                                        options);
        for (const auto &convVertexType : convVertexTypeCandidates) {
          const auto inChansPerGroup = convVertexType.inChansPerGroup;
          const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
          unsigned inChansPadding, partialChansPadding;
          auto paddedParams =
              calculatePaddedParams(flattenedParams, inChansPerGroup,
                                    partialChansPerGroup, inChansPadding,
                                    partialChansPadding);
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
                                           expandDims, outChanFlattenDims,
                                           inChansPadding,
                                           partialChansPadding, usedTiles);
          };
          std::tie(candidate, candidateCost) =
              choosePlan(target, hierarchy, perLevelExchangeBytesPerCycle,
                         fieldGrainSize, convVertexType,
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
  const auto numPartitions = plan.partitions.size();
  for (unsigned i = 0; i != numPartitions; ++i) {
    plan.partitions[i].inChanSplit = fwdPlan.partitions[i].outChanSplit;
    plan.partitions[i].outChanSplit = fwdPlan.partitions[i].inChanSplit;
    plan.partitions[i].outChanGrainSize = fwdPlan.partitions[i].inChanGrainSize;
    plan.partitions[i].inChanGrainSize = fwdPlan.partitions[i].outChanGrainSize;
  }
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
    plan.partitions.back().inChanGrainSize = plan.inChansPerGroup;
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
  for (auto &partition : plan.partitions) {
    std::swap(partition.fieldSplit.back(), partition.inChanSplit);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
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

std::uint64_t estimateConvCost(const poplar::Target &target,
                               const ConvParams &params,
                               const ConvOptions &options,
                               const Plan &plan,
                               PlanningCache *cache) {
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  std::vector<unsigned> flattenDims;
  const auto swappedParams = calculateSwappedParams(params, plan.swapOperands);
  const auto expandedParams =
      calculateExpandedParams(swappedParams, plan.expandDims);
  const auto flattenedParams =
      calculateFlattenedParams(expandedParams, plan.outChanFlattenDims,
                               flattenDims);
  unsigned inChansPadding, partialChansPadding;
  const auto paddedParams =
      calculatePaddedParams(flattenedParams, plan.inChansPerGroup,
                            plan.partialChansPerGroup, inChansPadding,
                            partialChansPadding);
  unsigned usedTiles = 1;
  for (const auto &partition :  plan.partitions) {
    usedTiles *= partition.batchSplit *
                 partition.outChanSplit *
                 partition.inChanSplit *
                 partition.convGroupSplit;
    usedTiles = std::accumulate(partition.fieldSplit.begin(),
                                partition.fieldSplit.end(),
                                usedTiles, std::multiplies<unsigned>());
    usedTiles = std::accumulate(partition.kernelSplit.begin(),
                                partition.kernelSplit.end(),
                                usedTiles, std::multiplies<unsigned>());
  }
  std::vector<double> perLevelExchangeBytesPerCycle;
  (void)getTileHierarchy(target, perLevelExchangeBytesPerCycle);
  assert(perLevelExchangeBytesPerCycle.size() ==
         plan.partitions.size());
  const auto transformCycles =
      estimateTransformCycles(target,
                              paddedParams, options, plan.swapOperands,
                              plan.expandDims, plan.outChanFlattenDims,
                              inChansPadding, partialChansPadding, usedTiles);
  const auto floatActivations = params.dType == poplar::FLOAT;
  const auto exchangeCycles =
      estimateExchangeCycles(target, perLevelExchangeBytesPerCycle,
                             floatActivations, paddedParams, plan);
  const auto zeroCycles =
      estimateZeroCycles(target, paddedParams, plan);
  const auto partialCalcCycles =
      estimatePartialCalcCycles(target, floatActivations, paddedParams,
                                plan, cacheImpl);
  const auto reduceCycles =
      estimateReduceCycles(target, paddedParams, plan);
  return transformCycles + exchangeCycles + zeroCycles +
         partialCalcCycles + reduceCycles;
}

} // end namespace conv
