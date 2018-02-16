#include "popconv/internal/ConvPlan.hpp"
#include "popconv/Convolution.hpp"
#include "poputil/exceptions.hpp"
#include "poplar/Graph.hpp"
#include "popconv/ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "poplibs_support/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "poplibs_support/Compiler.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <popsolver/Model.hpp>
#include "poplibs_support/print.hpp"

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

  struct ConvSizeVariables {
    std::vector<popsolver::Variable> numFieldGrains;
    popsolver::Variable batchSize;
    popsolver::Variable numOutChanGrains;
    std::vector<popsolver::Variable> kernelSize;
    popsolver::Variable numInChanGrains;
    popsolver::Variable numConvGroups;
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
    throw poputil::poplib_error(
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

std::ostream& operator<<(std::ostream &os, const ConvTransform &t) {
  os << "  Transform: swapOperands        " << t.swapOperands << "\n"
        "             expandDims          ";
  printContainer(t.expandDims, os);
  os << "\n"
     << "        outChanFlattenDims      ";
  printContainer(t.outChanFlattenDims, os);
  os << "\n"
     << "        flattenDims             ";
  printContainer(t.flattenDims, os);
  os << "\n"
     << "        inChansPadding          " << t.inChansPadding << "\n"
     << "        partialChansPadding     " << t.partialChansPadding << "\n";
  return os;
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
  os << p.transform << "\n";
  os << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n";
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
  const auto stride = params.outputTransform.stride[dim];
  const auto inputDilation = params.inputTransform.dilation[dim];
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

static popsolver::Variable
addZeroCycles(popsolver::Model &m,
              const poplar::Target &target,
              const popsolver::Variable partialsPerTile,
              const ConvSizeVariables &convSize,
              const ConvParams &params,
              bool floatPartials,
              Plan::Method method) {
  enum { Yes, No, Maybe } zeroPartials = Maybe;
  if (method == Plan::Method::MAC) {
    zeroPartials = Yes;
  } else {
    const auto numFieldDims = params.getNumFieldDims();
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      if (params.outputTransform.stride[dim] != 1 ||
          params.inputTransform.dilation[dim] != 1 ||
          params.kernelTransform.dilation[dim] != 1 ||
          params.inputTransform.paddingLower[dim] != 0 ||
          params.inputTransform.paddingUpper[dim] != 0 ||
          params.outputTransform.paddingLower[dim] != 0 ||
          params.outputTransform.paddingUpper[dim] != 0 ||
          params.kernelTransform.paddingLower[dim] != 0 ||
          params.kernelTransform.paddingUpper[dim] != 0) {
        zeroPartials = Yes;
        break;
      }
    }
    const auto kernelElements =
        std::accumulate(params.kernelShape.begin(),
                        params.kernelShape.end(), std::size_t(1UL),
                        std::multiplies<std::size_t>());
    if (kernelElements != 1) {
      zeroPartials = No;
    }
  }
  if (zeroPartials == No)
    return m.addConstant(0);
  const auto vectorWidth =
      m.addConstant(floatPartials ? target.getFloatVectorWidth() :
                                    target.getHalfVectorWidth());
  auto numCycles = m.ceildiv(partialsPerTile, vectorWidth);
  if (zeroPartials == Maybe) {
    const auto tileKernelElements = m.product(convSize.kernelSize);
    auto cycleMultiplier = m.call({tileKernelElements},
                                  [](const std::vector<unsigned> &values) {
      return values[0] > 1 ? 1 : 0;
    });
    m.lessOrEqual(cycleMultiplier, 1);
    numCycles = m.product({numCycles, cycleMultiplier});
  }
  return numCycles;
}

static bool
canUseConvPartial1x1Vertex(const ConvParams &params,
                           unsigned convUnitWeightHeight,
                           const std::vector<unsigned> &tileKernelShape) {
  if (convUnitWeightHeight != 1)
    return false;
  if (params.inputTransform.dilation != params.outputTransform.stride)
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
          getOutputRangeForKernelIndex(dim, outputRange, k, params);
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
      unsigned numEdges =
          tileNumInGroups + tileNumOutGroups +
          tileNumInGroups * tileNumOutGroups;
      bool useDeltasForEdges = useDeltaEdgesForConvPartials(numEdges);
      if (canUseConvPartial1x1Vertex(params, convUnitWeightHeight,
                                     tileKernelShape)) {
        auto innerLoopCycles =
            cache->mGetConvPartial1x1InnerLoopCycleEstimate(
              tileBatchElements, tileOutShape,
              target.getNumWorkerContexts(), params.inputTransform.dilation,
              params.outputTransform.stride);
        computeCycles =
          getConvPartial1x1SupervisorOuterLoopCycleEstimate(
              innerLoopCycles, tileNumGroupedConv, tileNumInGroups,
              tileNumOutGroups, convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              floatActivations, useDeltasForEdges);
      } else {
        auto innerLoopCycles =
            cache->mGetConvPartialnx1InnerLoopCycleEstimate(
              tileBatchElements, tileOutShape, tileKernelShape,
              convUnitWeightHeight, convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(), floatActivations,
              params.inputTransform.dilation, params.outputTransform.stride);
        computeCycles =
            getConvPartialnx1SupervisorCycleOuterLoopEstimate(
              innerLoopCycles, tileNumGroupedConv, tileNumOutGroups,
              tileNumInGroups, useDeltasForEdges);
      }
    }
    break;
  case Plan::Method::MAC:
    {
      const auto outputStrideX = params.inputTransform.dilation.back();
      unsigned numActiveOutRows = tileBatchElements;
      for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
        const auto dimActiveRows =
            (tileOutShape[dim] + params.inputTransform.dilation[dim] - 1) /
            params.inputTransform.dilation[dim];
        numActiveOutRows *= dimActiveRows;
      }
      const auto tileKernelWidth =
          getMaxTileKernelSize(params, plan, numFieldDims - 1);
      const auto tileOutWidth = getMaxTileOutSize(params, plan,
                                                  numFieldDims - 1);
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
      const auto tileOutWidth = getMaxTileOutSize(params, plan,
                                                  numFieldDims - 1);
      assert(tileOutWidth == tileOutElements);
      assert(tileBatchElements == 1);
      assert(tileNumInGroups == 1);
      assert(std::all_of(params.outputTransform.stride.begin(),
                         params.outputTransform.stride.end(), equalsOne));
      assert(std::all_of(params.inputTransform.dilation.begin(),
                         params.inputTransform.dilation.end(), equalsOne));
      const auto numContexts = target.getNumWorkerContexts();
      const auto workerOutWidth =
          (tileOutWidth + numContexts - 1) / numContexts;
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
getScaledExchangeBytesPerCycle(popsolver::Model &m,
                               double exchangeBytesPerCycle,
                               unsigned scaleFactor) {
  auto scaledExchangeBytesPerCycle =
      std::round(exchangeBytesPerCycle * scaleFactor);
  // Ensure scaled bytes per cycle is at least one to avoid divide by zero
  // errors.
  scaledExchangeBytesPerCycle = std::max(1.0, scaledExchangeBytesPerCycle);
  // Saturate to the half the maximum unsigned integer value (we avoid the
  // maximum value to avoid range problems with the intermediate variables used
  // to implement ceildiv).
  scaledExchangeBytesPerCycle =
      std::min(scaledExchangeBytesPerCycle,
               static_cast<double>(std::numeric_limits<unsigned>::max() / 2));
  return m.addConstant(static_cast<unsigned>(scaledExchangeBytesPerCycle));
}

static popsolver::Variable
addExchangeCycleEstimate(
    popsolver::Model &m,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSizes,
    const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const ConvParams &params,
    bool floatPartials,
    bool floatActivations,
    Plan::LinearizeTileOrder linearizeTileOrder) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto numLevelsOfHierarchy = convSizes.size();
  const auto activationSize = floatActivations ? 4 : 2;
  const auto partialSize = floatPartials ? 4 : 2;
  // Exchange bytes per cycle is given as a floating point value but the
  // constaint solver only supports unsigned integer variables. To reduce
  // quantization error in the calclation of the number of cycles we multiply
  // both the divisor (exchange bytes per cycle) and the dividend (the number of
  // bytes) by this scaling factor. Larger values of the scaling factor reduce
  // the quantization error but reduce the maximum number of bytes that can
  // be exchanged before running into the limits of the data type used to store
  // it.
  const auto exchangeBytesScalingFactor = 16;
  const auto scaledActivationSize =
      m.addConstant(activationSize * exchangeBytesScalingFactor);
  const auto scaledPartialSize =
      m.addConstant(partialSize * exchangeBytesScalingFactor);
  std::vector<popsolver::Variable> cycleSumOperands;
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    const auto &convSize = convSizes[level];
    const auto scaledExchangeBytesPerCycle =
        getScaledExchangeBytesPerCycle(m, perLevelExchangeBytesPerCycle[level],
                                       exchangeBytesScalingFactor);
    std::vector<popsolver::Variable> outputFieldSizes;
    std::vector<popsolver::Variable> inputFieldSizes;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto fieldGrainSize = partitionVars[level].fieldGrainSize[dim];
      auto outputFieldSize = convSize.numFieldGrains[dim];
      if (fieldGrainSize != 1) {
        outputFieldSize = m.product({outputFieldSize,
                                     m.addConstant(fieldGrainSize)});
      }
      auto inputFieldSize = m.call({outputFieldSize, convSize.kernelSize[dim]},
                                   [=](const std::vector<unsigned> &values) {
        return getMaxInputRangeSize(values[0], dim, params, values[1]);
      });
      outputFieldSizes.push_back(outputFieldSize);
      inputFieldSizes.push_back(inputFieldSize);
    }
    auto totalOutputFieldSize = m.product(outputFieldSizes);
    auto totalInputFieldSize = m.product(inputFieldSizes);
    auto totalKernelSize = m.product(convSize.kernelSize);
    auto numInChans =
        m.product({convSize.numInChanGrains,
                   m.addConstant(partitionVars[level].inChanGrainSize)});
    auto numOutChans =
        m.product({convSize.numOutChanGrains,
                   m.addConstant(partitionVars[level].outChanGrainSize)});
    auto numberOfInputElements =
        m.product({totalInputFieldSize, convSize.batchSize, numInChans,
                   convSize.numConvGroups});
    auto numberOfWeights =
        m.product({totalKernelSize, numInChans, numOutChans,
                   convSize.numConvGroups});
    auto numberOfOutputElements =
        m.product({totalOutputFieldSize, convSize.batchSize,
                   numOutChans, convSize.numConvGroups});
    auto scaledInputElementsBytes = m.product({numberOfInputElements,
                                               scaledActivationSize});
    auto scaledWeightBytes = m.product({numberOfWeights, scaledActivationSize});
    const auto numberOfPartialSums = numberOfOutputElements;
    const auto scaledPartialSumBytes = m.product({numberOfPartialSums,
                                                  scaledPartialSize});

    const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();
    auto scaledInputElementBytesPerCycle = scaledExchangeBytesPerCycle;
    if (target.supportsExchangeBusSharing() &&
        level + 1 == numLevelsOfHierarchy &&
        linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD) {
      auto multiplier =
          m.call({partitionVars[level].outChanSplit},
                 [=](const std::vector<unsigned> &values) {
            return values[0] % tilesPerSuperTile == 0 ? 2 : 1;
          });
      scaledInputElementBytesPerCycle =
          m.product({scaledInputElementBytesPerCycle, multiplier});
    }
    cycleSumOperands.push_back(m.ceildiv(scaledInputElementsBytes,
                                         scaledInputElementBytesPerCycle));
    cycleSumOperands.push_back(m.ceildiv(scaledWeightBytes,
                                         scaledExchangeBytesPerCycle));
    cycleSumOperands.push_back(m.ceildiv(scaledPartialSumBytes,
                                         scaledExchangeBytesPerCycle));
  }
  return m.sum(cycleSumOperands);
}

static popsolver::Variable
addReduceCycleEstimate(
    popsolver::Model &m,
    const std::vector<PartitionVariables> &partitionVars,
    popsolver::Variable partialsPerTile,
    const poplar::Target &target,
    bool floatPartials) {
  std::vector<popsolver::Variable> cycleSumOperands;
  const auto numLevelsOfHierarchy = partitionVars.size();
  const auto vectorWidth =
      m.addConstant(floatPartials ? target.getFloatVectorWidth() :
                                    target.getHalfVectorWidth());
  for (int level = numLevelsOfHierarchy - 1; level >= 0; --level) {
    auto reduceDimSizes = partitionVars[level].kernelSplit;
    reduceDimSizes.push_back(partitionVars[level].inChanSplit);
    const auto reductionDepth = m.product(reduceDimSizes);
    // Consider a group of tiles that compute partial sums for the same output
    // volume. The number of partial sums that to be reduced is
    // partialsPerTile * numTiles. Calculation of the output is spread evenly
    // across the tiles so the number of partial sums each tile must reduce is
    // (partialsPerTile * numTiles) / numTiles = partialsPerTile.
    auto reduceElementsPerTile = partialsPerTile;
    // Nothing to do if the reduction depth is one.
    const auto reduceMultiplier =
        m.call({reductionDepth},
               [](const std::vector<unsigned> &vars) {
      return vars[0] > 1 ? 1 : 0;
    });
    m.lessOrEqual(reduceMultiplier, 1);
    reduceElementsPerTile = m.product({reduceElementsPerTile,
                                       reduceMultiplier});
    cycleSumOperands.push_back(m.ceildiv(reduceElementsPerTile, vectorWidth));
    if (level != 0) {
      partialsPerTile = m.ceildiv(partialsPerTile, reductionDepth);
    }
  }
  return m.sum(cycleSumOperands);
}

static popsolver::Variable
addPartialsPerTile(popsolver::Model &m,
                   const std::vector<PartitionVariables> &partitionVars,
                   const std::vector<ConvSizeVariables> &convSize) {
  unsigned grainSizeProduct = partitionVars.back().outChanGrainSize;
  std::accumulate(partitionVars.back().fieldGrainSize.begin(),
                  partitionVars.back().fieldGrainSize.end(),
                  grainSizeProduct,
                  std::multiplies<unsigned>());
  auto partialDimSizes = convSize.back().numFieldGrains;
  partialDimSizes.push_back(convSize.back().batchSize);
  partialDimSizes.push_back(convSize.back().numConvGroups);
  partialDimSizes.push_back(convSize.back().numOutChanGrains);
  partialDimSizes.push_back(m.addConstant(grainSizeProduct));
  return m.product(partialDimSizes);
}

static popsolver::Variable
addCycleEstimate(popsolver::Model &m,
                 const std::vector<PartitionVariables> &partitionVars,
                 const std::vector<ConvSizeVariables> &convSize,
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
      addExchangeCycleEstimate(m, partitionVars, convSize, target,
                               perLevelExchangeBytesPerCycle, params,
                               floatPartials, floatActivations,
                               linearizeTileOrder);
  const auto partialsPerTile = addPartialsPerTile(m, partitionVars, convSize);
  const auto zeroCycles =
      addZeroCycles(m, target, partialsPerTile, convSize.back(), params,
                    floatPartials, method);
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
      addReduceCycleEstimate(m, partitionVars, partialsPerTile, target,
                             floatPartials);
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
static void
constructModel(const poplar::Target &target,
               const std::vector<unsigned> &hierarchy,
               const std::vector<double> &perLevelExchangeBytesPerCycle,
               const std::vector<unsigned> &fieldGrainSize,
               const ConvVertexType &convVertexType,
               const ConvParams &params,
               TransformCostFn &&transformCost,
               Cost bestCost,
               const CostBounds costBounds,
               PlanningCacheImpl *cache,
               const ConvOptions &options,
               popsolver::Model &m,
               std::vector<PartitionVariables> &partitionVars,
               popsolver::Variable &cycles) {
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
  const auto numFieldDims = params.getNumFieldDims();
  const auto numLevelsOfHierarchy = hierarchy.size();
  assert(numLevelsOfHierarchy >= 1);
  partitionVars.clear();
  const auto outChanGrains =
      (params.getNumOutputChansPerConvGroup() + outChanGrainSize - 1) /
      outChanGrainSize;
  const auto inChanGrains =
      (params.getNumInputChansPerConvGroup() + inChanGrainSize - 1)
      / inChanGrainSize;
  std::vector<ConvSizeVariables> convSize;
  convSize.emplace_back();
  convSize.back().numFieldGrains.reserve(numFieldDims);
  convSize.back().kernelSize.reserve(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const unsigned numGrains =
        (params.getOutputSize(dim) + fieldGrainSize[dim] - 1) /
        fieldGrainSize[dim];
    convSize.back().numFieldGrains.push_back(m.addConstant(numGrains));
    convSize.back().kernelSize.push_back(
      m.addConstant(params.kernelShape[dim])
    );
  }
  convSize.back().batchSize = m.addConstant(params.getBatchSize());
  convSize.back().numConvGroups = m.addConstant(params.getNumConvGroups());
  convSize.back().numOutChanGrains = m.addConstant(outChanGrains);
  convSize.back().numInChanGrains = m.addConstant(inChanGrains);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    const auto &prevConvSize = convSize.back();
    ConvSizeVariables nextConvSize;
    convSize.back().numFieldGrains.reserve(numFieldDims);
    convSize.back().kernelSize.reserve(numFieldDims);
    const auto levelMaxSplit = hierarchy[level];
    PartitionVariables p;
    p.fieldSplit.reserve(numFieldDims);
    p.kernelSplit.reserve(numFieldDims);
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      p.fieldSplit.push_back(m.addVariable(1, levelMaxSplit));
      m.lessOrEqual(p.fieldSplit.back(), prevConvSize.numFieldGrains[dim]);
      // Currenlty the implementation doesn't support splitting the innermost
      // kernel dimension. TODO lift this restriction.
      if (dim == numFieldDims - 1) {
        p.kernelSplit.push_back(m.addConstant(1));
      } else {
        p.kernelSplit.push_back(m.addVariable(1, levelMaxSplit));
        m.lessOrEqual(p.kernelSplit.back(), prevConvSize.kernelSize[dim]);
      }
      nextConvSize.numFieldGrains.push_back(
        m.ceildiv(prevConvSize.numFieldGrains[dim], p.fieldSplit.back())
      );
      nextConvSize.kernelSize.push_back(
        m.ceildiv(prevConvSize.kernelSize[dim], p.kernelSplit.back())
      );
    }
    p.batchSplit = m.addVariable(1, levelMaxSplit);
    m.lessOrEqual(p.batchSplit, prevConvSize.batchSize);
    p.convGroupSplit = m.addVariable(1, levelMaxSplit);
    m.lessOrEqual(p.convGroupSplit, prevConvSize.numConvGroups);
    // The joint planning cost function assumes that no exchange is required to
    // rearrange weights between passes. Because of the way we derive the
    // backward and weight update plans from the forward plan this is guaranteed
    // to be the case if each weight is used on exactly one tile in the forward
    // pass. Disallow splitting of fully connected batch (or equivalently the
    // convolutional output channels) across tiles to ensure this holds.
    if (options.pass == Pass::FC_TRAINING_FWD) {
      p.outChanSplit = m.addConstant(1);
    } else {
      p.outChanSplit = m.addVariable(1, levelMaxSplit);
      m.lessOrEqual(p.outChanSplit, prevConvSize.numOutChanGrains);
    }
    p.inChanSplit = m.addVariable(1, levelMaxSplit);
    m.lessOrEqual(p.inChanSplit, prevConvSize.numInChanGrains);
    p.outChanGrainSize = outChanGrainSize;
    p.inChanGrainSize = inChanGrainSize;
    p.fieldGrainSize = fieldGrainSize;
    nextConvSize.batchSize = m.ceildiv(prevConvSize.batchSize, p.batchSplit);
    nextConvSize.numConvGroups =
        m.ceildiv(prevConvSize.numConvGroups, p.convGroupSplit);
    nextConvSize.numOutChanGrains =
        m.ceildiv(prevConvSize.numOutChanGrains, p.outChanSplit);
    nextConvSize.numInChanGrains =
        m.ceildiv(prevConvSize.numInChanGrains, p.inChanSplit);
    partitionVars.push_back(std::move(p));
    convSize.push_back(std::move(nextConvSize));
  }

  std::vector<Variable> perLevelSplits;
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
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
  convSize.erase(convSize.begin());

  cycles =
      addCycleEstimate(m, partitionVars, convSize, usedTiles, target,
                       perLevelExchangeBytesPerCycle, params,
                       inChansPerGroup, partialChansPerGroup,
                       convVertexType.floatPartials,
                       floatActivations, convVertexType.method,
                       Plan::LinearizeTileOrder::STANDARD, cache);
  if (options.pass == Pass::FC_TRAINING_FWD) {
    auto bwdParams = params;
    std::swap(bwdParams.inputFieldShape.back(), bwdParams.inputChannels);
    std::vector<PartitionVariables> bwdPartitionVars;
    std::vector<ConvSizeVariables> bwdConvSize;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      const auto &p = partitionVars[level];
      auto bwdP = p;
      bwdP.fieldSplit.back() = p.inChanSplit;
      bwdP.inChanSplit = p.fieldSplit.back();
      bwdP.inChanGrainSize = p.fieldGrainSize.back();
      bwdP.fieldGrainSize.back() = inChansPerGroup;
      bwdPartitionVars.push_back(bwdP);

      const auto &s = convSize[level];
      auto bwdS = s;
      bwdS.numFieldGrains.back() = s.numInChanGrains;
      bwdS.numInChanGrains = s.numFieldGrains.back();
      bwdConvSize.push_back(bwdS);
    }
    const auto bwdInChansPerGroup = bwdPartitionVars.back().inChanGrainSize;
    const auto bwdMethod =
        getFullyConnectedBwdMethod(params,
                                   convVertexType.method);
    const auto bwdCycles =
        addCycleEstimate(m, bwdPartitionVars, bwdConvSize,
                         usedTiles, target, perLevelExchangeBytesPerCycle,
                         params, bwdInChansPerGroup, partialChansPerGroup,
                         convVertexType.floatPartials,
                         floatActivations, bwdMethod,
                         Plan::LinearizeTileOrder::FC_BWD_AS_CONV, cache);
    auto wuParams = params;
    std::swap(wuParams.inputChannels, wuParams.outputChannels);
    std::vector<PartitionVariables> wuPartitionVars;
    std::vector<ConvSizeVariables> wuConvSize;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      const auto &p = partitionVars[level];
      auto wuP = p;
      wuP.outChanSplit = p.inChanSplit;
      wuP.inChanSplit = p.outChanSplit;
      wuP.inChanGrainSize = p.outChanGrainSize;
      wuP.outChanGrainSize = p.inChanGrainSize;
      wuP.fieldGrainSize = std::vector<unsigned>(numFieldDims, 1);
      wuPartitionVars.push_back(wuP);

      const auto &s = convSize[level];
      auto wuS = s;
      wuS.numInChanGrains = s.numOutChanGrains;
      wuS.numOutChanGrains = s.numInChanGrains;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto fieldGrainSize = partitionVars[level].fieldGrainSize[dim];
        if (partitionVars[level].fieldGrainSize[dim] != 1) {
          wuS.numFieldGrains[dim] =
              m.product({s.numFieldGrains[dim], m.addConstant(fieldGrainSize)});
        }
      }
      wuConvSize.push_back(wuS);
    }
    const auto wuInChansPerGroup = partialChansPerGroup;
    const auto wuPartialChansPerGroup = inChansPerGroup;
    const auto wuMethod =
        getFullyConnectedWUMethod(params,
                                  convVertexType.method,
                                  partialChansPerGroup,
                                  inChansPerGroup);
    const auto wuCycles =
        addCycleEstimate(m, wuPartitionVars, wuConvSize,
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
             [=](const std::vector<unsigned> &values) {
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
}

template <class TransformCostFn>
static std::pair<Plan, Cost>
choosePlan(const poplar::Target &target,
           const std::vector<unsigned> &hierarchy,
           const std::vector<double> &perLevelExchangeBytesPerCycle,
           const std::vector<unsigned> &fieldGrainSize,
           const ConvVertexType &convVertexType,
           const ConvParams &params,
           TransformCostFn &&transformCost,
           Cost bestCost,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  popsolver::Variable cycles;
  constructModel(target, hierarchy, perLevelExchangeBytesPerCycle,
                 fieldGrainSize, convVertexType, params, transformCost,
                 bestCost, costBounds, cache, options, m, partitionVars,
                 cycles);
  popsolver::Solution s;
  try {
    assert(costBounds.primaryCheckIsCycles);
    s = m.minimize(cycles);
  } catch (popsolver::NoSolution) {
    return {Plan(), highestCost};
  }
  std::vector<Partition> partitions;
  for (const auto &p : partitionVars) {
    partitions.push_back(makePartition(s, p));
  }
  Plan plan(std::move(partitions),
            convVertexType.inChansPerGroup, convVertexType.partialChansPerGroup,
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
    if (params.getTransformedKernelSize(dim) != 1) {
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
         std::all_of(params.outputTransform.truncationLower.begin(),
                     params.outputTransform.truncationLower.end(),
                     equalsZero) &&
         std::all_of(params.outputTransform.truncationUpper.begin(),
                     params.outputTransform.truncationUpper.end(),
                     equalsZero) &&
         std::all_of(params.outputTransform.stride.begin(),
                     params.outputTransform.stride.end(), equalsOne) &&
         std::all_of(params.outputTransform.paddingLower.begin(),
                     params.outputTransform.paddingLower.end(), equalsZero) &&
         std::all_of(params.outputTransform.paddingUpper.begin(),
                     params.outputTransform.paddingUpper.end(), equalsZero) &&
         std::all_of(params.inputTransform.dilation.begin(),
                     params.inputTransform.dilation.end(), equalsOne) &&
         std::all_of(params.inputTransform.flip.begin(),
                     params.inputTransform.flip.end(), equalsZero) &&
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
  params.inputTransform.truncationLower[dim] = 0;
  params.inputTransform.truncationUpper[dim] = 0;
  params.inputTransform.dilation[dim] = 1;
  params.inputTransform.paddingLower[dim] = 0;
  params.inputTransform.paddingUpper[dim] = 0;
  params.inputTransform.flip[dim] = false;
  params.kernelTransform.truncationLower[dim] = 0;
  params.kernelTransform.truncationUpper[dim] = 0;
  params.kernelTransform.dilation[dim] = 1;
  params.kernelTransform.paddingLower[dim] = 0;
  params.kernelTransform.paddingUpper[dim] = 0;
  params.kernelTransform.flip[dim] = false;
  params.outputTransform.truncationLower[dim] = 0;
  params.outputTransform.truncationUpper[dim] = 0;
  params.outputTransform.stride[dim] = 1;
  params.outputTransform.paddingLower[dim] = 0;
  params.outputTransform.paddingUpper[dim] = 0;
}

static unsigned
estimateTransformCycles(const poplar::Target &target,
                        const ConvParams &transformedParams,
                        const ConvOptions &options,
                        const ConvTransform &transform,
                        unsigned usedTiles) {
  assert(options.pass != Pass::FC_TRAINING_WU &&
         options.pass != Pass::FC_TRAINING_BWD);
  bool isWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool rearrangeInput = isWeightUpdate || !transform.expandDims.empty() ||
                        transform.swapOperands || transform.inChansPadding;
  bool rearrangeWeights = isWeightUpdate || !transform.expandDims.empty() ||
                          !transform.outChanFlattenDims.empty() ||
                          transform.swapOperands || transform.inChansPadding ||
                          transform.partialChansPadding;
  bool rearrangeOutput = (!isWeightUpdate && transform.swapOperands) ||
                         (isWeightUpdate && !transform.swapOperands) ||
                         !transform.outChanFlattenDims.empty() ||
                         transform.partialChansPadding;
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
  return params.getTransformedKernelSize(dim) == 1 &&
         params.inputTransform.truncationLower[dim] == 0 &&
         params.inputTransform.truncationUpper[dim] == 0 &&
         params.inputTransform.dilation[dim] == 1 &&
         params.inputTransform.paddingLower[dim] == 0 &&
         params.inputTransform.paddingUpper[dim] == 0 &&
         !params.inputTransform.flip[dim] &&
         params.outputTransform.truncationLower[dim] == 0 &&
         params.outputTransform.truncationUpper[dim] == 0 &&
         params.outputTransform.stride[dim] == 1 &&
         params.outputTransform.paddingLower[dim] == 0 &&
         params.outputTransform.paddingUpper[dim] == 0;
}

void
swapOperands(ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> extraInputPadding(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto transformedInputSize = params.getTransformedInputSize(dim);
    const auto transformedKernelSize = params.getTransformedKernelSize(dim);
    extraInputPadding[dim] = transformedInputSize - transformedKernelSize;
  }
  std::swap(params.inputFieldShape, params.kernelShape);
  std::swap(params.inputTransform, params.kernelTransform);
  std::swap(params.batchSize, params.outputChannels);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    params.inputTransform.flip[dim] = !params.inputTransform.flip[dim];
    params.kernelTransform.flip[dim] = !params.kernelTransform.flip[dim];
    params.inputTransform.paddingLower[dim] += extraInputPadding[dim];
    params.inputTransform.paddingUpper[dim] += extraInputPadding[dim];
  }
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
  insertAtFront(params.inputFieldShape, extraDims, std::size_t(1));
  insertAtFront(params.kernelShape, extraDims, std::size_t(1));

  insertAtFront(params.inputTransform.truncationLower, extraDims, 0U);
  insertAtFront(params.inputTransform.truncationUpper, extraDims, 0U);
  insertAtFront(params.inputTransform.dilation, extraDims, 1U);
  insertAtFront(params.inputTransform.paddingLower, extraDims, 0U);
  insertAtFront(params.inputTransform.paddingUpper, extraDims, 0U);
  insertAtFront(params.inputTransform.flip, extraDims, false);

  insertAtFront(params.kernelTransform.truncationLower, extraDims, 0U);
  insertAtFront(params.kernelTransform.truncationUpper, extraDims, 0U);
  insertAtFront(params.kernelTransform.dilation, extraDims, 1U);
  insertAtFront(params.kernelTransform.paddingLower, extraDims, 0U);
  insertAtFront(params.kernelTransform.paddingUpper, extraDims, 0U);
  insertAtFront(params.kernelTransform.flip, extraDims, false);

  insertAtFront(params.outputTransform.truncationLower, extraDims, 0U);
  insertAtFront(params.outputTransform.truncationUpper, extraDims, 0U);
  insertAtFront(params.outputTransform.stride, extraDims, 1U);
  insertAtFront(params.outputTransform.paddingLower, extraDims, 0U);
  insertAtFront(params.outputTransform.paddingUpper, extraDims, 0U);
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
  flattenDims.clear();
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
                      unsigned partialChansPerGroup, unsigned &inChansPadding,
                      unsigned &partialChansPadding) {
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
  const auto clockFrequency = target.getTileClockFrequency();
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
  ConvTransform transform;
  transform.extraFieldDims = addedFieldDims;
  for (bool swapOperands : getSwapOperandCandidates(options)) {
    transform.swapOperands = swapOperands;
    auto swappedParams = calculateSwappedParams(params, swapOperands);
    for (std::vector<unsigned> expandDims :
         getExpandDimsCandidates(swappedParams)) {
      transform.expandDims = expandDims;
      auto expandedParams = calculateExpandedParams(swappedParams, expandDims);
      for (std::vector<unsigned> outChanFlattenDims :
           getOutChanFlattenDimsCandidates(expandedParams)) {
        transform.outChanFlattenDims = outChanFlattenDims;
        auto flattenedParams = calculateFlattenedParams(expandedParams,
                                                        outChanFlattenDims,
                                                        transform.flattenDims);
        const bool floatActivations = params.dType == poplar::FLOAT;
        const bool floatPartials = partialsType == poplar::FLOAT;
        const auto convVertexTypeCandidates =
            getConvVertexTypeCandidates(target, floatActivations,
                                        floatPartials, flattenedParams,
                                        options);
        for (const auto &convVertexType : convVertexTypeCandidates) {
          const auto inChansPerGroup = convVertexType.inChansPerGroup;
          const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
          auto paddedParams =
              calculatePaddedParams(flattenedParams, inChansPerGroup,
                                    partialChansPerGroup,
                                    transform.inChansPadding,
                                    transform.partialChansPadding);
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
                                           paddedParams, options, transform,
                                           usedTiles);
          };
          std::tie(candidate, candidateCost) =
              choosePlan(target, hierarchy, perLevelExchangeBytesPerCycle,
                         fieldGrainSize, convVertexType,
                         paddedParams, transformCostFn, bestCost, costBounds,
                         cache, options);
          if (candidateCost == highestCost)
            continue;
          candidate.transform = transform;
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
  assert(params.inputTransform.truncationLower[0] == 0);
  assert(params.inputTransform.truncationUpper[0] == 0);
  assert(params.inputTransform.dilation[0] == 1);
  assert(params.inputTransform.paddingLower[0] == 0);
  assert(params.inputTransform.paddingUpper[0] == 0);
  assert(params.kernelTransform.truncationLower[0] == 0);
  assert(params.kernelTransform.truncationUpper[0] == 0);
  assert(params.kernelShape[0] == 1);
  assert(params.kernelTransform.truncationLower[0] == 0);
  assert(params.kernelTransform.truncationUpper[0] == 0);
  assert(params.outputTransform.truncationLower[0] == 0);
  assert(params.outputTransform.truncationUpper[0] == 0);
  assert(params.outputTransform.stride[0] == 1);
  assert(params.outputTransform.paddingLower[0] == 0);
  assert(params.outputTransform.paddingUpper[0] == 0);
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
                    1,                         // batchSize
                    {outputSize},              // inputShape
                    {1},                       // kernelShape
                    inputSize,                 // input channels
                    batchSize,                 // output channels
                    params.getNumConvGroups(), // conv groups
                    {0},                       // input truncation lower
                    {0},                       // input truncation upper
                    {1},                       // input dilation
                    {0},                       // input padding lower
                    {0},                       // input padding upper
                    {false},                   // flip input
                    {0},                       // kernel truncation lower
                    {0},                       // kernel truncation upper
                    {1},                       // kernel dilation
                    {0},                       // kernel padding lower
                    {0},                       // kernel padding upper
                    {false},                   // flip kernel
                    {0},                       // output truncation lower
                    {0},                       // output truncation upper
                    {1},                       // stride
                    {0},                       // output padding lower
                    {0}                        // output padding upper
                    );
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
  assert(!fwdPlan.transform.swapOperands);
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
  plan.transform.partialChansPadding = fwdPlan.transform.inChansPadding;
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  plan.transform.inChansPadding = fwdPlan.transform.partialChansPadding;

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
  assert(!fwdPlan.transform.swapOperands);
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdParams, fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  for (auto &partition : plan.partitions) {
    std::swap(partition.fieldSplit.back(), partition.inChanSplit);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
  const auto bwdInputChans = fwdParams.inputFieldShape.back();
  const auto bwdPaddedInputChans =
      ((bwdInputChans + plan.inChansPerGroup - 1) / plan.inChansPerGroup) *
      plan.inChansPerGroup;
  plan.transform.inChansPadding = bwdPaddedInputChans - bwdInputChans;
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
    assert(params.outputTransform.stride.size() == 2);
    assert(params.inputTransform.paddingLower.size() == 2);
    assert(params.inputTransform.paddingUpper.size() == 2);
    if (options.winogradPatchSize != 4 ||
        params.outputTransform.stride[0] != 1 ||
        params.outputTransform.stride[1] != 1 ||
        params.inputTransform.dilation[0] != 1 ||
        params.inputTransform.dilation[1] != 1 ||
        params.kernelShape[0] != 3 || params.kernelShape[1] != 3 ||
        params.getNumConvGroups() == 1) {
      throw poputil::poplib_error("Attempt to force winograd convolution for "
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
    throw poputil::poplib_error("Optimizing for memory is not supported");
  }
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(plan)));
    auto res = plans.emplace(std::make_pair(key, std::move(pPlan)));
    return *res.first->second;
  }
  return plan;
}

static void
constrainVariable(popsolver::Model &m, popsolver::Variable v, unsigned value) {
  m.lessOrEqual(v, value);
  m.lessOrEqual(value, v);
}

static void
constrainPartitionVars(popsolver::Model &m,
                       const PartitionVariables &vars,
                       const Partition &partition) {
  const auto numFieldDims = vars.fieldSplit.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    constrainVariable(m, vars.fieldSplit[dim], partition.fieldSplit[dim]);
    constrainVariable(m, vars.kernelSplit[dim], partition.kernelSplit[dim]);
  }
  constrainVariable(m, vars.batchSplit, partition.batchSplit);
  constrainVariable(m, vars.outChanSplit, partition.outChanSplit);
  constrainVariable(m, vars.inChanSplit, partition.inChanSplit);
  constrainVariable(m, vars.convGroupSplit, partition.convGroupSplit);
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
  const auto swappedParams =
      calculateSwappedParams(params, plan.transform.swapOperands);
  const auto expandedParams =
      calculateExpandedParams(swappedParams, plan.transform.expandDims);
  const auto flattenedParams =
      calculateFlattenedParams(expandedParams,
                               plan.transform.outChanFlattenDims,
                               flattenDims);
  unsigned inChansPadding, partialChansPadding;
  const auto paddedParams =
      calculatePaddedParams(flattenedParams, plan.inChansPerGroup,
                            plan.partialChansPerGroup, inChansPadding,
                            partialChansPadding);
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy =
      getTileHierarchy(target, perLevelExchangeBytesPerCycle);
  assert(perLevelExchangeBytesPerCycle.size() ==
         plan.partitions.size());
  auto transformCostFn = [&](unsigned usedTiles) {
    return estimateTransformCycles(target,
                                   paddedParams, options, plan.transform,
                                   usedTiles);
  };
  CostBounds costBounds(0, 0);
  bool floatActivations = params.dType == poplar::FLOAT;
  ConvVertexType convVertexType(plan.method, floatActivations,
                                plan.floatPartials,
                                plan.inChansPerGroup,
                                plan.partialChansPerGroup);
  const auto fieldGrainSize = plan.partitions.back().fieldAxisGrainSize;
  // Check grain size is the same at each level.
  for (const auto &p : plan.partitions) {
    assert(p.fieldAxisGrainSize == fieldGrainSize);
  }
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  popsolver::Variable cycles;
  constructModel(target, hierarchy, perLevelExchangeBytesPerCycle,
                 fieldGrainSize, convVertexType, paddedParams, transformCostFn,
                 highestCost, costBounds, cacheImpl, options, m, partitionVars,
                 cycles);
  const auto numLevelsOfHierarchy = plan.partitions.size();
  assert(partitionVars.size() == numLevelsOfHierarchy);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    constrainPartitionVars(m, partitionVars[level], plan.partitions[level]);
  }
  popsolver::Solution s;
  try {
    s = m.minimize(cycles);
  } catch (popsolver::NoSolution) {
    return highestCost.cycles;
  }
  return s[cycles];
}

} // end namespace conv
