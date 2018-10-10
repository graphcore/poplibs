#include "poplin/internal/ConvPlan.hpp"

#include "ConvUtilInternal.hpp"
#include "poplin/internal/ConvOptions.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/exceptions.hpp"
#include "poplar/Graph.hpp"
#include "poplin/ConvUtil.hpp"
#include "ConvValidation.hpp"
#include "poplibs_support/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/Compiler.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <iostream>
#include <popsolver/Model.hpp>
#include "poplibs_support/print.hpp"

namespace poplin {

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

  struct ConvSize {
    std::vector<unsigned> numFieldGrains;
    unsigned batchSize;
    unsigned numOutChanGrains;
    std::vector<unsigned> kernelSize;
    unsigned numInChanGrains;
    unsigned numConvGroups;
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
  poplar::Type dType;
  poplar::Type partialType;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  ConvVertexType(Plan::Method method, poplar::Type dType,
                 poplar::Type partialType, unsigned inChansPerGroup,
                 unsigned partialChansPerGroup) :
    method(method),
    dType(dType),
    partialType(partialType),
    inChansPerGroup(inChansPerGroup),
    partialChansPerGroup(partialChansPerGroup) {}
};

static const char *asString(Plan::Method m) {
  switch (m) {
  case Plan::Method::AMP: return "AMP";
  case Plan::Method::MAC: return "MAC";
  case Plan::Method::OUTER_PRODUCT: return "OUTER_PRODUCT";
  }
  POPLIB_UNREACHABLE();
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
  os << "  Transform:\n";
  os << "        dilatePostConv          ";
  printContainer(t.dilatePostConv, os);
  os << "\n"
     << "        swapOperands            "
     << t.swapOperands
     << "\n"
     << "             expandDims          ";
  printContainer(t.expandDims, os);
  os << "\n"
     << "        outChanFlattenDims      ";
  printContainer(t.outChanFlattenDims, os);
  os << "\n"
     << "        flattenDims             ";
  printContainer(t.flattenDims, os);
  os << "\n";
  return os;
}

std::ostream& operator<<(std::ostream &os, const ConvTypes &t) {
  os << "  Types: partialType        " << t.partialType << "\n";
  os << "         resultType         " << t.resultType << "\n";
  return os;
}

std::ostream& operator<<(std::ostream &os, Plan::Method m) {
  os << asString(m);
  return os;
}

std::ostream& operator<<(std::ostream &os, const Plan &p)
{
  os << "  Plan:";
  const auto numLevels = p.transforms.size();
  for (unsigned i = 0; i != numLevels; ++i) {
    os << "        transform #" << i << "\n";
    os << p.transforms[i] << "\n";
    if (i + 1 != numLevels) {
      os << "        partition #" << i << "\n";
      os << p.partitions[i];
    }
    os << "        types #" << i << "\n";
    os << p.types[i];
  }
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
    unsigned outChansPerGroup,
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
  const auto kernelOuterElems = numKernelPositions / positionsOuter;
  const auto kernelInnerElems = positionsOuter;

  cycles = getConvPartialnx1SupervisorCycleInnerLoopEstimate(
             workList, kernelInnerElems, kernelOuterElems, filterHeight,
             outChansPerGroup, convUnitInputLoadElemsPerCycle,
             numConvUnitsPerTile, convUnitCoeffLoadBytesPerCycle,
             numWorkerContexts, floatWeights);
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
                                                        numWorkerContexts,
                                                        false);
  return cycles;
}

static std::uint64_t
estimateConvReduceCycles(unsigned outputSize,
                         unsigned reductionDepth,
                         bool floatOutput,
                         bool floatPartials,
                         unsigned numWorkers,
                         unsigned dataPathWidth) {
  if (reductionDepth == 0)
    return 0;
  return getReduceCycleEstimate(outputSize,
                                reductionDepth,
                                dataPathWidth,
                                floatOutput,
                                floatPartials,
                                numWorkers);
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
                                                unsigned outChansPerGroup,
                                                unsigned dataPathWidth);
class PlanningCacheImpl {
public:
  decltype(memoize(getConvPartial1x1InnerLoopCycleEstimate))
    mGetConvPartial1x1InnerLoopCycleEstimate;
  decltype(memoize(getConvPartialnx1InnerLoopCycleEstimate))
    mGetConvPartialnx1InnerLoopCycleEstimate;
  decltype(memoize(estimateConvPartialHorizontalMacInnerLoopCycles))
    mEstimateConvPartialHorizontalMacInnerLoopCycles;
  decltype(memoize(estimateConvReduceCycles))
    mEstimateConvReduceCycles;
  decltype(memoize(getNumberOfMACs))
    mGetNumberOfMACs;
  PlanningCacheImpl() :
    mGetConvPartial1x1InnerLoopCycleEstimate(
      memoize(getConvPartial1x1InnerLoopCycleEstimate)
    ),
    mGetConvPartialnx1InnerLoopCycleEstimate(
      memoize(getConvPartialnx1InnerLoopCycleEstimate)
    ),
    mEstimateConvPartialHorizontalMacInnerLoopCycles(
      memoize(estimateConvPartialHorizontalMacInnerLoopCycles)
    ),
    mEstimateConvReduceCycles(
      memoize(estimateConvReduceCycles)
    ),
    mGetNumberOfMACs(
      memoize(getNumberOfMACs)
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

static std::uint64_t
estimateConvPartialHorizontalMacInnerLoopCycles(unsigned numOutRows,
                                                unsigned tileOutWidth,
                                                unsigned outputStrideX,
                                                unsigned tileKernelHeight,
                                                unsigned tileKernelWidth,
                                                unsigned numWorkers,
                                                bool floatActivations,
                                                unsigned inChansPerGroup,
                                                unsigned outChansPerGroup,
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
    outChansPerGroup,
    dataPathWidth,
    numWorkers,
    floatActivations);
}

static popsolver::Variable
addZeroCycles(popsolver::Model &m,
              const poplar::Target &target,
              const popsolver::Variable partialsPerTile,
              const ConvSizeVariables &convSize,
              const std::unordered_set<unsigned> &transformedDims,
              const ConvParams &params,
              poplar::Type partialType,
              Plan::Method method) {
  enum { Yes, No, Maybe } zeroPartials = Maybe;
  if (method == Plan::Method::MAC) {
    zeroPartials = Yes;
  } else {
    const auto numFieldDims = params.getNumFieldDims();
    unsigned kernelElements = 1;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      if (transformedDims.count(dim))
        continue;
      kernelElements *= params.kernelShape[dim];
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
    if (kernelElements == 1) {
      zeroPartials = No;
    }
  }
  if (zeroPartials == No)
    return m.addConstant(0);
  const auto vectorWidth = m.addConstant(target.getVectorWidth(partialType));
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
canUseConvPartial1x1Vertex(
    const ConvParams &params,
    const std::unordered_set<unsigned> &transformedDims,
    const std::vector<unsigned> &transformedInputDilation,
    const std::vector<unsigned> &transformedOutputStride,
    unsigned convUnitWeightHeight,
    const std::vector<unsigned> &tileKernelShape) {
  if (convUnitWeightHeight != 1)
    return false;
  if (transformedInputDilation != transformedOutputStride)
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
    if (transformedDims.count(dim))
      continue;
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

static popsolver::Variable
addPartialCalcCycleEstimate(
    popsolver::Model &m,
    const std::vector<unsigned> &fieldGrainSize,
    unsigned inChanGrainSize,
    unsigned outChanGrainSize,
    const ConvSizeVariables &convSizeVars,
    const std::unordered_set<unsigned> &transformedDims,
    const poplar::Target &target,
    const ConvParams &params,
    unsigned inChansPerGroup,
    unsigned outChansPerGroup,
    poplar::Type inputType,
    poplar::Type partialType,
    Plan::Method method,
    PlanningCacheImpl *cache) {
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);
  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  const auto numFieldDims = convSizeVars.numFieldGrains.size();
  std::vector<popsolver::Variable> convSizeVarsVector = {
    convSizeVars.batchSize,
    convSizeVars.numOutChanGrains,
    convSizeVars.numInChanGrains,
    convSizeVars.numConvGroups,
  };
  convSizeVarsVector.insert(convSizeVarsVector.end(),
                            convSizeVars.numFieldGrains.begin(),
                            convSizeVars.numFieldGrains.end());
  convSizeVarsVector.insert(convSizeVarsVector.end(),
                            convSizeVars.kernelSize.begin(),
                            convSizeVars.kernelSize.end());
  auto makeConvSize = [](const std::vector<unsigned> &values,
                         unsigned numFieldDims) {
    ConvSize convSize;
    convSize.batchSize = values[0];
    convSize.numOutChanGrains = values[1];
    convSize.numInChanGrains = values[2];
    convSize.numConvGroups = values[3];
    convSize.numFieldGrains.insert(convSize.numFieldGrains.begin(),
                                   values.begin() + 4,
                                   values.begin() + 4 + numFieldDims);
    convSize.kernelSize.insert(convSize.kernelSize.begin(),
                               values.begin() + 4 + numFieldDims,
                               values.begin() + 4 + 2 * numFieldDims);
    return convSize;
  };
  auto makeTileFieldSize = [](const ConvSize &convSize,
                              const std::vector<unsigned> &fieldGrainSize) {
    const auto numFieldDims = convSize.numFieldGrains.size();
    std::vector<unsigned> tileFieldSize;
    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      tileFieldSize.push_back(convSize.numFieldGrains[dim] *
                              fieldGrainSize[dim]);
    }
    return tileFieldSize;
  };
  auto transformedInputDilation = params.inputTransform.dilation;
  auto transformedOutputStride = params.outputTransform.stride;
  for (const auto dim : transformedDims) {
    transformedInputDilation[dim] = 1;
    transformedOutputStride[dim] = 1;
  }
  switch (method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
    {
      assert(target.getWeightsPerConvUnit(floatActivations) %
           inChansPerGroup == 0);
      const auto convUnitWeightHeight =
         target.getWeightsPerConvUnit(floatActivations) / inChansPerGroup;
      const auto numConvUnits = getNumConvUnits(floatActivations,
                                               floatPartials,
                                               target);
      const auto convUnitInputLoadElemsPerCycle =
         target.getConvUnitInputLoadElemsPerCycle(floatActivations);

      return m.call(convSizeVarsVector,
          [=,&target](const std::vector<unsigned> &values) {
        auto convSize = makeConvSize(values, numFieldDims);
        auto tileFieldSize = makeTileFieldSize(convSize, fieldGrainSize);
        const auto tileNumInChans = convSize.numInChanGrains * inChanGrainSize;
        const auto tileNumInGroups =
            (tileNumInChans + inChansPerGroup - 1) / inChansPerGroup;
        const auto tileNumOutChans = convSize.numOutChanGrains *
                                     outChanGrainSize;
        const auto tileNumOutGroups =
            (tileNumOutChans + outChansPerGroup - 1) / outChansPerGroup;


        if (canUseConvPartial1x1Vertex(params, transformedDims,
                                       transformedInputDilation,
                                       transformedOutputStride,
                                       convUnitWeightHeight,
                                       convSize.kernelSize)) {
          auto innerLoopCycles =
              cache->mGetConvPartial1x1InnerLoopCycleEstimate(
                convSize.batchSize, tileFieldSize,
                target.getNumWorkerContexts(), transformedInputDilation,
                transformedOutputStride);
          // cycles cost assumes that cost of zeroing partials and overhead
          // of splitting vertices is negligible.
          return
              getConvPartial1x1SupervisorOuterLoopCycleEstimate(
                innerLoopCycles, innerLoopCycles,
                convSize.numConvGroups, tileNumInGroups,
                tileNumOutGroups, outChansPerGroup,
                convUnitInputLoadElemsPerCycle, numConvUnits,
                target.getConvUnitCoeffLoadBytesPerCycle(),
                floatActivations);
        }
        auto innerLoopCycles =
            cache->mGetConvPartialnx1InnerLoopCycleEstimate(
              convSize.batchSize, tileFieldSize, convSize.kernelSize,
              convUnitWeightHeight, outChansPerGroup,
              convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(), floatActivations,
              transformedInputDilation, transformedOutputStride);
        return
            getConvPartialnx1SupervisorCycleOuterLoopEstimate(
              innerLoopCycles, convSize.numConvGroups,
              tileNumOutGroups, tileNumInGroups, outChansPerGroup,
              numConvUnits);
      });
    }
  case Plan::Method::MAC:
    {
      const auto outputStrideX = transformedInputDilation.back();
      return m.call(convSizeVarsVector,
          [=,&target](const std::vector<unsigned> &values) {
        auto convSize = makeConvSize(values, numFieldDims);
        auto tileOutShape = makeTileFieldSize(convSize, fieldGrainSize);
        const auto tileNumInChans = convSize.numInChanGrains * inChanGrainSize;
        const auto tileNumInGroups =
            (tileNumInChans + inChansPerGroup - 1) / inChansPerGroup;
        const auto tileNumOutChans = convSize.numOutChanGrains *
                                     outChanGrainSize;
        const auto tileNumOutGroups =
            (tileNumOutChans + outChansPerGroup - 1) / outChansPerGroup;
        const auto tileKernelElements =
            std::accumulate(convSize.kernelSize.begin(),
                            convSize.kernelSize.end(),
                            1U,
                            std::multiplies<unsigned>());
        unsigned numActiveOutRows = convSize.batchSize;
        for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
          const auto dimActiveRows =
              (tileOutShape[dim] + transformedInputDilation[dim] - 1) /
              transformedInputDilation[dim];
          numActiveOutRows *= dimActiveRows;
        }
        const auto tileKernelWidth =
            convSize.kernelSize.back();
        const auto tileOutWidth = tileOutShape.back();
        auto innerLoopCycles =
            cache->mEstimateConvPartialHorizontalMacInnerLoopCycles(
              numActiveOutRows,
              tileOutWidth,
              outputStrideX,
              tileKernelElements / tileKernelWidth,
              tileKernelWidth,
              target.getNumWorkerContexts(),
              floatActivations,
              inChansPerGroup,
              outChansPerGroup,
              target.getDataPathWidth());
        return
            getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
              innerLoopCycles,
              convSize.numConvGroups,
              tileNumInGroups,
              tileNumOutGroups);
      });
    }
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      assert(params.getBatchSize() == 1);
      assert(params.getNumInputChansPerConvGroup() == 1);
      assert(std::all_of(transformedOutputStride.begin(),
                         transformedOutputStride.end(), equalsOne));
      assert(std::all_of(transformedInputDilation.begin(),
                         transformedInputDilation.end(), equalsOne));
      const auto numContexts = target.getNumWorkerContexts();
      return m.call(convSizeVarsVector,
          [=](const std::vector<unsigned> &values) {
        auto convSize = makeConvSize(values, numFieldDims);
        const auto tileOutWidth =
            convSize.numFieldGrains.back() * fieldGrainSize.back();
        const auto workerOutWidth =
            (tileOutWidth + numContexts - 1) / numContexts;
        const auto tileNumOutChans = convSize.numOutChanGrains *
                                     outChanGrainSize;
        auto vertexRuntime =
            getOuterProductCycleEstimate(floatActivations, workerOutWidth,
                                         tileNumOutChans *
                                             convSize.numConvGroups,
                                         outChansPerGroup,
                                         target.getDataPathWidth());
        return vertexRuntime * numContexts;
      });
    }
    break;
  }
}

unsigned getMaxMACsPerCyclePerTile(const poplar::Target &target,
                                   poplar::Type partialType,
                                   poplar::Type inputType,
                                   Plan::Method method) {
  auto vectorWidth = target.getVectorWidth(inputType);
  switch (method) {
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT:
    return vectorWidth;
  case Plan::Method::AMP:
    {
      assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
      assert(inputType == poplar::HALF || inputType == poplar::FLOAT);
      bool floatActivations = inputType == poplar::FLOAT;
      bool floatPartials = partialType == poplar::FLOAT;
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
  POPLIB_UNREACHABLE();
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
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const ConvParams &params,
    const std::vector<ConvTypes> &types,
    Plan::LinearizeTileOrder linearizeTileOrder) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto numLevelsOfHierarchy = convSizes.size();
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
      m.addConstant(target.getTypeSize(params.dType) *
                    exchangeBytesScalingFactor);
  std::vector<popsolver::Variable> cycleSumOperands;
  for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
    const auto scaledPartialSize =
        m.addConstant(target.getTypeSize(types[level + 1].resultType) *
                      exchangeBytesScalingFactor);
    const auto &convSize = convSizes[level + 1];
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
      outputFieldSizes.push_back(outputFieldSize);
      if (transformedDims[level].count(dim)) {
        inputFieldSizes.push_back(outputFieldSize);
      } else {
        auto inputFieldSize =
            m.call({outputFieldSize, convSize.kernelSize[dim]},
                   [=](const std::vector<unsigned> &values) {
          return getMaxInputRangeSize(values[0], dim, params, values[1]);
        });
        inputFieldSizes.push_back(inputFieldSize);
      }
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
        level + 2 == numLevelsOfHierarchy &&
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
    const std::vector<ConvTypes> &types,
    PlanningCacheImpl *cache) {
  std::vector<popsolver::Variable> cycleSumOperands;
  const auto numLevelsOfHierarchy = partitionVars.size();
  for (int level = numLevelsOfHierarchy - 1; level >= 0; --level) {
    auto reduceDimSizes = partitionVars[level].kernelSplit;
    reduceDimSizes.push_back(partitionVars[level].inChanSplit);
    const auto reductionDepth = m.product(reduceDimSizes);
    auto tileOutSize = m.ceildiv(partialsPerTile, reductionDepth);
    bool floatPartials = types[level + 1].resultType == poplar::FLOAT;
    bool floatOutput = types[level].resultType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    const auto numWorkers = target.getNumWorkerContexts();
    const auto cycleEstimate =
        m.call({tileOutSize, reductionDepth},
               [=](const std::vector<unsigned> &vars) -> unsigned {
      return cache->mEstimateConvReduceCycles(vars[0], vars[1], floatOutput,
                                              floatPartials, numWorkers,
                                              dataPathWidth);
    });
    cycleSumOperands.push_back(cycleEstimate);
    if (level != 0) {
      partialsPerTile = m.ceildiv(partialsPerTile, reductionDepth);
    }
  }
  return m.sum(cycleSumOperands);
}

static popsolver::Variable
addPartialsPerTile(popsolver::Model &m,
                   const PartitionVariables &partitionVars,
                   unsigned partialChansPerGroup,
                   const ConvSizeVariables &convSize) {
  unsigned grainSizeProduct = partialChansPerGroup;
  std::accumulate(partitionVars.fieldGrainSize.begin(),
                  partitionVars.fieldGrainSize.end(),
                  grainSizeProduct,
                  std::multiplies<unsigned>());
  auto partialDimSizes = convSize.numFieldGrains;
  partialDimSizes.push_back(convSize.batchSize);
  partialDimSizes.push_back(convSize.numConvGroups);
  partialDimSizes.push_back(convSize.numOutChanGrains);
  partialDimSizes.push_back(m.addConstant(grainSizeProduct));
  return m.product(partialDimSizes);
}

static popsolver::Variable
addCycleEstimate(popsolver::Model &m,
                 const std::vector<PartitionVariables> &partitionVars,
                 const std::vector<ConvSizeVariables> &convSize,
                 const std::vector<ConvSizeVariables> &transformedConvSize,
                 popsolver::Variable usedTiles,
                 const std::vector<
                   std::unordered_set<unsigned>
                 > &transformedDims,
                 const poplar::Target &target,
                 const std::vector<double> &perLevelExchangeBytesPerCycle,
                 const ConvParams &params,
                 unsigned inChansPerGroup,
                 unsigned partialChansPerGroup,
                 const std::vector<ConvTypes> &types,
                 Plan::Method method,
                 Plan::LinearizeTileOrder linearizeTileOrder,
                 PlanningCacheImpl *cache) {
  // popsolver takes into account whether a variable is an operand of a call
  // when deciding the order to set variables. Add a dummy call to ensure the
  // split variables are prioritized as this reduces the amount of time spent
  // in the planner. TODO Improve popsolver's heuristics for ordering variables
  // so this hack is no longer necessary (or provide a proper mechanism for
  // ordering hints).
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
  (void)m.call(variables, [](const std::vector<unsigned> &) {
    return 0U;
  });
  const auto exchangeCycles =
      addExchangeCycleEstimate(m, partitionVars, convSize, transformedDims,
                               target, perLevelExchangeBytesPerCycle, params,
                               types, linearizeTileOrder);
  const auto partialsPerTile = addPartialsPerTile(m, partitionVars.back(),
                                                  partialChansPerGroup,
                                                  transformedConvSize.back());
  const auto zeroCycles =
      addZeroCycles(m, target, partialsPerTile, transformedConvSize.back(),
                    transformedDims.back(), params, types.back().partialType,
                    method);
  const auto partialCalcCycles =
      addPartialCalcCycleEstimate(m, partitionVars.back().fieldGrainSize,
                                  inChansPerGroup,
                                  partialChansPerGroup,
                                  transformedConvSize.back(),
                                  transformedDims.back(),
                                  target, params, inChansPerGroup,
                                  partialChansPerGroup, params.dType,
                                  types.back().partialType, method, cache);
  // Add a redunant inequality that relates the cycles required to calculate the
  // partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile =
      getMaxMACsPerCyclePerTile(target, types.back().partialType, params.dType,
                                method);
  const auto totalMacs = cache->mGetNumberOfMACs(params);
  m.lessOrEqual(totalMacs / maxMACsPerCyclePerTile,
                m.product({usedTiles, partialCalcCycles}));
  const auto reduceCycles =
      addReduceCycleEstimate(m, partitionVars, partialsPerTile, target, types,
                             cache);
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
  if (fwdMethod == Plan::Method::OUTER_PRODUCT) {
    return Plan::Method::MAC;
  }
  return fwdMethod;
}

static Plan::Method
getFullyConnectedBwdMethod(Plan::Method fwdMethod) {
  if (fwdMethod == Plan::Method::OUTER_PRODUCT) {
    return Plan::Method::MAC;
  }
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

/// Return whether the dilation can be sunk until after the striding (before
/// output padding is applied).
static bool canDeferDilation(const ConvParams &params, unsigned dim) {
  return params.inputTransform.paddingLower[dim] == 0 &&
         params.inputTransform.paddingUpper[dim] == 0 &&
         params.outputTransform.stride[dim] == 1 &&
         params.outputTransform.truncationLower[dim] == 0 &&
         params.outputTransform.truncationUpper[dim] == 0 &&
         params.getTransformedKernelSize(dim) == 1;
}

ConvParams
calculateParamsWithDeferredDilation(
    const ConvParams &params,
    const std::vector<unsigned> &dilatePostConv) {
  auto paramsWithDeferredDilation = params;
  for (const auto dim : dilatePostConv) {
    assert(canDeferDilation(params, dim));
    paramsWithDeferredDilation.inputTransform.dilation[dim] = 1;
    paramsWithDeferredDilation.outputTransform.paddingLower[dim] = 0;
    paramsWithDeferredDilation.outputTransform.paddingUpper[dim] = 0;
  }
  return paramsWithDeferredDilation;
}

static ConvParams
calculateSwappedParams(const ConvParams &params, bool swapOperands) {
  auto swappedParams = params;
  if (swapOperands) {
    poplin::swapOperands(swappedParams);
  }
  return swappedParams;
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

static ConvParams
calculateExpandedParams(const ConvParams &params,
                        const std::vector<unsigned> &expandDims) {
  auto expandedParams = params;
  for (unsigned dim : expandDims) {
    expandDim(expandedParams, dim);
  }
  return expandedParams;
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

static ConvParams
calculateFlattenedParams(const ConvParams &params,
                         const std::vector<unsigned> &outChanFlattenDims,
                         std::vector<unsigned> &flattenDims) {
  flattenDims.clear();
  auto flattenedParams = params;
  if (!outChanFlattenDims.empty()) {
    poplin::swapOperands(flattenedParams);
    for (unsigned dim : outChanFlattenDims) {
      expandDim(flattenedParams, dim);
      // Flatten into the batch axis (this will become the output channel
      // axis when we swap back).
      flattenedParams.batchSize *= flattenedParams.inputFieldShape[dim];
      flattenedParams.inputFieldShape[dim] = 1;
    }
    poplin::swapOperands(flattenedParams);
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
calculatePaddedParams(const ConvParams &params, unsigned inChanGrainSize,
                      unsigned partialChanGrainSize, unsigned &inChansPadding,
                      unsigned &partialChansPadding) {
  auto paddedParams = params;
  const auto inChans = params.getNumInputChansPerConvGroup();
  paddedParams.inputChannels =
      ((inChans + inChanGrainSize - 1) / inChanGrainSize) *
      inChanGrainSize;
  inChansPadding = paddedParams.inputChannels - inChans;
  const auto partialChans =
      params.getNumOutputChansPerConvGroup();
  paddedParams.outputChannels =
      ((partialChans + partialChanGrainSize - 1) / partialChanGrainSize) *
      partialChanGrainSize;
  partialChansPadding = paddedParams.outputChannels - partialChans;
  return paddedParams;
}

static ConvParams applyTransform(const ConvParams &params,
                                 const ConvTransform &transform,
                                 unsigned inChanGrainSize,
                                 unsigned outChanGrainSize) {
  auto paramsWithExtraDims = params;
  addExtraDims(paramsWithExtraDims, transform.extraFieldDims);
  auto paramsWithDeferredDilation =
      calculateParamsWithDeferredDilation(paramsWithExtraDims,
                                          transform.dilatePostConv);
  auto swappedParams =
      calculateSwappedParams(paramsWithDeferredDilation,
                             transform.swapOperands);
  const auto expandedParams =
      calculateExpandedParams(swappedParams, transform.expandDims);
  std::vector<unsigned> flattenDims;
  const auto flattenedParams =
      calculateFlattenedParams(expandedParams, transform.outChanFlattenDims,
                               flattenDims);
  unsigned inChansPadding, outChansPadding;
  auto paddedParams = calculatePaddedParams(flattenedParams, inChanGrainSize,
                                            outChanGrainSize, inChansPadding,
                                            outChansPadding);
  return paddedParams;
}

static void getTransformedDims(const ConvTransform &transform,
                               std::unordered_set<unsigned> &transformed) {
  for (const auto dim : transform.expandDims) {
    transformed.insert(dim);
  }
  for (const auto dim : transform.outChanFlattenDims) {
    transformed.insert(dim);
  }
  for (const auto dim : transform.flattenDims) {
    if (dim == 0)
      continue;
    transformed.insert(dim - 1);
  }
}

static std::vector<unsigned>
getOutChanGrainSizes(const std::vector<ConvTransform> &transforms,
                     unsigned partialChansPerGroup) {
  assert(transforms.size() >= 1);
  std::vector<unsigned> outChanGrainSizes(transforms.size());
  // The grain size at the last level is equal to partialChansPerGroup.
  // To avoid rearrangement we use the same grain size at upper levels
  // unless these is a transform that rearranges the output channel axis.
  outChanGrainSizes[transforms.size() - 1] = partialChansPerGroup;
  for (int i = static_cast<int>(transforms.size()) - 2; i >= 0; --i) {
    outChanGrainSizes[i] = (i == 0 ||
                            transforms[i].outChanFlattenDims.empty()) ?
                           outChanGrainSizes[i + 1] : 1;
  }
  return outChanGrainSizes;
}

static std::vector<unsigned>
getInChanGrainSizes(const std::vector<ConvTransform> &transforms,
                    unsigned inChansPerGroup) {
  assert(transforms.size() >= 1);
  std::vector<unsigned> inChanGrainSizes(transforms.size());
  // The grain size at the last level is equal to inChansPerGroup.
  // To avoid rearrangement we use the same grain size at upper levels
  // unless these is a transform that rearranges the input channel axis.
  inChanGrainSizes[transforms.size() - 1] = inChansPerGroup;
  for (int i = static_cast<int>(transforms.size()) - 2; i >= 0; --i) {
    inChanGrainSizes[i] = (transforms[i + 1].outChanFlattenDims.empty() &&
                           transforms[i + 1].expandDims.empty()) ?
                          inChanGrainSizes[i + 1] : 1;
  }
  return inChanGrainSizes;
}

popsolver::Variable
addTransformCycleEstimate(
    popsolver::Model &m,
    const ConvParams &params,
    const ConvParams &transformedOnceParams,
    const std::vector<ConvTransform> &transforms,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &transformedConvSizes,
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    unsigned inChansPerGroup,
    unsigned partialChansPerGroup,
    const std::vector<ConvTypes> &types,
    const ConvOptions &options,
    const poplar::Target &target) {
  assert(options.pass != Pass::FC_TRAINING_WU &&
         options.pass != Pass::FC_TRAINING_BWD);
  bool isWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool expandDims = false;
  bool swapOperands = false;
  bool outChanFlattenDims = false;
  assert(transforms.size() >= 2);
  const auto ipuLevel = transforms.size() - 2;
  for (unsigned level = 0; level <= ipuLevel; ++level) {
    if (transforms[level].swapOperands)
      swapOperands = true;
    if (!transforms[level].expandDims.empty())
      expandDims = true;
    if (!transforms[level].outChanFlattenDims.empty())
      outChanFlattenDims = true;
  }
  bool padInChannels = params.inputChannels % inChansPerGroup;
  bool padPartialChannels = params.outputChannels % partialChansPerGroup;
  bool rearrangeInput = isWeightUpdate || expandDims ||
                        swapOperands || padInChannels;
  bool rearrangeWeights = isWeightUpdate || expandDims ||
                          outChanFlattenDims ||
                          swapOperands || padInChannels ||
                          padPartialChannels;
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(params.dType == poplar::FLOAT);
  bool rearrangeOutput = (!isWeightUpdate && swapOperands) ||
                         (isWeightUpdate && !swapOperands) ||
                         outChanFlattenDims ||
                         padPartialChannels;
  // We assume the next layer uses an input channel grouping of
  // weightsPerConvUnit and apply a small cost if the output channel
  // grouping of this layer doesn't match.
  bool regroupOutput = options.pass != Pass::FC_TRAINING_FWD &&
                       partialChansPerGroup != weightsPerConvUnit;
  // If the input channel grouping of the backward pass doesn't divide the
  // output channel grouping of the forward pass the block size for the
  // cross-tile rearrangement of weights between the forward and backward pass
  // will be small. We assume the backward pass uses an input channel grouping
  // of weightsPerConvUnit and apply a small cost if the output channel grouping
  // of this layer isn't a multiple of this weightsPerConvUnit.
  bool regroupWeights = options.pass == Pass::TRAINING_FWD &&
                        partialChansPerGroup % weightsPerConvUnit != 0;
  const auto bytesPerElement = target.getTypeSize(params.dType);
  const auto regroupBytesPerCycle =
      std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                         partialChansPerGroup * bytesPerElement);
  if (!rearrangeInput && !rearrangeOutput && !rearrangeWeights &&
      !regroupOutput && !regroupWeights)
    return m.addConstant(0);
  const auto &convSize = transformedConvSizes[ipuLevel];
  std::vector<popsolver::Variable> outputFieldSizes;
  std::vector<popsolver::Variable> inputFieldSizes;
  const auto numFieldDims = partitionVars[ipuLevel].fieldSplit.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto fieldGrainSize = partitionVars[ipuLevel].fieldGrainSize[dim];
    auto outputFieldSize = convSize.numFieldGrains[dim];
    if (fieldGrainSize != 1) {
      outputFieldSize = m.product({outputFieldSize,
                                   m.addConstant(fieldGrainSize)});
    }
    outputFieldSizes.push_back(outputFieldSize);
    if (transformedDims[ipuLevel].count(dim)) {
      inputFieldSizes.push_back(outputFieldSize);
    } else {
      auto inputFieldSize =
          m.call({outputFieldSize, convSize.kernelSize[dim]},
                 [=](const std::vector<unsigned> &values) {
        return getMaxInputRangeSize(values[0], dim, transformedOnceParams,
                                    values[1]);
      });
      inputFieldSizes.push_back(inputFieldSize);
    }
  }
  auto numInChans =
      m.product({convSize.numInChanGrains,
                 m.addConstant(partitionVars[ipuLevel].inChanGrainSize)});
  auto numOutChans =
      m.product({convSize.numOutChanGrains,
                 m.addConstant(partitionVars[ipuLevel].outChanGrainSize)});
  std::vector<popsolver::Variable> ipuSplits = {
    partitionVars[ipuLevel].batchSplit,
    partitionVars[ipuLevel].convGroupSplit,
    partitionVars[ipuLevel].inChanSplit,
    partitionVars[ipuLevel].outChanSplit
  };
  ipuSplits.insert(ipuSplits.end(),
                   partitionVars[ipuLevel].fieldSplit.begin(),
                   partitionVars[ipuLevel].fieldSplit.end());
  ipuSplits.insert(ipuSplits.end(),
                   partitionVars[ipuLevel].kernelSplit.begin(),
                   partitionVars[ipuLevel].kernelSplit.end());
  auto ipuUsedTiles = m.product(ipuSplits);
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  std::vector<popsolver::Variable> cyclesOperands;
  if (rearrangeInput || rearrangeWeights || regroupWeights) {
    const auto reorderBytesPerCycle =
        std::min<unsigned>(target.getMemcpyBytesPerCycle(), bytesPerElement);
    std::vector<popsolver::Variable> numElementsOperands;
    if (rearrangeInput) {
      auto totalInputFieldSize = m.product(inputFieldSizes);
      auto numInputElements =
          m.product({totalInputFieldSize, convSize.batchSize, numInChans,
                     convSize.numConvGroups});
      numElementsOperands.push_back(numInputElements);
    }
    if (rearrangeWeights || regroupWeights) {
      auto totalKernelSize = m.product(convSize.kernelSize);
      auto numWeightElements =
          m.product({totalKernelSize, numInChans, numOutChans,
                     convSize.numConvGroups});
      if (rearrangeWeights) {
        numElementsOperands.push_back(numWeightElements);
      } else if (regroupWeights) {
        auto numElementsPerTile = m.ceildiv(numWeightElements, ipuUsedTiles);
        auto bytesPerTile = m.product({numElementsPerTile,
                                       m.addConstant(bytesPerElement)});
        cyclesOperands.push_back(
          m.ceildiv(bytesPerTile, m.addConstant(regroupBytesPerCycle))
        );
      }
    }
    auto numElements = m.sum(numElementsOperands);
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    auto bytesPerTile = m.product({numElementsPerTile,
                                   m.addConstant(bytesPerElement)});
    cyclesOperands.push_back(m.ceildiv(bytesPerTile,
                                       m.addConstant(exchangeBytesPerCycle)));
    cyclesOperands.push_back(m.ceildiv(bytesPerTile,
                                       m.addConstant(reorderBytesPerCycle)));
  }
  if (rearrangeOutput || regroupOutput) {
    auto totalOutputFieldSize = m.product(outputFieldSizes);
    auto numElements =
        m.product({totalOutputFieldSize, convSize.batchSize,
                   numOutChans, convSize.numConvGroups});
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    const auto outputBytesPerElement =
        target.getTypeSize(types[ipuLevel].resultType);
    auto bytesPerTile = m.product({numElementsPerTile,
                                   m.addConstant(outputBytesPerElement)});
    if (rearrangeOutput) {
      const auto outputReorderBytesPerCycle =
          std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                             outputBytesPerElement);
      cyclesOperands.push_back(m.ceildiv(bytesPerTile,
                                         m.addConstant(exchangeBytesPerCycle)));
      cyclesOperands.push_back(
        m.ceildiv(bytesPerTile, m.addConstant(outputReorderBytesPerCycle))
      );
    } else if (regroupOutput) {
      cyclesOperands.push_back(
        m.ceildiv(bytesPerTile, m.addConstant(regroupBytesPerCycle))
      );
    }
  }
  auto cycles = m.sum(cyclesOperands);
  // Apply an experimentally determined fudge factor to account for other
  // overheads that aren't modeled.
  cycles = m.ceildiv(m.product({cycles, m.addConstant(3)}), m.addConstant(2));
  return cycles;
}

static void
constructModel(const poplar::Target &target,
               const std::vector<ConvTransform> &transforms,
               const std::vector<ConvTypes> &types,
               const std::vector<unsigned> &hierarchy,
               const std::vector<double> &perLevelExchangeBytesPerCycle,
               const std::vector<unsigned> &fieldGrainSize,
               const ConvVertexType &convVertexType,
               const ConvParams &params_,
               Cost bestCost,
               const CostBounds costBounds,
               PlanningCacheImpl *cache,
               const ConvOptions &options,
               popsolver::Model &m,
               std::vector<PartitionVariables> &partitionVars,
               popsolver::Variable &cycles) {
  const auto inChansPerGroup = convVertexType.inChansPerGroup;
  const auto partialChansPerGroup = convVertexType.partialChansPerGroup;
  const auto outChanGrainSize = getOutChanGrainSizes(transforms,
                                                     partialChansPerGroup);
  const auto inChanGrainSize = getInChanGrainSizes(transforms,
                                                   inChansPerGroup);
  // Apply the top level transform to the parameters. The top level transform is
  // the only transform that can add dimensions / swap operands. Applying the
  // top level transform to the parameters here means we don't need to support
  // adding dimensions / swapping operands in the generic code that handles
  // transforms different levels.
  auto params = applyTransform(params_, transforms[0], inChanGrainSize[0],
                               outChanGrainSize[0]);
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
  const auto numLevelsOfHierarchy = hierarchy.size() + 1;
  assert(numLevelsOfHierarchy >= 1);
  partitionVars.clear();

  const auto outChanGrains = params.getNumOutputChansPerConvGroup() ?
      (params.getNumOutputChansPerConvGroup() + outChanGrainSize[0] - 1) /
      outChanGrainSize[0] : 1;
  const auto inChanGrains = params.getNumInputChansPerConvGroup() ?
      (params.getNumInputChansPerConvGroup() + inChanGrainSize[0] - 1)
      / inChanGrainSize[0] : 1;

  // For each level the set of dimensions that are flattened / expanded.
  std::vector<std::unordered_set<unsigned>> transformedDims;
  std::vector<ConvSizeVariables> convSize;
  std::vector<ConvSizeVariables> transformedConvSize;
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
    if (level == 0) {
      transformedDims.emplace_back();
    } else {
      transformedDims.emplace_back(transformedDims.back());
    }
    getTransformedDims(transforms[level], transformedDims.back());
    transformedConvSize.push_back(convSize.back());
    // Don't transform level 0 since this transform has already been applied to
    // the parameters.
    if (level != 0) {
      assert(!transforms[level].swapOperands);
      assert(transforms[level].extraFieldDims == 0);
      assert(transforms[level].dilatePostConv.empty());
      for (const auto dim : transforms[level].expandDims) {
        transformedConvSize.back().numInChanGrains =
            m.product({transformedConvSize.back().numInChanGrains,
                       transformedConvSize.back().kernelSize[dim]});
        transformedConvSize.back().kernelSize[dim] = m.addConstant(1);
      }
      for (const auto dim : transforms[level].outChanFlattenDims) {
        popsolver::Variable outputSize =
            transformedConvSize.back().numFieldGrains[dim];
        if (fieldGrainSize[dim] != 1) {
          outputSize = m.product({outputSize,
                                  m.addConstant(fieldGrainSize[dim])});
        }
        transformedConvSize.back().numOutChanGrains =
            m.product({transformedConvSize.back().numOutChanGrains,
                       outputSize});
        popsolver::Variable inputSize;
        if (level != 0 && transformedDims[level - 1].count(dim)) {
          inputSize = outputSize;
        } else {
          inputSize = m.call({outputSize,
                              transformedConvSize.back().kernelSize[dim]},
                             [=](const std::vector<unsigned> &values) {
            return getMaxInputRangeSize(values[0], dim, params, values[1]);
          });
        }
        transformedConvSize.back().numInChanGrains =
            m.product({transformedConvSize.back().numInChanGrains, inputSize});
        transformedConvSize.back().numFieldGrains[dim] = m.addConstant(1);
      }
      if (!transforms[level].flattenDims.empty()) {
        std::vector<Variable> vars;
        unsigned multiplier = 1;
        for (const auto dim : transforms[level].flattenDims) {
          if (dim == 0) {
            vars.push_back(transformedConvSize.back().batchSize);
            transformedConvSize.back().batchSize = m.addConstant(1);
          } else {
            vars.push_back(transformedConvSize.back().numFieldGrains[dim - 1]);
            multiplier *= fieldGrainSize[dim - 1];
            transformedConvSize.back().numFieldGrains[dim - 1] =
                m.addConstant(1);
          }
        }
        const auto toDim = transforms[level].flattenDims.back();
        if (toDim != 0) {
          multiplier /= fieldGrainSize[toDim - 1];
        }
        if (multiplier != 1)
          vars.push_back(m.addConstant(multiplier));
        if (toDim == 0) {
          transformedConvSize.back().batchSize = m.product(vars);
        } else {
          transformedConvSize.back().numFieldGrains[toDim - 1] =
              m.product(vars);
        }
      }
      if (outChanGrainSize[level] > outChanGrainSize[level - 1]) {
        assert(outChanGrainSize[level] % outChanGrainSize[level - 1] == 0);
        const auto divisor = outChanGrainSize[level] /
                             outChanGrainSize[level - 1];
        transformedConvSize.back().numOutChanGrains =
            m.ceildiv(transformedConvSize.back().numOutChanGrains,
                      m.addConstant(divisor));
      } else if (outChanGrainSize[level] < outChanGrainSize[level - 1]) {
        assert(outChanGrainSize[level - 1] % outChanGrainSize[level] == 0);
        const auto multiplier = outChanGrainSize[level - 1] /
                                outChanGrainSize[level];
        transformedConvSize.back().numOutChanGrains =
            m.product({transformedConvSize.back().numOutChanGrains,
                       m.addConstant(multiplier)});
      }
      if (inChanGrainSize[level] != inChanGrainSize[level - 1]) {
        assert(inChanGrainSize[level] % inChanGrainSize[level - 1] == 0);
        const auto divisor = inChanGrainSize[level] /
                             inChanGrainSize[level - 1];
        transformedConvSize.back().numInChanGrains =
            m.ceildiv(transformedConvSize.back().numInChanGrains,
                      m.addConstant(divisor));
      }
    }
    if (level + 1 == numLevelsOfHierarchy)
      break;
    const auto &prevConvSize = transformedConvSize.back();
    ConvSizeVariables nextConvSize;
    convSize.back().numFieldGrains.reserve(numFieldDims);
    convSize.back().kernelSize.reserve(numFieldDims);
    const auto levelMaxSplit = hierarchy[level];
    PartitionVariables p;
    p.fieldSplit.reserve(numFieldDims);
    p.kernelSplit.reserve(numFieldDims);
    // Return m.ceildiv(dividend, divisor) and constrain the divisor so it is
    // the smallest divisor that gives us that result. This reduces the size of
    // the search space without sacrificing the quality of the plan since the
    // maximum amount of work / data on any one tile stays the same.
    auto ceildivConstrainDivisor =
        [](Model &m, popsolver::Variable dividend,
           popsolver::Variable divisor) -> popsolver::Variable {
      auto isSmallestDivisorTheGivesResult =
          [](const std::vector<unsigned> &values) -> unsigned {
        auto dividend = values[0];
        auto divisor = values[1];
        // The divisor is the smallest divisor that gives this result if
        // it is 1 or if dividing by (divisor - 1) would gives a larger
        // result.
        if (divisor == 1)
          return 1;
        if ((dividend + divisor - 1) / divisor <
            (dividend + divisor - 2) / (divisor - 1))
          return 1;
        return 0;
      };
      auto isSmallest = m.call({dividend, divisor},
                               isSmallestDivisorTheGivesResult);
      // Add constraint that isSmallestDivisorTheGivesResult > 0, i.e.
      // it returns true.
      m.less(0, isSmallest);
      return m.ceildiv(dividend, divisor);
    };
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
        ceildivConstrainDivisor(m, prevConvSize.numFieldGrains[dim],
                                p.fieldSplit.back())
      );
      nextConvSize.kernelSize.push_back(
        ceildivConstrainDivisor(m, prevConvSize.kernelSize[dim],
                                p.kernelSplit.back())
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
    p.outChanGrainSize = outChanGrainSize[level];
    p.inChanGrainSize = inChanGrainSize[level];
    p.fieldGrainSize = fieldGrainSize;
    nextConvSize.batchSize =
        ceildivConstrainDivisor(m, prevConvSize.batchSize, p.batchSplit);
    nextConvSize.numConvGroups =
        ceildivConstrainDivisor(m, prevConvSize.numConvGroups,
                                p.convGroupSplit);
    nextConvSize.numOutChanGrains =
        ceildivConstrainDivisor(m, prevConvSize.numOutChanGrains,
                                p.outChanSplit);
    nextConvSize.numInChanGrains =
        ceildivConstrainDivisor(m, prevConvSize.numInChanGrains, p.inChanSplit);
    partitionVars.push_back(std::move(p));
    convSize.push_back(std::move(nextConvSize));
  }

  std::vector<Variable> perLevelSplits;
  for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
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

  cycles =
      addCycleEstimate(m, partitionVars, convSize, transformedConvSize,
                       usedTiles, transformedDims,
                       target, perLevelExchangeBytesPerCycle, params,
                       inChansPerGroup, partialChansPerGroup,
                       types, convVertexType.method,
                       Plan::LinearizeTileOrder::STANDARD, cache);
  if (options.pass == Pass::FC_TRAINING_FWD) {
    auto bwdParams = params;
    std::swap(bwdParams.inputFieldShape.back(), bwdParams.inputChannels);
    std::vector<PartitionVariables> bwdPartitionVars;
    std::vector<ConvSizeVariables> bwdConvSize;
    std::vector<ConvSizeVariables> bwdTransformedConvSize;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      if (level + 1 < numLevelsOfHierarchy) {
        const auto &p = partitionVars[level];
        auto bwdP = p;
        bwdP.fieldSplit.back() = p.inChanSplit;
        bwdP.inChanSplit = p.fieldSplit.back();
        bwdP.inChanGrainSize = p.fieldGrainSize.back();
        bwdP.fieldGrainSize.back() = inChansPerGroup;
        bwdPartitionVars.push_back(bwdP);
      }

      const auto &s = convSize[level];
      auto bwdS = s;
      bwdS.numFieldGrains.back() = s.numInChanGrains;
      bwdS.numInChanGrains = s.numFieldGrains.back();
      bwdConvSize.push_back(bwdS);

      const auto &tS = convSize[level];
      auto bwdTS = tS;
      bwdTS.numFieldGrains.back() = tS.numInChanGrains;
      bwdTS.numInChanGrains = tS.numFieldGrains.back();
      bwdTransformedConvSize.push_back(bwdTS);
    }
    const auto bwdInChansPerGroup = bwdPartitionVars.back().inChanGrainSize;
    const auto bwdMethod =
        getFullyConnectedBwdMethod(convVertexType.method);
    const auto bwdCycles =
        addCycleEstimate(m, bwdPartitionVars, bwdConvSize,
                         bwdTransformedConvSize,
                         usedTiles,
                         std::vector<
                           std::unordered_set<unsigned>
                         >(numLevelsOfHierarchy), target,
                         perLevelExchangeBytesPerCycle,
                         params, bwdInChansPerGroup, partialChansPerGroup,
                         types, bwdMethod,
                         Plan::LinearizeTileOrder::FC_BWD_AS_CONV, cache);
    auto wuParams = params;
    std::swap(wuParams.inputChannels, wuParams.outputChannels);
    std::vector<PartitionVariables> wuPartitionVars;
    std::vector<ConvSizeVariables> wuConvSize;
    std::vector<ConvSizeVariables> wuTransformedConvSize;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      if (level + 1 < numLevelsOfHierarchy) {
        const auto &p = partitionVars[level];
        auto wuP = p;
        wuP.outChanSplit = p.inChanSplit;
        wuP.inChanSplit = p.outChanSplit;
        wuP.inChanGrainSize = p.outChanGrainSize;
        wuP.outChanGrainSize = p.inChanGrainSize;
        wuP.fieldGrainSize = std::vector<unsigned>(numFieldDims, 1);
        wuPartitionVars.push_back(wuP);
      }

      const auto &s = convSize[level];
      auto wuS = s;
      wuS.numInChanGrains = s.numOutChanGrains;
      wuS.numOutChanGrains = s.numInChanGrains;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto fieldGrainSize =
            level > 0 ? partitionVars[level - 1].fieldGrainSize[dim] :
                        partitionVars[level].fieldGrainSize[dim];
        if (fieldGrainSize != 1) {
          wuS.numFieldGrains[dim] =
              m.product({s.numFieldGrains[dim], m.addConstant(fieldGrainSize)});
        }
      }
      wuConvSize.push_back(wuS);

      const auto &tS = convSize[level];
      auto wuTS = tS;
      wuTS.numInChanGrains = tS.numOutChanGrains;
      wuTS.numOutChanGrains = tS.numInChanGrains;
      for (unsigned dim = 0; dim != numFieldDims; ++dim) {
        const auto fieldGrainSize =
            level + 1 < numLevelsOfHierarchy ?
              partitionVars[level].fieldGrainSize[dim] :
              partitionVars[level - 1].fieldGrainSize[dim];
        if (fieldGrainSize != 1) {
          wuTS.numFieldGrains[dim] =
              m.product({tS.numFieldGrains[dim],
                         m.addConstant(fieldGrainSize)});
        }
      }
      wuTransformedConvSize.push_back(wuTS);
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
                         wuTransformedConvSize,
                         usedTiles,
                         std::vector<
                           std::unordered_set<unsigned>
                         >(numLevelsOfHierarchy), target,
                         perLevelExchangeBytesPerCycle,
                         wuParams, wuInChansPerGroup, wuPartialChansPerGroup,
                         types, wuMethod,
                         Plan::LinearizeTileOrder::FC_WU,
                         cache);
    cycles = m.sum({cycles, bwdCycles, wuCycles});
  }
  auto transformCycles =
      addTransformCycleEstimate(m, params_, params, transforms, partitionVars,
                                transformedConvSize, transformedDims,
                                inChansPerGroup, partialChansPerGroup,
                                types, options, target);
  cycles = m.sum({cycles, transformCycles});
  auto cycleBound = bestCost.cycles;
  if (costBounds.cycles > 0) {
    cycleBound = std::min(cycleBound, costBounds.cycles);
  }
  if (cycleBound < std::numeric_limits<unsigned>::max()) {
    m.lessOrEqual(cycles, cycleBound);
  }
}

static std::pair<Plan, Cost>
choosePlan(const poplar::Target &target,
           const std::vector<ConvTransform> &transforms,
           const std::vector<ConvTypes> &types,
           const std::vector<unsigned> &hierarchy,
           const std::vector<double> &perLevelExchangeBytesPerCycle,
           const std::vector<unsigned> &fieldGrainSize,
           const ConvVertexType &convVertexType,
           const ConvParams &params,
           Cost bestCost,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  popsolver::Variable cycles;
  constructModel(target, transforms, types, hierarchy,
                 perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType,
                 params, bestCost, costBounds, cache, options, m, partitionVars,
                 cycles);
  popsolver::Solution s;

  assert(costBounds.primaryCheckIsCycles);
  s = m.minimize(cycles);
  if (!s.validSolution()) {
    return {Plan(), highestCost};
  }
  std::vector<Partition> partitions;
  for (const auto &p : partitionVars) {
    partitions.push_back(makePartition(s, p));
  }
  Plan plan(std::move(partitions),
            std::move(types),
            convVertexType.inChansPerGroup,
            convVertexType.partialChansPerGroup,
            convVertexType.method,
            Plan::LinearizeTileOrder::STANDARD);
  plan.transforms = transforms;
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
                            poplar::Type inputType,
                            poplar::Type partialType,
                            const ConvParams &params,
                            const ConvOptions &options) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (canUseOuterProductMethod(params)) {
    const auto partialChansPerGroup = target.getVectorWidth(inputType);
    convVertexTypeCandidates.emplace_back(Plan::Method::OUTER_PRODUCT,
                                          inputType, inputType, 1,
                                          partialChansPerGroup);
  }
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);
  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
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
  auto ampPartialType = ampFloatPartials ? poplar::FLOAT : poplar::HALF;
  const bool isFullyConnectedFwd =
      options.pass == Pass::FC_TRAINING_FWD;
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   target)) {
    const auto weightsPerConvUnit =
        target.getWeightsPerConvUnit(floatActivations);
    for (unsigned inChansPerGroup = 1; inChansPerGroup <= weightsPerConvUnit;
         ++inChansPerGroup) {
      bool isFullyConnected = options.pass == Pass::FC_INFERENCE_FWD ||
                              options.pass == Pass::FC_TRAINING_BWD ||
                              options.pass == Pass::FC_TRAINING_FWD ||
                              options.pass == Pass::FC_TRAINING_WU;
      for (unsigned partialChansPerGroup : {numConvUnits, weightsPerConvUnit}) {
        if (!floatActivations && inChansPerGroup % 2 != 0)
          continue;
        if (isFullyConnected && partialChansPerGroup != numConvUnits)
          continue;
        if (!canUseConvolutionInstruction(floatActivations,
                                          floatPartials,
                                          inChansPerGroup, target))
          continue;
        if (isFullyConnectedFwd) {
          // The input channels in the forward pass become the output channels
          // of the weight update pass. Make sure it is a multiple of the
          // supported output channels per group.
          if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
            continue;
        }
        convVertexTypeCandidates.emplace_back(Plan::Method::AMP, inputType,
                                              ampPartialType, inChansPerGroup,
                                              partialChansPerGroup);
      }
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
    convVertexTypeCandidates.emplace_back(Plan::Method::MAC, inputType,
                                          partialType, inChansPerGroup, 1);
    previousInChanGroups = inChanGroups;
  }
  return convVertexTypeCandidates;
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
  for (unsigned i = 0; i < (1u << numItems); ++i) {
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
  if (params.outputChannels)
    poplin::swapOperands(swappedParams);
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

static std::vector<bool> getSwapOperandCandidates(const ConvParams &params,
                                                  const ConvOptions &options) {
  // Avoid swapping operands when output channels could be swapped with batch
  // size
  if (!params.outputChannels) {
    return {false};
  }

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

static std::vector<ConvTypes> getConvTypes(const poplar::Target &target,
                                           unsigned numLevels,
                                           poplar::Type resultType,
                                           const ConvOptions &options) {
  std::vector<ConvTypes> types(numLevels);
  for (int level = numLevels - 1; level >= 0; --level) {
    types[level].partialType = options.partialsType;
    if (level == 0) {
      types[level].resultType = resultType;
    } else {
      bool isTileLevel = static_cast<unsigned>(level) == numLevels - 1;
      auto levelResultType = isTileLevel ?
                               options.interTilePartialsType :
                               options.interIpuPartialsType;
      // Use the result type of the previous level if it is smaller than the
      // requested result type. This means that if a user wants to use half
      // partials they only need to set the option for the first level that
      // should use half partials.
      if (!isTileLevel &&
          target.getTypeSize(levelResultType) >
          target.getTypeSize(types[level + 1].resultType)) {
        levelResultType = types[level + 1].resultType;
      }
      // There is no point in using a result type larger than the partial type.
      if (target.getTypeSize(levelResultType) >
          target.getTypeSize(types[level].partialType)) {
        levelResultType = types[level].partialType;
      }
      types[level].resultType = levelResultType;
    }
  }
  return types;
}

static std::vector<unsigned>
getDilatePostConvDims(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> dilateAfterConv;
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (params.inputTransform.dilation[dim] != 1 &&
        canDeferDilation(params, dim)) {
      dilateAfterConv.push_back(dim);
    }
  }
  std::reverse(dilateAfterConv.begin(), dilateAfterConv.end());
  return dilateAfterConv;
}

static std::pair<Plan, Cost>
createPlan(ConvParams params,
           const ConvOptions &options,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlanningCacheImpl *cache) {
  validateLayerParams(params, options);
  params = canonicalizeParams(params);
  const auto &target = graph.getTarget();
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy = getTileHierarchy(target,
                                          perLevelExchangeBytesPerCycle);
  const auto numLevels = hierarchy.size() + 1;
  Cost bestCost = highestCost;
  Plan bestPlan;
  std::vector<ConvTransform> transforms(numLevels);
  auto convTypes = getConvTypes(graph.getTarget(), numLevels, params.dType,
                                options);
  const auto ipuLevel = transforms.size() - 2;
  unsigned addedFieldDims = 0;
  auto numFieldDims = params.getNumFieldDims();
  auto paramsWithExtraDims = params;
  if (numFieldDims < 2) {
    // Various places assume there are at least two dimensions. In particular
    // code related to the nx1ConvPartial vertex has special handling for the
    // outermost dimension and special handling for the innermost dimension
    // and there is an assumption that these two dimensions are distinct.
    addedFieldDims = 2 - numFieldDims;
    addExtraDims(paramsWithExtraDims, addedFieldDims);
    numFieldDims = 2;
  }
  transforms[0].extraFieldDims = addedFieldDims;
  transforms[0].dilatePostConv = getDilatePostConvDims(paramsWithExtraDims);
  auto paramsWithDeferredDilation =
      calculateParamsWithDeferredDilation(paramsWithExtraDims,
                                          transforms[0].dilatePostConv);
  for (bool swapOperands : getSwapOperandCandidates(paramsWithDeferredDilation,
                                                    options)) {
    transforms[0].swapOperands = swapOperands;
    auto swappedParams = calculateSwappedParams(paramsWithDeferredDilation,
                                                swapOperands);
    for (const std::vector<unsigned> &expandDims :
         getExpandDimsCandidates(swappedParams)) {
      transforms[ipuLevel].expandDims = expandDims;
      auto expandedParams = calculateExpandedParams(swappedParams, expandDims);
      for (const std::vector<unsigned> &outChanFlattenDims :
           getOutChanFlattenDimsCandidates(expandedParams)) {
        transforms[ipuLevel].outChanFlattenDims = outChanFlattenDims;
        auto flattenedParams =
            calculateFlattenedParams(expandedParams, outChanFlattenDims,
                                     transforms[ipuLevel].flattenDims);
        const auto convVertexTypeCandidates =
            getConvVertexTypeCandidates(target, params.dType,
                                        convTypes.back().partialType,
                                        flattenedParams, options);
        for (const auto &convVertexType : convVertexTypeCandidates) {
          std::vector<unsigned> fieldGrainSize(numFieldDims, 1);
          if (options.pass == Pass::FC_TRAINING_FWD) {
            // The innermost grain size becomes the inChansPerGroup in the
            // backward pass. For now assume the same grouping in both passes.
            // TODO search for the optimal grouping in each pass.
            fieldGrainSize.back() = convVertexType.inChansPerGroup;
          }
          Plan candidate;
          Cost candidateCost;
          convTypes.back().partialType = convVertexType.partialType;
          std::tie(candidate, candidateCost) =
              choosePlan(target, transforms, convTypes, hierarchy,
                         perLevelExchangeBytesPerCycle, fieldGrainSize,
                         convVertexType, params, bestCost, costBounds, cache,
                         options);
          if (candidateCost == highestCost)
            continue;
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

  // In FC backward pass, input channels becomes input shape. Avoid this if
  // number of input channels are zero
  if (options.pass == Pass::FC_TRAINING_BWD &&
      params.getNumInputChansPerConvGroup() == 0)
    return params;

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
  assert(!fwdPlan.transforms[0].swapOperands);
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
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;

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

  // Set the partials type to the output type as there are no reductions
  // required
  if (fwdParams.dType == poplar::HALF &&
      plan.method == Plan::Method::OUTER_PRODUCT) {
    for (auto &x : plan.types) {
      x.partialType = x.resultType = poplar::HALF;
    }
  }
  return plan;
}

static Plan getFullyConnectedBwdPlan(const ConvParams &fwdParams,
                                     const Plan &fwdPlan) {
  assert(!fwdPlan.transforms[0].swapOperands);
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  for (auto &partition : plan.partitions) {
    std::swap(partition.fieldSplit.back(), partition.inChanSplit);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
  return plan;
}

Plan getPlan(const poplar::Graph &graph, const ConvParams &params,
             const ConvOptions &options, PlanningCache *cache) {
  const auto &target = graph.getTarget();
  if (options.pass == Pass::FC_TRAINING_WU ||
      options.pass == Pass::FC_TRAINING_BWD) {
    auto fwdParams = getFullyConnectedFwdParams(params, options);
    auto fwdOptions = getFullyConnectedFwdOptions(options);
    const auto fwdPlan =
        getPlan(graph, fwdParams, fwdOptions, cache);
    if (options.pass == Pass::FC_TRAINING_WU)
      return getFullyConnectedWUPlan(target, fwdParams, fwdOptions,
                                     fwdPlan);
    assert(options.pass == Pass::FC_TRAINING_BWD);
    return getFullyConnectedBwdPlan(fwdParams, fwdPlan);
  }
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cacheImpl) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  PlanningCacheImpl::Key key(params, options, false, 0, 0);
  if (!tempCache.get()) {
    auto &plans = cacheImpl->plans;
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

  std::tie(plan, cost) = poplin::createPlan(params,
                                             options,
                                             costBounds, graph,
                                             cacheImpl);
  if (options.percentageCyclesExcessForMemOptim) {
    throw poputil::poplib_error("Optimizing for memory is not supported");
  }
  if (!tempCache.get()) {
    auto &plans = cacheImpl->plans;
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

/// Estimate the cost of a convololution. This is not used by poplibs/enigma.
std::uint64_t estimateConvCost(const poplar::Target &target,
                               const ConvParams &params,
                               const ConvOptions &options,
                               PlanningCache *cache,
                               const Plan &plan) {
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy =
      getTileHierarchy(target, perLevelExchangeBytesPerCycle);
  assert(perLevelExchangeBytesPerCycle.size() ==
         plan.partitions.size());
  CostBounds costBounds(0, 0);
  ConvVertexType convVertexType(plan.method, params.dType,
                                plan.types.back().partialType,
                                plan.inChansPerGroup,
                                plan.partialChansPerGroup);
  const auto fieldGrainSize = plan.partitions.back().fieldAxisGrainSize;
  // Check grain size is the same at each level.
#ifndef NDEBUG
  for (const auto &p : plan.partitions) {
    assert(p.fieldAxisGrainSize == fieldGrainSize);
  }
#endif
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  popsolver::Variable cycles;
  constructModel(target, plan.transforms, plan.types, hierarchy,
                 perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType,
                 params, highestCost, costBounds, cacheImpl,
                 options, m, partitionVars, cycles);
  const auto numLevelsOfHierarchy = plan.partitions.size();
  assert(partitionVars.size() == numLevelsOfHierarchy);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    constrainPartitionVars(m, partitionVars[level], plan.partitions[level]);
  }
  popsolver::Solution s;
  s = m.minimize(cycles);
  if (!s.validSolution()) {
    return highestCost.cycles;
  }
  return s[cycles];
}

} // end namespace conv
