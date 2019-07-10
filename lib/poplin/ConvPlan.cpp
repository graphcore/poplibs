#include "poplin/internal/ConvPlan.hpp"
#include "ConvReducePlan.hpp"
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
#include "poplibs_support/logging.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/VectorUtils.hpp"
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
#include "tbb/parallel_for.h"
#include "tbb/concurrent_unordered_map.h"
#include "CanonicalConvParams.hpp"

using namespace poplibs_support;

namespace hash_tuple{
  template <typename TT>
  struct hash {
    size_t operator()(TT const& tt) const {
      return std::hash<TT>()(tt);
    }
  };

  template <class T>
  inline void hash_combine(std::size_t& seed, T const& v) {
    seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
  }

  template <typename TT>
  struct hash<std::vector<TT>> {
    size_t operator()(const std::vector<TT> &tt) const {
      size_t hash = 0;
      for (const auto e:tt)
        hash_combine(hash, e);
      return hash;
    }
  };

  namespace details {
    template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
    struct HashValueImpl {
      void operator()(size_t& seed, Tuple const& tuple)const{
        HashValueImpl<Tuple, Index-1>{}(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
      }
    };
    template <class Tuple>
    struct HashValueImpl<Tuple,0> {
      void operator()(size_t& seed, Tuple const& tuple)const{
        hash_combine(seed, std::get<0>(tuple));
      }
    };
  }

  template <typename ... TT>
  struct hash<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
      size_t seed = 0;
      details::HashValueImpl<std::tuple<TT...> >{}(seed, tt);
      return seed;
    }
  };
}

namespace poplin {

namespace {
  struct PartitionVariables {
    std::vector<popsolver::Variable> fieldSplit;
    Split<popsolver::Variable> batchSplit;
    Split<popsolver::Variable> outChanSplit;
    std::vector<popsolver::Variable> kernelSplit;
    Split<popsolver::Variable> inChanSplit;
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
    unsigned fieldMACs = 0;
    auto kernelSize = params.kernelShape[dim];
    auto kernelTruncationLower = params.kernelTransform.truncationLower[dim];
    auto kernelTruncationUpper = params.kernelTransform.truncationUpper[dim];
    auto outputSize = params.getOutputSize(dim);
    auto outputStride = params.outputTransform.stride[dim];
    auto inputDilation = params.inputTransform.dilation[dim];
    // For a fixed kernel index the distance between elements in the output
    // whose calculation involves that kernel index.
    auto MACStride = lcm(outputStride, inputDilation) / outputStride;
    for (unsigned k = kernelTruncationLower;
         k != kernelSize - kernelTruncationUpper; ++k) {
      auto outRange = getOutputRangeForKernelIndex(dim, {0, outputSize}, k,
                                                   params);
      auto outRangeSize = outRange.second - outRange.first;
      fieldMACs += (outRangeSize + MACStride - 1) / MACStride;
    }
    numMACs *= fieldMACs;
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
    throw poputil::poplibs_error(
      "Unknown weight update method <" + token + ">");
  return is;

}

// A simple function to memoize other functions. Any recursive calls
// with the function are non memoized
template <typename Ret, typename... Args>
class Memo {
  using Key = std::tuple<typename std::remove_reference<Args>::type...>;
public:
  tbb::concurrent_unordered_map<Key,
           Ret, hash_tuple::hash<Key>> table;
  Ret (*fn)(Args...);
 public:
  Memo(Ret (*fn)(Args...)) : fn(fn) {}
  Ret operator()(Args... args) {
    const auto key = std::make_tuple(args...);
    const auto match = table.find(key);
    if(match == table.end()) {
      auto result = fn(args...);
      auto insertRes = table.insert({key, result});
      // another thread may have updated with the same key - in which case
      // it should be with the same value
      if (insertRes.second == false)
        assert(insertRes.first->second == result);
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
  poplar::Type inputType;
  poplar::Type partialType;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;
  ConvVertexType(Plan::Method method,
                 poplar::Type inputType,
                 poplar::Type outputType,
                 poplar::Type partialType,
                 unsigned inChansPerGroup,
                 unsigned partialChansPerGroup) :
    method(method),
    inputType(inputType),
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
     << "             batchSplit.serial     " << p.batchSplit.serial << "\n"
     << "             batchSplit.parallel   " << p.batchSplit.parallel << "\n"
     << "             outChanSplit.serial   " << p.outChanSplit.serial << "\n"
     << "             outChanSplit.parallel " << p.outChanSplit.parallel << "\n"
     << "             kernelSplit           ";
  printContainer(p.kernelSplit, os);
  os << "\n"
     << "             inChanSplit.serial    " << p.inChanSplit.serial << "\n"
     << "             inChanSplit.parallel  " << p.inChanSplit.parallel << "\n"
     << "             convGroupSplit        " << p.convGroupSplit << "\n"
     << "             fieldAxisGrainSize    ";
  printContainer(p.fieldAxisGrainSize, os);
  os << "\n"
     << "             inChanGrainSize       " << p.inChanGrainSize << "\n"
     << "             outChanGrainSize      " << p.outChanGrainSize << "\n";
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

std::ostream& operator<<(std::ostream &os, const Plan::Method &m) {
  os << asString(m);
  return os;
}

std::istream& operator>>(std::istream &is, Plan::Method &m) {
  std::string token;
  is >> token;
  if (token == "MAC") {
    m = Plan::Method::MAC;
  } else if (token == "AMP") {
    m = Plan::Method::AMP;
  } else if (token == "OUTER_PRODUCT") {
    m = Plan::Method::OUTER_PRODUCT;
  } else {
    throw poputil::poplibs_error("Unrecognised convolution method '" +
                                 token + "'");
  }
  return is;
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
  os << "        outChanSerialSplit      " << p.outChanSerialSplit << "\n"
     << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n"
     << "        isJointPlan             " << p.isJointPlan << "\n";
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
    bool floatPartials,
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
             numWorkerContexts, floatWeights, floatPartials);
  return cycles;
}


static std::uint64_t
getConvPartial1x1InnerLoopCycleEstimate(
    unsigned batchElements,
    const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride,
    bool floatActivations,
    bool floatPartials,
    bool zeroPartials) {
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
                                                        zeroPartials,
                                                        floatActivations,
                                                        floatPartials);
  return cycles;
}

static std::uint64_t
getConvPartial1x1InnerLoopCycleEstimateWithZeroing(
    unsigned batchElements,
    const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride,
    bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(batchElements,
                                                 outShape,
                                                 numWorkerContexts,
                                                 inputDilation,
                                                 stride,
                                                 floatActivations,
                                                 floatPartials,
                                                 true);
}

static std::uint64_t
getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
    unsigned batchElements,
    const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride,
    bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(batchElements,
                                                 outShape,
                                                 numWorkerContexts,
                                                 inputDilation,
                                                 stride,
                                                 floatActivations,
                                                 floatPartials,
                                                 false);
}

static std::uint64_t estimateCastCycles(unsigned outputSize,
                                        unsigned partialsVectorWidth,
                                        unsigned outputVectorWidth,
                                        unsigned numWorkers) {
  const auto outputPerWorker = (outputSize + numWorkers - 1) / numWorkers;
  std::uint64_t loadPartialsCycles =
      (outputPerWorker + partialsVectorWidth - 1) / partialsVectorWidth;
  std::uint64_t writeOutputCycles =
      (outputPerWorker + outputVectorWidth - 1) / outputVectorWidth;
  std::uint64_t cycles = std::max(loadPartialsCycles, writeOutputCycles);
  return (cycles + 26) * numWorkers;
}

static std::uint64_t
estimateConvReduceCycles(unsigned outputSize,
                         unsigned reductionDepth,
                         bool floatOutput,
                         bool floatPartials,
                         unsigned numWorkers,
                         unsigned dataPathWidth,
                         unsigned partialsVectorWidth,
                         unsigned outputVectorWidth) {
  if (reductionDepth == 0)
    return 0;

  if (reductionDepth == 1) {
    if (floatOutput == floatPartials)
      return 0;
    else
      return estimateCastCycles(outputSize, partialsVectorWidth,
                                outputVectorWidth, numWorkers);
  }

  // Determine number of stages used in the reduction
  auto reductionPlan = getMultiStageReducePlan(reductionDepth);
  std::uint64_t cycles = 0;

  unsigned remainingDepth = reductionDepth;
  // Output size depends on the depth used in the reduction
  unsigned outputSizeThisStage = outputSize * reductionDepth;
  for (auto d : reductionPlan) {
    const auto depthThisStage = (remainingDepth + d - 1) / d;
    outputSizeThisStage =
        (outputSizeThisStage + depthThisStage - 1) / depthThisStage;
    cycles += getReduceCycleEstimate(outputSizeThisStage,
                                     depthThisStage,
                                     dataPathWidth,
                                     floatOutput,
                                     floatPartials,
                                     numWorkers);
    remainingDepth = (remainingDepth + depthThisStage - 1) / depthThisStage;
  }

  if (remainingDepth > 1) {
      outputSizeThisStage =
          (outputSizeThisStage + remainingDepth - 1) / remainingDepth;
      cycles += getReduceCycleEstimate(outputSizeThisStage,
                                       remainingDepth,
                                       dataPathWidth,
                                       floatOutput,
                                       floatPartials,
                                       numWorkers);
  }
  return cycles;
}

static std::uint64_t
estimateZeroSupervisorCycles(unsigned fieldSize,
                             unsigned numOutGroups,
                             unsigned numConvGroups,
                             unsigned outChansPerGroup,
                             unsigned dataPathWidth,
                             unsigned numWorkerContexts) {
  std::vector<unsigned> zeroWorkList;
  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    zeroWorkList.push_back((fieldSize * outChansPerGroup +
                            numWorkerContexts - 1) / numWorkerContexts);
  }
  return
    getZeroSupervisorVertexCycleEstimate(zeroWorkList,
                                         numOutGroups * numConvGroups,
                                         dataPathWidth,
                                         numWorkerContexts,
                                         true);
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
  struct Key {
    CanonicalConvParams convParams;
    ConvOptions options;
    Key(CanonicalConvParams params, ConvOptions options) :
      convParams(std::move(params)), options(std::move(options)) {}
    bool operator<(const Key &other) const {
      return std::tie(convParams, options) <
             std::tie(other.convParams, other.options);
    }
  };
  class CycleEstimationImpl {
  public:
    decltype(memoize(getConvPartial1x1InnerLoopCycleEstimateWithZeroing))
      mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing;
    decltype(memoize(getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing))
      mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing;
    decltype(memoize(getConvPartialnx1InnerLoopCycleEstimate))
      mGetConvPartialnx1InnerLoopCycleEstimate;
    decltype(memoize(estimateConvPartialHorizontalMacInnerLoopCycles))
      mEstimateConvPartialHorizontalMacInnerLoopCycles;
    decltype(memoize(estimateConvReduceCycles))
      mEstimateConvReduceCycles;
    decltype(memoize(getNumberOfMACs))
      mGetNumberOfMACs;
    CycleEstimationImpl() :
      mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing(
        memoize(getConvPartial1x1InnerLoopCycleEstimateWithZeroing)
      ),
      mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
        memoize(getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing)
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

  };
  // The plan's cycleEstimation can be used and updated in parallel.
  CycleEstimationImpl cycleEstimation;
  // Updates to plans must be single-threaded.
  std::map<Key, std::unique_ptr<Plan>> plans;
};

PlanningCache::PlanningCache() {
  impl = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl());
}

PlanningCache::~PlanningCache() = default;

struct Cost {
  unsigned cycles;
  // Maximum amount of temporary memory used on a tile in bytes.
  unsigned tileTempMemory;

  Cost(unsigned cycles, unsigned tileTempMemory) :
    cycles(cycles),
    tileTempMemory(tileTempMemory) {}
  Cost() {}
};

inline bool operator==(Cost a, Cost b) {
  return a.cycles == b.cycles && a.tileTempMemory == b.tileTempMemory;
}

inline bool operator!=(Cost a, Cost b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const Cost &c) {
  os << "Cost{cycles=" << c.cycles << ", memory=" << c.tileTempMemory << "}";
  return os;
}

class PlanningObjective {
public:
  enum Type {
    MINIMIZE_CYCLES,
    MINIMIZE_TILE_TEMP_MEMORY
  };
private:
  Type type;
  unsigned cyclesBound = 0;
  unsigned tileTempMemoryBound = 0;
  PlanningObjective(Type type) : type(type) {}
public:
  PlanningObjective() {}
  static PlanningObjective minimizeCycles() {
    return PlanningObjective(MINIMIZE_CYCLES);
  }
  static PlanningObjective minimizeTileTempMemory() {
    return PlanningObjective(MINIMIZE_TILE_TEMP_MEMORY);
  }
  PlanningObjective &setCyclesBound(unsigned bound) {
    assert(type != MINIMIZE_CYCLES);
    assert(bound > 0);
    cyclesBound = bound;
    return *this;
  }
  PlanningObjective &setTileTempMemoryBound(unsigned bound) {
    assert(type != MINIMIZE_TILE_TEMP_MEMORY);
    assert(bound > 0);
    tileTempMemoryBound = bound;
    return *this;
  }
  unsigned getCyclesBound() const {
    return cyclesBound;
  }
  unsigned getTileTempMemoryBound() const {
    return tileTempMemoryBound;
  }
  Type getType() const {
    return type;
  }
  bool lowerCost(Cost a, Cost b) const {
    bool aCyclesOutOfBounds = a.cycles >= cyclesBound;
    bool bCyclesOutOfBounds = b.cycles >= cyclesBound;
    bool aMemoryOutOfBounds = a.tileTempMemory >= tileTempMemoryBound;
    bool bMemoryOutOfBounds = b.tileTempMemory >= tileTempMemoryBound;
    switch (type) {
    case MINIMIZE_CYCLES:
      return std::tie(aCyclesOutOfBounds, aMemoryOutOfBounds, a.cycles,
                      a.tileTempMemory) <
             std::tie(bCyclesOutOfBounds, bMemoryOutOfBounds, b.cycles,
                      b.tileTempMemory);
    case MINIMIZE_TILE_TEMP_MEMORY:
      return std::tie(aMemoryOutOfBounds, aCyclesOutOfBounds, a.tileTempMemory,
                      a.cycles) <
             std::tie(bMemoryOutOfBounds, bCyclesOutOfBounds, b.tileTempMemory,
                      b.cycles);
    }
    POPLIB_UNREACHABLE();
  }
};

static Cost highestCost(std::numeric_limits<unsigned>::max(),
                        std::numeric_limits<unsigned>::max());

// Pick a tile to start laying out the convolution on. We pick a "random" tile
// by hashing the convolution parameters in an attempt to evenly distribute
// across the entire tile range. If we always start from the same tile we will
// see the higher tiles getting much less data than everything else.
static unsigned getStartTile(const ConvParams &params,
                             const ConvOptions &options,
                             bool isJointPlan) {
  // Use a start tile of 0 for joint plans to avoid the risk of exchanging
  // weights. TODO: investigate whether this is necessary.
  if (isJointPlan) {
    return 0;
  }

  // Always start on an even tile because the convolutions rely on 64-bit sends.
  if (options.startTileMultiplier % 2 != 0) {
    throw poputil::poplibs_error(
      "Must start distributing convolutions on an even tile.");
  }

  // A multiplier of zero effectively disables the dithering.
  if (options.startTileMultiplier == 0) {
    return 0;
  }

  const auto numTiles = options.tilesPerIPU * options.numIPUs;
  const auto numEvenTiles =
    std::max(1U, numTiles / options.startTileMultiplier);
  return (std::hash<ConvParams>()(params) % numEvenTiles)
    * options.startTileMultiplier;
}

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

  const auto wholeInputRange =
      getInputRange(dim, {0, params.getOutputSize(dim)}, params);
  const auto wholeInputRangeSize =
      wholeInputRange.second - wholeInputRange.first;

  if (outputRangeSize == params.getOutputSize(dim) &&
      tileKernelSize == params.kernelShape[dim]) {
    return wholeInputRangeSize;
  }
  const auto stride = params.outputTransform.stride[dim];
  const auto inputDilation = params.inputTransform.dilation[dim];
  const auto preDownSampleOutputSize = (outputRangeSize - 1) * stride + 1;
  const auto dilatedInputSize = preDownSampleOutputSize + tileKernelSize - 1;
  const auto inputRangeSize = (dilatedInputSize - 1) / inputDilation + 1;

  // If inputRangeSize expands  beyond the input data range, clip the padding
  return std::min(inputRangeSize, wholeInputRangeSize);
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
    numWorkers,
    floatActivations);
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
    poplar::Type partialType,
    Plan::Method method,
    const ConvOptions &options,
    PlanningCacheImpl::CycleEstimationImpl *cache) {
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(params.inputType == poplar::HALF || params.inputType == poplar::FLOAT);
  bool floatActivations = params.inputType == poplar::FLOAT;
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
      auto convUnitInputLoadElemsPerCycle =
         target.getConvUnitInputLoadElemsPerCycle(floatActivations);
      if (!options.use128BitConvUnitLoad)
        convUnitInputLoadElemsPerCycle /= 2;

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
        const auto floatPartials = partialType == poplar::FLOAT;

        if (canUseConvPartial1x1Vertex(params, transformedDims,
                                       transformedInputDilation,
                                       transformedOutputStride,
                                       convUnitWeightHeight,
                                       convSize.kernelSize)) {
          auto innerLoopCyclesWithZeroing =
              cache->mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing(
                convSize.batchSize,
                tileFieldSize,
                target.getNumWorkerContexts(),
                transformedInputDilation,
                transformedOutputStride,
                floatActivations,
                floatPartials);
          auto innerLoopCyclesWithoutZeroing =
              cache->mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
                convSize.batchSize,
                tileFieldSize,
                target.getNumWorkerContexts(),
                transformedInputDilation,
                transformedOutputStride,
                floatActivations,
                floatPartials);

          return
              getConvPartial1x1SupervisorOuterLoopCycleEstimate(
                innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
                convSize.numConvGroups, tileNumInGroups,
                tileNumOutGroups, outChansPerGroup,
                convUnitInputLoadElemsPerCycle, numConvUnits,
                target.getConvUnitCoeffLoadBytesPerCycle(),
                floatActivations,
                floatPartials);
        }
        auto zeroCycles =
            estimateZeroSupervisorCycles(product(tileFieldSize) *
                                           convSize.batchSize,
                                         tileNumOutGroups,
                                         convSize.numConvGroups,
                                         outChansPerGroup,
                                         target.getDataPathWidth(),
                                         target.getNumWorkerContexts());

        auto innerLoopCycles =
            cache->mGetConvPartialnx1InnerLoopCycleEstimate(
              convSize.batchSize, tileFieldSize, convSize.kernelSize,
              convUnitWeightHeight, outChansPerGroup,
              convUnitInputLoadElemsPerCycle,
              numConvUnits, target.getConvUnitCoeffLoadBytesPerCycle(),
              target.getNumWorkerContexts(), floatActivations, floatPartials,
              transformedInputDilation, transformedOutputStride);
        return
            getConvPartialnx1SupervisorCycleOuterLoopEstimate(
              innerLoopCycles, convSize.numConvGroups,
              tileNumOutGroups, tileNumInGroups, outChansPerGroup,
              numConvUnits) +
            zeroCycles;
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
        const auto zeroCycles =
            estimateZeroSupervisorCycles(numActiveOutRows * tileOutWidth,
                                         tileNumOutChans,
                                         convSize.numConvGroups,
                                         outChansPerGroup,
                                         target.getDataPathWidth(),
                                         target.getNumWorkerContexts());
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
              tileNumOutGroups,
              floatActivations) +
            zeroCycles;
      });
    }
    break;
  case Plan::Method::OUTER_PRODUCT:
    {
      assert(inChansPerGroup == 1);
      assert(std::all_of(transformedOutputStride.begin(),
                         transformedOutputStride.end(), equalsOne));
      assert(std::all_of(transformedInputDilation.begin(),
                         transformedInputDilation.end(), equalsOne));
      const auto numContexts = target.getNumWorkerContexts();
      return m.call(convSizeVarsVector,
          [=](const std::vector<unsigned> &values) {
        auto convSize = makeConvSize(values, numFieldDims);
        assert(convSize.batchSize == 1);
        assert(convSize.numInChanGrains == 1);
        const auto tileOutWidth =
            convSize.numFieldGrains.back() * fieldGrainSize.back();
        const auto workerOutWidth =
            (tileOutWidth + numContexts - 1) / numContexts;
        const auto tileNumOutChans = convSize.numOutChanGrains *
                                     outChanGrainSize;
        auto vertexRuntime = getOuterProductCycleEstimate(
          floatActivations || params.outputType == poplar::FLOAT,
          workerOutWidth,
          tileNumOutChans * convSize.numConvGroups,
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
addTempMemoryEstimate(
    popsolver::Model &m,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSizes,
    const popsolver::Variable &inputsPerTile,
    const popsolver::Variable &weightsPerTile,
    const popsolver::Variable &partialsPerTile,
    const poplar::Target &target,
    const ConvParams &params,
    const std::vector<ConvTypes> &types)
{
  std::vector<popsolver::Variable> memorySumOperands;
  auto elementBytes = target.getTypeSize(params.inputType);
  auto inputStorage = m.product({m.addConstant(elementBytes), inputsPerTile});
  auto weightStorage = m.product({m.addConstant(elementBytes), weightsPerTile});
  auto partialStorage =
      m.product({m.addConstant(target.getTypeSize(types.back().partialType)),
                 partialsPerTile});
  auto convStorage = m.sum({inputStorage, weightStorage, partialStorage});
  // Rearrangements can require both pre- and post-rearranged inputs and/or
  // weights to be required. This may be bigger than the storage need during the
  // convolution.
  return convStorage;
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
    Plan::LinearizeTileOrder linearizeTileOrder,
    std::vector<popsolver::Variable> &inputsPerLevel,
    std::vector<popsolver::Variable> &weightsPerLevel) {
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
      m.addConstant(target.getTypeSize(params.inputType) *
                    exchangeBytesScalingFactor);
  std::vector<popsolver::Variable> cycleSumOperands;
  inputsPerLevel.clear();
  weightsPerLevel.clear();
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
    inputsPerLevel.push_back(numberOfInputElements);
    weightsPerLevel.push_back(numberOfWeights);

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
      // TODO: handle outChanSplit.serial
      auto multiplier =
          m.call({partitionVars[level].outChanSplit.parallel},
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
    PlanningCacheImpl::CycleEstimationImpl *cache) {
  std::vector<popsolver::Variable> cycleSumOperands;
  const auto numLevelsOfHierarchy = partitionVars.size();
  for (int level = numLevelsOfHierarchy - 1; level >= 0; --level) {
    auto reduceDimSizes = partitionVars[level].kernelSplit;
    // TODO: handle inChanSplit.serial
    reduceDimSizes.push_back(partitionVars[level].inChanSplit.parallel);
    const auto reductionDepth = m.product(reduceDimSizes);
    auto tileOutSize = m.ceildiv(partialsPerTile, reductionDepth);
    bool floatPartials = types[level + 1].resultType == poplar::FLOAT;
    bool floatOutput = types[level].resultType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    const auto numWorkers = target.getNumWorkerContexts();
    const auto partialsVectorWidth =
        target.getVectorWidth(floatPartials ? poplar::FLOAT : poplar::HALF);
    const auto outputVectorWidth =
        target.getVectorWidth(floatOutput ? poplar::FLOAT : poplar::HALF);
    const auto cycleEstimate =
        m.call({tileOutSize, reductionDepth},
               [=](const std::vector<unsigned> &vars) -> unsigned {
      return cache->mEstimateConvReduceCycles(vars[0], vars[1], floatOutput,
                                              floatPartials, numWorkers,
                                              dataPathWidth,
                                              partialsVectorWidth,
                                              outputVectorWidth);
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

static std::pair<popsolver::Variable, popsolver::Variable>
addEstimates(popsolver::Model &m,
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
             const ConvOptions &options,
             PlanningCacheImpl::CycleEstimationImpl *cache) {
  // popsolver takes into account whether a variable is an operand of a call
  // when deciding the order to set variables. Add a dummy call to ensure the
  // split variables are prioritized as this reduces the amount of time spent
  // in the planner. TODO Improve popsolver's heuristics for ordering variables
  // so this hack is no longer necessary (or provide a proper mechanism for
  // ordering hints).
  std::vector<popsolver::Variable> variables;
  for (const auto &vars : partitionVars) {
    // TODO: handle {batchSplit,outChanSplit,inChanSplit}.serial
    variables.push_back(vars.batchSplit.parallel);
    variables.push_back(vars.outChanSplit.parallel);
    variables.push_back(vars.inChanSplit.parallel);
    variables.push_back(vars.convGroupSplit);
    variables.insert(variables.end(), vars.fieldSplit.begin(),
                     vars.fieldSplit.end());
    variables.insert(variables.end(), vars.kernelSplit.begin(),
                     vars.kernelSplit.end());
  };
  (void)m.call(variables, [](const std::vector<unsigned> &) {
    return 0U;
  });
  std::vector<popsolver::Variable> inputsPerLevel, weightsPerLevel;
  const auto exchangeCycles =
      addExchangeCycleEstimate(m, partitionVars, convSize, transformedDims,
                               target, perLevelExchangeBytesPerCycle, params,
                               types, linearizeTileOrder, inputsPerLevel,
                               weightsPerLevel);
  const auto partialsPerTile = addPartialsPerTile(m, partitionVars.back(),
                                                  partialChansPerGroup,
                                                  transformedConvSize.back());
  const auto tempBytes =
      addTempMemoryEstimate(m, partitionVars, convSize,
                            inputsPerLevel.back(), weightsPerLevel.back(),
                            partialsPerTile, target, params, types);
  const auto partialCalcCycles =
      addPartialCalcCycleEstimate(m, partitionVars.back().fieldGrainSize,
                                  inChansPerGroup,
                                  partialChansPerGroup,
                                  transformedConvSize.back(),
                                  transformedDims.back(),
                                  target, params, inChansPerGroup,
                                  partialChansPerGroup,
                                  types.back().partialType, method,
                                  options, cache);
  // Add a redundant inequality that relates the cycles required to calculate
  // the partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile = getMaxMACsPerCyclePerTile(
        target, types.back().partialType, params.inputType, method);
  const auto totalMacs = cache->mGetNumberOfMACs(params);
  m.lessOrEqual(totalMacs / maxMACsPerCyclePerTile,
                m.product({usedTiles, partialCalcCycles}));
  const auto reduceCycles =
      addReduceCycleEstimate(m, partitionVars, partialsPerTile, target, types,
                             cache);
  return {m.sum({exchangeCycles, partialCalcCycles, reduceCycles}), tempBytes};
}

static Plan::Method
getFullyConnectedBwdMethod(Plan::Method fwdMethod) {
  if (fwdMethod == Plan::Method::OUTER_PRODUCT) {
    return Plan::Method::MAC;
  }
  return fwdMethod;
}

static std::pair<popsolver::Variable, popsolver::Variable>
addBwdEstimates(popsolver::Model &m,
                const ConvParams &params,
                const std::size_t numLevelsOfHierarchy,
                const std::vector<PartitionVariables> &partitionVars,
                const std::vector<ConvSizeVariables> &convSize,
                Plan::Method method,
                const popsolver::Variable usedTiles,
                const poplar::Target &target,
                const std::vector<double> &perLevelExchangeBytesPerCycle,
                const std::vector<ConvTypes> &types,
                const unsigned partialChansPerGroup,
                const unsigned inChansPerGroup,
                const ConvOptions &options,
                PlanningCacheImpl::CycleEstimationImpl *cache) {
  auto bwdParams = params;
  std::swap(bwdParams.inputFieldShape.back(), bwdParams.inputChannels);
  std::vector<PartitionVariables> bwdPartitionVars;
  std::vector<ConvSizeVariables> bwdConvSize;
  std::vector<ConvSizeVariables> bwdTransformedConvSize;
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    if (level + 1 < numLevelsOfHierarchy) {
      const auto &p = partitionVars[level];
      auto bwdP = p;
      // TODO: handle inChanSplit.serial
      bwdP.fieldSplit.back() = p.inChanSplit.parallel;
      bwdP.inChanSplit.parallel = p.fieldSplit.back();
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
  const auto bwdMethod = getFullyConnectedBwdMethod(method);

  std::vector<std::unordered_set<unsigned>>
    transformedDims(numLevelsOfHierarchy);
  return addEstimates(m, bwdPartitionVars, bwdConvSize,
                      bwdTransformedConvSize,
                      usedTiles,
                      transformedDims,
                      target,
                      perLevelExchangeBytesPerCycle,
                      params, bwdInChansPerGroup, partialChansPerGroup,
                      types, bwdMethod,
                      Plan::LinearizeTileOrder::FC_BWD_AS_CONV,
                      options, cache);
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

static std::pair<popsolver::Variable, popsolver::Variable>
addWuEstimates(popsolver::Model &m,
               const ConvParams &params,
               const std::size_t numLevelsOfHierarchy,
               const std::vector<PartitionVariables> &partitionVars,
               const std::vector<ConvSizeVariables> &convSize,
               Plan::Method method,
               const popsolver::Variable usedTiles,
               const poplar::Target &target,
               const unsigned numFieldDims,
               const std::vector<double> &perLevelExchangeBytesPerCycle,
               const std::vector<ConvTypes> &types,
               const unsigned partialChansPerGroup,
               const unsigned inChansPerGroup,
               const ConvOptions &options,
               PlanningCacheImpl::CycleEstimationImpl *cache) {
  auto wuParams = params;
  std::swap(wuParams.inputChannels, wuParams.outputChannels);
  std::vector<PartitionVariables> wuPartitionVars;
  std::vector<ConvSizeVariables> wuConvSize;
  std::vector<ConvSizeVariables> wuTransformedConvSize;
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    if (level + 1 < numLevelsOfHierarchy) {
      const auto &p = partitionVars[level];
      auto wuP = p;
      // TODO: handle {inChanSplit,outChanSplit}.serial
      wuP.outChanSplit.parallel = p.inChanSplit.parallel;
      wuP.inChanSplit.parallel = p.outChanSplit.parallel;
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
                                method,
                                partialChansPerGroup,
                                inChansPerGroup);

  std::vector<std::unordered_set<unsigned>>
    transformedDims(numLevelsOfHierarchy);
  return addEstimates(m, wuPartitionVars, wuConvSize,
                      wuTransformedConvSize,
                      usedTiles,
                      transformedDims,
                      target,
                      perLevelExchangeBytesPerCycle,
                      wuParams, wuInChansPerGroup, wuPartialChansPerGroup,
                      types, wuMethod,
                      Plan::LinearizeTileOrder::FC_WU,
                      options, cache);
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
  Partition partition(std::move(fieldSplitValues),
                      {s[vars.batchSplit.serial],
                       s[vars.batchSplit.parallel]},
                      {s[vars.outChanSplit.serial],
                       s[vars.outChanSplit.parallel]},
                      std::move(kernelSplitValues),
                      {s[vars.inChanSplit.serial],
                       s[vars.inChanSplit.parallel]},
                      s[vars.convGroupSplit],
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
  params.inputChannels *= params.getTruncatedKernelSize(dim);
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
  // Transformed input must be greater than or equal to the transformed kernel
  // size.
  if (params.inputFieldShape[dim] == 0) {
    params.inputTransform.paddingUpper[dim] = 1;
    params.outputTransform.truncationUpper[dim] = 1;
  }
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

static std::pair<ConvParams, ConvParams>
applyTransform(const ConvParams &params,
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
  return std::make_pair(paddedParams, flattenedParams);
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
    outChanGrainSizes[i] = (transforms[i + 1].outChanFlattenDims.empty()) ?
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

// A fudge factor to apply to the transform cycle cost.
// The two sets of costs were computed using a few layers of RESNET-50. The
// useful case is the 7x7 field size WU in RESNET-50 where some transforms
// result in tensors which cannot be regrouped efficiently.
static std::array<unsigned, 2>
getScaleFactorForTransform(const poplar::Type &type, unsigned dimSize) {
  const auto granularity = type == poplar::FLOAT ? 2U : 4U;
  if (dimSize % granularity == 0)
    return {5U, 4U};
  else
    return {5U, 3U};
}

popsolver::Variable
addTransformCycleEstimate(
    popsolver::Model &m,
    const ConvParams &params,
    const ConvParams &transformedOnceParams,
    const ConvParams &transformedOnceUnpaddedParams,
    const std::vector<ConvTransform> &transforms,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &transformedConvSizes,
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    unsigned inChansPerGroup,
    unsigned partialChansPerGroup,
    const std::vector<ConvTypes> &types,
    bool isJointPlan,
    const ConvOptions &options,
    const poplar::Target &target) {
  bool isConvWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool isFullyConnectedLayer =
      options.pass == Pass::FC_INFERENCE_FWD ||
      options.pass == Pass::FC_TRAINING_FWD ||
      options.pass == Pass::FC_TRAINING_BWD ||
      options.pass == Pass::FC_TRAINING_WU;
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
  bool rearrangeInput = isConvWeightUpdate || expandDims ||
                        swapOperands || padInChannels ||
                        options.pass == Pass::FC_TRAINING_WU ||
                        (options.pass == Pass::FC_TRAINING_BWD && !isJointPlan);
  bool rearrangeWeights = isConvWeightUpdate ||
                          expandDims ||
                          outChanFlattenDims ||
                          swapOperands || padInChannels ||
                          padPartialChannels;
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(params.inputType == poplar::FLOAT);
  bool rearrangeOutput = (!isConvWeightUpdate && swapOperands) ||
                         (isConvWeightUpdate && !swapOperands) ||
                         outChanFlattenDims ||
                         padPartialChannels ||
                         (options.pass == Pass::FC_TRAINING_WU && !isJointPlan);
  // We assume the next layer uses an input channel grouping of
  // weightsPerConvUnit and apply a small cost if the output channel
  // grouping of this layer doesn't match.
  bool regroupOutput = !isFullyConnectedLayer &&
                       partialChansPerGroup != weightsPerConvUnit;
  // If the input channel grouping of the backward pass doesn't divide the
  // output channel grouping of the forward pass the block size for the
  // cross-tile rearrangement of weights between the forward and backward pass
  // will be small. We assume the backward pass uses an input channel grouping
  // of weightsPerConvUnit and apply a small cost if the output channel grouping
  // of this layer isn't a multiple of this weightsPerConvUnit.
  bool regroupWeights = options.pass == Pass::TRAINING_FWD &&
                        partialChansPerGroup % weightsPerConvUnit != 0;
  const auto inputBytesPerElement = target.getTypeSize(params.outputType);
  const auto regroupBytesPerCycle =
      std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                         partialChansPerGroup * inputBytesPerElement);
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
        return getMaxInputRangeSize(values[0],
                                    dim,
                                    transformedOnceParams,
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
  // TODO: handle {batchSplit,outChanSplit,inChanSplit}.serial
  std::vector<popsolver::Variable> ipuSplits = {
    partitionVars[ipuLevel].batchSplit.parallel,
    partitionVars[ipuLevel].convGroupSplit,
    partitionVars[ipuLevel].inChanSplit.parallel,
    partitionVars[ipuLevel].outChanSplit.parallel
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
        std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                           inputBytesPerElement);
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
                                       m.addConstant(inputBytesPerElement)});
        const auto factor =
            getScaleFactorForTransform(
              transformedOnceUnpaddedParams.inputType,
              transformedOnceUnpaddedParams.outputChannels);
        auto cycles =
            m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                      m.addConstant(factor[1] * regroupBytesPerCycle));
        cyclesOperands.push_back(cycles);
      }
    }
    auto numElements = m.sum(numElementsOperands);
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    auto bytesPerTile = m.product({numElementsPerTile,
                                   m.addConstant(inputBytesPerElement)});
    cyclesOperands.push_back(m.ceildiv(bytesPerTile,
                                       m.addConstant(exchangeBytesPerCycle)));
    const auto factor =
        getScaleFactorForTransform(
            transformedOnceUnpaddedParams.inputType,
            transformedOnceUnpaddedParams.inputChannels *
              transformedOnceUnpaddedParams.outputChannels);

    cyclesOperands.push_back(
        m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                             m.addConstant(reorderBytesPerCycle * factor[1])));
  }
  if (rearrangeOutput || regroupOutput) {
    auto totalOutputFieldSize = m.product(outputFieldSizes);
    auto numElements =
        m.product({totalOutputFieldSize, convSize.batchSize,
                   numOutChans, convSize.numConvGroups});
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    const auto outputBytesPerElement =
        target.getTypeSize(types[ipuLevel].resultType);
    const auto outputRegroupBytesPerCycle =
        std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                           partialChansPerGroup * outputBytesPerElement);
    auto bytesPerTile = m.product({numElementsPerTile,
                                   m.addConstant(outputBytesPerElement)});
    if (rearrangeOutput) {
      const auto outputReorderBytesPerCycle =
          std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                             outputBytesPerElement);
      cyclesOperands.push_back(m.ceildiv(bytesPerTile,
                                         m.addConstant(exchangeBytesPerCycle)));
      const auto factor =
          getScaleFactorForTransform(
              transformedOnceUnpaddedParams.outputType,
              transformedOnceUnpaddedParams.outputChannels);
      cyclesOperands.push_back(
        m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                            m.addConstant(outputReorderBytesPerCycle *
                                          factor[1]))
      );
    } else if (regroupOutput) {
      const auto factor =
          getScaleFactorForTransform(
                transformedOnceUnpaddedParams.outputType,
                transformedOnceUnpaddedParams.outputChannels);
      cyclesOperands.push_back(m.ceildiv(
        m.product({bytesPerTile, m.addConstant(factor[0])}),
                  m.addConstant(outputRegroupBytesPerCycle * factor[1]))
      );
    }
  }
  auto cycles = m.sum(cyclesOperands);
  return cycles;
}

static void
applyPartitionPlanConstraint(popsolver::Model &m,
                             const ConvOptions &options, unsigned level,
                             const PartitionVariables &p) {
  const auto &planConstraints = options.planConstraints;
  const auto &thisPartition =
    planConstraints.get_child_optional(std::to_string(level) + ".partition");
  if (thisPartition) {
    const auto constrainVar = [&](const std::string &pathSuffix,
                                  const popsolver::Variable &var) {
      const auto constraint =
        thisPartition.get().get_optional<unsigned>(pathSuffix);
      if (constraint) {
        m.equal(var, *constraint);
      }
    };
    const auto constrainSplitVar = [&](const std::string &pathSuffix,
                                       const Split<popsolver::Variable> &var) {
      constrainVar(pathSuffix + ".parallel", var.parallel);
      constrainVar(pathSuffix + ".serial", var.serial);
    };
    const auto constrainVars =
      [&](const std::string &pathSuffix,
          const std::vector<popsolver::Variable> &vars) {
      // Constraints are objects with keys as indices that may be sparse,
      // and values that are the constraints for those indices in `vars`.
      for (std::size_t i = 0; i < vars.size(); ++i) {
        constrainVar(pathSuffix + "." + std::to_string(i), vars[i]);
      }
    };
    constrainVars("fieldSplit", p.fieldSplit);
    constrainSplitVar("batchSplit", p.batchSplit);
    constrainSplitVar("outChanSplit", p.outChanSplit);
    constrainVars("kernelSplit", p.kernelSplit);
    constrainSplitVar("inChanSplit", p.inChanSplit);
    constrainVar("convGroupSplit", p.convGroupSplit);
    // All other PartitionVariables members are dependent on these splits.
  }
}

// Mostly for testing purposes. We have some constants fixed to a value which
// has no effect (serial partitioning currently) while functionality is
// implemented but which we want to be able to force to a different value
// for development purposes. This function adds such a constant with a default
// value, but which can be overridden by the plan constraints given in the
// ConvOptions.
static popsolver::Variable
addPartitionConstant(popsolver::Model &m,
                     const ConvOptions &options,
                     unsigned level,
                     const std::string &pathSuffix,
                     unsigned defaultVal) {
  const auto val = options.planConstraints.get_optional<unsigned>(
    std::to_string(level) + ".partition." + pathSuffix
  );
  return m.addConstant(val ? *val : defaultVal);
}

// Return m.ceildiv(dividend, divisor) and constrain the divisor so it is
// the smallest divisor that gives us that result. This reduces the size of
// the search space without sacrificing the quality of the plan since the
// maximum amount of work / data on any one tile stays the same.
static popsolver::Variable
ceildivConstrainDivisor(popsolver::Model &m,
                        const popsolver::Variable dividend,
                        const popsolver::Variable divisor) {
  const auto isSmallestDivisorTheGivesResult =
      [](const std::vector<unsigned> &values) -> unsigned {
    auto dividend = values[0];
    auto divisor = values[1];
    // The divisor is the smallest divisor that gives this result if
    // it is 1 or if dividing by (divisor - 1) would gives a larger
    // result.
    if (divisor == 1) {
      return 1;
    }

    if ((dividend + divisor - 1) / divisor <
        (dividend + divisor - 2) / (divisor - 1)) {
      return 1;
    }

    return 0;
  };

  const auto isSmallest = m.call({dividend, divisor},
                                 isSmallestDivisorTheGivesResult);

  // Add constraint that isSmallestDivisorTheGivesResult > 0, i.e.
  // it returns true.
  m.less(0, isSmallest);
  return m.ceildiv(dividend, divisor);
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
               bool isJointPlan,
               Cost bestCost,
               const PlanningObjective &objective,
               PlanningCacheImpl::CycleEstimationImpl *cache,
               const ConvOptions &options,
               popsolver::Model &m,
               std::vector<PartitionVariables> &partitionVars,
               popsolver::Variable &cycles,
               popsolver::Variable &maxTempBytes) {
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
  const auto paramPair = applyTransform(params_,
                                        transforms[0],
                                        inChanGrainSize[0],
                                        outChanGrainSize[0]);
  const auto &params = paramPair.first;
  const auto &unpaddedParams = paramPair.second;
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
    convSize.back().numFieldGrains.push_back(
      m.addConstant(std::max(numGrains, 1U))
    );
    convSize.back().kernelSize.push_back(
      m.addConstant(std::max(params.kernelShape[dim], 1UL))
    );
  }
  convSize.back().batchSize =
      m.addConstant(std::max(params.getBatchSize(), 1UL));
  convSize.back().numConvGroups =
      m.addConstant(std::max(params.getNumConvGroups(), 1UL));
  convSize.back().numOutChanGrains =
      m.addConstant(std::max(outChanGrains, 1UL));
  convSize.back().numInChanGrains =
      m.addConstant(std::max(inChanGrains, 1UL));
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
    p.batchSplit.parallel = m.addVariable(1, levelMaxSplit);
    p.batchSplit.serial =
      addPartitionConstant(m, options, level, "batchSplit.serial", 1);
    auto batchSplit = m.product({p.batchSplit.parallel, p.batchSplit.serial});
    m.lessOrEqual(batchSplit, prevConvSize.batchSize);
    p.convGroupSplit = m.addVariable(1, levelMaxSplit);
    m.lessOrEqual(p.convGroupSplit, prevConvSize.numConvGroups);
    // The joint planning cost function assumes that no exchange is required to
    // rearrange weights between passes. Because of the way we derive the
    // backward and weight update plans from the forward plan this is guaranteed
    // to be the case if each weight is used on exactly one tile in the forward
    // pass. Disallow splitting of fully connected batch (or equivalently the
    // convolutional output channels) across tiles to ensure this holds.
    if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
      p.outChanSplit.parallel = m.addConstant(1);
      p.outChanSplit.serial = m.addConstant(1);
    } else {
      assert(!isJointPlan);
      p.outChanSplit.parallel = m.addVariable(1, levelMaxSplit);
      p.outChanSplit.serial =
        addPartitionConstant(m, options, level, "outChanSplit.serial", 1);
    }
    auto outChanSplit =
      m.product({p.outChanSplit.parallel, p.outChanSplit.serial});
    m.lessOrEqual(outChanSplit, prevConvSize.numOutChanGrains);
    p.inChanSplit.parallel = m.addVariable(1, levelMaxSplit);
    p.inChanSplit.serial =
      addPartitionConstant(m, options, level, "inChanSplit.serial", 1);
    auto inChanSplit =
      m.product({p.inChanSplit.parallel, p.inChanSplit.serial});
    m.lessOrEqual(inChanSplit, prevConvSize.numInChanGrains);
    p.outChanGrainSize = outChanGrainSize[level];
    p.inChanGrainSize = inChanGrainSize[level];
    p.fieldGrainSize = fieldGrainSize;
    nextConvSize.batchSize =
      ceildivConstrainDivisor(m, prevConvSize.batchSize, batchSplit);
    nextConvSize.numConvGroups =
      ceildivConstrainDivisor(m, prevConvSize.numConvGroups, p.convGroupSplit);
    nextConvSize.numOutChanGrains =
      ceildivConstrainDivisor(m, prevConvSize.numOutChanGrains, outChanSplit);
    nextConvSize.numInChanGrains =
      ceildivConstrainDivisor(m, prevConvSize.numInChanGrains, inChanSplit);
    applyPartitionPlanConstraint(m, options, level, p);
    partitionVars.push_back(std::move(p));
    convSize.push_back(std::move(nextConvSize));
  }

  std::vector<Variable> perLevelSplits;
  for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
    const auto &p = partitionVars[level];
    std::vector<Variable> splits;
    // TODO: handle {batchSplit,outChanSplit,inChanSplit}.serial
    splits.push_back(p.batchSplit.parallel);
    splits.push_back(p.outChanSplit.parallel);
    splits.push_back(p.inChanSplit.parallel);
    splits.push_back(p.convGroupSplit);
    splits.insert(splits.end(), p.fieldSplit.begin(), p.fieldSplit.end());
    splits.insert(splits.end(), p.kernelSplit.begin(), p.kernelSplit.end());
    const auto levelSplit = m.product(splits);
    m.lessOrEqual(levelSplit, hierarchy[level]);
    perLevelSplits.push_back(levelSplit);
  }
  const auto usedTiles = m.product(perLevelSplits);

  popsolver::Variable tempBytes;
  std::tie(cycles, tempBytes) =
      addEstimates(m, partitionVars, convSize, transformedConvSize,
                   usedTiles, transformedDims,
                   target, perLevelExchangeBytesPerCycle, params,
                   inChansPerGroup, partialChansPerGroup,
                   types, convVertexType.method,
                   Plan::LinearizeTileOrder::STANDARD, options, cache);
  maxTempBytes = tempBytes;
  if (isJointPlan) {
    assert(options.pass == Pass::FC_TRAINING_FWD);

    popsolver::Variable bwdCycles, bwdTempBytes;
    std::tie(bwdCycles, bwdTempBytes) =
      addBwdEstimates(m, params, numLevelsOfHierarchy, partitionVars, convSize,
                      convVertexType.method, usedTiles, target,
                      perLevelExchangeBytesPerCycle, types,
                      partialChansPerGroup, inChansPerGroup, options, cache);

    popsolver::Variable wuCycles, wuTempBytes;
    std::tie(wuCycles, wuTempBytes) =
      addWuEstimates(m, params, numLevelsOfHierarchy, partitionVars, convSize,
                     convVertexType.method, usedTiles, target, numFieldDims,
                     perLevelExchangeBytesPerCycle, types,
                     partialChansPerGroup, inChansPerGroup, options, cache);

    cycles = m.sum({cycles, bwdCycles, wuCycles});
    if (objective.getTileTempMemoryBound() > 0) {
      auto bound = objective.getTileTempMemoryBound();
      // fwd temp bytes constrained below
      m.lessOrEqual(bwdTempBytes, bound);
      m.lessOrEqual(wuTempBytes, bound);
    }

    // report the max requirement of all three phases
    maxTempBytes = m.call({tempBytes, bwdTempBytes, wuTempBytes},
                          [](const std::vector<unsigned> &values) {
      auto max = 0u;
      for (auto &v : values)
        if (max < v)
          max = v;
      return max;
    });
  }

  auto transformCycles =
      addTransformCycleEstimate(m, params_, params, unpaddedParams, transforms,
                                partitionVars, transformedConvSize,
                                transformedDims, inChansPerGroup,
                                partialChansPerGroup, types, isJointPlan,
                                options, target);
  cycles = m.sum({cycles, transformCycles});
  auto cyclesBound = bestCost.cycles;
  if (objective.getCyclesBound() > 0) {
    cyclesBound = std::min(cyclesBound, objective.getCyclesBound());
  }
  if (cyclesBound < std::numeric_limits<unsigned>::max()) {
    m.lessOrEqual(cycles, cyclesBound);
  }
  if (objective.getTileTempMemoryBound() > 0) {
    m.lessOrEqual(tempBytes, objective.getTileTempMemoryBound());
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
           bool isJointPlan,
           Cost bestCost,
           const PlanningObjective &objective,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           const ConvOptions &options) {
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  popsolver::Variable cycles, tempBytes;
  constructModel(target, transforms, types, hierarchy,
                 perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType,
                 params, isJointPlan, bestCost, objective, cache, options, m,
                 partitionVars, cycles, tempBytes);
  popsolver::Solution s;

  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    s = m.minimize({cycles, tempBytes});
    break;
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    s = m.minimize({tempBytes, cycles});
    break;
  }
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
            Plan::LinearizeTileOrder::STANDARD,
            getStartTile(params, options, isJointPlan),
            isJointPlan);
  plan.transforms = transforms;

  Cost cost = {s[cycles], s[tempBytes]};

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

static void
getConvVertexMACCandidates(const poplar::Target &target,
                           const poplar::Type &inputType,
                           const poplar::Type &outputType,
                           const poplar::Type &partialType,
                           const ConvParams &params,
                           const ConvOptions &options,
                           bool isJointPlan,
                           std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
    planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
    planConstraints.get_optional<unsigned>("partialChansPerGroup");

  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnits = getNumConvUnits(floatActivations,
                                      ampFloatPartials,
                                      target);

  // Constrain the input channel grouping to a mulitple of two if the activation
  // type is half. This ensures that we never need to apply padding when sending
  // activations over the exchange.
  auto grainSize = floatActivations ? 1 : 2;
  const auto roundedNumInChans =
      ((params.getNumInputChansPerConvGroup() + grainSize - 1) / grainSize) *
      grainSize;

  unsigned inChansLower = grainSize;
  unsigned inChansUpper = roundedNumInChans;
  if (constrainedInChansPerGroup) {
    // Must be within bounds of the input channels and divisible by
    // the grain size for this type to use this vertex.
    if (*constrainedInChansPerGroup > roundedNumInChans ||
        *constrainedInChansPerGroup % grainSize != 0) {
      return;
    }
    inChansLower = inChansUpper = *constrainedInChansPerGroup;
  }

  const unsigned partialChansPerGroup = 1;
  // This is the only supported partialChansPerGroup for this method.
  if (constrainedPartialChansPerGroup &&
      *constrainedPartialChansPerGroup != partialChansPerGroup) {
    return;
  }

  unsigned previousInChanGroups = 0;
  for (unsigned inChansPerGroup = inChansLower;
       inChansPerGroup <= inChansUpper;
       inChansPerGroup += grainSize) {
    unsigned inChanGroups = (roundedNumInChans + inChansPerGroup - 1) /
                            inChansPerGroup;
    if (inChanGroups == previousInChanGroups) {
      // There is no point considering a larger group size if it doesn't
      // decrease the number of groups - the zero padding increases the
      // amount of work per group and we can't use fewer groups per tile.
      continue;
    }
    if (isJointPlan) {
      assert(options.pass == Pass::FC_TRAINING_FWD);
      // The input channels in the forward pass become the output channels of
      // the weight update pass. Make sure it is a multiple of the supported
      // output channels per group.
      if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
        continue;
    }
    candidates.emplace_back(Plan::Method::MAC,
                            inputType,
                            outputType,
                            partialType,
                            inChansPerGroup,
                            partialChansPerGroup);
    previousInChanGroups = inChanGroups;
  }
}

static void
getConvVertexAMPCandidates(const poplar::Target &target,
                           const poplar::Type &inputType,
                           const poplar::Type &outputType,
                           const poplar::Type &partialType,
                           const ConvParams &params,
                           const ConvOptions &options,
                           bool isJointPlan,
                           std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
    planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
    planConstraints.get_optional<unsigned>("partialChansPerGroup");

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
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   target)) {
    unsigned inChansLower = 0;
    unsigned inChansUpper = std::numeric_limits<unsigned>::max();
    if (constrainedInChansPerGroup) {
      inChansLower = inChansUpper = *constrainedInChansPerGroup;
    }

    const unsigned weightsPerConvUnit =
        target.getWeightsPerConvUnit(floatActivations);
    inChansLower = std::max(inChansLower, 1u);
    inChansUpper = std::min(inChansUpper, weightsPerConvUnit);

    std::vector<unsigned> partialChansPerGroupCandidates;
    if (constrainedPartialChansPerGroup) {
      if (*constrainedPartialChansPerGroup == numConvUnits ||
          *constrainedPartialChansPerGroup == weightsPerConvUnit) {
        partialChansPerGroupCandidates.push_back(
          *constrainedPartialChansPerGroup);
      }
    } else {
      partialChansPerGroupCandidates = {
        numConvUnits,
        weightsPerConvUnit
      };
    }

    for (unsigned inChansPerGroup = inChansLower;
         inChansPerGroup <= inChansUpper;
         ++inChansPerGroup) {
      for (unsigned partialChansPerGroup : partialChansPerGroupCandidates) {
        if (!floatActivations && inChansPerGroup % 2 != 0) {
          continue;
        }
        // There are two reasons we might choose to make partialChansPerGroup
        // not equal to numConvUnits:
        // - The output of a convolution is likely to be fed into another
        //   convolution that wants its input grouped by weightsPerConvUnit
        //   so there will be a small cost (estimated by the planner) if
        //   partialChansPerGroup != weightsPerConvUnit
        // - The output channel grouping of a fully connected forward pass
        //   becomes the input channel grouping of the fully connected weight
        //   update pass and so if partialChansPerGroup != weightsPerConvUnit
        //   we can't fully utilize AMP in the weight update pass.
        // Neither of these reasons apply to fully connected inference (we
        // must always rearrange the output regardless of the grouping and
        // there is no weight update pass).
        if (options.pass == Pass::FC_INFERENCE_FWD &&
            partialChansPerGroup != numConvUnits) {
          continue;
        }
        if (!canUseConvolutionInstruction(floatActivations,
                                          floatPartials,
                                          inChansPerGroup, target)) {
          continue;
        }
        if (isJointPlan) {
           assert(options.pass == Pass::FC_TRAINING_FWD);
          // The input channels in the forward pass become the output channels
          // of the weight update pass. Make sure it is a multiple of the
          // supported output channels per group.
          if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0) {
            continue;
          }
        }
        candidates.emplace_back(Plan::Method::AMP,
                                inputType,
                                outputType,
                                ampPartialType,
                                inChansPerGroup,
                                partialChansPerGroup);
      }
    }
  }
}

static void
getConvVertexOuterProductCandidates(const poplar::Target &target,
                                    const poplar::Type &inputType,
                                    const poplar::Type &outputType,
                                    const poplar::Type &partialType,
                                    const ConvParams &params,
                                    const ConvOptions &options,
                                    bool isJointPlan,
                                    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
    planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
    planConstraints.get_optional<unsigned>("partialChansPerGroup");

  if (canUseOuterProductMethod(params)) {
    const auto inChansPerGroup = 1u;
    const auto partialChansPerGroup = target.getVectorWidth(inputType);
    // Only one supported inChansPerGroup or partialChansPerGroup
    // for this method.
    if (constrainedInChansPerGroup &&
        *constrainedInChansPerGroup != inChansPerGroup) {
      return;
    }
    if (constrainedPartialChansPerGroup &&
        *constrainedPartialChansPerGroup != partialChansPerGroup) {
      return;
    }
    candidates.emplace_back(Plan::Method::OUTER_PRODUCT,
                            inputType,
                            outputType,
                            inputType,
                            inChansPerGroup,
                            partialChansPerGroup);
  }
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            poplar::Type inputType,
                            poplar::Type outputType,
                            poplar::Type partialType,
                            const ConvParams &params,
                            const ConvOptions &options,
                            bool isJointPlan) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedMethod = [&]() -> boost::optional<Plan::Method> {
    const auto constraint =
      planConstraints.get_optional<std::string>("method");
    if (constraint) {
      Plan::Method m;
      std::stringstream ss(*constraint);
      ss >> m;
      return m;
    }
    return boost::none;
  }();

  std::vector<Plan::Method> methodCandidates;
  if (constrainedMethod) {
    methodCandidates.push_back(*constrainedMethod);
  } else {
    methodCandidates = {
      Plan::Method::MAC,
      Plan::Method::AMP,
      Plan::Method::OUTER_PRODUCT,
    };
  }

  // All the following methods assume half or float input/partial types.
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);

  std::vector<ConvVertexType> convVertexTypeCandidates;
  for (const auto &method : methodCandidates) {
    switch (method) {
    case Plan::Method::MAC: {
      getConvVertexMACCandidates(target,
                                 inputType,
                                 outputType,
                                 partialType,
                                 params,
                                 options,
                                 isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::AMP: {
      getConvVertexAMPCandidates(target,
                                 inputType,
                                 outputType,
                                 partialType,
                                 params,
                                 options,
                                 isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::OUTER_PRODUCT: {
      getConvVertexOuterProductCandidates(target,
                                          inputType,
                                          outputType,
                                          partialType,
                                          params,
                                          options,
                                          isJointPlan,
                                          convVertexTypeCandidates);
      break;
    }
    default: {
      throw poputil::poplibs_error("Unknown Plan::Method");
    }
    }
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
getExpandDimsCandidates(unsigned ipuLevel,
                        const ConvParams &params,
                        const ConvOptions &options) {
  const auto &planConstraints = options.planConstraints;
  const auto constraint =
    planConstraints.get_child_optional(std::to_string(ipuLevel) +
                                       ".transform.expandDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    for (const auto &child : *constraint) {
      const auto dim = child.second.get_value<unsigned>();
      if (dim >= params.getNumFieldDims()) {
        throw poputil::poplibs_error("Trying to force expansion of spatial "
                                     "dimension " + std::to_string(dim) +
                                     " but there are only " +
                                     std::to_string(params.getNumFieldDims()) +
                                     " spatial dimensions");
      }
      forcedDims.push_back(dim);
    }
    std::sort(forcedDims.begin(), forcedDims.end());
    forcedDims.erase(std::unique(forcedDims.begin(), forcedDims.end()),
                     forcedDims.end());
    std::reverse(forcedDims.begin(), forcedDims.end());
    candidateDimSets.emplace_back(std::move(forcedDims));
  } else {
    std::vector<unsigned> candidateDims;
    for (unsigned i = 0; i != params.getNumFieldDims(); ++i) {
      if (!expandingDimChangesParams(params, i)) {
        continue;
      }
      // Don't expand this dimension if the number of non zero kernel entries
      // is larger than the number of non zero input entries as it is unlikely
      // to be profitable. This heuristic cuts down the size of the search
      // space.
      //
      // TODO investigate better heuristics.
      if (params.inputFieldShape[i] < params.kernelShape[i])
        continue;
      candidateDims.push_back(i);
    }
    candidateDimSets = getPowerSet(candidateDims);
    for (auto &subset : candidateDimSets) {
      // The subsets returned by getPowerSet have the outermost dimension first
      // but it is more efficient to expand the innermost dimension first.
      std::reverse(subset.begin(), subset.end());
    }
  }
  return candidateDimSets;
}

static std::vector<std::vector<unsigned>>
getOutChanFlattenDimsCandidates(unsigned ipuLevel,
                                const ConvParams &params,
                                const ConvOptions &options) {
  auto swappedParams = params;
  const auto &planConstraints = options.planConstraints;
  const auto constraint =
    planConstraints.get_child_optional(std::to_string(ipuLevel) +
                                       ".transform.outChanFlattenDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    for (const auto &child : *constraint) {
      const auto dim = child.second.get_value<unsigned>();
      if (dim >= params.getNumFieldDims()) {
        throw poputil::poplibs_error("Trying to force expansion of spatial "
                                     "dimension " + std::to_string(dim) +
                                     " but there are only " +
                                     std::to_string(params.getNumFieldDims()) +
                                     " spatial dimensions");
      }
      forcedDims.push_back(dim);
    }
    std::sort(forcedDims.begin(), forcedDims.end());
    forcedDims.erase(std::unique(forcedDims.begin(), forcedDims.end()),
                     forcedDims.end());
    std::reverse(forcedDims.begin(), forcedDims.end());
    candidateDimSets.emplace_back(std::move(forcedDims));
  } else {
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
    candidateDimSets = getPowerSet(candidateDims);
    for (auto &subset : candidateDimSets) {
      // The subsets returned by getPowerSet have the outermost dimension first
      // but it is more efficient to expand the innermost dimension first.
      std::reverse(subset.begin(), subset.end());
    }
  }
  return candidateDimSets;
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
  params = params.canonicalize();
}

static std::vector<bool> getSwapOperandCandidates(const ConvParams &params,
                                                  const ConvOptions &options,
                                                  bool isJointPlan) {
  std::vector<bool> validValues;
  if (isJointPlan) {
    // The joint planning logic doesn't yet handle swapped operands.
    // TODO lift this restriction.
    validValues = {false};
  } else {
    validValues = {false, true};
  }

  // Check for explicitly forced swapped operands in the options.
  const auto &planConstraints = options.planConstraints;
  const auto constraint =
    planConstraints.get_optional<bool>("0.transform.swapOperands");
  if (constraint) {
    if (std::find(validValues.begin(), validValues.end(), *constraint) ==
        validValues.end()) {
      throw poputil::poplibs_error(
          "0.transform.swapOperands was constrained to be '" +
          std::string(*constraint ? "true" : "false") +
          "' but this is not valid for these parameters");
    }
    validValues = {*constraint};
  } else if (!params.outputChannels) {
    // Avoid swapping operands when output channels could be swapped with batch
    // size
    validValues = {false};
  }

  return validValues;
}


std::vector<unsigned> getTileHierarchy(const poplar::Target &target,
                                       const ConvOptions &options) {
  std::vector<double> dummy;
  return poplibs::getTileHierarchy(target, options.numIPUs, options.tilesPerIPU,
                                   dummy);
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
           bool isJointPlan,
           const PlanningObjective &objective,
           const poplar::Target &target,
           PlanningCacheImpl::CycleEstimationImpl *cache) {
  validateLayerParams(params, options, target);

  // T8972: It is currently assumed that the parameters for all the training
  // passes can be derived from one pass, but this is no longer the case since a
  // different outputType can be specified for each pass. To avoid a costly
  // exchange of weights, we plan with the assumption that
  // outputType == inputType for FC_TRAINING.
  const auto originalOutputType = params.outputType;
  if (isJointPlan) params.outputType = params.inputType;

  // TODO: (T9459) Validate planConstraints in ConvOptions. These are validated
  // for syntax but not against e.g. no. of levels of hierarchy or no. of
  // dimensions etc. so this is the point at which this should be validated.
  unsigned outChanSerialSplit = 1;
  // Apply a heuristic that if the number of output elements is large to
  // split the out channels into chunks to be calculated sequentially.
  // Currently the planner will only do this if the channels are a multiple of
  // 8 and the out channels are only split into multiples of 8. This is
  // to still try and ensure efficiency of the individual chunks.
  auto outShape = params.getOutputFieldShape();
  auto numOutputs = std::accumulate(outShape.begin(), outShape.end(), 1UL,
                                    std::multiplies<std::size_t>())
                       * params.getNumOutputChans()
                       * params.getBatchSize();

  auto outputBytesPerTile =
    (numOutputs * target.getTypeSize(params.outputType)) / target.getNumTiles();
  auto largeOutputBytesPerTile =
      options.maxOutputMemoryProportion * target.getBytesPerTile();
  if (largeOutputBytesPerTile != 0 &&
      outputBytesPerTile > largeOutputBytesPerTile) {
    // We want to split the output channels with the smallest split that
    // reduces the number of outputs to less than the threshold.
    auto maxSplit = params.outputChannels;
    for (unsigned split = 1; split <= maxSplit; ++split) {
      if (params.outputChannels % split != 0)
        continue;
      if (outputBytesPerTile / split < largeOutputBytesPerTile ||
          split == maxSplit) {
        outChanSerialSplit = split;
        break;
      }
    }
    params.outputChannels = params.outputChannels / outChanSerialSplit;
  }

  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy =
      poplibs::getTileHierarchy(target, options.numIPUs, options.tilesPerIPU,
                                perLevelExchangeBytesPerCycle);
  const auto numLevels = hierarchy.size() + 1;
  Cost bestCost = highestCost;
  Plan bestPlan;
  std::vector<ConvTransform> transforms(numLevels);
  auto convTypes = getConvTypes(target, numLevels, params.outputType, options);
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
                                                    options,
                                                    isJointPlan)) {
    transforms[0].swapOperands = swapOperands;
    auto swappedParams = calculateSwappedParams(paramsWithDeferredDilation,
                                                swapOperands);
    for (const std::vector<unsigned> &expandDims :
         getExpandDimsCandidates(ipuLevel, swappedParams, options)) {
      transforms[ipuLevel].expandDims = expandDims;
      auto expandedParams = calculateExpandedParams(swappedParams, expandDims);
      for (const std::vector<unsigned> &outChanFlattenDims :
           getOutChanFlattenDimsCandidates(ipuLevel, expandedParams, options)) {
        transforms[ipuLevel].outChanFlattenDims = outChanFlattenDims;
        auto flattenedParams =
            calculateFlattenedParams(expandedParams, outChanFlattenDims,
                                     transforms[ipuLevel].flattenDims);
        const auto convVertexTypeCandidates =
            getConvVertexTypeCandidates(target,
                                        params.inputType,
                                        params.outputType,
                                        convTypes.back().partialType,
                                        flattenedParams,
                                        options,
                                        isJointPlan);
        for (const auto &convVertexType : convVertexTypeCandidates) {
          std::vector<unsigned> fieldGrainSize(numFieldDims, 1);
          if (isJointPlan) {
            assert(options.pass == Pass::FC_TRAINING_FWD);
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
                         convVertexType, params, isJointPlan, bestCost,
                         objective, cache, options);
          if (candidateCost == highestCost) {
            continue;
          }

          if (objective.lowerCost(candidateCost, bestCost)) {
            logging::debug("Found new best candidate plan: {}", candidateCost);
            bestPlan = candidate;
            bestCost = candidateCost;
          }
        }
      }
    }
  }

  bestPlan.outChanSerialSplit = outChanSerialSplit;
  if (largeOutputBytesPerTile == 0) {
    assert(bestPlan.outChanSerialSplit == 1);
  }

  if (isJointPlan) {
    // If we created a plan with the assumption that inputType == outputType,
    // we now restore resultType to ensure bestPlan is valid.
    const auto numLevelsOfHierarchy = hierarchy.size() + 1;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      const auto outputTypeSize = target.getTypeSize(originalOutputType);
      auto &types = bestPlan.types[level];

      if (target.getTypeSize(types.resultType) < outputTypeSize || 0 == level) {
        types.resultType = originalOutputType;
      }
      if (target.getTypeSize(types.partialType) < outputTypeSize) {
        types.partialType = originalOutputType;
      }
    }
  }

  return {bestPlan, bestCost};
}
static CanonicalConvParams
getFullyConnectedPassParams(const CanonicalConvParams &params,
                            const ConvOptions &options,
                            Pass pass) {
  assert(params->getNumFieldDims() == 1);
  assert(params->batchSize == 1);
  assert(params->inputTransform.flip[0] == false);
  assert(params->inputTransform.dilation[0] == 1);
  assert(params->kernelTransform.flip[0] == false);
  assert(params->kernelTransform.truncationLower[0] == 0);
  if (params->inputFieldShape[0] == 0) {
    // for a zero convolution the canonical form is to provide a kernel of size
    // 1 and then truncate it back to zero.
    assert(params->kernelTransform.truncationUpper[0] == 1);
    assert(params->outputTransform.truncationUpper[0] == 1);
  } else {
    assert(params->kernelTransform.truncationUpper[0] == 0);
    assert(params->outputTransform.truncationUpper[0] == 0);
  }
  assert(params->kernelShape[0] == 1);
  assert(params->outputTransform.stride[0] == 1);
  assert(params->outputTransform.paddingLower[0] == 0);
  assert(params->outputTransform.paddingUpper[0] == 0);

  // Translate convolution parameters to parameters of the fully connected layer
  // forward pass.
  unsigned fwdOutputSize, fwdInputSize, fwdBatchSize;
  switch (options.pass) {
  default: assert(0 && "Unexpected pass");
  case Pass::FC_TRAINING_FWD:
    fwdInputSize = params->getNumInputChansPerConvGroup();
    fwdBatchSize = params->getNumOutputChansPerConvGroup();
    fwdOutputSize = params->getInputSize(0);
    break;
  case Pass::FC_TRAINING_BWD:
    fwdInputSize = params->getInputSize(0);
    fwdBatchSize = params->getNumOutputChansPerConvGroup();
    fwdOutputSize = params->getNumInputChansPerConvGroup();
    break;
  case Pass::FC_TRAINING_WU:
    fwdOutputSize = params->getInputSize(0);
    fwdBatchSize = params->getNumInputChansPerConvGroup();
    fwdInputSize = params->getNumOutputChansPerConvGroup();
    break;
  }
  // Translate fully connected layer forward pass parameters back into
  // convolution parameters for the specified pass.
  unsigned convFieldSize, convInputChannels, convOutputChannels,
           inputPadding = 0, outputTruncation = 0;
  switch (pass) {
  default: assert(0 && "Unexpected pass");
  case Pass::FC_TRAINING_FWD:
    convInputChannels = fwdInputSize;
    convFieldSize = fwdOutputSize;
    convOutputChannels = fwdBatchSize;
    break;
  case Pass::FC_TRAINING_BWD:
    convInputChannels = fwdOutputSize;
    convFieldSize = fwdInputSize;
    convOutputChannels = fwdBatchSize;
    break;
  case Pass::FC_TRAINING_WU:
    convInputChannels = fwdBatchSize;
    convFieldSize = fwdOutputSize;
    convOutputChannels = fwdInputSize;
    break;
  }
  if (convFieldSize == 0) {
    // Transformed input must be greater than or equal to the transformed kernel
    // size.
    inputPadding = 1;
    outputTruncation = 1;
  }
  ConvParams newParams{
    params->inputType,
    params->outputType,
    1,                          // batchSize
    {convFieldSize},            // inputShape
    {1},                        // kernelShape
    convInputChannels,          // input channels
    convOutputChannels,         // output channels
    params->getNumConvGroups()  // conv groups
  };
  newParams.inputTransform.paddingUpper = {inputPadding};
  newParams.outputTransform.truncationUpper = {outputTruncation};

  return newParams;
}

static ConvOptions getFullyConnectedPassOptions(const ConvOptions &options,
                                                Pass pass) {
  auto newOptions = options;
  newOptions.pass = pass;
  return newOptions;
}

static std::pair<Plan, Cost>
createPlan(const ConvParams &params,
           const ConvOptions &options,
           const PlanningObjective &objective,
           const poplar::Target &target,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           std::vector<std::pair<PlanningCacheImpl::Key, Plan>> *
               additionalPlansToCache) {
  if (options.pass != Pass::FC_TRAINING_FWD)
    return createPlan(params, options, false, objective, target, cache);
  // It doesn't make sense to compare joint and separate planning when the
  // number of cycles is bounded since we can't easily derive bounds for each
  // individual pass from a bound on the total number of cycles.
  assert(objective.getCyclesBound() == 0);
  Plan jointPlan;
  Cost jointCost;
  std::tie(jointPlan, jointCost) =
      createPlan(params, options, true, objective, target, cache);
  Plan fwdPlan, bwdPlan, wuPlan;
  Cost fwdCost, bwdCost, wuCost;
  std::tie(fwdPlan, fwdCost) =
      createPlan(params, options, false, objective, target, cache);
  auto bwdParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_BWD);
  auto bwdOptions =
      getFullyConnectedPassOptions(options, Pass::FC_TRAINING_BWD);
  std::tie(bwdPlan, bwdCost) = createPlan(bwdParams.getParams(), bwdOptions,
                                          false, objective, target, cache);
  auto wuParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_WU);
  auto wuOptions =
      getFullyConnectedPassOptions(options, Pass::FC_TRAINING_WU);
  std::tie(wuPlan, wuCost) = createPlan(wuParams.getParams(), wuOptions, false,
                                        objective, target, cache);
  auto separateCost = fwdCost;
  for (const auto &cost : {bwdCost, wuCost}) {
    if (separateCost == highestCost || cost == highestCost) {
      separateCost = highestCost;
      break;
    }
    separateCost.cycles += cost.cycles;
    separateCost.tileTempMemory = std::max(separateCost.tileTempMemory,
                                           cost.tileTempMemory);
  }
  if (objective.lowerCost(separateCost, jointCost)) {
    if (additionalPlansToCache) {
      using Key = PlanningCacheImpl::Key;
      additionalPlansToCache->emplace_back(Key(std::move(bwdParams),
                                               std::move(bwdOptions)),
                                           std::move(bwdPlan));
      additionalPlansToCache->emplace_back(Key(std::move(wuParams),
                                               std::move(wuOptions)),
                                           std::move(wuPlan));
    }
    return {fwdPlan, fwdCost};
  }
  return {jointPlan, jointCost};
}

// Plan the specified convolution in one of three possible modes:
// cycle cost is the priority
// memory cost is the priority
// optimised for memory, constrained to have cycles cost no worse than some
// multiple of the minimimum possible cycle cost.
// Planning a particular training pass (forward / backward / weight update) may
// create plans for the other training passes as a side effect. There plans
// are appended to the end of additionalPlansToCache if it is not null.
static std::pair<Plan, Cost>
runPlanner(
    const CanonicalConvParams &params,
    const ConvOptions &options_,
    const poplar::Target &target,
    PlanningCacheImpl::CycleEstimationImpl *cache,
    std::vector<std::pair<PlanningCacheImpl::Key, Plan>> *
        additionalPlansToCache) {
  // TODO: enable the exception.  This warning is a workaround for current
  // frameworks / tests that set the tempMemoryBudget without overriding the
  // new default cycleBackoffPercent
  auto options = options_;
  if (options.tempMemoryBudget != 0 && options.cycleBackoffPercent != 0) {
    if (true) {
      logging::warn("Warning: both tempMemoryBudget and cycleBackoffPercent"
                    " set; ignoring cycleBackoffPercent");
      options.cycleBackoffPercent = 0;
    } else {
      throw poputil::poplibs_error(
            "ConvPlan: Specifying both tempMemoryBudget and "
            "cycleBackoffPercent not supported");
    }
  }
  Plan plan;
  Cost cost;

  if (options.cycleBackoffPercent != 0) {
    logging::info("Planning convolution with a cycle backoff of {}%",
                  options.cycleBackoffPercent);

    // A cycle backoff is allowed, so first plan for cycles with no memory
    // constraint
    std::tie(plan, cost) = createPlan(params.getParams(), options,
                                      PlanningObjective::minimizeCycles(),
                                      target, cache, nullptr);
    if (cost.cycles == ~0u) {
      throw poputil::poplibs_error(
          "No base plan found for unbounded plan");
    }

    const auto cyclesBound =
      cost.cycles * (100.0 + options.cycleBackoffPercent) / 100.0;
    logging::info("Found fastest plan: {}.", cost);
    logging::info("Applying back-off to find the smallest plan that takes at "
                  "most {} cycles.", cost, cyclesBound);
    logging::trace("{}", plan);

    // Now replan for minimimum memory, with the cycles limited.
    // This is guaranteed to find a solution (which might be the same as the
    // original one). Force the planner to use the same plan type (joint or
    // separate) as we can't use the cycle estimate of one type of plan to
    // constrain a plan of the other type.
    auto objective = PlanningObjective::minimizeTileTempMemory();
    objective.setCyclesBound(cyclesBound);
    std::tie(plan, cost) = createPlan(params.getParams(), options,
                                      plan.isJointPlan, objective, target,
                                      cache);
    if (cost.cycles == ~0u) {
      throw poputil::poplibs_error(
          "No base plan found for cycle-backoff convolution????");
    }

    logging::info("Found smallest plan: {}.", cost);
    logging::trace("{}", plan);

    return {plan, cost};
  }

  // Find a plan that minimizes cycles within the memory budget.
  // If the budget is tiny we skip this stage and go straight to minimizing
  // memory. Don't warn if we skip this stage as the target was obviously set
  // low to achieve this.
  if (options.tempMemoryBudget == 0 || options.tempMemoryBudget > 10000) {
    logging::info("Planning convolution with a temporary memory budget of {} "
                  "bytes.", options.tempMemoryBudget);

    auto objective = PlanningObjective::minimizeCycles();
    if (options.tempMemoryBudget)
      objective.setTileTempMemoryBound(options.tempMemoryBudget);
    std::tie(plan, cost) = createPlan(params.getParams(), options, objective,
                                      target, cache, additionalPlansToCache);

    if (cost.cycles != ~0u) {
      logging::info("Found smallest plan: {}.", cost);
      logging::trace("{}", plan);

      return {plan, cost};
    }

    logging::warn("Warning: convolution planner unable to meet memory target {}"
                  "; retrying targeting minimum memory.",
                  objective.getTileTempMemoryBound());
  }
  // No plan within the memory budget was found - find the plan that
  // minimizes memory instead.
  std::tie(plan, cost) = createPlan(params.getParams(), options,
                                    PlanningObjective::minimizeTileTempMemory(),
                                    target, cache, additionalPlansToCache);
  if (cost.cycles == ~0u) {
    throw poputil::poplibs_error(
        "No mem-priority plan found for convolution");
  }

  logging::info("Found smallest plan: {}.", cost);
  logging::trace("{}", plan);
  return {plan, cost};
}

static Plan getFullyConnectedWUPlan(const poplar::Target &target,
                                    const CanonicalConvParams &fwdParams,
                                    const ConvOptions &fwdOptions,
                                    const Plan &fwdPlan) {
  assert(fwdPlan.isJointPlan);
  assert(!fwdPlan.transforms[0].swapOperands);
  auto plan = fwdPlan;
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_WU;
  const auto numPartitions = plan.partitions.size();
  for (unsigned i = 0; i != numPartitions; ++i) {
    // TODO: handle {inChanSplit,outChanSplit}.serial
    plan.partitions[i].inChanSplit.parallel =
      fwdPlan.partitions[i].outChanSplit.parallel;
    plan.partitions[i].outChanSplit.parallel =
      fwdPlan.partitions[i].inChanSplit.parallel;
    plan.partitions[i].outChanGrainSize = fwdPlan.partitions[i].inChanGrainSize;
    plan.partitions[i].inChanGrainSize = fwdPlan.partitions[i].outChanGrainSize;
  }
  plan.partialChansPerGroup = fwdPlan.inChansPerGroup;
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;

  plan.method = getFullyConnectedWUMethod(fwdParams.getParams(), fwdPlan.method,
                                          fwdPlan.partialChansPerGroup,
                                          fwdPlan.inChansPerGroup);
  // TODO make the fwd pass aware that it would be good to use a grouping of
  // 16 if possible.
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  if (plan.method == Plan::Method::AMP &&
      !canUseConvolutionInstruction(fwdParams->inputType == poplar::FLOAT,
                                    fwdOptions.partialsType == poplar::FLOAT,
                                    plan.inChansPerGroup, target)) {
    plan.inChansPerGroup =
        target.getWeightsPerConvUnit(fwdParams->inputType == poplar::FLOAT);
    plan.partitions.back().inChanGrainSize = plan.inChansPerGroup;
  }

  // If the result type is half and all the reduction is done within a single
  // pass of the AMP unit then there is no reason to use a higher precision
  // partial type.
  if (fwdParams->outputType == poplar::HALF &&
      fwdParams->getNumOutputChansPerConvGroup() == plan.inChansPerGroup &&
      target.getFp16InFp16OutConvUnitsPerTile() ==
      target.getFp16InFp32OutConvUnitsPerTile()) {
    for (auto &x : plan.types) {
      x.partialType = x.resultType = poplar::HALF;
    }
  }

  // Set the partials type to the output type as there are no reductions
  // required
  if (fwdParams->outputType == poplar::HALF &&
      plan.method == Plan::Method::OUTER_PRODUCT) {
    for (auto &x : plan.types) {
      x.partialType = x.resultType = poplar::HALF;
    }
  }
  return plan;
}

static Plan getFullyConnectedBwdPlan(const Plan &fwdPlan) {
  assert(fwdPlan.isJointPlan);
  assert(!fwdPlan.transforms[0].swapOperands);
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  for (auto &partition : plan.partitions) {
    std::swap(partition.fieldSplit.back(), partition.inChanSplit.parallel);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
  return plan;
}

void preplanConvolutionsImpl(
    const poplar::Target &target,
    const std::set<std::pair<ConvParams, ConvOptions>> &paramSet,
    PlanningCache &cache) {
  // convert to a vector for efficient tbb looping
  struct Job {
    const std::pair<ConvParams, ConvOptions> *input;
    std::vector<std::pair<PlanningCacheImpl::Key, Plan>> output;
  };
  std::vector<Job> jobs(paramSet.size());

  auto pIt = paramSet.cbegin();
  for (unsigned i = 0u; i != paramSet.size(); ++i, ++pIt) {
    jobs[i].input = &*pIt;
  }
  // create plans in parallel

  tbb::parallel_for(0u, unsigned(paramSet.size()), [&](unsigned i) {
    const auto &params = jobs[i].input->first;
    const auto &options = jobs[i].input->second;
    Plan plan;
    Cost cost;
    std::tie(plan, cost) = runPlanner(params, options, target,
                                      &cache.impl->cycleEstimation,
                                      &jobs[i].output);
    auto key = PlanningCacheImpl::Key(jobs[i].input->first,
                                      jobs[i].input->second);
    jobs[i].output.emplace_back(key, std::move(plan));
  });
  // sequential insert into the cache
  for (unsigned i = 0u; i != jobs.size(); ++i) {
    for (auto &entry : jobs[i].output) {
      auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(entry.second)));
      cache.impl->plans.emplace(std::move(entry.first), std::move(pPlan));
    }
  }
}

Plan getPlan(const poplar::Target &target, const CanonicalConvParams &params,
             const ConvOptions &options, PlanningCache *cache) {
  if (options.pass == Pass::FC_TRAINING_WU ||
      options.pass == Pass::FC_TRAINING_BWD) {
    auto fwdParams = getFullyConnectedPassParams(params, options,
                                                 Pass::FC_TRAINING_FWD);
    auto fwdOptions = getFullyConnectedPassOptions(options,
                                                   Pass::FC_TRAINING_FWD);
    const auto fwdPlan =
        getPlan(target, fwdParams, fwdOptions, cache);
    if (fwdPlan.isJointPlan) {
      if (options.pass == Pass::FC_TRAINING_WU)
        return getFullyConnectedWUPlan(target, fwdParams, fwdOptions,
                                       fwdPlan);
      assert(options.pass == Pass::FC_TRAINING_BWD);
      return getFullyConnectedBwdPlan(fwdPlan);
    }
  }
  Plan plan;
  Cost cost;
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cacheImpl) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  PlanningCacheImpl::Key key(params, options);
  if (!tempCache.get()) {
    auto &plans = cacheImpl->plans;
    auto match = plans.find(key);
    if (match != plans.end()) {
      return *match->second;
    }
  }

  std::vector<std::pair<PlanningCacheImpl::Key, Plan>> plansToCache;
  std::tie(plan, cost) = runPlanner(params, options, target,
                                    &cacheImpl->cycleEstimation,
                                    &plansToCache);
  if (!tempCache.get()) {
    plansToCache.emplace_back(key, plan);
    auto &plans = cacheImpl->plans;
    for (const auto &entry : plansToCache) {
      auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(entry.second)));
      plans.emplace(entry.first, std::move(pPlan));
    }
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
  // TODO: handle {batchSplit,outChanSplit,inChanSplit}.serial
  constrainVariable(m, vars.batchSplit.parallel,
                    partition.batchSplit.parallel);
  constrainVariable(m, vars.outChanSplit.parallel,
                    partition.outChanSplit.parallel);
  constrainVariable(m, vars.inChanSplit.parallel,
                    partition.inChanSplit.parallel);
  constrainVariable(m, vars.convGroupSplit, partition.convGroupSplit);
}

/// Estimate the cost of a convololution. This is not used by poplibs/enigma.
std::pair<std::uint64_t, std::uint64_t>
estimateConvCost(const poplar::Target &target,
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
      poplibs::getTileHierarchy(target,
                                options.numIPUs,
                                options.tilesPerIPU,
                                perLevelExchangeBytesPerCycle);
  assert(perLevelExchangeBytesPerCycle.size() ==
         plan.partitions.size());
  auto objective = PlanningObjective::minimizeCycles();
  ConvVertexType convVertexType(plan.method,
                                params.inputType,
                                params.outputType,
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
  popsolver::Variable cycles, tempBytes;
  constructModel(target, plan.transforms, plan.types, hierarchy,
                 perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType,
                 params, plan.isJointPlan, highestCost, objective,
                 &cacheImpl->cycleEstimation, options, m, partitionVars, cycles,
                 tempBytes);
  const auto numLevelsOfHierarchy = plan.partitions.size();
  assert(partitionVars.size() == numLevelsOfHierarchy);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    constrainPartitionVars(m, partitionVars[level], plan.partitions[level]);
  }
  popsolver::Solution s;
  s = m.minimize(cycles);
  if (!s.validSolution()) {
    return {highestCost.cycles, highestCost.tileTempMemory};
  }
  return {s[cycles], s[tempBytes]};
}

} // end namespace conv
