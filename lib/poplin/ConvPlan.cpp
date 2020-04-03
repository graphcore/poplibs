// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "ConvPlan.hpp"
#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include "ConvReducePlan.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvValidation.hpp"
#include "PerformanceEstimation.hpp"
#include "poplar/Graph.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/print.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Convolution.hpp"
#include "poputil/exceptions.hpp"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/parallel_for.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <popsolver/Model.hpp>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>

using namespace poplibs_support;

namespace hash_tuple {
template <typename TT> struct hash {
  size_t operator()(TT const &tt) const { return std::hash<TT>()(tt); }
};

template <class T> inline void hash_combine(std::size_t &seed, T const &v) {
  seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename TT> struct hash<std::vector<TT>> {
  size_t operator()(const std::vector<TT> &tt) const {
    size_t hash = 0;
    for (const auto e : tt)
      hash_combine(hash, e);
    return hash;
  }
};

namespace details {
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
  void operator()(size_t &seed, Tuple const &tuple) const {
    HashValueImpl<Tuple, Index - 1>{}(seed, tuple);
    hash_combine(seed, std::get<Index>(tuple));
  }
};
template <class Tuple> struct HashValueImpl<Tuple, 0> {
  void operator()(size_t &seed, Tuple const &tuple) const {
    hash_combine(seed, std::get<0>(tuple));
  }
};
} // namespace details

template <typename... TT> struct hash<std::tuple<TT...>> {
  size_t operator()(std::tuple<TT...> const &tt) const {
    size_t seed = 0;
    details::HashValueImpl<std::tuple<TT...>>{}(seed, tt);
    return seed;
  }
};
} // namespace hash_tuple

namespace poplin {

namespace {

// constraint variables that represent how each item is split for a particular
// level in the hierarchy.
struct PartitionVariables {
  // indexed by field dimension.
  std::vector<popsolver::Variable> fieldSplit;
  popsolver::Variable batchSplit;
  Split<popsolver::Variable> outChanSplit;
  // indexed by kernel dimension.
  std::vector<popsolver::Variable> kernelSplit;
  popsolver::Variable inChanSplit;
  popsolver::Variable convGroupSplit;
  std::vector<unsigned> fieldGrainSize;

  unsigned convGroupGrainSize;
  unsigned inChanGrainSize;
  unsigned outChanGrainSize;
};

// constraint variables that specify the grain sizes of each dimension.
struct ConvSizeVariables {
  // indexed by field dimension.
  std::vector<popsolver::Variable> numFieldGrains;
  popsolver::Variable batchSize;
  // indexed by kernel dimension.
  std::vector<popsolver::Variable> kernelSize;

  popsolver::Variable numConvGroupGrains;
  popsolver::Variable numInChanGrains;
  popsolver::Variable numOutChanGrains;
};

// a description of a (sub-)convolution at a particular level in the hierarchy.
template <typename T> struct ConvSize {
  T convGroupSize;
  T batchSize;
  std::vector<T> fieldSize;
  std::vector<T> kernelSize;
  T inChanSize;
  T outChanSize;
};

class ExchangeEstimator {
  // Exchange bytes per cycle is given as a floating point value but the
  // constaint solver only supports unsigned integer variables. To reduce
  // quantization error in the calculation of the number of cycles we multiply
  // both the divisor (exchange bytes per cycle) and the dividend (the number of
  // bytes) by this scaling factor. Larger values of the scaling factor reduce
  // the quantization error but reduce the maximum number of bytes that can
  // be exchanged before running into the limits of the data type used to store
  // it.
  constexpr static unsigned exchangeBytesScalingFactor = 16u;

public:
  ExchangeEstimator(popsolver::Model &m, const poplar::Target &target,
                    const std::vector<double> &perLevelExchangeBytesPerCycle,
                    const unsigned numLevelsOfHierarchy,
                    const std::vector<PartitionVariables> &partitionVars,
                    const Plan::LinearizeTileOrder linearizeTileOrder)
      : m(m), target(target), numLevelsOfHierarchy(numLevelsOfHierarchy) {
    for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
      const auto scaledBytesPerCycle = getScaledExchangeBytesPerCycle(
          m, perLevelExchangeBytesPerCycle[level], exchangeBytesScalingFactor);

      perLevelScaledExchangeBytesPerCycle.push_back(scaledBytesPerCycle);
      perLevelScaledExchangeBytesPerCycleVar.push_back(
          m.addConstant(scaledBytesPerCycle));
    }

    const unsigned ipuLevel = numLevelsOfHierarchy - 2;
    scaledInputElementBytesPerCycle =
        perLevelScaledExchangeBytesPerCycleVar[ipuLevel];

    // when we lay the data out on the tiles (assuming the standard linearlize
    // tile order) we make the grouped output channels the innermost dimension.
    // this means that consecutive output channels will be distributed across
    // consecutive tiles. this is advantageous because when we parallel split by
    // output channels we need to broadcast out the same input elements to these
    // tiles. therefore the tiles that receive the same input elements will be
    // next to each other and therefore part of the same super tile. this
    // enables a higher bandwidth for receiving as both tiles can receive the
    // same data in the same cycle. we teach the planner about this here so that
    // it will bias splits towards making this happen and therefore produce
    // faster convolutions. for the implementation side of this see the function
    // `linearizeConvIndices` in Convolution.cpp
    //
    // it is worth mentioning that this decision to share inputs rather than
    // weights is arbitrary -- in the future we may want to let the planner
    // decide which is the innermost dimension and therefore gets a faster
    // exchange speed.
    if (target.supportsExchangeBusSharing() &&
        linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD) {
      const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();

      // don't care about the serial split here as that does not change the
      // tiles that the input elements are mapped to.
      const auto outChanSplit = partitionVars[ipuLevel].outChanSplit.parallel;
      const auto multiplier =
          m.call({outChanSplit}, [tilesPerSuperTile](const auto &values) {
            return values[0] % tilesPerSuperTile == 0 ? 2 : 1;
          });

      scaledInputElementBytesPerCycle =
          m.product({scaledInputElementBytesPerCycle, multiplier});
    }
  }

  popsolver::Variable
  getInputElementCycles(const popsolver::Variable numInputElements,
                        const poplar::Type inputElementType,
                        const unsigned level,
                        const std::string &debugName = "") const {
    const auto scaledInputElementSize = m.addConstant(
        target.getTypeSize(inputElementType) * exchangeBytesScalingFactor);

    const auto scaledInputElementBytes =
        m.product({numInputElements, scaledInputElementSize});

    if (level + 2 == numLevelsOfHierarchy) {
      return m.ceildiv(scaledInputElementBytes, scaledInputElementBytesPerCycle,
                       debugName);
    } else {
      return m.ceildiv(scaledInputElementBytes,
                       perLevelScaledExchangeBytesPerCycleVar[level],
                       debugName);
    }
  }

  popsolver::Variable getCycles(const popsolver::Variable numElements,
                                const poplar::Type elementType,
                                const unsigned level,
                                const std::string &debugName = "") const {
    const auto scaledSize = m.addConstant(target.getTypeSize(elementType) *
                                          exchangeBytesScalingFactor);

    const auto scaledElementBytes = m.product({numElements, scaledSize});
    return m.ceildiv(scaledElementBytes,
                     perLevelScaledExchangeBytesPerCycleVar[level], debugName);
  }

  unsigned getCycles(unsigned numElements, const poplar::Type elementType,
                     unsigned level) const {
    const unsigned scaledSize =
        target.getTypeSize(elementType) * exchangeBytesScalingFactor;
    const auto scaledElementBytes = numElements * scaledSize;
    return ceildiv(scaledElementBytes,
                   perLevelScaledExchangeBytesPerCycle[level]);
  }

private:
  static unsigned getScaledExchangeBytesPerCycle(popsolver::Model &m,
                                                 double exchangeBytesPerCycle,
                                                 unsigned scaleFactor) {
    auto scaledExchangeBytesPerCycle =
        std::round(exchangeBytesPerCycle * scaleFactor);
    // Ensure scaled bytes per cycle is at least one to avoid divide by zero
    // errors.
    scaledExchangeBytesPerCycle = std::max(1.0, scaledExchangeBytesPerCycle);
    // Saturate to the half the maximum unsigned integer value (we avoid the
    // maximum value to avoid range problems with the intermediate variables
    // used to implement ceildiv).
    scaledExchangeBytesPerCycle =
        std::min(scaledExchangeBytesPerCycle,
                 static_cast<double>(std::numeric_limits<unsigned>::max() / 2));
    return static_cast<unsigned>(scaledExchangeBytesPerCycle);
  }

  popsolver::Model &m;
  const poplar::Target &target;
  unsigned numLevelsOfHierarchy;
  std::vector<unsigned> perLevelScaledExchangeBytesPerCycle;
  std::vector<popsolver::Variable> perLevelScaledExchangeBytesPerCycleVar;

  // input elements can sometimes benefit from a fast bandwidth. see comment
  // in the constructor about why this is the case.
  popsolver::Variable scaledInputElementBytesPerCycle;
};

} // End anonymous namespace

std::uint64_t getNumberOfMACs(const ConvParams &params) {
  std::uint64_t numMACs = params.getNumConvGroups() * params.getBatchSize() *
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
      auto outRange =
          getOutputRangeForKernelIndex(dim, {0, outputSize}, k, params);
      auto outRangeSize = outRange.second - outRange.first;
      fieldMACs += (outRangeSize + MACStride - 1) / MACStride;
    }
    numMACs *= fieldMACs;
  }
  return numMACs;
}

// A simple function to memoize other functions. Any recursive calls
// with the function are non memoized
template <typename Ret, typename... Args> class Memo {
  using Key = std::tuple<typename std::remove_reference<Args>::type...>;

public:
  tbb::concurrent_unordered_map<Key, Ret, hash_tuple::hash<Key>> table;
  Ret (*fn)(Args...);

public:
  Memo(Ret (*fn)(Args...)) : fn(fn) {}
  Ret operator()(Args... args) {
    const auto key = std::make_tuple(args...);
    const auto match = table.find(key);
    if (match == table.end()) {
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
  void clearTable() { table.clear(); }
};

template <typename Ret, typename... Args>
static Memo<Ret, Args...> memoize(Ret (*fn)(Args...)) {
  return Memo<Ret, Args...>(fn);
}

static unsigned getNumConvUnits(bool floatActivations, bool floatPartial,
                                const poplar::Target &target) {
  if (floatActivations) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  } else {
    return floatPartial ? target.getFp16InFp32OutConvUnitsPerTile()
                        : target.getFp16InFp16OutConvUnitsPerTile();
  }
}

struct ConvVertexType {
  Plan::Method method;
  poplar::Type inputType;
  poplar::Type partialType;

  unsigned convGroupsPerGroup;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;

  // TODO: these variables are only valid for certain methods, it might be
  // better to use a variant here instead.
  //
  // The width of the kernel that slides over the input. Only 4 is currently
  // supported in the software but the SLIC engine also supports 3.
  unsigned slicWindowWidth;
  // Number of engines enabled. Allowed options: 4 or 8
  unsigned numConvUnitsRequired;

  ConvVertexType(Plan::Method method, poplar::Type inputType,
                 poplar::Type outputType, poplar::Type partialType,
                 unsigned convGroupsPerGroup, unsigned inChansPerGroup,
                 unsigned partialChansPerGroup, unsigned slicWindowWidth,
                 unsigned numConvUnitsRequired)
      : method(method), inputType(inputType), partialType(partialType),
        convGroupsPerGroup(convGroupsPerGroup),
        inChansPerGroup(inChansPerGroup),
        partialChansPerGroup(partialChansPerGroup),
        slicWindowWidth(slicWindowWidth),
        numConvUnitsRequired(numConvUnitsRequired) {}
};

static const char *asString(Plan::Method m) {
  switch (m) {
  case Plan::Method::AMP:
    return "AMP";
  case Plan::Method::SLIC:
    return "SLIC";
  case Plan::Method::MAC:
    return "MAC";
  case Plan::Method::OUTER_PRODUCT:
    return "OUTER_PRODUCT";
  }
  POPLIB_UNREACHABLE();
}

std::ostream &operator<<(std::ostream &os, const Partition &p) {
  // T10408: Splitting the batch and in channel dimensions serially has not been
  // implemented yet so we don't bother printing them out for now.
  os << "  Partition: fieldSplit            ";
  printContainer(p.fieldSplit, os);
  os << "\n"
     << "             batchSplit            " << p.batchSplit << "\n"
     << "             outChanSplit.serial   " << p.outChanSplit.serial << "\n"
     << "             outChanSplit.parallel " << p.outChanSplit.parallel << "\n"
     << "             kernelSplit           ";
  printContainer(p.kernelSplit, os);
  os << "\n"
     << "             inChanSplit           " << p.inChanSplit << "\n"
     << "             convGroupSplit        " << p.convGroupSplit << "\n"
     << "             fieldAxisGrainSize    ";
  printContainer(p.fieldAxisGrainSize, os);
  os << "\n"
     << "             inChanGrainSize       " << p.inChanGrainSize << "\n"
     << "             outChanGrainSize      " << p.outChanGrainSize << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ConvTransform &t) {
  os << "  Transform:\n"
        "        extraFieldDims          "
     << t.extraFieldDims
     << "\n"
        "        dilatePostConv          ";
  printContainer(t.dilatePostConv, os);
  os << "\n"
     << "        swapOperands            "
     << (t.swapOperands ? "true" : "false") << "\n"
     << "        expandDims              ";
  printContainer(t.expandDims, os);
  os << "\n"
     << "        outChanFlattenDims      ";
  printContainer(t.outChanFlattenDims, os);
  os << "\n"
     << "        flattenDims             ";
  printContainer(t.flattenDims, os);
  os << "\n"
     << "        combineConvGroupsFactor       " << t.combineConvGroupsFactor
     << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ConvTypes &t) {
  os << "  Types: partialType        " << t.partialType << "\n";
  os << "         resultType         " << t.resultType << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Plan::Method &m) {
  os << asString(m);
  return os;
}

std::istream &operator>>(std::istream &is, Plan::Method &m) {
  std::string token;
  is >> token;
  if (token == "MAC") {
    m = Plan::Method::MAC;
  } else if (token == "AMP") {
    m = Plan::Method::AMP;
  } else if (token == "SLIC") {
    m = Plan::Method::SLIC;
  } else if (token == "OUTER_PRODUCT") {
    m = Plan::Method::OUTER_PRODUCT;
  } else {
    throw poputil::poplibs_error("Unrecognised convolution method '" + token +
                                 "'");
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, const Plan &p) {
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
  os << "        convGroupsPerGroup      " << p.convGroupsPerGroup << "\n"
     << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n"
     << "        isJointPlan             " << p.isJointPlan << "\n";
  return os;
}

static std::uint64_t getConvPartialnx1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    const std::vector<unsigned> &kernelShape, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatWeights, bool floatPartials,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride) {
  const auto kernelElements = product(kernelShape);
  const auto partition = partitionConvPartialByWorker(
      batchElements, outShape, numWorkerContexts, inputDilation, stride);

  // use conv nx1 vertex
  // workList is indexed by [context][numKernelPositions][numPartitions]
  std::vector<std::vector<std::vector<unsigned>>> workList;
  const unsigned positionsOuter = ceildiv(kernelShape[0], filterHeight);
  const unsigned numKernelPositions =
      (positionsOuter * kernelElements / kernelShape[0]);
  const auto outStrideX =
      inputDilation.back() / gcd(inputDilation.back(), stride.back());
  for (unsigned context = 0; context < numWorkerContexts; ++context) {
    workList.emplace_back();
    for (auto k = 0U; k != numKernelPositions; ++k) {
      workList.back().emplace_back();
      for (const auto &partialRow : partition[context]) {
        const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
        const auto numFieldPos = ceildiv(workerOutWidth, outStrideX);
        if (numFieldPos) {
          workList.back().back().push_back(numFieldPos);
        }
      }
    }
  }
  const auto kernelOuterElems = numKernelPositions / positionsOuter;
  const auto kernelInnerElems = positionsOuter;

  return getConvPartialnx1SupervisorCycleInnerLoopEstimate(
      workList, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, convUnitInputLoadElemsPerCycle, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights,
      floatPartials);
}

static std::uint64_t getConvPartial1x1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials, bool zeroPartials) {
  assert(inputDilation == stride);
  std::vector<std::vector<PartialRow>> partition = partitionConvPartialByWorker(
      batchElements, outShape, numWorkerContexts, inputDilation, stride);
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
  return getConvPartial1x1SupervisorInnerLoopCycleEstimate(
      worklist, numWorkerContexts, numConvUnits, zeroPartials, floatActivations,
      floatPartials);
}

static std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, floatActivations, floatPartials, true);
}

static std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, floatActivations, floatPartials, false);
}

static std::uint64_t getConvPartialSlicInnerLoopCycles(
    bool implicitZeroing, unsigned batchElements,
    const std::vector<unsigned> &outShape, unsigned numWorkerContexts,
    unsigned slicWindowWidth, bool floatActivations, bool floatPartials) {
  // SLIC doesn't support output striding and input dilation.
  const std::vector<unsigned> inputDilation(outShape.size(), 1);
  const auto &outputStride = inputDilation;

  const auto partition = partitionConvPartialByWorker(
      batchElements, outShape, numWorkerContexts, inputDilation, outputStride);
  std::vector<std::vector<unsigned>> worklist(numWorkerContexts);
  for (unsigned context = 0; context != numWorkerContexts; ++context) {
    for (const auto &partialRow : partition[context]) {
      const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
      if (workerOutWidth == 0) {
        continue;
      }

      worklist[context].push_back(workerOutWidth);
    }
  }

  return getConvPartialSlicSupervisorCycleInnerLoopEstimate(
      worklist, numWorkerContexts, slicWindowWidth, floatActivations,
      floatPartials, implicitZeroing);
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

static std::uint64_t estimateConvReduceCycles(
    unsigned outputSize, unsigned reductionDepth, bool floatOutput,
    bool floatPartials, unsigned numWorkers, unsigned dataPathWidth,
    unsigned partialsVectorWidth, unsigned outputVectorWidth,
    bool enableMultiStageReduce) {
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
  auto reductionPlan =
      getMultiStageReducePlan(reductionDepth, enableMultiStageReduce);
  std::uint64_t cycles = 0;

  unsigned remainingDepth = reductionDepth;
  // Output size depends on the depth used in the reduction
  unsigned outputSizeThisStage = outputSize * reductionDepth;
  for (auto d : reductionPlan) {
    const auto depthThisStage = (remainingDepth + d - 1) / d;
    outputSizeThisStage =
        (outputSizeThisStage + depthThisStage - 1) / depthThisStage;
    cycles += getReduceCycleEstimate(outputSizeThisStage, depthThisStage,
                                     dataPathWidth, floatOutput, floatPartials,
                                     numWorkers);
    remainingDepth = (remainingDepth + depthThisStage - 1) / depthThisStage;
  }

  if (remainingDepth > 1) {
    outputSizeThisStage =
        (outputSizeThisStage + remainingDepth - 1) / remainingDepth;
    cycles += getReduceCycleEstimate(outputSizeThisStage, remainingDepth,
                                     dataPathWidth, floatOutput, floatPartials,
                                     numWorkers);
  }
  return cycles;
}

static std::uint64_t estimateZeroSupervisorCycles(unsigned fieldSize,
                                                  unsigned numOutGroups,
                                                  unsigned numConvGroups,
                                                  unsigned outChansPerGroup,
                                                  unsigned dataPathWidth,
                                                  unsigned numWorkerContexts) {
  std::vector<unsigned> zeroWorkList;
  zeroWorkList.reserve(numWorkerContexts);

  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    zeroWorkList.push_back(
        (fieldSize * outChansPerGroup + numWorkerContexts - 1) /
        numWorkerContexts);
  }
  return getZeroSupervisorVertexCycleEstimate(
      zeroWorkList, numOutGroups * numConvGroups, dataPathWidth,
      numWorkerContexts, true);
}

static std::uint64_t estimateConvPartialHorizontalMacInnerLoopCycles(
    unsigned numOutRows, unsigned tileOutWidth, unsigned outputStrideX,
    unsigned tileKernelHeight, unsigned tileKernelWidth, unsigned numWorkers,
    bool floatActivations, unsigned inChansPerGroup, unsigned outChansPerGroup,
    unsigned dataPathWidth);

class PlanningCacheImpl {
public:
  struct Key {
    CanonicalConvParams convParams;
    ConvOptions options;
    Key(CanonicalConvParams params, ConvOptions options)
        : convParams(std::move(params)), options(std::move(options)) {}
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
    decltype(memoize(estimateConvReduceCycles)) mEstimateConvReduceCycles;
    decltype(memoize(getNumberOfMACs)) mGetNumberOfMACs;
    decltype(
        memoize(estimateZeroSupervisorCycles)) mEstimateZeroSupervisorCycles;
    decltype(memoize(getConvPartialSlicSupervisorCycleOuterLoopEstimate))
        mGetConvPartialSlicSupervisorCycleOuterLoopEstimate;
    decltype(memoize(
        getConvPartialSlicInnerLoopCycles)) mGetConvPartialSlicInnerLoopCycles;

    CycleEstimationImpl()
        : mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing(
              getConvPartial1x1InnerLoopCycleEstimateWithZeroing),
          mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
              getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing),
          mGetConvPartialnx1InnerLoopCycleEstimate(
              getConvPartialnx1InnerLoopCycleEstimate),
          mEstimateConvPartialHorizontalMacInnerLoopCycles(
              estimateConvPartialHorizontalMacInnerLoopCycles),
          mEstimateConvReduceCycles(estimateConvReduceCycles),
          mGetNumberOfMACs(getNumberOfMACs),
          mEstimateZeroSupervisorCycles(estimateZeroSupervisorCycles),
          mGetConvPartialSlicSupervisorCycleOuterLoopEstimate(
              getConvPartialSlicSupervisorCycleOuterLoopEstimate),
          mGetConvPartialSlicInnerLoopCycles(
              getConvPartialSlicInnerLoopCycles) {}
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

template <typename T> struct Estimates {
  Estimates() = default;
  Estimates(const T totalCycles, const T totalTempBytes)
      : totalCycles(totalCycles), totalTempBytes(totalTempBytes) {}

  // the two values we support minimizing on.
  T totalCycles;
  T totalTempBytes;

  // extra information that can be used for logging.
  T rearrangeBeforeSliceCycles;
  T dynamicSliceCycles;
  T transformCycles;
  T exchangeCycles;
  T tileLevelTransformCycles;
  T partialCalcCycles;
  T reduceCycles;
  T dynamicUpdateCycles;

  T rearrangeBeforeSliceTempBytes;
  T rearrangeBeforeSliceTempDuringRearrangeBytes;
  T transformTempBytes;
  T tileLevelTransformTempBytes;
  T convTempBytes;
  T reduceTempBytes;
};

using Cost = Estimates<unsigned>;

inline bool operator==(Cost a, Cost b) {
  return a.totalCycles == b.totalCycles && a.totalTempBytes == b.totalTempBytes;
}

inline bool operator!=(Cost a, Cost b) { return !(a == b); }

std::ostream &operator<<(std::ostream &os, const Cost &c) {
  os << "Cost{cycles=" << c.totalCycles << ", memory=" << c.totalTempBytes
     << "}";
  return os;
}

class PlanningObjective {
public:
  enum Type { MINIMIZE_CYCLES, MINIMIZE_TILE_TEMP_MEMORY };

private:
  Type type;
  unsigned cyclesBound = std::numeric_limits<unsigned>::max();
  unsigned tileTempMemoryBound = std::numeric_limits<unsigned>::max();
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
  unsigned getCyclesBound() const { return cyclesBound; }
  unsigned getTileTempMemoryBound() const { return tileTempMemoryBound; }
  Type getType() const { return type; }
  bool lowerCost(Cost a, Cost b) const {
    bool aCyclesOutOfBounds = a.totalCycles >= cyclesBound;
    bool bCyclesOutOfBounds = b.totalCycles >= cyclesBound;
    bool aMemoryOutOfBounds = a.totalTempBytes >= tileTempMemoryBound;
    bool bMemoryOutOfBounds = b.totalTempBytes >= tileTempMemoryBound;
    switch (type) {
    case MINIMIZE_CYCLES:
      return std::tie(aCyclesOutOfBounds, aMemoryOutOfBounds, a.totalCycles,
                      a.totalTempBytes) <
             std::tie(bCyclesOutOfBounds, bMemoryOutOfBounds, b.totalCycles,
                      b.totalTempBytes);
    case MINIMIZE_TILE_TEMP_MEMORY:
      return std::tie(aMemoryOutOfBounds, aCyclesOutOfBounds, a.totalTempBytes,
                      a.totalCycles) <
             std::tie(bMemoryOutOfBounds, bCyclesOutOfBounds, b.totalTempBytes,
                      b.totalCycles);
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
static unsigned getStartTile(const poplar::Target &target,
                             const ConvParams &params,
                             const ConvOptions &options, bool isJointPlan) {
  // Use a start tile of 0 for joint plans to avoid the risk of exchanging
  // weights. TODO: T12875 Investigate whether this is necessary.
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

  const auto numTiles = target.getNumTiles();
  const auto numEvenTiles =
      std::max(1U, numTiles / options.startTileMultiplier);
  return (std::hash<ConvParams>()(params) % numEvenTiles) *
         options.startTileMultiplier;
}

static unsigned getConvUnitsPerTile(const poplar::Target &target,
                                    bool floatActivations, bool floatPartials) {
  if (floatActivations) {
    return floatPartials ? target.getFp32InFp32OutConvUnitsPerTile() : 0;
  }
  return floatPartials ? target.getFp16InFp32OutConvUnitsPerTile()
                       : target.getFp16InFp16OutConvUnitsPerTile();
}

static bool canUseConvolutionInstruction(bool floatActivations,
                                         bool floatPartials,
                                         const poplar::Target &target) {
  if (getConvUnitsPerTile(target, floatActivations, floatPartials) == 0) {
    return false;
  }

  if (floatActivations) {
    // the case where activations are float but partials are not is handled by
    // getConvUnitsPerTile above.
    assert(floatPartials);
  }

  return true;
}

static bool canUseConvolutionInstruction(bool floatActivations,
                                         bool floatPartials,
                                         unsigned inChansPerGroup,
                                         unsigned numConvUniteRequired,
                                         unsigned outChansPerGroup,
                                         const poplar::Target &target) {
  if (!canUseConvolutionInstruction(floatActivations, floatPartials, target)) {
    return false;
  }
  unsigned usedWeightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);
  // Any other configuration than 4 uses full set of weights hence no need for
  // extra constraint
  if (numConvUniteRequired == 4) {
    usedWeightsPerConvUnit =
        (usedWeightsPerConvUnit * numConvUniteRequired) /
        getConvUnitsPerTile(target, floatActivations, floatPartials);
  }
  if (usedWeightsPerConvUnit % inChansPerGroup != 0) {
    return false;
  }
  // Output channels grouping shall be great or equal to number of engines
  if ((outChansPerGroup % numConvUniteRequired) != 0) {
    return false;
  }
  // Check we can use aligned loads.
  if ((inChansPerGroup * (floatActivations ? 32 : 16)) %
          target.getDataPathWidth() !=
      0) {
    return false;
  }
  return true;
}

static unsigned getMaxInputRangeSize(unsigned outputRangeSize, unsigned dim,
                                     const ConvParams &params,
                                     unsigned tileKernelSize) {
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

static std::uint64_t estimateConvPartialHorizontalMacInnerLoopCycles(
    unsigned numOutRows, unsigned tileOutWidth, unsigned outputStrideX,
    unsigned tileKernelHeight, unsigned tileKernelWidth, unsigned numWorkers,
    bool floatActivations, unsigned inChansPerGroup, unsigned outChansPerGroup,
    unsigned dataPathWidth) {
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numOutRows);
  unsigned numPartRows = numOutRows * rowSplitFactor;
  const auto maxPartRows = (numPartRows + numWorkers - 1) / numWorkers;
  const auto workerWholeRows = maxPartRows / rowSplitFactor;
  const auto workerPartRows = maxPartRows % rowSplitFactor;
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
        auto convSize = workerPartRows *
                        (wholeRowConvSize + rowSplitFactor - 1) /
                        rowSplitFactor;
        workerPartitions.back().back().push_back(convSize);
      }
    }
  }

  return getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
      workerPartitions, kernelSize, inChansPerGroup, outChansPerGroup,
      numWorkers, floatActivations);
}

static bool canUseConvPartial1x1Vertex(
    const ConvParams &params,
    const std::unordered_set<unsigned> &transformedDims,
    const std::vector<unsigned> &transformedInputDilation,
    const std::vector<unsigned> &transformedOutputStride,
    unsigned convUnitWeightHeight,
    const std::vector<unsigned> &tileKernelShape) {
  if (convUnitWeightHeight != 1) {
    return false;
  }

  if (transformedInputDilation != transformedOutputStride) {
    return false;
  }

  const auto tileKernelElements = product(tileKernelShape);
  if (tileKernelElements != 1) {
    return false;
  }

  // To save memory the 1x1 vertex only supports a single worklist therefore
  // all dimensions up-to the innermost spatial dimension must be singular (not
  // including the group dimension as that is looped over in the supervisor part
  // of this vertex). If they aren't then additional worklist items are needed
  // for each one. This matches the logic in `createConvPartialAmpVertex` which
  // switches to the nx1 vertex if a context has more than one partition.
  assert(!params.inputFieldShape.empty());
  const auto isNotOne = [](const auto &x) { return x != 1; };
  if (params.batchSize != 1 ||
      std::any_of(std::begin(params.inputFieldShape),
                  std::end(params.inputFieldShape) - 1, isNotOne)) {
    return false;
  }

  // We can only use the 1x1 vertex if every output value is written. It may be
  // the case every output value is written on some tiles but not others - we
  // return false in this case since we are interested in the worse case
  // and we assume the nx1 vertex is always slower.
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (transformedDims.count(dim)) {
      continue;
    }

    std::pair<unsigned, unsigned> outputRange = {0, params.getOutputSize(dim)};
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

// mapping between ConvSizeVariables and the std::vector<unsigned> that is
// passed to the callback for an m.call constraint.
template <typename T> class ConvSizeVariablesVector {
  // offsets for all of the variables.
  constexpr static unsigned batchSizeOffset = 0;
  constexpr static unsigned numConvGroupGrainsOffset = 1;
  constexpr static unsigned numInChanGrainsOffset = 2;
  constexpr static unsigned numOutChanGrainsOffset = 3;
  constexpr static unsigned numFieldGrainsOffset = 4;

public:
  ConvSizeVariablesVector(const ConvSizeVariables &convSizeVars)
      : values(numFieldGrainsOffset),
        numFieldDims(convSizeVars.numFieldGrains.size()) {
    assert(numFieldDims == convSizeVars.kernelSize.size());
    values.at(batchSizeOffset) = convSizeVars.batchSize;
    values.at(numConvGroupGrainsOffset) = convSizeVars.numConvGroupGrains;
    values.at(numInChanGrainsOffset) = convSizeVars.numInChanGrains;
    values.at(numOutChanGrainsOffset) = convSizeVars.numOutChanGrains;

    values.insert(std::end(values), std::begin(convSizeVars.numFieldGrains),
                  std::end(convSizeVars.numFieldGrains));
    values.insert(std::end(values), std::begin(convSizeVars.kernelSize),
                  std::end(convSizeVars.kernelSize));
  }

  ConvSizeVariablesVector(std::vector<T> values, unsigned numFieldDims)
      : values(std::move(values)), numFieldDims(numFieldDims) {}

  operator const std::vector<T> &() const { return values; }

  T batchSize() const { return values.at(batchSizeOffset); }
  T numConvGroupGrains() const { return values.at(numConvGroupGrainsOffset); }
  T numInChanGrains() const { return values.at(numInChanGrainsOffset); }
  T numOutChanGrains() const { return values.at(numOutChanGrainsOffset); }

  poplar::ArrayRef<T> numFieldGrains() const {
    return {values.data() + numFieldGrainsOffset, numFieldDims};
  }

  poplar::ArrayRef<T> kernelSize() const {
    return {values.data() + numFieldGrainsOffset + numFieldDims, numFieldDims};
  }

private:
  std::vector<T> values;
  unsigned numFieldDims;
};

static ConvSize<unsigned>
makeConvSize(std::vector<unsigned> values,
             const std::vector<unsigned> &fieldGrainSize,
             const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
             const unsigned outChansPerGroup) {
  const unsigned numFieldDims = fieldGrainSize.size();
  ConvSizeVariablesVector<unsigned> convSizeVarsVector(std::move(values),
                                                       numFieldDims);

  ConvSize<unsigned> convSize;
  convSize.batchSize = convSizeVarsVector.batchSize();
  convSize.outChanSize =
      convSizeVarsVector.numOutChanGrains() * outChansPerGroup;
  convSize.inChanSize = convSizeVarsVector.numInChanGrains() * inChansPerGroup;
  convSize.convGroupSize =
      convSizeVarsVector.numConvGroupGrains() * convGroupsPerGroup;

  const auto numFieldGrains = convSizeVarsVector.numFieldGrains();
  for (unsigned d = 0; d < numFieldDims; ++d) {
    convSize.fieldSize.push_back(numFieldGrains[d] * fieldGrainSize[d]);
  }

  const auto kernelSize = convSizeVarsVector.kernelSize();
  convSize.kernelSize.insert(std::begin(convSize.kernelSize),
                             std::begin(kernelSize), std::end(kernelSize));
  return convSize;
}

static popsolver::Variable addPartialCalcCycleEstimate(
    popsolver::Model &m, const std::vector<unsigned> &fieldGrainSize,
    const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
    const unsigned outChansPerGroup, const ConvSizeVariables &convSizeVars,
    const std::unordered_set<unsigned> &transformedDims,
    const poplar::Target &target, const ConvParams &params,
    poplar::Type partialType, Plan::Method method, unsigned slicWindowWidth,
    unsigned numConvUnitsRequired, const ConvOptions &options,
    PlanningCacheImpl::CycleEstimationImpl *cache) {
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(params.inputType == poplar::HALF || params.inputType == poplar::FLOAT);
  bool floatActivations = params.inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;

  ConvSizeVariablesVector<popsolver::Variable> convSizeVarsVector(convSizeVars);

  auto transformedInputDilation = params.inputTransform.dilation;
  auto transformedOutputStride = params.outputTransform.stride;
  for (const auto dim : transformedDims) {
    transformedInputDilation[dim] = 1;
    transformedOutputStride[dim] = 1;
  }

  auto convUnitInputLoadElemsPerCycle =
      target.getConvUnitInputLoadElemsPerCycle(floatActivations);
  if (!options.use128BitConvUnitLoad) {
    convUnitInputLoadElemsPerCycle /= 2;
  }

  const std::string debugName = "partialCalcCycleEstimate";
  switch (method) {
  default: {
    std::stringstream ss;
    ss << "Unexpected convolution method <" << method << ">";
    throw poputil::poplibs_error(ss.str());
  }
  case Plan::Method::AMP: {
    assert(target.getWeightsPerConvUnit(floatActivations) % inChansPerGroup ==
           0);
    auto numConvUnitsOnIpu =
        getNumConvUnits(floatActivations, floatPartials, target);
    auto weightsPerConvUnit = target.getWeightsPerConvUnit(floatActivations);

    assert(numConvUnitsRequired != 0);
    assert(numConvUnitsOnIpu % numConvUnitsRequired == 0);
    weightsPerConvUnit /= numConvUnitsOnIpu / numConvUnitsRequired;

    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;

    return m.call(
        convSizeVarsVector,
        [&target, fieldGrainSize, convGroupsPerGroup, inChansPerGroup,
         outChansPerGroup, partialType, params, transformedDims,
         transformedInputDilation, transformedOutputStride,
         convUnitWeightHeight, cache, floatActivations,
         convUnitInputLoadElemsPerCycle,
         numConvUnitsRequired](const std::vector<unsigned> &values) {
          const auto convSize =
              makeConvSize(values, fieldGrainSize, convGroupsPerGroup,
                           inChansPerGroup, outChansPerGroup);

          // AMP currently only expects a single convGroup grouping.
          assert(convGroupsPerGroup == 1);

          const auto tileNumInGroups =
              ceildiv(convSize.inChanSize, inChansPerGroup);
          const auto tileNumOutGroups =
              ceildiv(convSize.outChanSize, outChansPerGroup);
          const auto tileNumConvGroups =
              ceildiv(convSize.convGroupSize, convGroupsPerGroup);

          const auto floatPartials = partialType == poplar::FLOAT;

          if (canUseConvPartial1x1Vertex(
                  params, transformedDims, transformedInputDilation,
                  transformedOutputStride, convUnitWeightHeight,
                  convSize.kernelSize)) {
            const auto innerLoopCyclesWithZeroing =
                cache->mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing(
                    convSize.batchSize, convSize.fieldSize,
                    target.getNumWorkerContexts(), numConvUnitsRequired,
                    transformedInputDilation, transformedOutputStride,
                    floatActivations, floatPartials);
            const auto innerLoopCyclesWithoutZeroing =
                cache->mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
                    convSize.batchSize, convSize.fieldSize,
                    target.getNumWorkerContexts(), numConvUnitsRequired,
                    transformedInputDilation, transformedOutputStride,
                    floatActivations, floatPartials);

            return getConvPartial1x1SupervisorOuterLoopCycleEstimate(
                innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
                tileNumConvGroups, tileNumInGroups, tileNumOutGroups,
                outChansPerGroup, convUnitInputLoadElemsPerCycle,
                numConvUnitsRequired,
                target.getConvUnitCoeffLoadBytesPerCycle(), floatActivations,
                floatPartials, target.getNumWorkerContexts());
          }
          const auto zeroCycles = cache->mEstimateZeroSupervisorCycles(
              product(convSize.fieldSize) * convSize.batchSize,
              tileNumOutGroups, tileNumConvGroups, outChansPerGroup,
              target.getDataPathWidth(), target.getNumWorkerContexts());

          const auto innerLoopCycles =
              cache->mGetConvPartialnx1InnerLoopCycleEstimate(
                  convSize.batchSize, convSize.fieldSize, convSize.kernelSize,
                  convUnitWeightHeight, outChansPerGroup,
                  convUnitInputLoadElemsPerCycle, numConvUnitsRequired,
                  target.getConvUnitCoeffLoadBytesPerCycle(),
                  target.getNumWorkerContexts(), floatActivations,
                  floatPartials, transformedInputDilation,
                  transformedOutputStride);
          return getConvPartialnx1SupervisorCycleOuterLoopEstimate(
                     innerLoopCycles, tileNumConvGroups, tileNumOutGroups,
                     tileNumInGroups, outChansPerGroup, numConvUnitsRequired,
                     target.getNumWorkerContexts(), floatActivations,
                     floatPartials) +
                 zeroCycles;
        },
        debugName);
  }
  case Plan::Method::SLIC: {
    return m.call(
        convSizeVarsVector,
        [&target, fieldGrainSize, convGroupsPerGroup, inChansPerGroup,
         outChansPerGroup, transformedInputDilation, transformedOutputStride,
         slicWindowWidth, floatActivations, floatPartials,
         cache](const auto &values) {
          const auto convSize =
              makeConvSize(values, fieldGrainSize, convGroupsPerGroup,
                           inChansPerGroup, outChansPerGroup);

#ifndef NDEBUG
          const auto isOne = [](const auto &x) { return x == 1; };
#endif
          assert(std::all_of(std::begin(transformedOutputStride),
                             std::end(transformedOutputStride), isOne));

          // current vertex requirements
          assert(inChansPerGroup == outChansPerGroup);
          assert(convGroupsPerGroup * inChansPerGroup == 4);

          assert(ceildiv(convSize.inChanSize, inChansPerGroup) == 1);
          assert(ceildiv(convSize.outChanSize, outChansPerGroup) == 1);

          const auto tileNumConvGroups =
              ceildiv(convSize.convGroupSize, convGroupsPerGroup);

          // we process kernel width in 1x4 blocks (rounding up to the nearest
          // multiple of the SLIC kernel width) and then do this for each other
          // kernel dimension.
          const unsigned numWeightBlocks = [&] {
            assert(convSize.kernelSize.size() >= 2);

            // width is the inner-most dimension in kernelSize.
            const unsigned widthDim = convSize.kernelSize.size() - 1;
            const unsigned otherDims =
                product(convSize.kernelSize) / convSize.kernelSize[widthDim];
            return ceildiv(convSize.kernelSize[widthDim], slicWindowWidth) *
                   otherDims;
          }();

          const auto implicitZeroInnerLoopCycles =
              cache->mGetConvPartialSlicInnerLoopCycles(
                  /* implicitZeroing */ true, convSize.batchSize,
                  convSize.fieldSize, target.getNumWorkerContexts(),
                  slicWindowWidth, floatActivations, floatPartials);
          const auto innerLoopCycles =
              cache->mGetConvPartialSlicInnerLoopCycles(
                  /* implicitZeroing */ false, convSize.batchSize,
                  convSize.fieldSize, target.getNumWorkerContexts(),
                  slicWindowWidth, floatActivations, floatPartials);
          const auto weightLoadCycles =
              getConvPartialSlicSupervisorCycleWeightLoadEstimate(
                  convGroupsPerGroup, inChansPerGroup,
                  target.getNumWorkerContexts(), slicWindowWidth);

          return cache->mGetConvPartialSlicSupervisorCycleOuterLoopEstimate(
              implicitZeroInnerLoopCycles, innerLoopCycles, weightLoadCycles,
              tileNumConvGroups, numWeightBlocks, slicWindowWidth,
              floatActivations, floatPartials);
        });
  }
  case Plan::Method::MAC: {
    const auto outputStrideX = transformedInputDilation.back();
    return m.call(
        convSizeVarsVector,
        [&target, fieldGrainSize, inChansPerGroup, convGroupsPerGroup,
         outChansPerGroup, transformedInputDilation, cache, outputStrideX,
         floatActivations](const std::vector<unsigned> &values) {
          const auto convSize =
              makeConvSize(values, fieldGrainSize, convGroupsPerGroup,
                           inChansPerGroup, outChansPerGroup);

          // MAC currently only expects a single convGroup grouping.
          assert(convGroupsPerGroup == 1);

          const auto tileNumInGroups =
              ceildiv(convSize.inChanSize, inChansPerGroup);
          const auto tileNumOutGroups =
              ceildiv(convSize.outChanSize, outChansPerGroup);
          const auto tileNumConvGroups =
              ceildiv(convSize.convGroupSize, convGroupsPerGroup);
          const auto tileKernelElements = product(convSize.kernelSize);

          unsigned numActiveOutRows = convSize.batchSize;
          const unsigned numFieldDims = convSize.fieldSize.size();
          for (unsigned dim = 0; dim + 1 < numFieldDims; ++dim) {
            const auto dimActiveRows =
                (convSize.fieldSize[dim] + transformedInputDilation[dim] - 1) /
                transformedInputDilation[dim];
            numActiveOutRows *= dimActiveRows;
          }

          const auto tileKernelWidth = convSize.kernelSize.back();
          const auto tileOutWidth = convSize.fieldSize.back();
          const auto zeroCycles = estimateZeroSupervisorCycles(
              numActiveOutRows * tileOutWidth, tileNumOutGroups,
              tileNumConvGroups, outChansPerGroup, target.getDataPathWidth(),
              target.getNumWorkerContexts());
          const auto innerLoopCycles =
              cache->mEstimateConvPartialHorizontalMacInnerLoopCycles(
                  numActiveOutRows, tileOutWidth, outputStrideX,
                  tileKernelElements / tileKernelWidth, tileKernelWidth,
                  target.getNumWorkerContexts(), floatActivations,
                  inChansPerGroup, outChansPerGroup, target.getDataPathWidth());
          return getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
                     innerLoopCycles, tileNumConvGroups, tileNumInGroups,
                     tileNumOutGroups, target.getNumWorkerContexts(),
                     floatActivations) +
                 zeroCycles;
        },
        debugName);
  } break;
  case Plan::Method::OUTER_PRODUCT: {
    assert(inChansPerGroup == 1);
    const auto numContexts = target.getNumWorkerContexts();
    const auto outputIsFloat = params.outputType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    return m.call(
        convSizeVarsVector,
        [fieldGrainSize, numContexts, convGroupsPerGroup, outChansPerGroup,
         inChansPerGroup, floatActivations, outputIsFloat,
         dataPathWidth](const std::vector<unsigned> &values) {
          const auto convSize =
              makeConvSize(values, fieldGrainSize, convGroupsPerGroup,
                           inChansPerGroup, outChansPerGroup);
          assert(convSize.batchSize == 1);
          assert(convSize.inChanSize == 1);

          // OuterProduct currently only expects a single convGroup grouping.
          assert(convGroupsPerGroup == 1);

          const auto tileNumConvGroups =
              ceildiv(convSize.convGroupSize, convGroupsPerGroup);
          const auto tileOutWidth = convSize.fieldSize.back();
          const auto workerOutWidth = ceildiv(tileOutWidth, numContexts);
          const auto vertexRuntime = getOuterProductCycleEstimate(
              floatActivations || outputIsFloat, workerOutWidth,
              convSize.outChanSize * tileNumConvGroups, outChansPerGroup,
              dataPathWidth);
          return vertexRuntime * numContexts;
        },
        debugName);
  } break;
  }
}

unsigned getMaxMACsPerCyclePerTile(const poplar::Target &target,
                                   poplar::Type partialType,
                                   poplar::Type inputType, Plan::Method method,
                                   unsigned slicWindowWidth) {
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);
  const bool floatActivations = inputType == poplar::FLOAT;
  const bool floatPartials = partialType == poplar::FLOAT;

  auto vectorWidth = target.getVectorWidth(inputType);
  switch (method) {
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT:
    return vectorWidth;
  case Plan::Method::SLIC:
    assert(!floatActivations);
    assert(floatPartials);
    return vectorWidth * slicWindowWidth * 2;
  case Plan::Method::AMP: {
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

static popsolver::Variable addConvTempMemoryEstimate(
    popsolver::Model &m, const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSizes,
    const popsolver::Variable inputsPerTile,
    const popsolver::Variable weightsPerTile,
    const popsolver::Variable partialsPerTile, const poplar::Target &target,
    const ConvParams &params, const std::vector<ConvTypes> &types,
    const Plan::Method method) {
  std::vector<popsolver::Variable> memorySumOperands;
  auto elementBytes = target.getTypeSize(params.inputType);
  auto inputStorage = m.product({m.addConstant(elementBytes), inputsPerTile},
                                "tempConvInputBytes");
  auto weightStorage = m.product({m.addConstant(elementBytes), weightsPerTile},
                                 "tempConvWeightBytes");
  auto partialStorage =
      m.product({m.addConstant(target.getTypeSize(types.back().partialType)),
                 partialsPerTile},
                "tempConvPartialBytes");

  // the SLIC vertex uses an extra temporary buffer of size:
  //    (sizeof(output)/numConvGroupGroups) + 8.
  if (method == Plan::Method::SLIC) {
    const auto buffer =
        m.sum({m.ceildiv(partialStorage, convSizes.back().numConvGroupGrains),
               m.addConstant(200)});

    partialStorage = m.sum({partialStorage, buffer});
  }

  auto convStorage =
      m.sum({inputStorage, weightStorage, partialStorage}, "tempConvBytes");

  // Rearrangements can require both pre- and post-rearranged inputs and/or
  // weights to be required. This may be bigger than the storage need during the
  // convolution.
  return convStorage;
}

// calculates how many zeros are added for padding for the kernel and input
// fields by the equivalent function defined in `Convolution.cpp`

static void
padKernelSpatialDim(popsolver::Model &m, const ConvParams &params,
                    const std::vector<ConvSizeVariables> &transformedSizes,
                    const std::vector<PartitionVariables> &partitionVars,
                    std::vector<popsolver::Variable> &kernelPadding,
                    std::vector<popsolver::Variable> &inputPadding,
                    const unsigned padToMultipleOf, const unsigned dim) {
  assert(dim < kernelPadding.size());
  assert(dim < inputPadding.size());

  if (padToMultipleOf == 1) {
    return;
  }

  assert(transformedSizes.size() >= 2);
  const auto numLevelsOfHierarchy = transformedSizes.size();
  const auto ipuLevel = numLevelsOfHierarchy - 2;

  // Here we need to calculate how much padding (P) is required for the
  // kernel. We do this by taking the size of the kernel dim we want to
  // pad (D) of this sub-convolution and the amount of kernel splits (S)
  // and do the following:
  //
  //  P = (X - max(floor(D, S) % X, ceil(D, S) % X)) % X
  //
  // where X is the multiple we want to pad up to.
  //
  // We do both floor and ceil here and take the max because if the split
  // does not evenly divide the kernel dimension, some tiles will need
  // more padding than others. This max here takes the larger padding
  // number to be used for estimates on all tiles so it may cause the
  // overall cycle/memory estimates to be somewhat pessimistic.
  const auto x = m.addConstant(padToMultipleOf);

  assert(transformedSizes[ipuLevel].kernelSize.size() > dim);
  assert(partitionVars[ipuLevel].kernelSplit.size() > dim);

  // TODO: T12876 There is an added complexity here as either rounding up or
  // down produces the most padding at each level of the hierarchy. Therefore,
  // we need to walk over the entire hierarchy to find the padding required
  // for the lowest level.
  const auto h = transformedSizes[ipuLevel].kernelSize[dim];
  const auto s = partitionVars[ipuLevel].kernelSplit[dim];

  // This is how many elements the kernel size has increased by in
  // the given dimension. To get the number of bytes we need to multiply
  // this number by the number of elements per element of that dimension
  // and the no. of bytes to represent the element type.
  const auto kernelElemsToPadInDim = m.mod(
      m.sub(x, m.max({m.mod(m.floordiv(h, s), x), m.mod(m.ceildiv(h, s), x)})),
      x, "kernelPadding");

  // kernel dilation may result in extra input padding.
  const auto kernelDilation =
      m.addConstant(params.kernelTransform.dilation[dim], "kernelDilation");
  const auto inputElemsToPadInDim = m.product(
      {kernelElemsToPadInDim, kernelDilation}, "extraInputPaddingRows");

  kernelPadding[dim] = m.sum({kernelPadding[dim], kernelElemsToPadInDim});
  inputPadding[dim] = m.sum({inputPadding[dim], inputElemsToPadInDim});
}

popsolver::Variable getDilatedSize(popsolver::Model &m,
                                   popsolver::Variable size,
                                   unsigned dilation) {
  const auto one = m.addConstant(1);
  const auto sizeOrOne = m.max({one, size});

  // dilatedSize = 1 + (size - 1) * dilation
  const auto dilatedSize =
      m.sum({one, m.product({m.sub(sizeOrOne, one), m.addConstant(dilation)})});

  // x = 1 if size != 0 else 0
  const auto x = m.ceildiv(size, sizeOrOne);

  // return dilatedSize if size != 0 else 0
  return m.product({x, dilatedSize});
}

// this function models the function of the same name in Convolution.cpp. we do
// this by using very rough estimates of how many zeros padding or dilation
// needs and deriving memory and cycle costs from those, this doesn't take into
// account anything like grouping or layouts or which copy vertices are
// available which can change the result. we also don't do anything for
// truncation for now. estimating these values more accurately is covered by
// T7132 and once that is done we should use that library here instead.
static void truncateDilateAndPadInput(
    popsolver::Model &m, const ConvParams &params,
    const std::vector<ConvSizeVariables> &transformedSizes,
    const std::vector<PartitionVariables> &partitionVars,
    std::vector<popsolver::Variable> &inputPadding, const unsigned dim) {
  assert(dim < inputPadding.size());

  assert(transformedSizes.size() >= 2);
  const auto numLevelsOfHierarchy = transformedSizes.size();
  const auto ipuLevel = numLevelsOfHierarchy - 2;
  const auto tileLevel = numLevelsOfHierarchy - 1;

  // field size for this dim include any zero padding already applied
  const auto fieldGrainSize =
      m.addConstant(partitionVars[ipuLevel].fieldGrainSize[dim]);
  const auto fieldSize =
      m.sum({m.product({transformedSizes[tileLevel].numFieldGrains[dim],
                        fieldGrainSize}),
             inputPadding[dim]});

  // calculate how many elements are removed by the truncation.
  // TODO T10104: add modelling for truncation.

  // calculate how many zeroes are added by the dilation.
  const auto dilation = params.inputTransform.dilation[dim];
  const auto dilationZeros =
      m.sub(getDilatedSize(m, fieldSize, dilation), fieldSize);
  inputPadding[dim] = m.sum({inputPadding[dim], dilationZeros});

  // calculate how many zeroes are added by the padding.
  const auto padding = params.inputTransform.paddingUpper[dim] +
                       params.inputTransform.paddingLower[dim];
  if (padding != 0) {
    inputPadding[dim] = m.sum({inputPadding[dim], m.addConstant(padding)});
  }
}

// returns a pair of cycles and memory that estimate the cost of applying the
// passed in kernel and input padding. currently uses a very basic of model
// based around the nunber of zeros.
static std::pair<popsolver::Variable, popsolver::Variable>
applyPadding(popsolver::Model &m, const poplar::Target &target,
             const poplar::Type inputType,
             const std::vector<ConvSizeVariables> &transformedSizes,
             const std::vector<PartitionVariables> &partitionVars,
             const ExchangeEstimator &exchangeEstimator,
             const std::vector<popsolver::Variable> &kernelPadding,
             const std::vector<popsolver::Variable> &inputPadding) {
  assert(transformedSizes.size() >= 2);
  const auto numLevelsOfHierarchy = transformedSizes.size();
  const auto ipuLevel = numLevelsOfHierarchy - 2;
  const auto tileLevel = numLevelsOfHierarchy - 1;

  const auto convGroupSize =
      m.product({transformedSizes[tileLevel].numConvGroupGrains,
                 m.addConstant(partitionVars[ipuLevel].convGroupGrainSize)});
  const auto batchSize = transformedSizes[tileLevel].batchSize;
  const auto inChanSize =
      m.product({transformedSizes[tileLevel].numInChanGrains,
                 m.addConstant(partitionVars[ipuLevel].inChanGrainSize)});
  const auto outChanSize =
      m.product({transformedSizes[tileLevel].numOutChanGrains,
                 m.addConstant(partitionVars[ipuLevel].outChanGrainSize)});

  // estimate cycles and temp memory by total number of zeroes from all of
  // the transformations.
  const auto kernelZeros = [&] {
    const auto numKernelDims = transformedSizes[tileLevel].kernelSize.size();

    std::vector<popsolver::Variable> kernelDims;
    std::vector<popsolver::Variable> paddedKernelDims;
    for (unsigned d = 0; d < numKernelDims; ++d) {
      const auto kernelSize = transformedSizes[tileLevel].kernelSize[d];

      kernelDims.push_back(kernelSize);
      paddedKernelDims.push_back(m.sum({kernelSize, kernelPadding[d]}));
    }

    const auto padding = m.sub(m.product(std::move(paddedKernelDims)),
                               m.product(std::move(kernelDims)));
    return m.product({convGroupSize, padding, inChanSize, outChanSize});
  }();

  const auto inputZeros = [&] {
    const auto numFieldDims = transformedSizes[tileLevel].numFieldGrains.size();

    std::vector<popsolver::Variable> fieldDims;
    std::vector<popsolver::Variable> paddedFieldDims;
    for (unsigned d = 0; d < numFieldDims; ++d) {
      const auto fieldGrainSize =
          m.addConstant(partitionVars[ipuLevel].fieldGrainSize[d]);
      const auto fieldSize = m.product(
          {transformedSizes[tileLevel].numFieldGrains[d], fieldGrainSize});

      fieldDims.push_back(fieldSize);
      paddedFieldDims.push_back(m.sum({fieldSize, inputPadding[d]}));
    }

    const auto padding = m.sub(m.product(std::move(paddedFieldDims)),
                               m.product(std::move(fieldDims)));
    return m.product({convGroupSize, batchSize, padding, inChanSize});
  }();

  const auto kernelCycles =
      exchangeEstimator.getCycles(kernelZeros, inputType, ipuLevel);
  const auto inputCycles =
      exchangeEstimator.getInputElementCycles(inputZeros, inputType, ipuLevel);
  const auto extraCycles = m.sum({kernelCycles, inputCycles});

  // we sum the temp memory here as all of these transformations will be
  // alive while the vertex is running.
  const auto elementBytes = m.addConstant(target.getTypeSize(inputType));
  const auto allZeros = m.sum({kernelZeros, inputZeros});
  const auto extraTempBytes = m.product({allZeros, elementBytes});

  return std::make_pair(extraCycles, extraTempBytes);
}

// returns a pair of cycle estimate and temporary memory estimate as well as
// an updated ConvParams with the transformations applied.
static std::pair<popsolver::Variable, popsolver::Variable>
addTileLevelTransformEstimates(
    popsolver::Model &m, const poplar::Target &target, const ConvParams &params,
    poplar::Type partialType, unsigned inChansPerGroup,
    const std::vector<ConvSizeVariables> &transformedSizes,
    const std::vector<PartitionVariables> &partitionVars,
    const ExchangeEstimator &exchangeEstimator, Plan::Method method,
    unsigned slicWindowWidth, unsigned numConvUnitsRequired) {
  const auto numFieldDims = params.kernelShape.size();
  const auto zero = m.addConstant(0u);

  switch (method) {
  case Plan::Method::MAC:
  case Plan::Method::OUTER_PRODUCT: {
    return std::make_pair(zero, zero);
  }
  case Plan::Method::AMP: {
    // the logic in this case is designed to mirror the implementation found
    // in `Convolution.cpp:createConvPartialAmpVertices`
    auto weightsPerConvUnit =
        target.getWeightsPerConvUnit(params.inputType == poplar::FLOAT);
    assert(weightsPerConvUnit % inChansPerGroup == 0);

    const auto numConvUnitsonIpu =
        getNumConvUnits(params.inputType == poplar::FLOAT,
                        partialType == poplar::FLOAT, target);

    assert(numConvUnitsRequired != 0);
    assert(numConvUnitsonIpu % numConvUnitsRequired == 0);

    weightsPerConvUnit /= numConvUnitsonIpu / numConvUnitsRequired;

    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;

    // when we don't have 16 input chans per group then AMP pads the kernel
    // height dimension as well as applying the input transformations of the
    // outer-most spatial dimension, it then uses that dimension so make up for
    // the lack of input channels.
    if (convUnitWeightHeight != 1) {
      std::vector<popsolver::Variable> kernelPadding(numFieldDims, zero);
      std::vector<popsolver::Variable> inputPadding(numFieldDims, zero);

      // TODO: This method currently only calculates the kernel padding.
      // T10104 tracks extending these estimates with the other padding that
      // comes from the transforms (eg. dilation).
      const auto spatialDimToPad = 0;
      padKernelSpatialDim(m, params, transformedSizes, partitionVars,
                          kernelPadding, inputPadding, convUnitWeightHeight,
                          spatialDimToPad);

      return applyPadding(m, target, params.inputType, transformedSizes,
                          partitionVars, exchangeEstimator, kernelPadding,
                          inputPadding);
    } else {
      return std::make_pair(zero, zero);
    }
  }
  case Plan::Method::SLIC: {
    // the logic in this case is designed to mirror the implementation found
    // in `Convolution.cpp:createConvPartialSlicVertex`
    std::vector<popsolver::Variable> kernelPadding(numFieldDims, zero);
    std::vector<popsolver::Variable> inputPadding(numFieldDims, zero);

    // a SLIC kernel requires either a multiple of 1x3 or a multiple of 1x4.
    // for now we only support the 1x4 variant.
    assert(slicWindowWidth == 4);

    // SLIC pads the kernel width dimension which is the innermost spatial dim.
    const unsigned dimToPad = params.kernelShape.size() - 1;
    padKernelSpatialDim(m, params, transformedSizes, partitionVars,
                        kernelPadding, inputPadding, slicWindowWidth, dimToPad);

    // we also apply all input padding as the vertex cannot handle this.
    for (unsigned d = 0; d < numFieldDims; ++d) {
      truncateDilateAndPadInput(m, params, transformedSizes, partitionVars,
                                inputPadding, d);
    }

    return applyPadding(m, target, params.inputType, transformedSizes,
                        partitionVars, exchangeEstimator, kernelPadding,
                        inputPadding);
  }
  }

  throw poputil::poplibs_error("Unrecognised convolution method");
}

static popsolver::Variable addExchangeCycleEstimate(
    popsolver::Model &m, const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSizes,
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    const ExchangeEstimator &exchangeEstimator, const ConvParams &params,
    const ConvOptions &options, const std::vector<ConvTypes> &types,
    std::vector<popsolver::Variable> &inputsPerLevel,
    std::vector<popsolver::Variable> &weightsPerLevel) {
  const auto numFieldDims = params.getNumFieldDims();
  const auto numLevelsOfHierarchy = convSizes.size();

  assert(types.size() == numLevelsOfHierarchy);
  assert(partitionVars.size() == numLevelsOfHierarchy - 1);
  assert(transformedDims.size() == numLevelsOfHierarchy);

  // the number of cycles for exchange is the sum of the cycles for the input,
  // weights and partials for each level in the hierarchy (not including the
  // tile level). these are stored in this vector.
  std::vector<popsolver::Variable> cycleSumOperands;

  inputsPerLevel.clear();
  weightsPerLevel.clear();

  // this loop calculates the exchange cycles for each transition between a
  // hierarchy level, ie inter-IPU split to IPU level and then IPU level to tile
  // split (assuming there is more than one IPU).
  for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
    // the mapping of index to hierarchy level differs depending on the struct
    // we want to access so create references for all of them first and only
    // refer to them inside this loop. this makes it a bit easier to follow
    // the logic.
    const auto &sizesNextLevel = convSizes[level + 1];
    const auto &partitionsNextLevel = partitionVars[level];

    // transformations happen before partitioning therefore we need to take into
    // account the transformations that happen on the level we are exchange from
    // to be able to know how much data will be exchanged.
    const auto &transformedDimsPreviousLevel = transformedDims[level];

    // because we support an n-d convolution, we don't know how many input and
    // output field sizes we have and therefore the variables representing them
    // they must be stored in vectors.
    std::vector<popsolver::Variable> outputFieldSizes;
    std::vector<popsolver::Variable> inputFieldSizes;

    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      const auto fieldGrainSize = partitionsNextLevel.fieldGrainSize[dim];

      auto outputFieldSize = sizesNextLevel.numFieldGrains[dim];
      if (fieldGrainSize != 1) {
        outputFieldSize =
            m.product({outputFieldSize, m.addConstant(fieldGrainSize)});
      }
      outputFieldSizes.push_back(outputFieldSize);

      if (transformedDimsPreviousLevel.count(dim)) {
        inputFieldSizes.push_back(outputFieldSize);
      } else {
        auto inputFieldSize =
            m.call({outputFieldSize, sizesNextLevel.kernelSize[dim]},
                   [dim, params](const std::vector<unsigned> &values) {
                     const auto &outputFieldSize = values[0];
                     const auto &kernelSizeForThisDim = values[1];
                     return getMaxInputRangeSize(outputFieldSize, dim, params,
                                                 kernelSizeForThisDim);
                   });
        inputFieldSizes.push_back(inputFieldSize);
      }
    }

    const auto totalOutputFieldSize = m.product(outputFieldSizes);
    const auto totalInputFieldSize = m.product(inputFieldSizes);
    const auto totalKernelSize = m.product(sizesNextLevel.kernelSize);
    const auto numConvGroups =
        m.product({sizesNextLevel.numConvGroupGrains,
                   m.addConstant(partitionsNextLevel.convGroupGrainSize)});
    const auto numInChans =
        m.product({sizesNextLevel.numInChanGrains,
                   m.addConstant(partitionsNextLevel.inChanGrainSize)});
    const auto numOutChans =
        m.product({sizesNextLevel.numOutChanGrains,
                   m.addConstant(partitionsNextLevel.outChanGrainSize)});
    auto numberOfInputElements =
        m.product({totalInputFieldSize, sizesNextLevel.batchSize, numInChans,
                   numConvGroups});
    auto numberOfWeights =
        m.product({totalKernelSize, numInChans, numOutChans, numConvGroups});
    const auto numberOfOutputElements =
        m.product({totalOutputFieldSize, sizesNextLevel.batchSize, numOutChans,
                   numConvGroups});
    inputsPerLevel.push_back(numberOfInputElements);
    weightsPerLevel.push_back(numberOfWeights);

    const auto tilesUsedByWeights =
        m.product({m.product(partitionVars[level].fieldSplit),
                   partitionVars[level].batchSplit});

    const auto tilesUsedByInputElements =
        partitionVars[level].outChanSplit.parallel;

    // because we distribute the weights evenly across all tiles that require
    // them we can deduce that 1/Nth of the weights are already on the correct
    // tile. this needs to be calculated because each serial split will
    // introduce a certain amount of iterations where the data is exchanged onto
    // the tile and therefore the more splits the higher the cost. however, for
    // example, if the weights are split over a single tile we would expect a
    // zero exchange cost. we do this for both weights and inputs because of the
    // swap operands transformation.
    numberOfWeights =
        m.sub(numberOfWeights, m.floordiv(numberOfWeights, tilesUsedByWeights));
    numberOfInputElements =
        m.sub(numberOfInputElements,
              m.floordiv(numberOfInputElements, tilesUsedByInputElements));

    // partials here refers to the data that isn't either input (activations) or
    // weights. as we are calculating the exchange cost between two levels of
    // hierarchy we must be half way through a convolution and therefore have
    // some sort of partials. the size of the partials is the same as the output
    // of the next level of hierarchy. eg. the result type of the tile split
    // hierarchy will become the input of the IPU level which performs
    // a reduction of these partials across the device.
    const auto numberOfPartialSums = numberOfOutputElements;

    const auto inputElementCycles = exchangeEstimator.getInputElementCycles(
        numberOfInputElements, params.inputType, level);
    cycleSumOperands.push_back(inputElementCycles);

    const auto weightCycles =
        exchangeEstimator.getCycles(numberOfWeights, params.inputType, level);
    cycleSumOperands.push_back(weightCycles);

    // We do the first stage of any reduction separately so that we
    // can prune the search space based on this from previous best
    // cycles and because the first stage exchange cycles are independent
    // of the reduction plan.
    //
    // Any further stages are dependent on the reduction plan and their
    // cycle cost is added through a call.
    const auto partialSumCyclesFirstStage = exchangeEstimator.getCycles(
        numberOfPartialSums, types[level + 1].resultType, level);
    cycleSumOperands.push_back(partialSumCyclesFirstStage);

    auto reduceDimSizes = partitionsNextLevel.kernelSplit;
    reduceDimSizes.push_back(partitionsNextLevel.inChanSplit);
    const auto reductionDepth = m.product(reduceDimSizes);
    const auto resultType = types[level + 1].resultType;
    const auto partialSumCyclesRemainingStages = m.call(
        {numberOfPartialSums, reductionDepth},
        [exchangeEstimator, resultType, level,
         &options](const std::vector<unsigned> &vars) -> unsigned {
          const auto numPartialSums = vars[0];
          const auto reductionDepth = vars[1];

          if (reductionDepth <= 1) {
            return 0;
          }

          unsigned remainingDepth = reductionDepth;
          unsigned outputSizeThisStage = numPartialSums;
          unsigned cycles = 0;
          const auto reducePlan = getMultiStageReducePlan(
              reductionDepth, options.enableMultiStageReduce);
          bool firstStage = true;
          for (const auto d : reducePlan) {
            // We add first stage reduction exchange cycles separately above.
            if (!firstStage) {
              cycles += exchangeEstimator.getCycles(outputSizeThisStage,
                                                    resultType, level);
            }
            const auto depthThisStage = ceildiv(remainingDepth, d);
            outputSizeThisStage = ceildiv(outputSizeThisStage, depthThisStage);
            remainingDepth = ceildiv(remainingDepth, depthThisStage);
            firstStage = false;
          }
          // Final reduction
          if (remainingDepth > 1 && !firstStage) {
            cycles += exchangeEstimator.getCycles(outputSizeThisStage,
                                                  resultType, level);
          }
          return cycles;
        },
        "partialSumExchangeCycleEstimate");
    cycleSumOperands.push_back(partialSumCyclesRemainingStages);
  }

  return m.sum(cycleSumOperands, "exchangeCycleEstimate");
}

// Pair of cycles and temporary bytes for reductions
static std::pair<popsolver::Variable, popsolver::Variable>
addReduceCycleEstimate(popsolver::Model &m,
                       const std::vector<PartitionVariables> &partitionVars,
                       popsolver::Variable partialsPerTile,
                       const poplar::Target &target,
                       const std::vector<ConvTypes> &types,
                       std::vector<popsolver::Variable> &outputsPerLevel,
                       const ConvOptions &options,
                       PlanningCacheImpl::CycleEstimationImpl *cache) {
  std::vector<popsolver::Variable> cycleSumOperands;
  std::vector<popsolver::Variable> tempBytesMaxOperands;
  const auto numLevelsOfHierarchy = partitionVars.size();
  outputsPerLevel.clear();
  for (int level = numLevelsOfHierarchy - 1; level >= 0; --level) {
    auto reduceDimSizes = partitionVars[level].kernelSplit;
    reduceDimSizes.push_back(partitionVars[level].inChanSplit);
    const auto reductionDepth = m.product(reduceDimSizes);
    outputsPerLevel.push_back(m.ceildiv(partialsPerTile, reductionDepth));
    bool floatPartials = types[level + 1].resultType == poplar::FLOAT;
    bool floatOutput = types[level].resultType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    const auto numWorkers = target.getNumWorkerContexts();
    const auto partialsVectorWidth =
        target.getVectorWidth(floatPartials ? poplar::FLOAT : poplar::HALF);
    const auto outputVectorWidth =
        target.getVectorWidth(floatOutput ? poplar::FLOAT : poplar::HALF);
    const auto cycleEstimate =
        m.call({outputsPerLevel.back(), reductionDepth},
               [floatOutput, floatPartials, numWorkers, dataPathWidth,
                partialsVectorWidth, outputVectorWidth, &options,
                cache](const std::vector<unsigned> &vars) -> unsigned {
                 return cache->mEstimateConvReduceCycles(
                     vars[0], vars[1], floatOutput, floatPartials, numWorkers,
                     dataPathWidth, partialsVectorWidth, outputVectorWidth,
                     options.enableMultiStageReduce);
               });
    cycleSumOperands.push_back(cycleEstimate);
    // Temporary memory for the reduction will be given by the number of
    // outputs on a tile
    const unsigned elementBytes =
        target.getTypeSize(types[level + 1].resultType);
    const auto tempBytesEstimate = m.call(
        {outputsPerLevel.back(), reductionDepth},
        [elementBytes,
         &options](const std::vector<unsigned> &vars) -> unsigned {
          const auto numOutputs = vars[0];
          const auto reductionDepth = vars[1];
          if (reductionDepth <= 1) {
            return 0;
          }

          const auto reducePlan = getMultiStageReducePlan(
              reductionDepth, options.enableMultiStageReduce);
          unsigned remainingDepth = reductionDepth;
          unsigned numOutputsThisStage = numOutputs * reductionDepth;
          unsigned maxTempBytes = 0;
          for (const auto d : reducePlan) {
            const auto depthThisStage = ceildiv(remainingDepth, d);
            const auto tempBytesThisStage = numOutputsThisStage * elementBytes;
            maxTempBytes = std::max(maxTempBytes, tempBytesThisStage);
            numOutputsThisStage = ceildiv(numOutputsThisStage, depthThisStage);
            remainingDepth = ceildiv(remainingDepth, depthThisStage);
          }

          return maxTempBytes;
        });
    tempBytesMaxOperands.push_back(tempBytesEstimate);
    if (level != 0) {
      partialsPerTile = m.ceildiv(partialsPerTile, reductionDepth);
    }
  }
  return std::make_pair(
      m.sum(cycleSumOperands, "reduceCycleEstimate"),
      m.max(tempBytesMaxOperands, "reduceCycleTempBytesEstimate"));
}

// the number of weights in the tile level of the hierarchy is how many
// weights *after* broadcast, here we want to know how many there are before
// so take the number of weights at the hierarchy above and evenly split them.
static popsolver::Variable
addWeightsPerTile(popsolver::Model &m, const popsolver::Variable usedTiles,
                  const std::vector<popsolver::Variable> &weightsPerLevel,
                  const ConvParams &params) {
  assert(!weightsPerLevel.empty());
  const auto weightsPerIPU = [&] {
    // when there is only one IPU the "previous level" is actually the original
    // convolution parameters.
    if (weightsPerLevel.size() == 1) {
      // we don't need to take into account the kernel transforms here because
      // the transformation is applied after the dynamic slice, which is why
      // we want to calculate the number of weights per tile.
      const auto numberOfWeights =
          product(params.kernelShape) * params.inputChannelsPerConvGroup *
          params.outputChannelsPerConvGroup * params.numConvGroups;
      return m.addConstant(numberOfWeights);
    } else {
      return weightsPerLevel[weightsPerLevel.size() - 2];
    }
  }();

  return m.ceildiv(weightsPerIPU, usedTiles);
}

static popsolver::Variable
addPartialsPerTile(popsolver::Model &m, const PartitionVariables &partitionVars,
                   unsigned convGroupsPerGroup, unsigned partialChansPerGroup,
                   const ConvSizeVariables &convSize) {
  unsigned fieldGrainSizeProduct = product(partitionVars.fieldGrainSize);
  auto partialDimSizes = convSize.numFieldGrains;
  partialDimSizes.push_back(m.addConstant(fieldGrainSizeProduct));
  partialDimSizes.push_back(convSize.batchSize);
  partialDimSizes.push_back(m.product(
      {convSize.numConvGroupGrains, m.addConstant(convGroupsPerGroup)}));
  partialDimSizes.push_back(m.product(
      {convSize.numOutChanGrains, m.addConstant(partialChansPerGroup)}));
  return m.product(partialDimSizes, "partialsPerTile");
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

// returns a pair of the number of cycles and the number of bytes per tile.
static std::pair<popsolver::Variable, popsolver::Variable>
addTransformCycleEstimate(
    popsolver::Model &m, const ConvParams &params,
    const ConvParams &transformedOnceParams,
    const ConvParams &transformedOnceUnpaddedParams,
    const std::vector<ConvTransform> &transforms,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &transformedConvSizes,
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    unsigned inChansPerGroup, unsigned partialChansPerGroup,
    const std::vector<ConvTypes> &types, bool isJointPlan,
    const ConvOptions &options, const poplar::Target &target) {
  bool isConvWeightUpdate = options.pass == Pass::TRAINING_WU;
  bool isFullyConnectedLayer = options.pass == Pass::FC_INFERENCE_FWD ||
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
  bool padInChannels = params.inputChannelsPerConvGroup % inChansPerGroup;
  bool padPartialChannels =
      params.outputChannelsPerConvGroup % partialChansPerGroup;
  bool rearrangeInput = isConvWeightUpdate || expandDims || swapOperands ||
                        padInChannels || options.pass == Pass::FC_TRAINING_WU ||
                        (options.pass == Pass::FC_TRAINING_BWD && !isJointPlan);
  bool rearrangeWeights = isConvWeightUpdate || expandDims ||
                          outChanFlattenDims || swapOperands || padInChannels ||
                          padPartialChannels;
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(params.inputType == poplar::FLOAT);
  bool rearrangeOutput = (!isConvWeightUpdate && swapOperands) ||
                         (isConvWeightUpdate && !swapOperands) ||
                         outChanFlattenDims || padPartialChannels ||
                         (options.pass == Pass::FC_TRAINING_WU && !isJointPlan);
  // We assume the next layer uses an input channel grouping of
  // weightsPerConvUnit and apply a small cost if the output channel
  // grouping of this layer doesn't match.
  bool regroupOutput =
      !isFullyConnectedLayer && partialChansPerGroup != weightsPerConvUnit;
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
      !regroupOutput && !regroupWeights) {
    const auto zero = m.addConstant(0);
    return std::make_pair(zero, zero);
  }

  const auto &convSize = transformedConvSizes[ipuLevel];
  std::vector<popsolver::Variable> outputFieldSizes;
  std::vector<popsolver::Variable> inputFieldSizes;
  const auto numFieldDims = partitionVars[ipuLevel].fieldSplit.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto fieldGrainSize = partitionVars[ipuLevel].fieldGrainSize[dim];
    auto outputFieldSize = convSize.numFieldGrains[dim];
    if (fieldGrainSize != 1) {
      outputFieldSize =
          m.product({outputFieldSize, m.addConstant(fieldGrainSize)});
    }
    outputFieldSizes.push_back(outputFieldSize);
    if (transformedDims[ipuLevel].count(dim)) {
      inputFieldSizes.push_back(outputFieldSize);
    } else {
      auto inputFieldSize = m.call(
          {outputFieldSize, convSize.kernelSize[dim]},
          [dim, transformedOnceParams](const std::vector<unsigned> &values) {
            return getMaxInputRangeSize(values[0], dim, transformedOnceParams,
                                        values[1]);
          });
      inputFieldSizes.push_back(inputFieldSize);
    }
  }
  const auto numConvGroups =
      m.product({convSize.numConvGroupGrains,
                 m.addConstant(partitionVars[ipuLevel].convGroupGrainSize)});
  const auto numInChans =
      m.product({convSize.numInChanGrains,
                 m.addConstant(partitionVars[ipuLevel].inChanGrainSize)});
  const auto numOutChans =
      m.product({convSize.numOutChanGrains,
                 m.addConstant(partitionVars[ipuLevel].outChanGrainSize)});
  std::vector<popsolver::Variable> ipuSplits = {
      partitionVars[ipuLevel].batchSplit,
      partitionVars[ipuLevel].convGroupSplit,
      partitionVars[ipuLevel].inChanSplit,
      partitionVars[ipuLevel].outChanSplit.parallel};
  ipuSplits.insert(ipuSplits.end(), partitionVars[ipuLevel].fieldSplit.begin(),
                   partitionVars[ipuLevel].fieldSplit.end());
  ipuSplits.insert(ipuSplits.end(), partitionVars[ipuLevel].kernelSplit.begin(),
                   partitionVars[ipuLevel].kernelSplit.end());
  auto ipuUsedTiles = m.product(ipuSplits);
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  std::vector<popsolver::Variable> memoryUsage;
  std::vector<popsolver::Variable> cyclesOperands;

  if (rearrangeInput || rearrangeWeights || regroupWeights) {
    const auto reorderBytesPerCycle = std::min<unsigned>(
        target.getMemcpyBytesPerCycle(), inputBytesPerElement);
    std::vector<popsolver::Variable> numElementsOperands;
    if (rearrangeInput) {
      auto totalInputFieldSize = m.product(inputFieldSizes);
      auto numInputElements = m.product(
          {totalInputFieldSize, convSize.batchSize, numInChans, numConvGroups});
      numElementsOperands.push_back(numInputElements);
    }
    if (rearrangeWeights || regroupWeights) {
      auto totalKernelSize = m.product(convSize.kernelSize);
      auto numWeightElements =
          m.product({totalKernelSize, numInChans, numOutChans, numConvGroups});
      if (rearrangeWeights) {
        numElementsOperands.push_back(numWeightElements);
      } else if (regroupWeights) {
        auto numElementsPerTile = m.ceildiv(numWeightElements, ipuUsedTiles);
        auto bytesPerTile = m.product(
            {numElementsPerTile, m.addConstant(inputBytesPerElement)});
        const auto factor = getScaleFactorForTransform(
            transformedOnceUnpaddedParams.inputType,
            transformedOnceUnpaddedParams.outputChannelsPerConvGroup);
        auto cycles =
            m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                      m.addConstant(factor[1] * regroupBytesPerCycle));

        memoryUsage.push_back(bytesPerTile);
        cyclesOperands.push_back(cycles);
      }
    }
    auto numElements = m.sum(numElementsOperands);
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    auto bytesPerTile =
        m.product({numElementsPerTile, m.addConstant(inputBytesPerElement)});

    cyclesOperands.push_back(
        m.ceildiv(bytesPerTile, m.addConstant(exchangeBytesPerCycle)));
    const auto factor = getScaleFactorForTransform(
        transformedOnceUnpaddedParams.inputType,
        transformedOnceUnpaddedParams.inputChannelsPerConvGroup *
            transformedOnceUnpaddedParams.outputChannelsPerConvGroup);

    cyclesOperands.push_back(
        m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                  m.addConstant(reorderBytesPerCycle * factor[1])));
    memoryUsage.push_back(bytesPerTile);
  }
  if (rearrangeOutput || regroupOutput) {
    auto totalOutputFieldSize = m.product(outputFieldSizes);
    auto numElements = m.product(
        {totalOutputFieldSize, convSize.batchSize, numOutChans, numConvGroups});
    auto numElementsPerTile = m.ceildiv(numElements, ipuUsedTiles);
    const auto outputBytesPerElement =
        target.getTypeSize(types[ipuLevel].resultType);
    const auto outputRegroupBytesPerCycle =
        std::min<unsigned>(target.getMemcpyBytesPerCycle(),
                           partialChansPerGroup * outputBytesPerElement);
    auto bytesPerTile =
        m.product({numElementsPerTile, m.addConstant(outputBytesPerElement)});
    if (rearrangeOutput) {
      const auto outputReorderBytesPerCycle = std::min<unsigned>(
          target.getMemcpyBytesPerCycle(), outputBytesPerElement);
      cyclesOperands.push_back(
          m.ceildiv(bytesPerTile, m.addConstant(exchangeBytesPerCycle)));
      const auto factor = getScaleFactorForTransform(
          transformedOnceUnpaddedParams.outputType,
          transformedOnceUnpaddedParams.outputChannelsPerConvGroup);
      cyclesOperands.push_back(
          m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                    m.addConstant(outputReorderBytesPerCycle * factor[1])));
      memoryUsage.push_back(bytesPerTile);
    } else if (regroupOutput) {
      const auto factor = getScaleFactorForTransform(
          transformedOnceUnpaddedParams.outputType,
          transformedOnceUnpaddedParams.outputChannelsPerConvGroup);
      cyclesOperands.push_back(
          m.ceildiv(m.product({bytesPerTile, m.addConstant(factor[0])}),
                    m.addConstant(outputRegroupBytesPerCycle * factor[1])));
      memoryUsage.push_back(bytesPerTile);
    }
  }

  // the transforms happen serially therefore we sum the cycles and take the
  // max of the bytes. we also decide that the amount of temporary memory
  // required is two times the usage as the input and output must be live at the
  // same time. of course this assumes that the inputs and outputs are the same
  // size which is not always the case.
  const auto cycles =
      m.sum(std::move(cyclesOperands), "transformCycleEstimate");
  const auto tempBytes =
      m.product({m.max(std::move(memoryUsage)), m.addConstant(2u)},
                "transformTempBytesEstimate");

  return std::make_pair(cycles, tempBytes);
}

// estimation function for both dynamic slice and update.
template <typename F>
popsolver::Variable
addDynamicSliceEstimate(popsolver::Model &m, const unsigned numWorkers,
                        const popsolver::Variable &elementsPerTile,
                        const PartitionVariables &tileSplits,
                        const F &getElementsPerWord) {
  const auto &outChanSerialSplit = tileSplits.outChanSplit.serial;

  // assume we have to slice an even amount of weights on each tile for each
  // each split.
  const auto sliceSize = m.ceildiv(elementsPerTile, outChanSerialSplit);
  const auto elementsPerWord = getElementsPerWord();

  const std::vector<popsolver::Variable> vars = {outChanSerialSplit, sliceSize};
  return m.call(vars, [elementsPerWord, numWorkers](const auto &vars) {
    const auto &outChanSerialSplit = vars[0];
    const auto &sliceSize = vars[1];

    assert(outChanSerialSplit != 0);
    // when not splitting serially we require no dynamic slicing or updating.
    if (outChanSerialSplit == 1) {
      return 0u;
    }

    // rough estimate of vertex overhead plus assuming inner loop of 2 cycles
    // per word (one load, one store).
    const auto innerLoopCycles = 2 * sliceSize / elementsPerWord;
    return (30u + innerLoopCycles) * numWorkers;
  });
}

static popsolver::Variable
addDynamicSliceEstimate(popsolver::Model &m, const poplar::Target &target,
                        const popsolver::Variable &weightsPerTile,
                        const PartitionVariables &tileSplits,
                        const ConvParams &params) {
  const auto workers = target.getNumWorkerContexts();
  return addDynamicSliceEstimate(m, workers, weightsPerTile, tileSplits, [&] {
    // the weights type is always the same as the input type.
    const auto weightsType = params.inputType;

    // weights per word
    return target.getVectorWidth(weightsType) / 2;
  });
}

static popsolver::Variable
addDynamicUpdateEstimate(popsolver::Model &m, const poplar::Target &target,
                         const popsolver::Variable &outputsPerTile,
                         const PartitionVariables &tileSplits,
                         const std::vector<ConvTypes> &types) {
  const auto workers = target.getNumWorkerContexts();
  return addDynamicSliceEstimate(m, workers, outputsPerTile, tileSplits, [&] {
    // currently we only support splitting the output channels serially and only
    // when in the intra-IPU level. TODO: T12878 Assert that this is the case.
    assert(types.size() > 0);
    const unsigned intraTileLevel = types.size() - 1;

    const auto outputsType = types[intraTileLevel].resultType;
    const auto outputsPerWord = target.getVectorWidth(outputsType) / 2;

    return outputsPerWord;
  });
}

// cycles, temp persistent bytes for rearranged version of weights,
// temp bytes during the rearrange
static std::tuple<popsolver::Variable, popsolver::Variable, popsolver::Variable>
addRearrangeBeforeSliceEstimate(popsolver::Model &m,
                                const poplar::Target &target,
                                const ExchangeEstimator &exchangeEstimator,
                                const popsolver::Variable &weightsPerTile,
                                const PartitionVariables &tileSplits,
                                unsigned level, const ConvParams &params,
                                const ConvOptions &options, bool isJointPlan) {
  bool isFullyConnectedLayer = options.pass == Pass::FC_INFERENCE_FWD ||
                               options.pass == Pass::FC_TRAINING_FWD ||
                               options.pass == Pass::FC_TRAINING_BWD ||
                               options.pass == Pass::FC_TRAINING_WU;
  if (!isFullyConnectedLayer || isJointPlan) {
    const auto zero = m.addConstant(0u);
    return std::make_tuple(zero, zero, zero);
  }

  // Exchange cycle estimate, assume we are using a number of tiles equal to
  // the product of parallel splits, and exchanging all-to-all. We should be
  // able to achieve cycles:
  //
  // ceildiv(bytes, tilesUsed) / exchangeBytesPerCycle
  //
  // No super-tile send as we can't rely on sending+receiving tiles allowing
  // super-tile send/receive concurrently.
  //
  // isSeriallySplit is 0 when outChanSplit.serial == 1 and 1 otherwise.
  const auto isSeriallySplit = m.addVariable(0u, 1u, "isSeriallySplit");
  m.less(isSeriallySplit, tileSplits.outChanSplit.serial);

  const auto exchangeCycles =
      exchangeEstimator.getCycles(weightsPerTile, params.inputType, level);

  // We assume one element per-cycle as a rough estimate to rearrange on-tile
  // as we don't know what the layout of these could be.
  const auto rearrangeCycles = weightsPerTile;
  const auto totalCycles =
      m.product({isSeriallySplit, m.sum({exchangeCycles, rearrangeCycles})});

  const auto typeBytes = m.addConstant(target.getTypeSize(params.inputType),
                                       "weightBytesPerElement");
  const auto bytesPerTile = m.product({weightsPerTile, typeBytes});

  const auto extraWeightsTempBytes = m.product({bytesPerTile, isSeriallySplit});

  return std::make_tuple(totalCycles, extraWeightsTempBytes,
                         extraWeightsTempBytes);
}

static Estimates<popsolver::Variable> addEstimates(
    popsolver::Model &m, const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSize,
    const std::vector<ConvSizeVariables> &transformedConvSize,
    popsolver::Variable usedTiles,
    const std::vector<std::unordered_set<unsigned>> &transformedDims,
    const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const ConvParams &untransformedParams,
    const ConvParams &transformedOnceParams,
    const ConvParams &transformedOnceUnpaddedParams, const bool isJointPlan,
    const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
    const unsigned partialChansPerGroup, const std::vector<ConvTypes> &types,
    const std::vector<ConvTransform> &transforms, Plan::Method method,
    const unsigned slicWindowWidth, const unsigned numConvUnitsRequired,
    const Plan::LinearizeTileOrder linearizeTileOrder,
    const ConvOptions &options, PlanningCacheImpl::CycleEstimationImpl *cache) {
  const auto numLevelsOfHierarchy = convSize.size();
  ExchangeEstimator exchangeEstimator(m, target, perLevelExchangeBytesPerCycle,
                                      numLevelsOfHierarchy, partitionVars,
                                      linearizeTileOrder);

  // Popsolver takes into account whether a variable is an operand of a call
  // when deciding the order to set variables. Add a dummy call to ensure the
  // split variables are prioritised as this reduces the amount of time spent
  // in the planner. TODO: T12879 Improve Popsolver's heuristics for ordering
  // variables so this dummy call is no longer necessary (or provide a proper
  // mechanism for ordering hints).
  std::vector<popsolver::Variable> variables;
  for (const auto &vars : partitionVars) {
    variables.push_back(vars.batchSplit);
    variables.push_back(vars.outChanSplit.parallel);
    variables.push_back(vars.outChanSplit.serial);
    variables.push_back(vars.inChanSplit);
    variables.push_back(vars.convGroupSplit);
    variables.insert(variables.end(), vars.fieldSplit.begin(),
                     vars.fieldSplit.end());
    variables.insert(variables.end(), vars.kernelSplit.begin(),
                     vars.kernelSplit.end());
  };
  (void)m.call(variables, [](const std::vector<unsigned> &) { return 0U; });

  Estimates<popsolver::Variable> e;

  std::vector<popsolver::Variable> inputsPerLevel, weightsPerLevel;
  e.exchangeCycles = addExchangeCycleEstimate(
      m, partitionVars, convSize, transformedDims, exchangeEstimator,
      transformedOnceParams, options, types, inputsPerLevel, weightsPerLevel);

  std::tie(e.transformCycles, e.transformTempBytes) = addTransformCycleEstimate(
      m, untransformedParams, transformedOnceParams,
      transformedOnceUnpaddedParams, transforms, partitionVars,
      transformedConvSize, transformedDims, inChansPerGroup,
      partialChansPerGroup, types, isJointPlan, options, target);

  const auto &intraTileSplits = partitionVars.back();

  // create a variable that is the number of weights per tile before being
  // transformed and broadcast out. this is so we can calculate how much data
  // is dynamically sliced for serial convolutions. when calculating this we
  // assume the weights are distributed evenly.
  const auto weightsPerTile =
      addWeightsPerTile(m, usedTiles, weightsPerLevel, transformedOnceParams);

  // create a variable that represents that most amount of partials that will
  // live on a single tile. this is enough as a cycle estimate is how long the
  // longest tile would take to process it's part of a convolution.
  const auto partialsPerTile =
      addPartialsPerTile(m, intraTileSplits, convGroupsPerGroup,
                         partialChansPerGroup, transformedConvSize.back());

  // When splitting serially the temp memory should not outlive an iteration of
  // the loop and therefore we don't need to take into account and serial splits
  e.convTempBytes = addConvTempMemoryEstimate(
      m, partitionVars, convSize, inputsPerLevel.back(), weightsPerLevel.back(),
      partialsPerTile, target, transformedOnceParams, types, method);

  // it is possible that we may need to add zero padding to the activations
  // and weights so that we have the correct number of input channels for the
  // method we are planning to use (AMP, SLIC, etc.). this is synthesised by
  // exchanging the constant zero the amount of times, this can have a sizeable
  // effect on temporary memory and cycles and so we need to track it when
  // deciding on the optimal plan.
  std::tie(e.tileLevelTransformCycles, e.tileLevelTransformTempBytes) =
      addTileLevelTransformEstimates(
          m, target, transformedOnceParams, types.back().partialType,
          inChansPerGroup, transformedConvSize, partitionVars,
          exchangeEstimator, method, slicWindowWidth, numConvUnitsRequired);

  e.partialCalcCycles = addPartialCalcCycleEstimate(
      m, intraTileSplits.fieldGrainSize, convGroupsPerGroup, inChansPerGroup,
      partialChansPerGroup, transformedConvSize.back(), transformedDims.back(),
      target, transformedOnceParams, types.back().partialType, method,
      slicWindowWidth, numConvUnitsRequired, options, cache);

  const auto serialSplits = intraTileSplits.outChanSplit.serial;

  // Add a redundant inequality that relates the cycles required to calculate
  // the partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile = getMaxMACsPerCyclePerTile(
      target, types.back().partialType, transformedOnceParams.inputType, method,
      slicWindowWidth);
  const auto totalMacs = cache->mGetNumberOfMACs(transformedOnceParams);
  m.lessOrEqual(totalMacs / maxMACsPerCyclePerTile,
                m.product({usedTiles, e.partialCalcCycles, serialSplits}));

  std::vector<popsolver::Variable> outputsPerLevel;
  std::tie(e.reduceCycles, e.reduceTempBytes) =
      addReduceCycleEstimate(m, partitionVars, partialsPerTile, target, types,
                             outputsPerLevel, options, cache);

  // if this convolution has been split serially and we aren't sure the weights
  // are laid out well for a dynamic slice, we must also add a one-off cost
  // to rearrange the weights prior to slicing. The memory cost of this is
  // added to the temporary memory estimate rather than maxed because it will
  // remain live from before the serial loop begins to after it finishes.
  //
  // NOTE: Currently it is only possible for there to be a slice at the IPU
  // level so we always add rearrange estimates just for the ipu level. If
  // this capability was expanded for multi-IPU etc. this would have to change.
  const auto ipuLevel = transforms.size() - 2;
  std::tie(e.rearrangeBeforeSliceCycles, e.rearrangeBeforeSliceTempBytes,
           e.rearrangeBeforeSliceTempDuringRearrangeBytes) =
      addRearrangeBeforeSliceEstimate(
          m, target, exchangeEstimator, weightsPerTile, intraTileSplits,
          ipuLevel, transformedOnceParams, options, isJointPlan);

  // if this convolution has been split serially we must include the cycle cost
  // for performing the dynamic slice / update as well as multiplying our new
  // total by the amount of times we plan to execute this convolution. if the
  // outChanSplit.serial variable is 1 then these cycle counts should be zero.
  e.dynamicSliceCycles = addDynamicSliceEstimate(
      m, target, weightsPerTile, intraTileSplits, transformedOnceParams);

  const auto &outputsPerTile = outputsPerLevel.back();
  e.dynamicUpdateCycles = addDynamicUpdateEstimate(m, target, outputsPerTile,
                                                   intraTileSplits, types);

  e.totalCycles =
      m.sum({e.dynamicSliceCycles, e.transformCycles, e.exchangeCycles,
             e.tileLevelTransformCycles, e.partialCalcCycles, e.reduceCycles,
             e.dynamicUpdateCycles});
  e.totalCycles = m.product({e.totalCycles, serialSplits});
  e.totalCycles = m.sum({e.totalCycles, e.rearrangeBeforeSliceCycles});

  // take the total amount of temp bytes alive at the same time.
  e.totalTempBytes =
      m.sum({e.rearrangeBeforeSliceTempBytes,
             m.max({e.transformTempBytes,
                    m.sum({e.tileLevelTransformTempBytes, e.convTempBytes}),
                    e.reduceTempBytes,
                    e.rearrangeBeforeSliceTempDuringRearrangeBytes})});

  return e;
}

static Plan::Method getFullyConnectedBwdMethod(Plan::Method fwdMethod) {
  if (fwdMethod == Plan::Method::OUTER_PRODUCT) {
    return Plan::Method::MAC;
  }
  return fwdMethod;
}

static Estimates<popsolver::Variable> addBwdEstimates(
    popsolver::Model &m, ConvParams bwdUntransformedParams,
    ConvParams bwdTransformedOnceParams,
    ConvParams bwdTransformedOnceUnpaddedParams,
    const std::size_t numLevelsOfHierarchy,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSize,
    const std::vector<ConvTransform> &transforms, Plan::Method method,
    unsigned slicWindowWidth, unsigned numConvUnitsRequired,
    const popsolver::Variable usedTiles, const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<ConvTypes> &types, const bool isJointPlan,
    const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
    const unsigned partialChansPerGroup, const ConvOptions &options,
    PlanningCacheImpl::CycleEstimationImpl *cache) {
  // for the backwards pass the output shape will be Ci x Co (as defined in the
  // forward pass parameters) -- therefore if either of these are zero then
  // the backwards pass is a no-op and we can return zero.
  // note that, even though this is called the bwdTransformedOnceParams it is
  // still the forward params atm as we have not swapped the input channels and
  // field shape round yet (this happens after this check).
  if (bwdTransformedOnceParams.inputChannelsPerConvGroup == 0 ||
      bwdTransformedOnceParams.outputChannelsPerConvGroup == 0) {
    const auto zero = m.addConstant(0);
    return {zero, zero};
  }

  assert(!bwdTransformedOnceParams.inputFieldShape.empty());
  std::swap(bwdUntransformedParams.inputFieldShape.back(),
            bwdUntransformedParams.inputChannelsPerConvGroup);
  std::swap(bwdTransformedOnceParams.inputFieldShape.back(),
            bwdTransformedOnceParams.inputChannelsPerConvGroup);
  std::swap(bwdTransformedOnceUnpaddedParams.inputFieldShape.back(),
            bwdTransformedOnceUnpaddedParams.inputChannelsPerConvGroup);

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
  const auto bwdMethod = getFullyConnectedBwdMethod(method);

  std::vector<std::unordered_set<unsigned>> transformedDims(
      numLevelsOfHierarchy);
  return addEstimates(
      m, bwdPartitionVars, bwdConvSize, bwdTransformedConvSize, usedTiles,
      transformedDims, target, perLevelExchangeBytesPerCycle,
      bwdUntransformedParams, bwdTransformedOnceParams,
      bwdTransformedOnceUnpaddedParams, isJointPlan, convGroupsPerGroup,
      bwdInChansPerGroup, partialChansPerGroup, types, transforms, bwdMethod,
      slicWindowWidth, numConvUnitsRequired,
      Plan::LinearizeTileOrder::FC_BWD_AS_CONV, options, cache);
}

static Plan::Method getFullyConnectedWUMethod(const ConvParams &fwdParams,
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

static Estimates<popsolver::Variable> addWuEstimates(
    popsolver::Model &m, const ConvParams &untransformedParams,
    ConvParams wuTransformedOnceParams,
    ConvParams wuTransformedOnceUnpaddedParams,
    const std::size_t numLevelsOfHierarchy,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSize,
    const std::vector<ConvTransform> &transforms, Plan::Method method,
    unsigned slicWindowWidth, unsigned numConvUnitsRequired,
    const popsolver::Variable usedTiles, const poplar::Target &target,
    const unsigned numFieldDims,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<ConvTypes> &types, const bool isJointPlan,
    const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
    const unsigned partialChansPerGroup, const ConvOptions &options,
    PlanningCacheImpl::CycleEstimationImpl *cache) {
  // for the wu pass the output shape will be Ci x Fs (as defined in the
  // forward pass parameters) -- therefore if either of these are zero then
  // the weight update pass is a no-op and we can return zero.
  // note that, even though this is called the wuTransformedOnceParams it is
  // still the forward params atm as we have not swapped the input channels and
  // output channels round yet (this happens after this check).
  assert(!wuTransformedOnceParams.inputFieldShape.empty());
  if (wuTransformedOnceParams.inputChannelsPerConvGroup == 0 ||
      wuTransformedOnceParams.inputFieldShape.back() == 0) {
    const auto zero = m.addConstant(0);
    return {zero, zero};
  }

  auto wuUntransformedParams = untransformedParams;
  std::swap(wuUntransformedParams.inputChannelsPerConvGroup,
            wuUntransformedParams.outputChannelsPerConvGroup);
  std::swap(wuTransformedOnceParams.inputChannelsPerConvGroup,
            wuTransformedOnceParams.outputChannelsPerConvGroup);
  std::swap(wuTransformedOnceUnpaddedParams.inputChannelsPerConvGroup,
            wuTransformedOnceUnpaddedParams.outputChannelsPerConvGroup);

  std::vector<PartitionVariables> wuPartitionVars;
  std::vector<ConvSizeVariables> wuConvSize;
  std::vector<ConvSizeVariables> wuTransformedConvSize;
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    if (level + 1 < numLevelsOfHierarchy) {
      const auto &p = partitionVars[level];
      auto wuP = p;
      wuP.outChanSplit.parallel = p.inChanSplit;
      wuP.inChanSplit = p.outChanSplit.parallel;
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
          level > 0 ? partitionVars[level - 1].fieldGrainSize[dim]
                    : partitionVars[level].fieldGrainSize[dim];
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
          level + 1 < numLevelsOfHierarchy
              ? partitionVars[level].fieldGrainSize[dim]
              : partitionVars[level - 1].fieldGrainSize[dim];
      if (fieldGrainSize != 1) {
        wuTS.numFieldGrains[dim] =
            m.product({tS.numFieldGrains[dim], m.addConstant(fieldGrainSize)});
      }
    }
    wuTransformedConvSize.push_back(wuTS);
  }
  const auto wuInChansPerGroup = partialChansPerGroup;
  const auto wuPartialChansPerGroup = inChansPerGroup;
  const auto wuMethod = getFullyConnectedWUMethod(
      untransformedParams, method, partialChansPerGroup, inChansPerGroup);

  std::vector<std::unordered_set<unsigned>> transformedDims(
      numLevelsOfHierarchy);
  return addEstimates(m, wuPartitionVars, wuConvSize, wuTransformedConvSize,
                      usedTiles, transformedDims, target,
                      perLevelExchangeBytesPerCycle, wuUntransformedParams,
                      wuTransformedOnceParams, wuTransformedOnceUnpaddedParams,
                      isJointPlan, convGroupsPerGroup, wuInChansPerGroup,
                      wuPartialChansPerGroup, types, transforms, wuMethod,
                      slicWindowWidth, numConvUnitsRequired,
                      Plan::LinearizeTileOrder::FC_WU, options, cache);
}

static Partition makePartition(const popsolver::Solution &s,
                               const PartitionVariables &vars) {
  std::vector<unsigned> fieldSplitValues;
  for (const auto var : vars.fieldSplit) {
    fieldSplitValues.push_back(s[var]);
  }
  std::vector<unsigned> kernelSplitValues;
  for (const auto var : vars.kernelSplit) {
    kernelSplitValues.push_back(s[var]);
  }
  Partition partition(
      std::move(fieldSplitValues), s[vars.batchSplit],
      {s[vars.outChanSplit.serial], s[vars.outChanSplit.parallel]},
      std::move(kernelSplitValues), s[vars.inChanSplit], s[vars.convGroupSplit],
      vars.fieldGrainSize, vars.convGroupGrainSize, vars.inChanGrainSize,
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

ConvParams calculateParamsWithDeferredDilation(
    const ConvParams &params, const std::vector<unsigned> &dilatePostConv) {
  auto paramsWithDeferredDilation = params;
  for (const auto dim : dilatePostConv) {
    assert(canDeferDilation(params, dim));
    paramsWithDeferredDilation.inputTransform.dilation[dim] = 1;
    paramsWithDeferredDilation.outputTransform.paddingLower[dim] = 0;
    paramsWithDeferredDilation.outputTransform.paddingUpper[dim] = 0;
  }
  return paramsWithDeferredDilation;
}

static ConvParams calculateSwappedParams(const ConvParams &params,
                                         bool swapOperands) {
  auto swappedParams = params;
  if (swapOperands) {
    poplin::swapOperands(swappedParams);
  }
  return swappedParams;
}

static void expandDim(ConvParams &params, unsigned dim) {
  params.inputFieldShape[dim] = params.getOutputSize(dim);
  params.inputChannelsPerConvGroup *= params.getTruncatedKernelSize(dim);
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
  // TODO: T12880 Two dimensions can be flattened if they both have flipInput
  // set to true. To target this we would need to pass information about the two
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
  for (unsigned spatialDim = 0; spatialDim != flattenedParams.getNumFieldDims();
       ++spatialDim) {
    if (dimCanBeFlattened(flattenedParams, spatialDim)) {
      flattenDims.push_back(spatialDim + 1);
    }
  }
  if (flattenDims.size() > 1) {
    const auto innermostFlattenableDim = flattenDims.back();
    assert(innermostFlattenableDim > 0);
    for (auto it = std::next(flattenDims.rbegin()), end = flattenDims.rend();
         it != end; ++it) {
      const auto fromDimIndex = *it;
      auto &fromDimSize =
          fromDimIndex ? flattenedParams.inputFieldShape[fromDimIndex - 1]
                       : flattenedParams.batchSize;
      flattenedParams.inputFieldShape[innermostFlattenableDim - 1] *=
          fromDimSize;
      fromDimSize = 1;
    }
  } else {
    flattenDims.clear();
  }
  return flattenedParams;
}

unsigned convGroupCombineFactor(const unsigned factor,
                                const unsigned inputChannelsPerConvGroup) {
  return factor / inputChannelsPerConvGroup;
}

void combineConvGroups(const unsigned factor, ConvParams &params) {
  // divide the number of conv groups by the factor, rounding up in the process
  params.numConvGroups = ceildiv(params.numConvGroups, factor);

  // increase the number of input and output channels by the factor.
  params.inputChannelsPerConvGroup *= factor;
  params.outputChannelsPerConvGroup *= factor;
}

static ConvParams calculateGroupedParams(ConvParams groupedParams,
                                         unsigned combineConvGroups) {
  poplin::combineConvGroups(combineConvGroups, groupedParams);
  return groupedParams;
}

static ConvParams calculatePaddedParams(const ConvParams &params,
                                        const unsigned convGroupsGrainSize,
                                        const unsigned inChanGrainSize,
                                        const unsigned partialChanGrainSize) {
  auto paddedParams = params;

  const auto convGroups = params.getNumConvGroups();
  paddedParams.numConvGroups = roundUp(convGroups, convGroupsGrainSize);

  const auto inChans = params.getNumInputChansPerConvGroup();
  paddedParams.inputChannelsPerConvGroup = roundUp(inChans, inChanGrainSize);

  const auto partialChans = params.getNumOutputChansPerConvGroup();
  paddedParams.outputChannelsPerConvGroup =
      roundUp(partialChans, partialChanGrainSize);

  return paddedParams;
}

static std::tuple<ConvParams, ConvParams, ConvParams>
applyTransform(const ConvParams &params, const ConvTransform &transform,
               const unsigned convGroupGrainSize,
               const unsigned inChanGrainSize,
               const unsigned outChanGrainSize) {
  auto paramsWithExtraDims = params;
  addExtraDims(paramsWithExtraDims, transform.extraFieldDims);

  auto paramsWithDeferredDilation = calculateParamsWithDeferredDilation(
      paramsWithExtraDims, transform.dilatePostConv);

  auto swappedParams = calculateSwappedParams(paramsWithDeferredDilation,
                                              transform.swapOperands);
  const auto expandedParams =
      calculateExpandedParams(swappedParams, transform.expandDims);

  std::vector<unsigned> flattenDims;
  const auto flattenedParams = calculateFlattenedParams(
      expandedParams, transform.outChanFlattenDims, flattenDims);

  const auto groupedParams = calculateGroupedParams(
      std::move(flattenedParams), transform.combineConvGroupsFactor);

  auto paddedParams = calculatePaddedParams(groupedParams, convGroupGrainSize,
                                            inChanGrainSize, outChanGrainSize);

  return std::make_tuple(swappedParams, paddedParams, groupedParams);
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
getConvGroupGrainSizes(const std::vector<ConvTransform> &transforms,
                       unsigned convGroupsPerGroup) {
  assert(transforms.size() >= 1);
  std::vector<unsigned> convGroupGrainSizes(transforms.size());
  // The grain size at the last level is equal to convGroupsPerGroup.
  // To avoid rearrangement we use the same grain size at upper levels
  // unless these is a transform that rearranges the group axis.
  convGroupGrainSizes.back() = convGroupsPerGroup;

  for (int i = static_cast<int>(transforms.size()) - 2; i >= 0; --i) {
    convGroupGrainSizes[i] = transforms[i + 1].combineConvGroupsFactor == 1
                                 ? convGroupGrainSizes[i + 1]
                                 : 1;
  }
  return convGroupGrainSizes;
}

static std::vector<unsigned>
getOutChanGrainSizes(const std::vector<ConvTransform> &transforms,
                     unsigned partialChansPerGroup) {
  assert(transforms.size() >= 1);
  std::vector<unsigned> outChanGrainSizes(transforms.size());
  // The grain size at the last level is equal to partialChansPerGroup.
  // To avoid rearrangement we use the same grain size at upper levels
  // unless these is a transform that rearranges the output channel axis.
  outChanGrainSizes.back() = partialChansPerGroup;

  for (int i = static_cast<int>(transforms.size()) - 2; i >= 0; --i) {
    outChanGrainSizes[i] = (transforms[i + 1].outChanFlattenDims.empty() &&
                            (transforms[i + 1].combineConvGroupsFactor == 1))
                               ? outChanGrainSizes[i + 1]
                               : 1;
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
  inChanGrainSizes.back() = inChansPerGroup;

  for (int i = static_cast<int>(transforms.size()) - 2; i >= 0; --i) {
    inChanGrainSizes[i] = (transforms[i + 1].outChanFlattenDims.empty() &&
                           transforms[i + 1].expandDims.empty() &&
                           (transforms[i + 1].combineConvGroupsFactor == 1))
                              ? inChanGrainSizes[i + 1]
                              : 1;
  }
  return inChanGrainSizes;
}

static void applyPartitionPlanConstraint(popsolver::Model &m,
                                         const ConvOptions &options,
                                         unsigned level,
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
    constrainVar("batchSplit", p.batchSplit);
    constrainSplitVar("outChanSplit", p.outChanSplit);
    constrainVars("kernelSplit", p.kernelSplit);
    constrainVar("inChanSplit", p.inChanSplit);
    constrainVar("convGroupSplit", p.convGroupSplit);
    // All other PartitionVariables members are dependent on these splits.
  }
}

static inline std::string arrIndStr(unsigned level) {
  return "[" + std::to_string(level) + "]";
};

// Mostly for testing purposes. We have some constants fixed to a value which
// has no effect (serial partitioning currently) while functionality is
// implemented but which we want to be able to force to a different value
// for development purposes. This function creates a constant if specified in
// the plan constraints otherwise will call the provided function to create the
// variable normally.
template <typename F>
static popsolver::Variable
addPartitionConstant(popsolver::Model &m, const ConvOptions &options,
                     unsigned level, const std::string &pathSuffix,
                     const F &fn) {
  const auto val = options.planConstraints.get_optional<unsigned>(
      std::to_string(level) + ".partition." + pathSuffix);
  if (val) {
    return m.addConstant(*val);
  } else {
    return fn();
  }
}

static popsolver::Variable getInputChannelCount(popsolver::Model &m,
                                                const PartitionVariables &p,
                                                const ConvSizeVariables &s) {
  auto inputChannels = s.numInChanGrains;
  if (p.inChanGrainSize != 1) {
    inputChannels =
        m.product({inputChannels, m.addConstant(p.inChanGrainSize)});
  }
  return inputChannels;
}

static popsolver::Variable getInputFieldSize(popsolver::Model &m,
                                             const PartitionVariables &p,
                                             const ConvSizeVariables &s,
                                             const std::size_t dim) {
  const auto fieldGrainSize = p.fieldGrainSize[dim];
  auto inputFieldSize = s.numFieldGrains[dim];
  if (fieldGrainSize != 1) {
    inputFieldSize = m.product({inputFieldSize, m.addConstant(fieldGrainSize)});
  }
  return inputFieldSize;
}

// SLIC is not possible when the output has striding because this is implemented
// by striding the weights window across the input which cannot be done if SLIC
// handles the sliding. input dilation is also an issue because that is
// represented as output striding. kernel dilation would be possible if we
// realised the zeros in the weights before loading it into the CWEI registers,
// this is not currently modelled (and would incur a performance overhead) so
// is not supported either.
static void addSLICConstraints(popsolver::Model &m, const PartitionVariables &p,
                               const ConvSizeVariables &s,
                               const ConvParams &lvl1Params) {
  for (auto dim = 0U; dim < p.fieldGrainSize.size(); ++dim) {
    // TODO T14626: SLIC could handle these, we just need to implement them.
    // By expanding them out before the vertex.
    m.equal(m.addConstant(lvl1Params.inputTransform.flip[dim]), 0);

    // We don't handle kernel dilation, padding and flipping in the SLIC vertex
    // for now.
    m.equal(m.addConstant(lvl1Params.kernelTransform.dilation[dim]), 1);
    m.equal(m.addConstant(lvl1Params.kernelTransform.paddingLower[dim]), 0);
    m.equal(m.addConstant(lvl1Params.kernelTransform.paddingUpper[dim]), 0);
    m.equal(m.addConstant(lvl1Params.kernelTransform.flip[dim]), 0);

    m.equal(m.addConstant(lvl1Params.outputTransform.truncationLower[dim]), 0);
    m.equal(m.addConstant(lvl1Params.outputTransform.truncationUpper[dim]), 0);
    m.equal(m.addConstant(lvl1Params.outputTransform.stride[dim]), 1);
  }
}

// The Outer Product method can only be used if certain criteria are met (e.g.
// a batch size of 1 on any tile). See function implementation for a full list.
// The planner will not choose an Outer Product method unless all of these
// criteria are met.
static void addOuterProductConstaints(popsolver::Model &m,
                                      const PartitionVariables &p,
                                      const ConvSizeVariables &s,
                                      const ConvParams &lvl1Params) {
  m.equal(s.batchSize, 1);

  assert(lvl1Params.outputTransform.stride.size() == p.fieldGrainSize.size());
  assert(lvl1Params.inputTransform.dilation.size() == p.fieldGrainSize.size());
  assert(lvl1Params.inputTransform.flip.size() == p.fieldGrainSize.size());
  for (auto dim = 0U; dim < p.fieldGrainSize.size(); ++dim) {
    m.equal(s.kernelSize[dim], 1);
    m.equal(m.addConstant(lvl1Params.outputTransform.stride[dim]), 1);
    m.equal(m.addConstant(lvl1Params.inputTransform.dilation[dim]), 1);
    m.equal(m.addConstant(lvl1Params.inputTransform.flip[dim]), 0);
    m.equal(getInputChannelCount(m, p, s), 1);

    // Output size == (padded) input size (because kernelSize and stride are 1)
    m.equal(getInputFieldSize(m, p, s, dim), 1);
  }
}

static void addMethodConstraints(popsolver::Model &m, const Plan::Method method,
                                 const PartitionVariables &p,
                                 const ConvSizeVariables &s,
                                 const ConvParams &lvl1Params) {
  // TODO: T12881 We assume that the transformations applied to the
  // parameters (which are the transforms at level 1 in the hierarchy) are
  // referencing the tile level. This is only true for single IPU
  // convolutions, for multi-IPU there can be other transforms that make
  // these fields constrainable, therefore these constraints are currently
  // overly conserversative for the multi-IPU case.
  switch (method) {
  case Plan::Method::AMP:
  case Plan::Method::MAC:
    // these methods have no individual constraint requirements.
    break;
  case Plan::Method::SLIC:
    addSLICConstraints(m, p, s, lvl1Params);
    break;
  case Plan::Method::OUTER_PRODUCT:
    addOuterProductConstaints(m, p, s, lvl1Params);
    break;
  }
}

static popsolver::Variable
getUsedTiles(popsolver::Model &m,
             const std::vector<PartitionVariables> &partitionVars,
             const std::vector<unsigned> &hierarchy) {
  std::vector<popsolver::Variable> perLevelSplits;
  for (unsigned level = 0; level != hierarchy.size(); ++level) {
    const auto &p = partitionVars[level];
    // we only care about splits across tiles so don't include the serial splits
    std::vector<popsolver::Variable> splits;
    splits.push_back(p.batchSplit);
    splits.push_back(p.outChanSplit.parallel);
    splits.push_back(p.inChanSplit);
    splits.push_back(p.convGroupSplit);
    splits.insert(splits.end(), p.fieldSplit.begin(), p.fieldSplit.end());
    splits.insert(splits.end(), p.kernelSplit.begin(), p.kernelSplit.end());
    const auto levelSplit =
        m.product(splits, arrIndStr(level) + ".partition.total");
    m.lessOrEqual(levelSplit, hierarchy[level]);
    perLevelSplits.push_back(levelSplit);
  }

  return m.product(std::move(perLevelSplits));
}

static Estimates<popsolver::Variable> constructModel(
    const poplar::Target &target, const std::vector<ConvTransform> &transforms,
    const std::vector<ConvTypes> &types, const std::vector<unsigned> &hierarchy,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<unsigned> &fieldGrainSize,
    const ConvVertexType &convVertexType, const ConvParams &untransformedParams,
    bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
    PlanningCacheImpl::CycleEstimationImpl *cache, const ConvOptions &options,
    popsolver::Model &m, std::vector<PartitionVariables> &partitionVars) {
  using namespace popsolver;
  using poplibs_support::ceildiv;

  const auto convGroupsPerGroup = convVertexType.convGroupsPerGroup;
  const auto inChansPerGroup = convVertexType.inChansPerGroup;
  const auto partialChansPerGroup = convVertexType.partialChansPerGroup;

  const auto convGroupGrainSize =
      getConvGroupGrainSizes(transforms, convGroupsPerGroup);
  const auto outChanGrainSize =
      getOutChanGrainSizes(transforms, partialChansPerGroup);
  const auto inChanGrainSize = getInChanGrainSizes(transforms, inChansPerGroup);

  // Apply the top level transform to the parameters. The top level transform is
  // the only transform that can add dimensions / swap operands. Applying the
  // top level transform to the parameters here means we don't need to support
  // adding dimensions / swapping operands in the generic code that handles
  // transforms different levels.
  ConvParams transformedViewParams, transformedOnceParams,
      transformedOnceUnpaddedParams;
  std::tie(transformedViewParams, transformedOnceParams,
           transformedOnceUnpaddedParams) =
      applyTransform(untransformedParams, transforms[0], convGroupGrainSize[0],
                     inChanGrainSize[0], outChanGrainSize[0]);

  // If yTileSplit is greater than one we end up splitting across the y axis of
  // the output volume. The input elements required to compute output elements
  // on one side of the split will overlap with the input elements required for
  // the otherside of the split, increasing communication.
  // An alternative strategy would be to split across the y axis of
  // the input volume. Now there is no overlap in input elements read by each
  // tile, but nx1 convolutions for rows near the boundary must be summed
  // with nx1 convolutions for rows the other side the boundary. This results
  // to the communication for more partial sums.
  // Assuming a stride of 1, the alternative strategy reads
  // inputsChannelsPerTile * (filterSize - 1) fewer input rows per tile pair
  // but it needs to sends (outputChannelsPerTile * (filterSize - 1) / 2) extra
  // rows of partial sum per tile pair.
  // TODO: T12882 Investigate the alternative strategy outlined above.

  const auto numFieldDims = transformedOnceParams.getNumFieldDims();
  // the hierarchy vector contains how many agents there are on each level, in
  // other words how many IPUs in the multi-IPU split and how many tiles in the
  // tile split. we add one level of hierarchy here to represent the whole
  // system level which comes before the IPU split level. each level only
  // supports certain transforms and the tile level has no partition splits as
  // it is the last level (so there is nothing to split into).
  const auto numLevelsOfHierarchy = hierarchy.size() + 1;
  assert(numLevelsOfHierarchy >= 1);
  partitionVars.clear();

  const auto getNumGrains = [](const std::size_t total,
                               const std::size_t grainSize) {
    return total ? ceildiv(total, grainSize) : 1;
  };

  const auto convGroupGrains = getNumGrains(
      transformedOnceParams.getNumConvGroups(), convGroupGrainSize[0]);
  const auto outChanGrains =
      getNumGrains(transformedOnceParams.getNumOutputChansPerConvGroup(),
                   outChanGrainSize[0]);
  const auto inChanGrains = getNumGrains(
      transformedOnceParams.getNumInputChansPerConvGroup(), inChanGrainSize[0]);

  // transformedDims is the set of dimensions that are flattened / expanded,
  // indexed by level.
  std::vector<std::unordered_set<unsigned>> transformedDims;
  transformedDims.reserve(numLevelsOfHierarchy);

  std::vector<ConvSizeVariables> convSize;
  std::vector<ConvSizeVariables> transformedConvSize;

  convSize.emplace_back();
  convSize.back().numFieldGrains.reserve(numFieldDims);
  convSize.back().kernelSize.reserve(numFieldDims);

  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const unsigned numGrains =
        ceildiv(transformedOnceParams.getOutputSize(dim), fieldGrainSize[dim]);

    convSize.back().numFieldGrains.push_back(
        m.addConstant(std::max(numGrains, 1U),
                      arrIndStr(0) + ".size.numFieldGrains" + arrIndStr(dim)));
    convSize.back().kernelSize.push_back(
        m.addConstant(std::max(transformedOnceParams.kernelShape[dim], 1UL),
                      arrIndStr(0) + ".size.kernelShape" + arrIndStr(dim)));
  }

  convSize.back().batchSize =
      m.addConstant(std::max(transformedOnceParams.getBatchSize(), 1UL),
                    arrIndStr(0) + ".size.batchSize");

  convSize.back().numConvGroupGrains = m.addConstant(
      std::max(convGroupGrains, 1UL), arrIndStr(0) + ".size.convGroupGrains");
  convSize.back().numOutChanGrains = m.addConstant(
      std::max(outChanGrains, 1UL), arrIndStr(0) + ".size.outChanGrains");
  convSize.back().numInChanGrains = m.addConstant(
      std::max(inChanGrains, 1UL), arrIndStr(0) + ".size.inChanGrains");

  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    if (level == 0) {
      transformedDims.emplace_back();
    } else {
      assert(transformedDims.capacity() != transformedDims.size());
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

      // apply expandDims transformation
      for (const auto dim : transforms[level].expandDims) {
        transformedConvSize.back().numInChanGrains =
            m.product({transformedConvSize.back().numInChanGrains,
                       transformedConvSize.back().kernelSize[dim]},
                      arrIndStr(level) + ".size.inChanGrains");
        transformedConvSize.back().kernelSize[dim] = m.addConstant(
            1, arrIndStr(level) + ".size.kernelSize" + arrIndStr(dim));
      }

      // apply outChanFlattenDims transformation
      for (const auto dim : transforms[level].outChanFlattenDims) {
        popsolver::Variable outputSize =
            transformedConvSize.back().numFieldGrains[dim];
        if (fieldGrainSize[dim] != 1) {
          outputSize =
              m.product({outputSize, m.addConstant(fieldGrainSize[dim])});
        }
        transformedConvSize.back().numOutChanGrains =
            m.product({transformedConvSize.back().numOutChanGrains, outputSize},
                      arrIndStr(level) + ".size.outChanGrains");
        popsolver::Variable inputSize;
        if (level != 0 && transformedDims[level - 1].count(dim)) {
          inputSize = outputSize;
        } else {
          inputSize = m.call(
              {outputSize, transformedConvSize.back().kernelSize[dim]},
              [dim,
               transformedOnceParams](const std::vector<unsigned> &values) {
                return getMaxInputRangeSize(values[0], dim,
                                            transformedOnceParams, values[1]);
              },
              arrIndStr(level) + ".size.inputFieldSize" + arrIndStr(dim));
        }
        transformedConvSize.back().numInChanGrains =
            m.product({transformedConvSize.back().numInChanGrains, inputSize},
                      arrIndStr(level) + ".size.inChanGrains");
        transformedConvSize.back().numFieldGrains[dim] = m.addConstant(
            1, arrIndStr(level) + ".size.numFieldGrains" + arrIndStr(dim));
      }

      // apply flattenDims transformation
      if (!transforms[level].flattenDims.empty()) {
        std::vector<Variable> vars;
        unsigned multiplier = 1;
        for (const auto dim : transforms[level].flattenDims) {
          if (dim == 0) {
            vars.push_back(transformedConvSize.back().batchSize);
            transformedConvSize.back().batchSize =
                m.addConstant(1, arrIndStr(level) + ".size.batchSize");
          } else {
            vars.push_back(transformedConvSize.back().numFieldGrains[dim - 1]);
            multiplier *= fieldGrainSize[dim - 1];
            transformedConvSize.back().numFieldGrains[dim - 1] = m.addConstant(
                1, arrIndStr(level) + ".size.numFieldGrains" + arrIndStr(dim));
          }
        }
        const auto toDim = transforms[level].flattenDims.back();
        if (toDim != 0) {
          multiplier /= fieldGrainSize[toDim - 1];
        }
        if (multiplier != 1)
          vars.push_back(m.addConstant(multiplier));
        if (toDim == 0) {
          transformedConvSize.back().batchSize =
              m.product(vars, arrIndStr(level) + ".size.batchSize");
        } else {
          transformedConvSize.back().numFieldGrains[toDim - 1] =
              m.product(vars, arrIndStr(level) + ".size.numFieldGrains" +
                                  arrIndStr(toDim - 1));
        }
      }

      // apply combineConvGroups transformation
      if (transforms[level].combineConvGroupsFactor != 1) {
        assert(transforms[level].combineConvGroupsFactor != 0);
        // to know how many input channels we have on this level we must take
        // the grain size and number of grains from the previous level.
        assert(level > 0);
        const auto factor =
            m.addConstant(transforms[level].combineConvGroupsFactor);
        // divide by the factor, rounding up in the process.
        transformedConvSize.back().numConvGroupGrains =
            m.ceildiv(transformedConvSize.back().numConvGroupGrains, factor,
                      arrIndStr(level) + ".size.numConvGroupGrains");
        // multiply by the factor.
        transformedConvSize.back().numInChanGrains =
            m.product({transformedConvSize.back().numInChanGrains, factor},
                      arrIndStr(level) + ".size.numInChanGrains");
        transformedConvSize.back().numOutChanGrains =
            m.product({transformedConvSize.back().numOutChanGrains, factor},
                      arrIndStr(level) + ".size.numOutChanGrains");
      }

      // correct the number of grains in the case that the grain size has
      // changed between two levels in the hierarchy.
      if (outChanGrainSize[level] > outChanGrainSize[level - 1]) {
        assert(outChanGrainSize[level] % outChanGrainSize[level - 1] == 0);
        const auto divisor =
            outChanGrainSize[level] / outChanGrainSize[level - 1];
        transformedConvSize.back().numOutChanGrains = m.ceildiv(
            transformedConvSize.back().numOutChanGrains, m.addConstant(divisor),
            arrIndStr(level) + ".size.outChanGrains");
      } else if (outChanGrainSize[level] < outChanGrainSize[level - 1]) {
        assert(outChanGrainSize[level - 1] % outChanGrainSize[level] == 0);
        const auto multiplier =
            outChanGrainSize[level - 1] / outChanGrainSize[level];
        transformedConvSize.back().numOutChanGrains =
            m.product({transformedConvSize.back().numOutChanGrains,
                       m.addConstant(multiplier)},
                      arrIndStr(level) + ".size.outChanGrains");
      }
      if (inChanGrainSize[level] != inChanGrainSize[level - 1]) {
        // we have no transformations currently that should decrease the
        // input channel grain size between two levels of the hierarchy.
        assert(inChanGrainSize[level] > inChanGrainSize[level - 1]);

        assert(inChanGrainSize[level] % inChanGrainSize[level - 1] == 0);
        const auto divisor =
            inChanGrainSize[level] / inChanGrainSize[level - 1];
        transformedConvSize.back().numInChanGrains = m.ceildiv(
            transformedConvSize.back().numInChanGrains, m.addConstant(divisor),
            arrIndStr(level) + ".size.inChanGrains");
      }
    }

    // the last level in the hierarchy is always the tile split. this level does
    // not support partition splits so jump out the loop now.
    if (level + 1 == numLevelsOfHierarchy) {
      break;
    }

    const auto &prevConvSize = transformedConvSize.back();
    ConvSizeVariables nextConvSize;
    convSize.back().numFieldGrains.reserve(numFieldDims);
    convSize.back().kernelSize.reserve(numFieldDims);
    const auto levelMaxSplit = hierarchy[level];
    PartitionVariables p;
    p.fieldSplit.reserve(numFieldDims);
    p.kernelSplit.reserve(numFieldDims);

    for (unsigned dim = 0; dim != numFieldDims; ++dim) {
      p.fieldSplit.push_back(m.addVariable(
          1, levelMaxSplit,
          arrIndStr(level) + ".partition.fieldSplit" + arrIndStr(dim)));
      m.lessOrEqual(p.fieldSplit.back(), prevConvSize.numFieldGrains[dim]);
      // Currently the implementation doesn't support splitting the inner-most
      // kernel dimension. TODO: T12883 Lift this restriction.
      if (dim == numFieldDims - 1) {
        p.kernelSplit.push_back(m.addConstant(
            1, arrIndStr(level) + ".partition.kernelSplit" + arrIndStr(dim)));
      } else {
        p.kernelSplit.push_back(m.addVariable(
            1, levelMaxSplit,
            arrIndStr(level) + ".partition.kernelSplit" + arrIndStr(dim)));
        m.lessOrEqual(p.kernelSplit.back(), prevConvSize.kernelSize[dim]);
      }
      nextConvSize.numFieldGrains.push_back(m.ceildivConstrainDivisor(
          prevConvSize.numFieldGrains[dim], p.fieldSplit.back(),
          arrIndStr(level + 1) + ".size.numFieldGrains" + arrIndStr(dim)));
      nextConvSize.kernelSize.push_back(m.ceildivConstrainDivisor(
          prevConvSize.kernelSize[dim], p.kernelSplit.back(),
          arrIndStr(level + 1) + ".size.kernelSize" + arrIndStr(dim)));
    }
    p.batchSplit = m.addVariable(1, levelMaxSplit,
                                 arrIndStr(level) + ".partition.batchSplit");
    m.lessOrEqual(p.batchSplit, prevConvSize.batchSize);
    p.convGroupSplit = m.addVariable(
        1, levelMaxSplit, arrIndStr(level) + ".partition.convGroupSplit");
    m.lessOrEqual(p.convGroupSplit, prevConvSize.numConvGroupGrains);
    // The joint planning cost function assumes that no exchange is required to
    // rearrange weights between passes. Because of the way we derive the
    // backward and weight update plans from the forward plan this is guaranteed
    // to be the case if each weight is used on exactly one tile in the forward
    // pass. Disallow splitting of fully connected batch (or equivalently the
    // convolutional output channels) across tiles to ensure this holds.
    if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
      p.outChanSplit.parallel = m.addConstant(
          1, arrIndStr(level) + ".partition.outChanSplit.parallel");
    } else {
      assert(!isJointPlan);
      p.outChanSplit.parallel =
          m.addVariable(1, levelMaxSplit,
                        arrIndStr(level) + ".partition.outChanSplit.parallel");
    }

    // We only support splitting serially in the IPU level of the hierarchy.
    // This is always the penultimate level.
    // TODO: T10037 For now we do not attempt to serially split any plan
    // that has an inter-IPU level split.
    assert(numLevelsOfHierarchy >= 2);
    if (numLevelsOfHierarchy == 2 && level == numLevelsOfHierarchy - 2) {
      // TODO: T10408 We do not support splitting the output channels serially
      // during a joint plan as that will become an input channel serial split
      // during the weight update, which is not currently supported.
      if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
        p.outChanSplit.serial = m.addConstant(
            1, arrIndStr(level) + ".partition.outChanSplit.serial");
      } else {
        p.outChanSplit.serial =
            addPartitionConstant(m, options, level, "outChanSplit.serial", [&] {
              return m.addVariable(1, levelMaxSplit);
            });
      }

      // we must avoid splitting the convolutions serially when it will
      // produce different sized convolutions as this is implemented as a
      // repeat loop of the same sub-convolution. we enforce this by
      // requiring that the serial split is a factor of the total number of
      // output channels.
      const auto initialOutputChansPerGroup =
          transformedViewParams.getNumOutputChansPerConvGroup();
      m.factorOf(std::max(initialOutputChansPerGroup, 1ul),
                 p.outChanSplit.serial);
    } else {
      p.outChanSplit.serial =
          m.addConstant(1, arrIndStr(level) + ".partition.outChanSplit.serial");
    }

    auto totalOutChanSplit =
        m.product({p.outChanSplit.parallel, p.outChanSplit.serial});
    m.lessOrEqual(totalOutChanSplit, prevConvSize.numOutChanGrains);

    p.inChanSplit = m.addVariable(1, levelMaxSplit,
                                  arrIndStr(level) + ".partition.inChanSplit");
    m.lessOrEqual(p.inChanSplit, prevConvSize.numInChanGrains);

    p.convGroupGrainSize = convGroupGrainSize[level];
    p.outChanGrainSize = outChanGrainSize[level];
    p.inChanGrainSize = inChanGrainSize[level];
    p.fieldGrainSize = fieldGrainSize;

    nextConvSize.batchSize =
        m.ceildivConstrainDivisor(prevConvSize.batchSize, p.batchSplit,
                                  arrIndStr(level + 1) + ".size.batchSize");

    nextConvSize.numConvGroupGrains = m.ceildivConstrainDivisor(
        prevConvSize.numConvGroupGrains, p.convGroupSplit,
        arrIndStr(level + 1) + ".size.convGroupGrains");
    if (level == numLevelsOfHierarchy - 2 &&
        convVertexType.method == Plan::Method::SLIC) {
      // Like popsolver::Model::ceildiv(ConstrainDivisor) but with already
      // instantiated output variables.
      auto ceildivWithVars = [&](Model &m, const Variable &result,
                                 const Variable &left, const Variable &right) {
        auto resultTimesRight = m.product({result, right});
        m.lessOrEqual(left, resultTimesRight);
        m.less(resultTimesRight, m.sum({left, right}));
      };
      auto ceildivConstrainDivisorWithVars =
          [&](Model &m, const Variable &result, const Variable &left,
              const Variable &right) {
            ceildivWithVars(m, result, left, right);
            m.less(m.product({result, right}), m.sum({left, result}));
          };
      // Workaround for T17645. We would constrain in/out chan grains to be
      // equal 1 in the model except the call constraint later on that
      // calculates the cycle estimates for SLIC relies on never getting
      // invalid parameters, i.e. when there is greater than one input
      // or output channel group. The call constraint may be evaluated
      // before the equal constraint and so we get invalid parameters and
      // an assert.
      //
      // Ideally we would fix this by having the call constraint be able
      // to flag invalid parameters (T17632). We don't yet have this
      // and we instead construct the num{In,Out}ChanGrains as constants
      // and apply constraints backwards to ensure they are never not
      // equal 1.
      nextConvSize.numOutChanGrains =
          m.addConstant(1u, arrIndStr(level + 1) + ".size.outChanGrains");
      nextConvSize.numInChanGrains =
          m.addConstant(1u, arrIndStr(level + 1) + ".size.inChanGrains");
      ceildivConstrainDivisorWithVars(m, nextConvSize.numOutChanGrains,
                                      prevConvSize.numOutChanGrains,
                                      totalOutChanSplit);
      ceildivConstrainDivisorWithVars(m, nextConvSize.numInChanGrains,
                                      prevConvSize.numInChanGrains,
                                      p.inChanSplit);
    } else {
      nextConvSize.numOutChanGrains = m.ceildivConstrainDivisor(
          prevConvSize.numOutChanGrains, totalOutChanSplit,
          arrIndStr(level + 1) + ".size.outChanGrains");
      nextConvSize.numInChanGrains = m.ceildivConstrainDivisor(
          prevConvSize.numInChanGrains, p.inChanSplit,
          arrIndStr(level + 1) + ".size.inChanGrains");
    }

    convSize.push_back(std::move(nextConvSize));

    applyPartitionPlanConstraint(m, options, level, p);
    partitionVars.push_back(std::move(p));
  }

  {
    // We only apply these constraints at the tile-split level.
    const auto ipuLevel = numLevelsOfHierarchy - 2;
    const auto tileLevel = numLevelsOfHierarchy - 1;

    addMethodConstraints(m, convVertexType.method, partitionVars[ipuLevel],
                         convSize[tileLevel], transformedOnceParams);
  }

  const auto usedTiles = getUsedTiles(m, partitionVars, hierarchy);

  const auto method = convVertexType.method;
  const auto slicWindowWidth = convVertexType.slicWindowWidth;
  const auto numConvUnitsRequired = convVertexType.numConvUnitsRequired;

  auto e = addEstimates(
      m, partitionVars, convSize, transformedConvSize, usedTiles,
      transformedDims, target, perLevelExchangeBytesPerCycle,
      untransformedParams, transformedOnceParams, transformedOnceUnpaddedParams,
      isJointPlan, convGroupsPerGroup, inChansPerGroup, partialChansPerGroup,
      types, transforms, method, slicWindowWidth, numConvUnitsRequired,
      Plan::LinearizeTileOrder::STANDARD, options, cache);

  if (isJointPlan) {
    assert(options.pass == Pass::FC_TRAINING_FWD);

    const auto bwd = addBwdEstimates(
        m, untransformedParams, transformedOnceParams,
        transformedOnceUnpaddedParams, numLevelsOfHierarchy, partitionVars,
        convSize, transforms, method, slicWindowWidth, numConvUnitsRequired,
        usedTiles, target, perLevelExchangeBytesPerCycle, types, isJointPlan,
        convGroupsPerGroup, inChansPerGroup, partialChansPerGroup, options,
        cache);

    const auto wu = addWuEstimates(
        m, untransformedParams, transformedOnceParams,
        transformedOnceUnpaddedParams, numLevelsOfHierarchy, partitionVars,
        convSize, transforms, method, slicWindowWidth, numConvUnitsRequired,
        usedTiles, target, numFieldDims, perLevelExchangeBytesPerCycle, types,
        isJointPlan, convGroupsPerGroup, inChansPerGroup, partialChansPerGroup,
        options, cache);

    if (objective.getTileTempMemoryBound() > 0) {
      auto bound = objective.getTileTempMemoryBound();
      // fwd temp bytes constrained below
      m.lessOrEqual(bwd.totalTempBytes, bound);
      m.lessOrEqual(wu.totalTempBytes, bound);
    }

    // report the total cycles of all three phases.
    e.totalCycles =
        m.sum({e.totalCycles, bwd.totalCycles, wu.totalCycles}, "totalCycles");

    // report the max requirement of all three phases
    e.totalTempBytes =
        m.max({e.totalTempBytes, bwd.totalTempBytes, wu.totalTempBytes},
              "maxTempBytesPerTile");
  }

  // if an explicit cycle or memory bound has been added to the objective then
  // enforce that. additionally, depending on the object type prune the
  // relevant variable based upon the best plan found so far.
  auto cyclesBound = objective.getCyclesBound();
  auto memoryBound = objective.getTileTempMemoryBound();

  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    cyclesBound = std::min(cyclesBound, bestCost.totalCycles);
    break;
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    memoryBound = std::min(memoryBound, bestCost.totalTempBytes);
    break;
  }

  m.lessOrEqual(e.totalCycles, cyclesBound);
  m.lessOrEqual(e.totalTempBytes, memoryBound);

  return e;
}

static std::pair<Plan, Cost> choosePlan(
    const poplar::Target &target, const std::vector<ConvTransform> &transforms,
    const std::vector<ConvTypes> &types, const std::vector<unsigned> &hierarchy,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<unsigned> &fieldGrainSize,
    const ConvVertexType &convVertexType, const ConvParams &params,
    bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
    PlanningCacheImpl::CycleEstimationImpl *cache, const ConvOptions &options) {
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  Estimates<popsolver::Variable> e = constructModel(
      target, transforms, types, hierarchy, perLevelExchangeBytesPerCycle,
      fieldGrainSize, convVertexType, params, isJointPlan, bestCost, objective,
      cache, options, m, partitionVars);
  popsolver::Solution s;

  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    s = m.minimize({e.totalCycles, e.totalTempBytes});
    break;
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    s = m.minimize({e.totalTempBytes, e.totalCycles});
    break;
  }
  if (!s.validSolution()) {
    return {Plan(), highestCost};
  }
  std::vector<Partition> partitions;
  for (const auto &p : partitionVars) {
    partitions.push_back(makePartition(s, p));
  }
  Plan plan(std::move(partitions), std::move(types),
            convVertexType.convGroupsPerGroup, convVertexType.inChansPerGroup,
            convVertexType.partialChansPerGroup, convVertexType.slicWindowWidth,
            convVertexType.numConvUnitsRequired, convVertexType.method,
            Plan::LinearizeTileOrder::STANDARD,
            getStartTile(target, params, options, isJointPlan), isJointPlan);
  plan.transforms = transforms;

  Cost cost;
  cost.totalCycles = s[e.totalCycles];
  cost.totalTempBytes = s[e.totalTempBytes];

  cost.rearrangeBeforeSliceCycles = s[e.rearrangeBeforeSliceCycles];
  cost.dynamicSliceCycles = s[e.dynamicSliceCycles];
  cost.transformCycles = s[e.transformCycles];
  cost.exchangeCycles = s[e.exchangeCycles];
  cost.tileLevelTransformCycles = s[e.tileLevelTransformCycles];
  cost.partialCalcCycles = s[e.partialCalcCycles];
  cost.reduceCycles = s[e.reduceCycles];
  cost.dynamicUpdateCycles = s[e.dynamicUpdateCycles];

  cost.rearrangeBeforeSliceTempBytes = s[e.rearrangeBeforeSliceTempBytes];
  cost.rearrangeBeforeSliceTempDuringRearrangeBytes =
      s[e.rearrangeBeforeSliceTempDuringRearrangeBytes];
  cost.transformTempBytes = s[e.transformTempBytes];
  cost.tileLevelTransformTempBytes = s[e.tileLevelTransformTempBytes];
  cost.convTempBytes = s[e.convTempBytes];
  cost.reduceTempBytes = s[e.reduceTempBytes];

  return {plan, cost};
}

static void getConvVertexMACCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<unsigned>("convGroupsPerGroup");
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<unsigned>("partialChansPerGroup");

  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnits =
      getNumConvUnits(floatActivations, ampFloatPartials, target);

  // Constrain the input channel grouping to a multiple of two if the activation
  // type is half. This ensures that we never need to apply padding when sending
  // activations over the exchange.
  auto grainSize = floatActivations ? 1u : 2u;
  const auto roundedNumInChans =
      roundUp(params.getNumInputChansPerConvGroup(), grainSize);

  const unsigned convGroupsPerGroup = 1;
  // This is the only supported convGroupsPerGroup for this method.
  if (constrainedConvGroupsPerGroup &&
      *constrainedConvGroupsPerGroup != convGroupsPerGroup) {
    return;
  }

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
  for (unsigned inChansPerGroup = inChansLower; inChansPerGroup <= inChansUpper;
       inChansPerGroup += grainSize) {
    unsigned inChanGroups =
        (roundedNumInChans + inChansPerGroup - 1) / inChansPerGroup;
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

    // The MAC vertex does not require a grouping of the conv groups.
    const unsigned convGroupsPerGroup = 1;

    candidates.emplace_back(Plan::Method::MAC, inputType, outputType,
                            partialType, convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup, numConvUnits, numConvUnits);
    previousInChanGroups = inChanGroups;
  }
}

static void getConvVertexAMPCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<unsigned>("partialChansPerGroup");
  const auto constrainedNumConvUnits =
      planConstraints.get_optional<unsigned>("numAmpConvUnits");

  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnitsOnIpu =
      getNumConvUnits(floatActivations, ampFloatPartials, target);
  if (numConvUnitsOnIpu == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnitsOnIpu =
        getNumConvUnits(floatActivations, ampFloatPartials, target);
  }
  auto ampPartialType = ampFloatPartials ? poplar::FLOAT : poplar::HALF;
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   target)) {
    const unsigned weightsPerConvUnit =
        target.getWeightsPerConvUnit(floatActivations);

    std::vector<unsigned> partialChansCandidates = {numConvUnitsOnIpu,
                                                    weightsPerConvUnit};
    std::vector<unsigned> numConvUnitsCandidates = {numConvUnitsOnIpu};
    // On IPU1 we support half of conv units configuration for HALF types
    const bool canUseAmp4 = options.enableAmpHalfEnginesPlan &&
                            target.getFp16InFp16OutConvUnitsPerTile() == 8 &&
                            !floatActivations;

    // On IPU2 in case of half partials we need to enable 8 engines config as
    // well
    const bool canUseAmp8 = numConvUnitsOnIpu == 16;

    if (canUseAmp4 || canUseAmp8) {
      numConvUnitsCandidates.push_back(numConvUnitsOnIpu / 2);
      partialChansCandidates.push_back(numConvUnitsOnIpu / 2);
    }

    for (auto convUnits : numConvUnitsCandidates) {
      for (unsigned inputs = weightsPerConvUnit; inputs >= 1; inputs--) {
        for (auto partials : partialChansCandidates) {
          // Input channels constrain
          if (constrainedInChansPerGroup &&
              inputs != *constrainedInChansPerGroup) {
            continue;
          }

          // Partial channels constrain
          if (constrainedPartialChansPerGroup &&
              partials != *constrainedPartialChansPerGroup) {
            continue;
          }

          // Number of conv units constrain
          if (constrainedNumConvUnits &&
              convUnits != *constrainedNumConvUnits) {
            continue;
          }

          unsigned usedWeightsPerConvUnit =
              weightsPerConvUnit * convUnits / numConvUnitsOnIpu;
          if (!floatActivations &&
              (partials != convUnits && partials != usedWeightsPerConvUnit)) {
            continue;
          }

          if (!canUseConvolutionInstruction(floatActivations, floatPartials,
                                            inputs, convUnits, partials,
                                            target)) {
            continue;
          }

          // There are two reasons we might choose to make partialChansPerGroup
          // not equal to numConvUnitsOnIpu:
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
          if (options.pass == Pass::FC_INFERENCE_FWD && partials != convUnits) {
            continue;
          }

          if (isJointPlan) {
            assert(options.pass == Pass::FC_TRAINING_FWD);
            // The input channels in the forward pass become the output channels
            // of the weight update pass. Make sure it is a multiple of the
            // supported output channels per group.
            if (inputs % convUnits != 0) {
              continue;
            }
          }

          // AMP only supports a conv group grouping of 1.
          const unsigned convGroupsPerGroup = 1;

          candidates.emplace_back(Plan::Method::AMP, inputType, outputType,
                                  ampPartialType, convGroupsPerGroup, inputs,
                                  partials, 0, convUnits);
        }
      }
    }
  }
}

static void getConvVertexSLICCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  // TODO: SLIC is currently only supported for half->float operations.
  if (inputType != poplar::HALF || partialType != poplar::FLOAT) {
    return;
  }

  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<unsigned>("convGroupsPerGroup");
  const auto constrainedSlicWindowWidth =
      planConstraints.get_optional<unsigned>("slicWindowWidth");

  const auto constrainedChansPerGroup = [&]() -> boost::optional<unsigned> {
    const auto constrainedInChansPerGroup =
        planConstraints.get_optional<unsigned>("inChansPerGroup");
    const auto constrainedPartialChansPerGroup =
        planConstraints.get_optional<unsigned>("partialChansPerGroup");

    if (constrainedInChansPerGroup && constrainedPartialChansPerGroup &&
        *constrainedInChansPerGroup != *constrainedPartialChansPerGroup) {
      throw poputil::poplibs_error("SLIC requires the input and output channel "
                                   "grouping to be the same.");
    }

    if (constrainedInChansPerGroup) {
      return constrainedInChansPerGroup;
    } else if (constrainedPartialChansPerGroup) {
      return constrainedPartialChansPerGroup;
    } else {
      return boost::none;
    }
  }();

  const bool floatActivations = inputType == poplar::FLOAT;
  const bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnits =
      getNumConvUnits(floatActivations, ampFloatPartials, target);
  if (numConvUnits == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnits = getNumConvUnits(floatActivations, ampFloatPartials, target);
  }
  const auto ampPartialType = ampFloatPartials ? poplar::FLOAT : poplar::HALF;
  const unsigned weightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);

  // the numbers below are hardcoded but dependent on the expected machine
  // model that the real hardware models. ie. we expect 8 conv units and
  // 16 weights per conv unit (as SLIC currently only supports input type as
  // half and partial type as float).
  if (numConvUnits != 8) {
    throw poputil::poplibs_error(
        "Unsupported number of conv units for the SLIC instruction.");
  }

  if (weightsPerConvUnit != 16) {
    throw poputil::poplibs_error("Unsupported number of weights per conv "
                                 "unit for the SLIC instruction.");
  }

  // TODO: T14626, add a vertex for the the 1x3 kernel window size.
  const unsigned slicWindowWidth = constrainedSlicWindowWidth.value_or(4);

  if (isJointPlan) {
    assert(options.pass == Pass::FC_TRAINING_FWD);
    // There are a number of transformations between different passes when a
    // joint plan is being used which would need updating to handle SLIC.
    // T17666 tracks this. For the timebeing, don't allow joint plans with
    // SLIC.
    return;
  }

  struct Candidate {
    unsigned groups;
    unsigned channels;
  };
  std::array<Candidate, 3> groupings{Candidate{1u, 4u}, Candidate{2u, 2u},
                                     Candidate{4u, 1u}};

  for (const auto &grouping : groupings) {
    if (constrainedConvGroupsPerGroup &&
        *constrainedConvGroupsPerGroup != grouping.groups) {
      continue;
    }

    if (constrainedChansPerGroup &&
        *constrainedChansPerGroup != grouping.channels) {
      continue;
    }

    candidates.emplace_back(Plan::Method::SLIC, inputType, outputType,
                            ampPartialType, grouping.groups, grouping.channels,
                            grouping.channels, slicWindowWidth, 0);
  }
}

static void getConvVertexOuterProductCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<unsigned>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<unsigned>("partialChansPerGroup");

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
  // OuterProduct only implemented for when Tile.PartialType == input type.
  if (partialType != params.inputType) {
    return;
  }

  // The OuterProduct vertex does not require a grouping of the conv groups.
  const unsigned convGroupsPerGroup = 1;

  candidates.emplace_back(Plan::Method::OUTER_PRODUCT, inputType, outputType,
                          inputType, convGroupsPerGroup, inChansPerGroup,
                          partialChansPerGroup, 0, 0);
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            poplar::Type inputType, poplar::Type outputType,
                            poplar::Type partialType, const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedMethod = [&]() -> boost::optional<Plan::Method> {
    const auto constraint = planConstraints.get_optional<std::string>("method");
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

    // Disable SLIC until T18365 is fixed
    bool disableSLIC = options.pass == Pass::FC_INFERENCE_FWD ||
                       options.pass == Pass::FC_TRAINING_BWD ||
                       options.pass == Pass::FC_TRAINING_FWD ||
                       options.pass == Pass::FC_TRAINING_WU;

    // the order here should be in most-likely-best first for performance
    // because the planner constrains future models against the current best.
    methodCandidates = {
        Plan::Method::AMP,
        Plan::Method::SLIC,
        Plan::Method::MAC,
        Plan::Method::OUTER_PRODUCT,
    };

    if (disableSLIC) {
      methodCandidates.erase(methodCandidates.begin() + 1);
    }
  }

  // All the following methods assume half or float input/partial types.
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);

  std::vector<ConvVertexType> convVertexTypeCandidates;
  for (const auto &method : methodCandidates) {
    switch (method) {
    case Plan::Method::MAC: {
      getConvVertexMACCandidates(target, inputType, outputType, partialType,
                                 params, options, isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::AMP: {
      getConvVertexAMPCandidates(target, inputType, outputType, partialType,
                                 params, options, isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::SLIC: {
      getConvVertexSLICCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Plan::Method::OUTER_PRODUCT: {
      getConvVertexOuterProductCandidates(
          target, inputType, outputType, partialType, params, options,
          isJointPlan, convVertexTypeCandidates);
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
getExpandDimsCandidates(unsigned ipuLevel, const ConvParams &params,
                        const ConvOptions &options) {
  const auto &planConstraints = options.planConstraints;
  const auto constraint = planConstraints.get_child_optional(
      std::to_string(ipuLevel) + ".transform.expandDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    for (const auto &child : *constraint) {
      forcedDims.push_back(child.second.get_value<unsigned>());
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
      // TODO: T12884 Investigate better heuristics.
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
getOutChanFlattenDimsCandidates(unsigned ipuLevel, const ConvParams &params,
                                const ConvOptions &options) {
  auto swappedParams = params;
  const auto &planConstraints = options.planConstraints;
  const auto constraint = planConstraints.get_child_optional(
      std::to_string(ipuLevel) + ".transform.outChanFlattenDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    for (const auto &child : *constraint) {
      forcedDims.push_back(child.second.get_value<unsigned>());
    }
    std::sort(forcedDims.begin(), forcedDims.end());
    forcedDims.erase(std::unique(forcedDims.begin(), forcedDims.end()),
                     forcedDims.end());
    std::reverse(forcedDims.begin(), forcedDims.end());
    candidateDimSets.emplace_back(std::move(forcedDims));
  } else {
    if (params.outputChannelsPerConvGroup)
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
      // down the size of the search space. TODO: T12884 Investigate better
      // heuristics.
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

void swapOperands(ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> extraInputPadding(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto transformedInputSize = params.getTransformedInputSize(dim);
    const auto transformedKernelSize = params.getTransformedKernelSize(dim);
    extraInputPadding[dim] = transformedInputSize - transformedKernelSize;
  }
  std::swap(params.inputFieldShape, params.kernelShape);
  std::swap(params.inputTransform, params.kernelTransform);
  std::swap(params.batchSize, params.outputChannelsPerConvGroup);
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
    // TODO: T12885 Lift this restriction.
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
  } else if (!params.outputChannelsPerConvGroup) {
    // Avoid swapping operands when output channels could be swapped with batch
    // size
    validValues = {false};
  }

  return validValues;
}

std::vector<unsigned> getTileHierarchy(const poplar::Target &target) {
  std::vector<double> dummy;
  return poplibs::getTileHierarchy(target, dummy);
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
      auto levelResultType = isTileLevel ? options.interTilePartialsType
                                         : options.interIpuPartialsType;
      // Use the result type of the previous level if it is smaller than the
      // requested result type. This means that if a user wants to use half
      // partials they only need to set the option for the first level that
      // should use half partials.
      if (!isTileLevel && target.getTypeSize(levelResultType) >
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

static std::vector<unsigned> getDilatePostConvDims(const ConvParams &params) {
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

#ifndef NDEBUG
static bool isPowerOf2(const unsigned n) {
  if (n == 0) {
    return false;
  }
  return (n & (n - 1)) == 0;
}
#endif

static std::vector<unsigned> getCombineConvGroupCandidates(
    const unsigned level, const ConvParams &params, const ConvOptions &options,
    const poplar::Target &target, const bool isJointPlan) {

  std::string transform = std::to_string(level) + ".transform.";
  std::vector<unsigned> validValues = [&] {
    // when we have more than one conv group and one input channel we want to
    // try this transformation.
    const auto ci = params.inputChannelsPerConvGroup;
    const bool validInputChannelSize =
        (params.inputType == poplar::FLOAT && ci == 1) ||
        (params.inputType == poplar::HALF && (ci == 1 || ci == 2));

    // Joint plans may invalidate this transformation if they, for example, swap
    // the input channels with the batch size and the batch size does not
    // satisfy the constraint above. TODO: T12886 With a more advanced check
    // here we could support cases like this.
    if (validInputChannelSize && params.numConvGroups > 1 && !isJointPlan) {
      const auto baseLoadElements =
          params.inputType == poplar::HALF
              ? target.getFp16ConvUnitInputLoadElemsPerCycle()
              : target.getFp32ConvUnitInputLoadElemsPerCycle();

      unsigned minFactor = convGroupCombineFactor(baseLoadElements, ci);
      const unsigned maxFactor =
          (params.inputType == poplar::HALF ? 16U : 8U) / ci;

      assert(isPowerOf2(baseLoadElements));
      assert(isPowerOf2(ci));
      assert(isPowerOf2(maxFactor));
      std::vector<unsigned> result;
      result.push_back(1U); // 1 is noop transform
      while (minFactor <= maxFactor) {
        result.push_back(minFactor);
        minFactor *= 2;
      }
      return result;
    } else {
      return std::vector<unsigned>{1U};
    }
  }();

  const auto &planConstraints = options.planConstraints;
  const auto constraint_ =
      planConstraints.get_child_optional(transform + "combineConvGroupsFactor");
  if (constraint_) {
    std::set<unsigned> constraints;
    for (const auto &child : *constraint_) {
      constraints.insert(child.second.get_value<unsigned>());
    }
    if (std::any_of(constraints.begin(), constraints.end(),
                    [](unsigned i) { return i != 1U; })) {
      const auto expandDimsConstraint =
          planConstraints.get_child_optional(transform + "expandDims");
      const auto outChanFlattenDimsConstraint =
          planConstraints.get_child_optional(transform + "outChanFlattenDims");
      if ((expandDimsConstraint && !expandDimsConstraint->empty()) ||
          (outChanFlattenDimsConstraint &&
           !outChanFlattenDimsConstraint->empty())) {
        throw poputil::poplibs_error(
            "The combineConvGroups transformation is only valid when there is "
            "there is not another transformation that can increase the number "
            "of input channels (ie. expandDims or outChanFlattenDims");
      }
    }

    auto constrainedValues =
        boost::adaptors::filter(validValues, [&](unsigned i) {
          return static_cast<bool>(constraints.count(i));
        });
    return std::vector<unsigned>(constrainedValues.begin(),
                                 constrainedValues.end());
  }

  return validValues;
}

/*
 * Function ensures:
 * 1. Each level specified in plan constraints is within range of hierarchy.
 * 2. Each value within transform.expandDims and transform.outChanFlattenDims
 *    arrays are a valid field dimension.
 * 3. The key of each child of partition.fieldSplit and partition.kernelSplit
 *    is a valid field or kernel dimension, respectively.
 */
void validatePlanConstraints(
    const ConvParams &params,
    const poplibs_support::PlanConstraints &planConstraints,
    const std::size_t numLevels) {
  const struct {
    std::string key;
    bool checkKey; // If false, each element of value array will be validated.
    std::size_t maximum;
  } keysToCheck[] = {
      {"transform.expandDims", false, params.getNumFieldDims()},
      {"transform.outChanFlattenDims", false, params.getNumFieldDims()},
      {"partition.fieldSplit", true, params.getNumFieldDims()},
      {"partition.kernelSplit", true, params.kernelShape.size()},
  };

  auto isNumeric = [](const std::string &text) -> bool {
    return !text.empty() && std::all_of(text.begin(), text.end(), ::isdigit);
  };

  auto isValidKey = [&isNumeric](const std::string &key,
                                 const std::size_t maximum) -> bool {
    if (!isNumeric(key)) {
      throw poputil::poplibs_error("Invalid key - must be numeric: " + key);
    }
    return std::stoul(key) >= maximum;
  };

  for (const auto &kv : planConstraints) {
    if (!isNumeric(kv.first)) {
      continue; // No further checks for non-numeric keys.
    }

    if (std::stoul(kv.first) >= numLevels) {
      throw poputil::poplibs_error("Plan constraint " + kv.first +
                                   " is not a valid level of hierarchy.");
    }
    for (const auto &entry : keysToCheck) {
      if (const auto &child = kv.second.get_child_optional(entry.key)) {
        for (const auto &childKV : *child) {
          if (entry.checkKey
                  ? isValidKey(childKV.first, entry.maximum)
                  : childKV.second.get_value<unsigned>() >= entry.maximum) {
            throw poputil::poplibs_error(
                "Invalid plan constraint: " + kv.first + "." + entry.key + "." +
                childKV.first + (entry.checkKey ? " Key" : " Value") +
                " out-of-range -- maximum: " + std::to_string(entry.maximum));
          }
        }
      }
    }
  }
}

static void logPlanBreakdown(logging::Level l, const Plan &plan,
                             const Cost &cost) {
  logging::log(l, "  breakdown of memory and cycle estimates:");
  logging::log(l, "   - total parallel split: {}", plan.totalParallelSplit());
  logging::log(l, "   - total serial split: {}", plan.totalSerialSplit());
  logging::log(l,
               "   - rearrangement before slice: {} cycles, {} bytes ({} "
               "overhead, {} per-loop iteration)",
               cost.rearrangeBeforeSliceCycles,
               cost.rearrangeBeforeSliceTempBytes +
                   cost.rearrangeBeforeSliceTempDuringRearrangeBytes,
               cost.rearrangeBeforeSliceTempBytes,
               cost.rearrangeBeforeSliceTempDuringRearrangeBytes);
  logging::log(l, "   - dynamic slice: {} cycles, unknown bytes",
               cost.dynamicSliceCycles);
  logging::log(l, "   - transform: {} cycles, {} bytes", cost.transformCycles,
               cost.transformTempBytes);
  logging::log(l, "   - exchange: {} cycles, n/a bytes", cost.exchangeCycles);
  logging::log(l, "   - tile level transform: {} cycles, {} bytes",
               cost.tileLevelTransformCycles, cost.tileLevelTransformTempBytes);
  logging::log(l, "   - compute: {} cycles, {} bytes", cost.partialCalcCycles,
               cost.convTempBytes);
  logging::log(l, "   - reduction: {} cycles, {} bytes", cost.reduceCycles,
               cost.reduceTempBytes);
  logging::log(l, "   - dynamic update: {} cycles, unknown bytes",
               cost.dynamicUpdateCycles);
  logging::log(l, "   - total: {} cycles, {} bytes", cost.totalCycles,
               cost.totalTempBytes);
}

static std::pair<Plan, Cost>
createPlan(ConvParams params, const ConvOptions &options, bool isJointPlan,
           const PlanningObjective &objective, const poplar::Target &target,
           PlanningCacheImpl::CycleEstimationImpl *cache) {
  validateLayerParams(params, options, target);

  // T8972: It is currently assumed that the parameters for all the training
  // passes can be derived from one pass, but this is no longer the case since a
  // different outputType can be specified for each pass. To avoid a costly
  // exchange of weights, we plan with the assumption that
  // outputType == inputType for FC_TRAINING.
  const auto originalOutputType = params.outputType;
  if (isJointPlan) {
    params.outputType = params.inputType;
  }

  // perLevelExchangeBytesPerCycle is indexed by hierarchy (not including the
  // tile level), lower indices to higher hierarchies.
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy =
      poplibs::getTileHierarchy(target, perLevelExchangeBytesPerCycle);
  const auto numLevels = hierarchy.size() + 1;

  validatePlanConstraints(params, options.planConstraints, numLevels);

  Cost bestCost = highestCost;
  Plan bestPlan;
  std::vector<ConvTransform> transforms(numLevels);
  const auto convTypes =
      getConvTypes(target, numLevels, params.outputType, options);
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
  auto paramsWithDeferredDilation = calculateParamsWithDeferredDilation(
      paramsWithExtraDims, transforms[0].dilatePostConv);

  for (bool swapOperands : getSwapOperandCandidates(paramsWithDeferredDilation,
                                                    options, isJointPlan)) {
    transforms[0].swapOperands = swapOperands;
    auto swappedParams =
        calculateSwappedParams(paramsWithDeferredDilation, swapOperands);

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

        for (const unsigned combineConvGroups : getCombineConvGroupCandidates(
                 ipuLevel, flattenedParams, options, target, isJointPlan)) {
          transforms[ipuLevel].combineConvGroupsFactor = combineConvGroups;
          const auto groupedParams = calculateGroupedParams(
              flattenedParams, transforms[ipuLevel].combineConvGroupsFactor);

          const auto convVertexTypeCandidates = getConvVertexTypeCandidates(
              target, params.inputType, params.outputType,
              convTypes.back().partialType, groupedParams, options,
              isJointPlan);

          for (const auto &convVertexType : convVertexTypeCandidates) {
            std::vector<unsigned> fieldGrainSize(numFieldDims, 1);
            if (isJointPlan) {
              assert(options.pass == Pass::FC_TRAINING_FWD);
              // The innermost grain size becomes the inChansPerGroup in the
              // backward pass. For now assume the same grouping in both
              // passes.
              // TODO: T12887 Search for the optimal grouping in each pass.
              fieldGrainSize.back() = convVertexType.inChansPerGroup;
            } else if (groupedParams.outputType == poplar::HALF &&
                       convVertexType.partialChansPerGroup % 2 &&
                       groupedParams.getOutputSize(
                           groupedParams.getNumFieldDims() - 1) %
                               2 ==
                           0) {
              // If the number of output channels per group is odd then use a
              // field grain size of 2 to ensure the result has an even number
              // of elements on each tile since an odd number of elements
              // on a tile tends to cause costly rearrangements in the next
              // layer.
              fieldGrainSize.back() = 2;
            }
            Plan candidate;
            Cost candidateCost;
            // Override the partials type at the tile level with that chosen
            // for the vertex type as we may choose a lower precision to
            // implement the operation if we know the vertex can effectively
            // maintain the accuracy implied by the requested partials type.
            auto newConvTypes = convTypes;
            newConvTypes.back().partialType = convVertexType.partialType;
            std::tie(candidate, candidateCost) = choosePlan(
                target, transforms, newConvTypes, hierarchy,
                perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType,
                params, isJointPlan, bestCost, objective, cache, options);
            if (candidateCost == highestCost) {
              continue;
            }

            if (objective.lowerCost(candidateCost, bestCost)) {
              bestPlan = candidate;
              bestCost = candidateCost;

              logging::debug("Found new best candidate plan: {}",
                             candidateCost);
              logPlanBreakdown(logging::Level::Trace, bestPlan, bestCost);
            }
          }
        }
      }
    }
  }

  if (isJointPlan && bestCost != highestCost) {
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
                            const ConvOptions &options, Pass pass) {
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
  default:
    assert(0 && "Unexpected pass");
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
  default:
    assert(0 && "Unexpected pass");
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
      1,                         // batchSize
      {convFieldSize},           // inputShape
      {1},                       // kernelShape
      convInputChannels,         // input channels
      convOutputChannels,        // output channels
      params->getNumConvGroups() // conv groups
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
createPlan(const ConvParams &params, const ConvOptions &options,
           const PlanningObjective &objective, const poplar::Target &target,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           std::vector<std::pair<PlanningCacheImpl::Key, Plan>>
               *additionalPlansToCache) {
  if (options.pass != Pass::FC_TRAINING_FWD) {
    logging::debug("Creating plan for a non-joint plan...");
    return createPlan(params, options, false, objective, target, cache);
  }
  // It doesn't make sense to compare joint and separate planning when the
  // number of cycles is bounded since we can't easily derive bounds for each
  // individual pass from a bound on the total number of cycles.
  assert(objective.getCyclesBound() == std::numeric_limits<unsigned>::max());
  Plan jointPlan;
  Cost jointCost;

  logging::debug("Creating plan for a joint plan...");
  std::tie(jointPlan, jointCost) =
      createPlan(params, options, true, objective, target, cache);

  Plan fwdPlan, bwdPlan, wuPlan;
  Cost fwdCost, bwdCost, wuCost;

  logging::debug("Creating plan for a separate joint plan (fwd pass)...");
  std::tie(fwdPlan, fwdCost) =
      createPlan(params, options, false, objective, target, cache);
  auto bwdParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_BWD);
  auto bwdOptions =
      getFullyConnectedPassOptions(options, Pass::FC_TRAINING_BWD);

  logging::debug("Creating plan for a separate joint plan (bwd pass)...");
  std::tie(bwdPlan, bwdCost) = createPlan(bwdParams.getParams(), bwdOptions,
                                          false, objective, target, cache);
  auto wuParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_WU);
  auto wuOptions = getFullyConnectedPassOptions(options, Pass::FC_TRAINING_WU);

  logging::debug("Creating plan for a separate joint plan (wu pass)...");
  std::tie(wuPlan, wuCost) = createPlan(wuParams.getParams(), wuOptions, false,
                                        objective, target, cache);

  auto separateCost = fwdCost;
  for (const auto &cost : {bwdCost, wuCost}) {
    if (separateCost == highestCost || cost == highestCost) {
      separateCost = highestCost;
      break;
    }
    separateCost.totalCycles += cost.totalCycles;
    separateCost.totalTempBytes =
        std::max(separateCost.totalTempBytes, cost.totalTempBytes);
  }
  if (objective.lowerCost(separateCost, jointCost)) {
    if (additionalPlansToCache) {
      using Key = PlanningCacheImpl::Key;
      additionalPlansToCache->emplace_back(
          Key(std::move(bwdParams), std::move(bwdOptions)), std::move(bwdPlan));
      additionalPlansToCache->emplace_back(
          Key(std::move(wuParams), std::move(wuOptions)), std::move(wuPlan));
    }
    return {fwdPlan, fwdCost};
  }
  return {jointPlan, jointCost};
}

void writePlanConstraintsFile(const Plan &plan, const std::string filePath) {
  boost::property_tree::ptree constraints;
  const auto constrainValues = [&](const std::string &keySuffix,
                                   const std::vector<unsigned> &values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      constraints.add(keySuffix + "." + std::to_string(i), values[i]);
    }
  };

  const auto constrainArray = [&](const std::string &key,
                                  const std::vector<unsigned> &values) {
    using boost::property_tree::ptree;
    ptree array;
    for (const unsigned value : values) {
      array.push_back(ptree::value_type("", ptree(std::to_string(value))));
    }
    constraints.add_child(key, array);
  };

  // Transforms
  for (std::size_t i = 0; i < plan.transforms.size(); ++i) {
    const std::string keySuffix = std::to_string(i) + ".transform.";
    const ConvTransform &t = plan.transforms[i];
    constraints.add(keySuffix + "swapOperands", t.swapOperands);
    constrainArray(keySuffix + "expandDims", t.expandDims);
    constrainArray(keySuffix + "outChanFlattenDims", t.outChanFlattenDims);
    constraints.add(keySuffix + "combineConvGroups", t.combineConvGroupsFactor);
  }

  // Partitions
  for (std::size_t i = 0; i < plan.partitions.size(); ++i) {
    const std::string keySfx = std::to_string(i) + ".partition.";
    const Partition &p = plan.partitions[i];
    constrainValues(keySfx + "fieldSplit", p.fieldSplit);
    constraints.add(keySfx + "batchSplit", p.batchSplit);
    constraints.add(keySfx + "outChanSplit.serial", p.outChanSplit.serial);
    constraints.add(keySfx + "outChanSplit.parallel", p.outChanSplit.parallel);
    constrainValues(keySfx + "kernelSplit", p.kernelSplit);
    constraints.add(keySfx + "inChanSplit", p.inChanSplit);
    constraints.add(keySfx + "convGroupSplit", p.convGroupSplit);
  }

  // Other
  constraints.add("method", plan.method);
  constraints.add("convGroupsPerGroup", plan.convGroupsPerGroup);
  constraints.add("inChansPerGroup", plan.inChansPerGroup);
  constraints.add("partialChansPerGroup", plan.partialChansPerGroup);

  boost::property_tree::write_json(filePath, constraints);
}

std::string getPlanConstraintsOutputFile(const ConvOptions &options) {
  std::string path = options.planConstraintsOutputFilename;
  switch (options.pass) {
  case Pass::INFERENCE_FWD:
  case Pass::TRAINING_FWD:
  case Pass::FC_INFERENCE_FWD:
  case Pass::FC_TRAINING_FWD:
    path += "_FWD";
    break;
  case Pass::TRAINING_BWD:
  case Pass::FC_TRAINING_BWD:
    path += "_BWD";
    break;
  case Pass::TRAINING_WU:
  case Pass::FC_TRAINING_WU:
    path += "_WU";
    break;
  case Pass::NONE:
    break;
  }
  path += ".json";
  return path;
}

// Plan the specified convolution in one of three possible modes:
// cycle cost is the priority
// memory cost is the priority
// optimised for memory, constrained to have cycles cost no worse than some
// multiple of the minimum possible cycle cost.
// Planning a particular training pass (forward / backward / weight update) may
// create plans for the other training passes as a side effect. There plans
// are appended to the end of additionalPlansToCache if it is not null.
static std::pair<Plan, Cost>
runPlanner(const CanonicalConvParams &ccParams, const ConvOptions &options,
           const poplar::Target &target,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           std::vector<std::pair<PlanningCacheImpl::Key, Plan>>
               *additionalPlansToCache) {
  // we first attempt to find the fastest plan that we think will fit, if that
  // fails we replan, but minimising for memory instead. in an effort to fit in
  // memory we will apply an architecturally relevent memory limit to this first
  // plan. to calculate the limit we use a user-configured option called
  // `availableMemoryProportion` to state the proportion of memory which is
  // approximately available for this convolution. if the
  // `availableMemoryProportion` is 0 then we just optimise for memory.
  Plan plan;
  Cost cost = highestCost;
  const auto &params = ccParams.getParams();

  const unsigned availableTileMem =
      target.getBytesPerTile() * options.availableMemoryProportion;

  if (availableTileMem != 0) {
    logging::debug("Planning convolution with a per-tile memory limit of {} "
                   "bytes.",
                   availableTileMem);

    auto objective = PlanningObjective::minimizeCycles();
    objective.setTileTempMemoryBound(availableTileMem);

    std::tie(plan, cost) =
        createPlan(params, options, objective, target, cache, nullptr);
  }

  // if we can't find a plan within this limit this probably isn't going to
  // fit, therefore just try and find the smallest one.
  if (cost.totalCycles == ~0u) {
    if (availableTileMem != 0) {
      logging::warn("Warning: convolution planner unable to meet memory target;"
                    " retrying while targeting minimum memory.");
    } else {
      logging::debug("Planning convolution that uses the least amount of "
                     "temporary memory.");
    }

    auto objective = PlanningObjective::minimizeTileTempMemory();
    std::tie(plan, cost) =
        createPlan(params, options, objective, target, cache, nullptr);

    // if we still could not find a plan there's nothing else we can do.
    if (cost.totalCycles == ~0u) {
      throw poputil::poplibs_error("No base plan found for unbounded plan");
    }
  }

  logging::debug("Found best plan: {}.", cost);
  logging::debug(
      "  for input {}x({}x{}x{}), kernel {}, output = {}x({}x{}x{}), pass={}",
      params.inputFieldShape, params.getBatchSize(), params.getNumConvGroups(),
      params.getNumInputChansPerConvGroup(), params.kernelShape,
      params.getOutputFieldShape(), params.getBatchSize(),
      params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
      options.pass);
  logPlanBreakdown(logging::Level::Debug, plan, cost);

  logging::debug("{}", plan);
  logging::trace("for params: {}", params);

  if (!options.planConstraintsOutputFilename.empty()) {
    writePlanConstraintsFile(plan, getPlanConstraintsOutputFile(options));
  }

  return std::make_pair(std::move(plan), std::move(cost));
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
    plan.partitions[i].inChanSplit =
        fwdPlan.partitions[i].outChanSplit.parallel;
    plan.partitions[i].outChanSplit = {1, fwdPlan.partitions[i].inChanSplit};
    plan.partitions[i].outChanGrainSize = fwdPlan.partitions[i].inChanGrainSize;
    plan.partitions[i].inChanGrainSize = fwdPlan.partitions[i].outChanGrainSize;
  }
  plan.partialChansPerGroup = fwdPlan.inChansPerGroup;
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;

  plan.method = getFullyConnectedWUMethod(fwdParams.getParams(), fwdPlan.method,
                                          fwdPlan.partialChansPerGroup,
                                          fwdPlan.inChansPerGroup);
  // TODO: T12888 Make the forward pass aware that it would be good to use a
  // grouping of 16 if possible.
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  if (plan.method == Plan::Method::AMP &&
      !canUseConvolutionInstruction(
          fwdParams->inputType == poplar::FLOAT,
          fwdOptions.partialsType == poplar::FLOAT, plan.inChansPerGroup,
          plan.numConvUnitsRequired, plan.partialChansPerGroup, target)) {
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
    std::swap(partition.fieldSplit.back(), partition.inChanSplit);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
  return plan;
}

void preplanConvolutionsImpl(const poplar::Target &target,
                             const std::set<ConvPlanKey> &paramSet,
                             PlanningCache &cache) {
  // convert to a vector for efficient tbb looping
  struct Job {
    const ConvPlanKey *input;
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
    std::tie(plan, cost) = runPlanner(
        params, options, target, &cache.impl->cycleEstimation, &jobs[i].output);
    auto key =
        PlanningCacheImpl::Key(jobs[i].input->first, jobs[i].input->second);
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
    auto fwdParams =
        getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_FWD);
    auto fwdOptions =
        getFullyConnectedPassOptions(options, Pass::FC_TRAINING_FWD);
    const auto fwdPlan = getPlan(target, fwdParams, fwdOptions, cache);
    if (fwdPlan.isJointPlan) {
      if (options.pass == Pass::FC_TRAINING_WU) {
        return getFullyConnectedWUPlan(target, fwdParams, fwdOptions, fwdPlan);
      }
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
                                    &cacheImpl->cycleEstimation, &plansToCache);
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

static void constrainVariable(popsolver::Model &m, popsolver::Variable v,
                              unsigned value) {
  m.equal(v, value);
}

static void constrainVariable(popsolver::Model &m, Split<popsolver::Variable> v,
                              Split<unsigned> value) {
  constrainVariable(m, v.parallel, value.parallel);
  constrainVariable(m, v.serial, value.serial);
}

static void constrainPartitionVars(popsolver::Model &m,
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

/// Estimate the cost of a convolution. This is not used by poplibs/enigma.
std::pair<std::uint64_t, std::uint64_t>
estimateConvCost(const poplar::Target &target, const ConvParams &params,
                 const ConvOptions &options, PlanningCache *cache,
                 const Plan &plan) {
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  std::vector<double> perLevelExchangeBytesPerCycle;
  const auto hierarchy =
      poplibs::getTileHierarchy(target, perLevelExchangeBytesPerCycle);
  assert(perLevelExchangeBytesPerCycle.size() == plan.partitions.size());
  auto objective = PlanningObjective::minimizeCycles();
  ConvVertexType convVertexType(
      plan.method, params.inputType, params.outputType,
      plan.types.back().partialType, plan.convGroupsPerGroup,
      plan.inChansPerGroup, plan.partialChansPerGroup, plan.slicWindowWidth,
      plan.numConvUnitsRequired);
  const auto fieldGrainSize = plan.partitions.back().fieldAxisGrainSize;
  // Check grain size is the same at each level.
#ifndef NDEBUG
  for (const auto &p : plan.partitions) {
    assert(p.fieldAxisGrainSize == fieldGrainSize);
  }
#endif
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  const auto e = constructModel(
      target, plan.transforms, plan.types, hierarchy,
      perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType, params,
      plan.isJointPlan, highestCost, objective, &cacheImpl->cycleEstimation,
      options, m, partitionVars);
  const auto numLevelsOfHierarchy = plan.partitions.size();
  assert(partitionVars.size() == numLevelsOfHierarchy);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    constrainPartitionVars(m, partitionVars[level], plan.partitions[level]);
  }
  popsolver::Solution s;
  s = m.minimize(e.totalCycles);
  if (!s.validSolution()) {
    return {highestCost.totalCycles, highestCost.totalTempBytes};
  }
  return {s[e.totalCycles], s[e.totalTempBytes]};
}

} // namespace poplin
