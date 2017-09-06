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
#include <set>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <popsolver/Model.hpp>
#include "util/print.hpp"

namespace popconv {

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
  ConvVertexType(Plan::Method method, bool floatActivations,
                 bool floatPartials) :
    method(method),
    floatActivations(floatActivations),
    floatPartials(floatPartials) {}
};

static const char *asString(Plan::Method m) {
  switch (m) {
  case Plan::Method::AMP_ACCUMULATE_OVER_FIELD:
    return "AMP_ACCUMULATE_OVER_FIELD";
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
  os << "\n"
     << "        deltasAsCoefficents     "
     << (p.ampWUMethod == Plan::DELTAS_AS_COEFFICENTS) << "\n";
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
    inputRangeSize = (dilatedInputSize - 1 / inputDilation) + 1;
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
  for (unsigned i = 0; i != passesPerOutput; ++i) {
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

static Cost
estimateWeightUpdateByAmpReorderCost(const poplar::DeviceInfo &deviceInfo,
                                     bool floatActivations,
                                     const ConvParams &reorderedParams,
                                     const Plan &plan) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numInElements = reorderedParams.getBatchSize() *
                             reorderedParams.getInputDepthPerConvGroup() *
                             reorderedParams.getInputHeight() *
                             reorderedParams.getInputWidth();
  const auto numOutElements = reorderedParams.getBatchSize() *
                              reorderedParams.getOutputDepthPerConvGroup() *
                              reorderedParams.getOutputHeight() *
                              reorderedParams.getOutputWidth();
  const auto numCoefficients = reorderedParams.kernelShape[0] *
                               reorderedParams.kernelShape[1] *
                               reorderedParams.getInputDepthPerConvGroup() *
                               reorderedParams.getOutputDepthPerConvGroup();
  const auto tilesPerConvGroups = plan.tilesPerConvGroups;
  const auto tileNumConvGroups =
      (reorderedParams.getNumConvGroups() + tilesPerConvGroups - 1)
     / tilesPerConvGroups;
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
  cycles *= tileNumConvGroups;
  return {cycles, 0};
}

static ConvParams
weightUpdateByAmpTransformParams(const ConvParams &params,
                                 const poplar::DeviceInfo &deviceInfo,
                                 bool flattenXY,
                                 unsigned partialChansPerGroup,
                                 Plan::AmpWUMethod ampWUMethod) {
  const unsigned numConvGroups = params.getNumConvGroups();
  const unsigned numInputChannels = params.getInputDepthPerConvGroup();
  const unsigned numOutputChannels = params.getOutputDepthPerConvGroup();

  const auto canonicalParams = canonicalizeParams(params);
  // Padding the kernel in the forward pass requires truncation of the weight
  // deltas in the weight update pass. If flattenXY is true we can fold this
  // truncation into the flattening of the field. If flattenXY is false we
  // may not be able to represent the weight update as a convolution that can be
  // calculated with the convolution function.
  // TODO add support for output padding / truncation and remove this
  // restriction.
  assert(flattenXY ||
         (canonicalParams.kernelPaddingLower[0] == 0 &&
          canonicalParams.kernelPaddingUpper[0] == 0));
  bool floatActivations = canonicalParams.dType == "float";
  unsigned expandedFieldWidth;
  unsigned expandedActivationsHeight;
  unsigned expandedDeltasHeight;
  int expandedActivationsPaddingYLower;
  int expandedActivationsPaddingYUpper;
  unsigned expandedInputDepth;
  unsigned expandedDeltasDilationY;
  unsigned weightDeltasStrideY;
  if (flattenXY) {
    expandedFieldWidth =
       canonicalParams.getBatchSize() * canonicalParams.getOutputHeight() *
                                        canonicalParams.getOutputWidth();
    expandedActivationsHeight = 1;
    expandedDeltasHeight = 1;
    expandedActivationsPaddingYLower = 0;
    expandedActivationsPaddingYUpper = 0;
    expandedInputDepth =
        numInputChannels
        * canonicalParams.kernelShape[0] * canonicalParams.kernelShape[1];
    expandedDeltasDilationY = 1;
    weightDeltasStrideY = 1;
  } else {
    expandedFieldWidth = canonicalParams.getBatchSize() *
                         canonicalParams.getOutputWidth();
    expandedActivationsHeight = canonicalParams.getInputHeight();
    expandedDeltasHeight = canonicalParams.getOutputHeight();
    expandedActivationsPaddingYLower = canonicalParams.inputPaddingLower[0];
    expandedActivationsPaddingYUpper = canonicalParams.inputPaddingUpper[0];
    expandedInputDepth = numInputChannels
                         * canonicalParams.kernelShape[1];
    expandedDeltasDilationY = canonicalParams.stride[0];
    weightDeltasStrideY = canonicalParams.kernelDilation[0];
  }
  const auto fieldGroupSize =
      deviceInfo.getWeightsPerConvUnit(floatActivations);
  const auto paddedFieldWidth =
      ((expandedFieldWidth + fieldGroupSize - 1) / fieldGroupSize) *
      fieldGroupSize;
  ConvParams newParams;
  switch (ampWUMethod) {
  case Plan::DELTAS_AS_COEFFICENTS:
    {

      // weight update x-axis: fwd in chans
      // weight update y-axis: fwd y-axis
      // weight update in chans: fwd x-axis
      // weight update out chans: fwd out chans
      const auto paddedOutputDepth =
          ((numOutputChannels + partialChansPerGroup - 1) /
           partialChansPerGroup) * partialChansPerGroup;
      newParams = ConvParams(canonicalParams.dType,
                    // batch size
                    1,
                    // input field shape for each channel and batch
                    {expandedActivationsHeight,  expandedInputDepth},
                    // kernel shape for each input and output channel
                    {expandedDeltasHeight, 1},
                    // input channels
                    paddedFieldWidth,
                    // output channels
                    paddedOutputDepth,
                    // stride
                    {weightDeltasStrideY, 1},
                    // lower input padding
                    {expandedActivationsPaddingYLower, 0},
                    // upper input padding
                    {expandedActivationsPaddingYUpper, 0},
                    // input dilation
                    {1, 1},
                    // lower kernal padding
                    {0, 0},
                    // upper kernel padding
                    {0, 0},
                    // kernel dilation
                    {1, 1},
                    numConvGroups
                  );
    }
    break;
  case Plan::ACTIVATIONS_AS_COEFFICENTS:
    {
      // weight update x-axis: fwd out chans
      // weight update y-axis: fwd y-axis
      // weight update in chans: fwd x-axis
      // weight update out chans: fwd in chans
      const auto paddedExpandedInputDepth =
          ((expandedInputDepth + partialChansPerGroup - 1) /
           partialChansPerGroup) * partialChansPerGroup;
      newParams = ConvParams(canonicalParams.dType,
                    // batch size
                    1,
                    // input field shape for each channel and batch
                    {expandedDeltasHeight,
                     numOutputChannels},
                    // kernel shape for each input and output channel
                    {expandedActivationsHeight, 1},
                    // input channels
                    paddedFieldWidth,
                    // output channels
                    paddedExpandedInputDepth,
                    // stride
                    {weightDeltasStrideY, 1},
                    // lower input padding
                    {0, 0},
                    // upper input padding
                    {0, 0},
                    // input dilation
                    {expandedDeltasDilationY, 1},
                    // lower kernal padding
                    {expandedActivationsPaddingYLower, 0},
                    // upper kernel padding
                    {expandedActivationsPaddingYUpper, 0},
                    // kernel dilation
                    {1, 1},
                    numConvGroups
                   );
    }
    break;
  }
  return newParams;
}

ConvParams
weightUpdateByAmpTransformParams(const ConvParams &params,
                                 const poplar::DeviceInfo &deviceInfo,
                                 const Plan &plan) {
  assert(plan.method == Plan::Method::AMP_ACCUMULATE_OVER_FIELD);
  assert(plan.flattenDims.size() == 2 || plan.flattenDims.size() == 3);
  bool flattenXY = plan.flattenDims.size() == 3;
  return weightUpdateByAmpTransformParams(params, deviceInfo, flattenXY,
                                          plan.partialChansPerGroup,
                                          plan.ampWUMethod);
}

static popsolver::Variable
addCycleEstimate(popsolver::Model &m, popsolver::Variable tilesPerX,
                 popsolver::Variable tilesPerY,
                 popsolver::Variable tilesPerBatch,
                 popsolver::Variable tilesPerZ,
                 popsolver::Variable tilesPerKernelY,
                 popsolver::Variable tilesPerInZ,
                 popsolver::Variable tilesPerConvGroups,
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
  return m.sum({exchangeCycles, partialCalcCycles, reduceCycles});
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

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           unsigned inChansPerGroup,
           unsigned partialChansPerGroup,
           unsigned xAxisGrainSize,
           const ConvVertexType &convVertexType,
           const ConvParams &params, bool isWeightUpdate,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  const auto floatActivations = convVertexType.floatActivations;
  if (convVertexType.method == Plan::Method::AMP_ACCUMULATE_OVER_FIELD) {
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
      if (!flattenXY &&
          (params.kernelPaddingLower[0] != 0 ||
           params.kernelPaddingUpper[0] != 0))
        continue;
      for (Plan::AmpWUMethod method : {Plan::DELTAS_AS_COEFFICENTS,
                                       Plan::ACTIVATIONS_AS_COEFFICENTS}) {
        // There is currently no support for dilated convolutions.
        // TODO add support for this.
        if (!flattenXY && params.stride[0] != 1) {
          continue;
        }
        auto newParams =
            weightUpdateByAmpTransformParams(params, deviceInfo, flattenXY,
                                             partialChansPerGroup, method);
        auto newConvVertexType = convVertexType;
        newConvVertexType.method = Plan::Method::AMP;
        Plan plan;
        Cost cost;
        std::tie(plan, cost) = choosePlan(deviceInfo,
                                          inChansPerGroup,
                                          partialChansPerGroup,
                                          1,
                                          newConvVertexType, newParams, false,
                                          costBounds, cache,
                                          options);
        plan.method = Plan::Method::AMP_ACCUMULATE_OVER_FIELD;
        if (flattenXY) {
          plan.flattenDims = {0, 1, 2};
        } else {
          plan.flattenDims = {0, 2};
        }
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
                       deviceInfo, params,
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
                         deviceInfo, params,
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
                         deviceInfo, wuParams,
                         wuInChansPerGroup, wuPartialChansPerGroup,
                         1, convVertexType.floatPartials,
                         floatActivations, wuMethod,
                         Plan::LinearizeTileOrder::FC_WU,
                         cache);
    cycles = m.sum({cycles, bwdCycles, wuCycles});
  }
  if (costBounds.cycles > 0) {
    m.lessOrEqual(cycles, costBounds.cycles);
  }
  Solution s;
  try {
    assert(costBounds.primaryCheckIsCycles);
    s = m.minimize(cycles);
  } catch (NoSolution) {
    return {Plan(), highestCost};
  }
  Plan bestPlan(s[tilesPerX], s[tilesPerY], s[tilesPerBatch], s[tilesPerZ],
                s[tilesPerKernelY], s[tilesPerInZ], s[tilesPerConvGroups],
                inChansPerGroup, partialChansPerGroup,
                xAxisGrainSize,
                convVertexType.floatPartials,
                convVertexType.method,
                Plan::LinearizeTileOrder::STANDARD);
  // TODO estimate memory usage.
  unsigned memory = 0;
  Cost bestCost = {s[cycles], memory};
  return {bestPlan, bestCost};
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::DeviceInfo &deviceInfo,
                            bool floatActivations,
                            bool floatPartials,
                            const ConvParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  if (params.getInputDepthPerConvGroup() == 1 &&
      params.getPaddedDilatedKernelSize(0) == 1 &&
      params.getPaddedDilatedKernelSize(1) == 1 &&
      params.getOutputHeight() == 1 &&
      params.getBatchSize() == 1) {
    convVertexTypeCandidates.emplace_back(Plan::Method::OUTER_PRODUCT,
                                          floatActivations,
                                          floatActivations);
  }
  // We limit the use of the convolution instruction to cases where the number
  // of output channels is a multiple of the output channel grouping that would
  // be used.
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   params.stride[0], params.stride[1],
                                   deviceInfo)) {
    convVertexTypeCandidates.emplace_back(Plan::Method::AMP, floatActivations,
                                          floatPartials);
  } else if (!floatActivations && !floatPartials &&
             canUseConvolutionInstruction(false, true,
                                          params.stride[0], params.stride[1],
                                          deviceInfo) &&
             params.getOutputDepthPerConvGroup() %
             deviceInfo.fp16InFp32OutConvUnitsPerTile == 0) {
    convVertexTypeCandidates.emplace_back(Plan::Method::AMP, false, true);
  }
  convVertexTypeCandidates.emplace_back(Plan::Method::MAC, floatActivations,
                                        floatPartials);
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
      convVertexTypeCandidates.emplace_back(
            Plan::Method::AMP_ACCUMULATE_OVER_FIELD, floatActivations,
            floatPartials);
    } else if (!floatActivations && !floatPartials &&
               deviceInfo.fp16InFp32OutConvUnitsPerTile > 0) {
      convVertexTypeCandidates.emplace_back(
            Plan::Method::AMP_ACCUMULATE_OVER_FIELD, false, true);
    }
  }
  return convVertexTypeCandidates;
}

static std::vector<unsigned>
getInChansPerGroupCandidates(const ConvParams &params,
                             const ConvVertexType &convVertexType,
                             const poplar::DeviceInfo &deviceInfo,
                             const ConvOptions &options) {
  std::vector<unsigned> candidates;
  const auto numConvUnits = getNumConvUnits(convVertexType.floatActivations,
                                            convVertexType.floatPartials,
                                            deviceInfo);
  const bool isFullyConnectedFwd =
      options.fullyConnectedPass == FullyConnectedPass::FWD;
  if (convVertexType.method == Plan::Method::OUTER_PRODUCT) {
    assert(params.getInputDepthPerConvGroup() == 1);
    return {1};
  }
  assert(convVertexType.method == Plan::Method::AMP ||
         convVertexType.method == Plan::Method::MAC);
  bool useConvInstruction = convVertexType.method == Plan::Method::AMP;
  for (unsigned i = 1; i <= params.getInputDepthPerConvGroup(); ++i) {
    if (params.getInputDepthPerConvGroup() % i != 0)
      continue;
    if (!convVertexType.floatActivations && i % 2 != 0)
      continue;
    if (useConvInstruction &&
        !canUseConvolutionInstruction(convVertexType.floatActivations,
                                      convVertexType.floatPartials,
                                      params.stride[0], params.stride[1],
                                      i, deviceInfo))
      continue;
    if (isFullyConnectedFwd) {
      // The input channels in the forward pass become the output channels of
      // the weight update pass. Make sure it is a multiple of the supported
      // output channels per group.
      if (i % numConvUnits != 0)
        continue;
    }
    candidates.push_back(i);
  }
  if (candidates.empty()) {
    if (useConvInstruction) {
      // Drop the requirement that the input channel grouping must divide
      // the number of input channels. This causes the input to be zero padded
      // before the convolution.
      // TODO Currently we only consider input channel groupings that need
      // padding if we didn't find an input channel grouping that divides the
      // number of channels exactly. Ideally we would always consider all
      // input channel groupings and pick the one with the lowest cost.
      // We would need to check whether the cost model is sufficiently accurate
      // before making this change.
      for (unsigned i = 1; i <= params.getInputDepthPerConvGroup(); ++i) {
        if (!convVertexType.floatActivations && i % 2 != 0)
          continue;
        if (useConvInstruction &&
            !canUseConvolutionInstruction(convVertexType.floatActivations,
                                          convVertexType.floatPartials,
                                          params.stride[0], params.stride[1],
                                          i, deviceInfo))
          continue;
        if (isFullyConnectedFwd) {
          // The input channels in the forward pass become the output channels
          // of the weight update pass. Make sure it is a multiple of the
          // supported output channels per group.
          if (i % numConvUnits != 0)
            continue;
        }
        candidates.push_back(i);
      }
    } else {
      candidates.push_back(isFullyConnectedFwd ? numConvUnits :
                                         params.getInputDepthPerConvGroup());
    }
  }
  return candidates;
}

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           const ConvVertexType &convVertexType,
           unsigned tensorInChansPerGroup,
           unsigned tensorOutChansPerGroup,
           unsigned tensorWeightOutChansPerGroup,
           const ConvParams &params, bool isWeightUpdate,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  Plan best;
  Cost bestCost = highestCost;
  std::vector<unsigned> inChansPerGroupCandidates;
  if (convVertexType.method == Plan::Method::AMP_ACCUMULATE_OVER_FIELD) {
    inChansPerGroupCandidates.push_back(
      deviceInfo.getWeightsPerConvUnit(convVertexType.floatActivations)
    );
  } else {
    assert(tensorInChansPerGroup == 0);
    assert(tensorOutChansPerGroup == 0);
    assert(tensorWeightOutChansPerGroup == 0);
    inChansPerGroupCandidates =
        getInChansPerGroupCandidates(params, convVertexType, deviceInfo,
                                     options);
  }
  unsigned partialChansPerGroup;
  switch (convVertexType.method) {
  default: assert(0 && "Unexpected method");
  case Plan::Method::AMP:
  case Plan::Method::AMP_ACCUMULATE_OVER_FIELD:
    {
      const auto numConvUnits = getNumConvUnits(convVertexType.floatActivations,
                                                convVertexType.floatPartials,
                                                deviceInfo);
      // TODO take into account the best grouping for all the phases if
      // options.fullyConnectedFwd is set.
      partialChansPerGroup = numConvUnits;
    }
    break;
  case Plan::Method::MAC:
    partialChansPerGroup = 1;
    break;
  case Plan::Method::OUTER_PRODUCT:
    partialChansPerGroup = convVertexType.floatActivations ?
                           deviceInfo.getFloatVectorWidth() :
                           deviceInfo.getHalfVectorWidth();
    break;
  }

  for (auto inChansPerGroup : inChansPerGroupCandidates) {
    Plan candidate;
    Cost candidateCost;
    unsigned xAxisGrainSize = 1;
    if (options.fullyConnectedPass == FullyConnectedPass::FWD) {
      // The xAxisGrainSize becomes the inChansPerGroup in the backward pass.
      // For now assume the same grouping in both passes.
      // TODO search for the optimal grouping in each pass.
      xAxisGrainSize = inChansPerGroup;
    }
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, inChansPerGroup, partialChansPerGroup,
                   xAxisGrainSize, convVertexType, params, isWeightUpdate,
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
                   params, isWeightUpdate, costBounds, cache,
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

static void expandDim(ConvParams &params, unsigned dim) {
  if (params.getPaddedDilatedInputSize(dim) >=
      params.getPaddedDilatedKernelSize(dim)) {
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

static Cost
estimateTransformCost(const poplar::DeviceInfo &deviceInfo,
                      const ConvParams &params,
                      std::vector<unsigned> expandDims) {
  if (expandDims.empty()) {
    return {0, 0};
  }
  auto expandedParams = params;
  for (const auto dim : expandDims) {
    expandDim(expandedParams, dim);
  }
  if (expandedParams.getInputDepthPerConvGroup() ==
      params.getInputDepthPerConvGroup()) {
    return {0, 0};
  }
  unsigned expandedInputFieldSize = 1;
  unsigned expandedFilterSize = 1;
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    expandedInputFieldSize *= params.inputFieldShape[dim];
    expandedFilterSize *= params.kernelShape[dim];
  }
  const auto expandedInputChannelsPerGroup = params.getInputDepthPerConvGroup();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto expandedInputElements =
      expandedInputFieldSize * expandedInputChannelsPerGroup *
      params.getBatchSize() * params.numConvGroups;
  const auto expandedInputElementsPerTile =
      (expandedInputElements + numTiles - 1) / numTiles;
  const auto expandedFilterElements =
      expandedFilterSize * expandedInputChannelsPerGroup *
      params.getOutputDepthPerConvGroup() * params.numConvGroups;
  const auto expandedFilterElementsPerTile =
      (expandedFilterElements + numTiles - 1) / numTiles;
  const auto expandedElementsPerTile =
      expandedInputElementsPerTile + expandedFilterElementsPerTile;
  const auto bytesPerElement = params.dType == "float" ? 4U : 2U;
  const auto expandedBytesPerTile = expandedElementsPerTile * bytesPerElement;
  unsigned cycles = 0;
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
  return {cycles, 0};
}

static std::vector<std::vector<unsigned>>
getExpandDimsCandidates(bool isWeightUpdate) {
  if (isWeightUpdate)
    return {{}};
  return {
    {},
    {0},
    {1},
    {1, 0}
  };
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
  Cost bestCost = highestCost;
  Plan bestPlan;
  for (std::vector<unsigned> expandDims :
       getExpandDimsCandidates(isWeightUpdate)) {
    Cost transformCost =
        estimateTransformCost(graph.getDevice().getDeviceInfo(),
                              params, expandDims);
    auto newParams = params;
    for (unsigned dim : expandDims) {
      expandDim(newParams, dim);
    }
    std::vector<unsigned> outChanFlattenDims;
    if (!isWeightUpdate) {
      for (unsigned spatialDim = 0; spatialDim != newParams.getNumFieldDims();
           ++spatialDim) {
        if (dimCanBeFlattenedIntoOutChans(newParams, spatialDim) &&
            newParams.kernelShape[spatialDim] > 1) {
          outChanFlattenDims.push_back(spatialDim);
          newParams.outputChannels *= newParams.kernelShape[spatialDim];
          newParams.kernelShape[spatialDim] = 1;
        }
      }
      // Flatten from the innermost out.
      std::reverse(outChanFlattenDims.begin(), outChanFlattenDims.end());
    }
    std::vector<unsigned> flattenDims;
    if (!isWeightUpdate) {
      flattenDims.push_back(0);
      for (unsigned spatialDim = 0; spatialDim != newParams.getNumFieldDims();
           ++spatialDim) {
        if (dimCanBeFlattened(newParams, spatialDim)) {
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
              fromDimIndex ? newParams.inputFieldShape[fromDimIndex - 1] :
                             newParams.batchSize;
          newParams.inputFieldShape[innermostFlattenableDim - 1] *=
              fromDimSize;
          fromDimSize = 1;
        }
      } else {
        flattenDims.clear();
      }
    }
    const bool floatActivations = params.dType == "float";
    const bool floatPartials = partialsType == "float";
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, floatActivations,
                   floatPartials,
                   tensorInChansPerGroup,
                   tensorOutChansPerGroup,
                   tensorWeightOutChansPerGroup,
                   newParams, isWeightUpdate, costBounds,
                   cache,
                   options);
    candidateCost += transformCost;
    if (!isWeightUpdate) {
      candidate.flattenDims = flattenDims;
    }
    candidate.expandDims = expandDims;
    candidate.outChanFlattenDims = outChanFlattenDims;
    if (compareCost(candidateCost, bestCost, costBounds)) {
      bestPlan = candidate;
      bestCost = candidateCost;
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
                                    fwdParams.stride[0], fwdParams.stride[1],
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
                                             0, 0, 0,
                                             partialsType,
                                             false, options,
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

Plan getWeightUpdatePlan(const poplar::Graph &graph,
                         const poplar::Tensor &activations,
                         const poplar::Tensor &deltas,
                         const ConvParams &params,
                         ConvOptions options) {
  assert(activations.rank() == 4);
  assert(deltas.rank() == 4);
  assert(params.kernelShape.size() == 2);
  assert(params.stride.size() == 2);
  assert(params.inputPaddingLower.size() == 2);
  assert(params.inputPaddingUpper.size() == 2);
  assert(options.fullyConnectedPass == FullyConnectedPass::NONE);
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  const auto partialsType = options.partialsType;
  const auto fwdPlan = getPlan(graph, params, options);
  const auto actChansPerGroup = fwdPlan.inChansPerGroup;
  const auto deltasChansPerGroup = deltas.dim(3) / params.getNumConvGroups();
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
