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
  struct Params {
    std::string dType;
    std::vector<std::size_t> inShape;
    std::vector<std::size_t> weightsShape;
    std::vector<unsigned> stride;
    std::vector<unsigned> padding;
    bool isFractional;
    bool isWeightUpdate;
    unsigned actChansPerGroup;
    unsigned deltaChansPerGroup;
    ConvOptions options;
    Params(std::string dType,
           std::vector<std::size_t> inShape,
           std::vector<std::size_t> weightsShape,
           std::vector<unsigned> stride,
           std::vector<unsigned> padding,
           bool isFractional,
           bool isWeightUpdate, unsigned actChansPerGroup,
           unsigned deltaChansPerGroup,
           ConvOptions options) :
      dType(std::move(dType)),
      inShape(std::move(inShape)),
      weightsShape(std::move(weightsShape)),
      stride(std::move(stride)),
      padding(std::move(padding)),
      isFractional(isFractional),
      isWeightUpdate(isWeightUpdate),
      actChansPerGroup(isWeightUpdate ? actChansPerGroup : 0),
      deltaChansPerGroup(isWeightUpdate ? deltaChansPerGroup : 0),
      options(std::move(options)) {}
    bool operator<(const Params &other) const {
      return std::tie(dType, inShape, weightsShape, stride, padding,
                      isFractional, isWeightUpdate, options) <
               std::tie(other.dType, other.inShape, other.weightsShape,
                        other.stride, other.padding,
                        other.isFractional, other.isWeightUpdate,
                        other.options);
    }
  };
  std::map<Params, std::unique_ptr<Plan>> plans;
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
    if (isFractional) {
      return (inputWidth * strideX + kernelSizeX - 1) - (paddingX * 2);
    }
    return absdiff(inputWidth + paddingX * 2, kernelSizeX) / strideX + 1;
  }
  unsigned getOutputHeight() const {
    if (isFractional)
      return (inputHeight * strideY + kernelSizeY - 1) - (paddingY * 2);
    return absdiff(inputHeight + paddingY * 2, kernelSizeY) / strideY + 1;
  }
  ConvolutionParams() = default;
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
                     unsigned tileKernelSize,
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
                                      inputSize,
                                      {kernelSize - tileKernelSize, kernelSize},
                                      isFractional);
      inputRangeSize = inputRange.second - inputRange.first;
    }
    break;
  default:
    if (!isFractional) {
      inputRangeSize = (outputRangeSize - 1) * stride + 1 +
                       (tileKernelSize - 1);
    } else {
      inputRangeSize = (outputRangeSize - tileKernelSize) / stride + 1;
    }
    break;
  }
  if (!isFractional && !isWeightUpdate &&
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
                       bool floatActivations, const ConvolutionParams &params,
                       const Plan &plan) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroupAxis = plan.tilesPerInZGroupAxis;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;

  const auto tileKernelHeight = (params.kernelSizeY + tilesPerKernelYAxis - 1) /
                                tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelSizeX;
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
                           params.paddingX, tileKernelWidth, tilesPerX,
                           params.inputWidth, true, params.isFractional,
                           params.isWeightUpdate);
  const auto tileInHeight =
      getMaxInputRangeSize(tileOutHeight, params.strideY, params.kernelSizeY,
                           params.paddingY, tileKernelHeight, tilesPerY,
                           params.inputWidth, false, params.isFractional,
                           params.isWeightUpdate);
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
  return numCycles;
}

static unsigned
estimateWeightUpdatePartialCalcCycles(const poplar::DeviceInfo &deviceInfo,
                                      bool floatActivations,
                                      const ConvolutionParams &params,
                                      const Plan &plan) {
  assert(params.isWeightUpdate);
  assert(!plan.useConvolutionInstructions);
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
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
  const auto tileKernelHeight = (params.kernelSizeY + tilesPerKernelYAxis - 1) /
                                tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelSizeX;
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
  const auto vertexCycles =
      getWeightGradAopCycles(floatActivations, floatPartials, dataPathWidth,
                             inChansPerGroup, outChansPerGroup, shape,
                             useDeltasForEdges);
  unsigned totalCycles = vertexCycles * numWorkerContexts;
  return totalCycles;
}

static unsigned
estimateWeightUpdatePartialCalcMemory(const poplar::DeviceInfo &deviceInfo,
                                      bool floatActivations,
                                      const ConvolutionParams &params,
                                      const Plan &plan) {
  return 0;
}

static unsigned
estimatePartialCalcMemory(const poplar::DeviceInfo &deviceInfo,
                          bool floatActivations,
                          const ConvolutionParams &params,
                          const Plan &plan) {
  if (params.isWeightUpdate) {
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
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;

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
                          const ConvolutionParams &params,
                          const Plan &plan,
                          PlanningCacheImpl *cache) {
  if (params.isWeightUpdate) {
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
      (params.outputDepth + (outChansPerGroup - 1)) / outChansPerGroup;

  const auto tileNumOutGroups =
      (numOutGroups + tilesPerZ - 1) / tilesPerZ;

  // The use of supervisor vertices only affects vertices that use the
  // convolution instructions.
  unsigned numContexts = deviceInfo.numWorkerContexts;
  if (plan.useConvolutionInstructions) {
    numContexts = 1;
  }
  const auto numInGroups =
      (params.inputDepth + (inChansPerGroup - 1)) / inChansPerGroup;
  const auto tileNumInGroups =
      (numInGroups + tilesPerInZGroupAxis - 1) / tilesPerInZGroupAxis;

  const auto outputStrideY = params.isFractional ? params.strideY : 1;

  const auto tileKernelHeight =
      (params.kernelSizeY + tilesPerKernelYAxis - 1) / tilesPerKernelYAxis;
  const auto tileKernelWidth = params.kernelSizeX;
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
    const auto outputStrideX = params.isFractional ? params.strideX : 1;
    const auto outputStrideY = params.isFractional ? params.strideY : 1;
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
                                     const ConvolutionParams &params,
                                     const Plan &plan) {
  unsigned vertexFields = 0;
  unsigned edgePtrsPerVertex = 0;

  unsigned numTiles;
  if (params.isWeightUpdate) {
    assert(plan.batchesPerGroup == 1);
    numTiles = deviceInfo.getNumTiles();
  } else {
    const auto numBatchGroups = params.batchSize / plan.batchesPerGroup;
    numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(), numBatchGroups);
  }

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


static unsigned
estimateReduceCycles(const poplar::DeviceInfo &deviceInfo,
                          const ConvolutionParams &params,
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
                                     const ConvolutionParams &reorderedParams,
                                     const Plan &plan) {
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numInElements = reorderedParams.batchSize *
                             reorderedParams.inputDepth *
                             reorderedParams.inputHeight *
                             reorderedParams.inputWidth;
  const auto numOutElements = reorderedParams.batchSize *
                              reorderedParams.outputDepth *
                              reorderedParams.getOutputHeight() *
                              reorderedParams.getOutputWidth();
  const auto numCoefficients = reorderedParams.kernelSizeY *
                               reorderedParams.kernelSizeX *
                               reorderedParams.inputDepth *
                               reorderedParams.outputDepth;
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

static std::pair<Plan, Cost>
choosePlan(const poplar::DeviceInfo &deviceInfo,
           unsigned inChansPerGroup,
           unsigned partialChansPerGroup,
           const ConvVertexType &convVertexType,
           const ConvolutionParams &params,
           const unsigned batchesPerGroup,
           const CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  const auto floatActivations = convVertexType.floatActivations;
  if (convVertexType.useConvInstruction &&
      params.isWeightUpdate) {
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
        unsigned expandedFieldWidth;
        unsigned expandedActivationsHeight;
        unsigned expandedDeltasHeight;
        unsigned expandedActivationsPaddingY;
        unsigned expandedInputDepth;
        unsigned expandedDeltasUpsampleFactorY;
        if (flattenXY) {
          expandedFieldWidth =
              params.batchSize * params.getOutputHeight() *
                                 params.getOutputWidth();
          expandedActivationsHeight = 1;
          expandedDeltasHeight = 1;
          expandedActivationsPaddingY = 0;
          expandedInputDepth =
              params.inputDepth * params.kernelSizeX * params.kernelSizeY;
          expandedDeltasUpsampleFactorY = 1;
        } else {
          expandedFieldWidth = params.batchSize * params.getOutputWidth();
          expandedActivationsHeight = params.inputHeight;
          expandedDeltasHeight = params.getOutputHeight();
          expandedActivationsPaddingY = params.paddingY;
          expandedInputDepth =
              params.inputDepth * params.kernelSizeX;
          expandedDeltasUpsampleFactorY = params.strideY;
        }
        const auto fieldGroupSize =
            deviceInfo.getWeightsPerConvUnit(floatActivations);
        const auto paddedFieldWidth =
            ((expandedFieldWidth + fieldGroupSize - 1) / fieldGroupSize) *
            fieldGroupSize;
        ConvolutionParams newParams;
        switch (method) {
        case Plan::DELTAS_AS_COEFFICENTS:
          {
            // There is currently no support for dilated convolutions.
            // TODO add support for this.
            if (expandedDeltasUpsampleFactorY != 1) {
              continue;
            }
            // weight update x-axis: fwd in chans
            // weight update y-axis: fwd y-axis
            // weight update in chans: fwd x-axis
            // weight update out chans: fwd out chans
            const auto paddedOutputDepth =
                ((params.outputDepth + partialChansPerGroup - 1) /
                 partialChansPerGroup) * partialChansPerGroup;
            newParams = ConvolutionParams(
                          expandedDeltasHeight /* kernelSizeY */,
                          1 /* kernelSizeX */,
                          1,
                          1,
                          paddedFieldWidth,
                          expandedInputDepth,
                          expandedActivationsHeight,
                          expandedActivationsPaddingY /*paddingY*/,
                          0 /*paddingX*/,
                          paddedOutputDepth,
                          1, false, false
                        );
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
            newParams = ConvolutionParams(
                          /* kernelSizeY */
                          expandedActivationsHeight +
                          2 * expandedActivationsPaddingY,
                          1 /* kernelSizeX */,
                          expandedDeltasUpsampleFactorY,
                          1,
                          paddedFieldWidth,
                          params.outputDepth,
                          expandedDeltasHeight,
                          0 /*paddingY*/,
                          0 /*paddingX*/,
                          paddedExpandedInputDepth,
                          1, isFractional, false
                        );
          }
          break;
        }
        Plan plan;
        Cost cost;
        std::tie(plan, cost) = choosePlan(deviceInfo,
                                          inChansPerGroup,
                                          partialChansPerGroup,
                                          convVertexType, newParams,
                                          1, costBounds, cache,
                                          options);
        plan.flattenXY = flattenXY;
        plan.ampWUMethod = method;
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
  const auto numBatchGroups = params.batchSize / batchesPerGroup;
  const auto numTiles = calcNumUsableTiles(deviceInfo.getNumTiles(),
                                           numBatchGroups);
  const auto tilesPerX = m.addVariable(1, params.getOutputWidth());
  const auto tilesPerY = m.addVariable(1, params.getOutputHeight());
  const auto outChanGroups = (params.outputDepth + partialChansPerGroup - 1) /
                             partialChansPerGroup;
  const auto tilesPerZ = m.addVariable(1, outChanGroups);
  const auto tilesPerKernelY = m.addVariable(1, params.kernelSizeY);
  const auto inChanGroups = (params.inputDepth + inChansPerGroup - 1) /
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
                                     candidate, cache);
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
    return estimateReduceCycles(deviceInfo, params, candidate);
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
                                     candidate);
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
    return estimateReduceMemory(deviceInfo, params, candidate);
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
                            const ConvolutionParams &params) {
  std::vector<ConvVertexType> convVertexTypeCandidates;
  // We limit the use of the convolution instruction to cases where the number
  // of output channels is a multiple of the output channel grouping that would
  // be used.
  // TODO teach the convolution code to use smaller stores and / or zero pad
  // in this case.
  const auto convUnitsPerTile =
      getConvUnitsPerTile(deviceInfo, floatActivations, floatPartials);
  if (canUseConvolutionInstruction(floatActivations, floatPartials,
                                   params.strideY, params.strideX,
                                   deviceInfo) &&
      params.outputDepth % convUnitsPerTile == 0) {
    convVertexTypeCandidates.emplace_back(true, floatActivations,
                                          floatPartials);
  } else if (!floatActivations && !floatPartials &&
             canUseConvolutionInstruction(false, true,
                                          params.strideY, params.strideX,
                                          deviceInfo) &&
             params.outputDepth %
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
                                    const ConvolutionParams &params,
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
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
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
                   convVertexType, params, batchesPerGroup, costBounds, cache,
                   options);
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
           const ConvolutionParams &params,
           unsigned batchesPerGroup,
           CostBounds costBounds,
           PlanningCacheImpl *cache,
           const ConvOptions &options) {
  Cost bestCost = highestCost;
  Plan bestPlan;
  const auto convVertexTypeCandidates =
      params.isWeightUpdate
        ? getWeightUpdateVertexTypeCandidates(deviceInfo, floatActivations,
                                              floatPartials, params,
                                              options)
        : getConvVertexTypeCandidates(deviceInfo, floatActivations,
                                      floatPartials, params);
  for (const auto &convVertexType : convVertexTypeCandidates) {
    Plan candidate;
    Cost candidateCost;
    std::tie(candidate, candidateCost) =
        choosePlan(deviceInfo, convVertexType, tensorInChansPerGroup,
                   tensorOutChansPerGroup, tensorWeightOutChansPerGroup,
                   params, batchesPerGroup, costBounds, cache, options);
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
           const ConvOptions &options,
           const CostBounds costBounds,
           const poplar::Graph &graph,
           PlanningCacheImpl *cache) {
  validateLayerParams(inDimY, inDimX, inNumChans, kernelSizeY, kernelSizeX,
                      strideY, strideX, paddingY, paddingX,
                      numChannels, dType);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
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
                   tensorInChansPerGroup,
                   tensorOutChansPerGroup,
                   tensorWeightOutChansPerGroup,
                   params,
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

Plan getPlan(const poplar::Graph &graph,
             std::string dType,
             unsigned batchSize,
             unsigned inDimY, unsigned inDimX, unsigned inNumChans,
             std::vector<std::size_t> weightsShape,
             std::vector<unsigned> stride,
             std::vector<unsigned> padding,
             bool isFractional,
             ConvOptions options) {
  assert (weightsShape.size() == 3);
  assert (stride.size() == 2);
  assert (padding.size() == 2);
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  const auto strideY = stride[0];
  const auto strideX = stride[1];
  const auto paddingY = padding[0];
  const auto paddingX = padding[1];
  const auto kernelSizeY = weightsShape[0];
  const auto kernelSizeX = weightsShape[1];
  const auto numChannels = weightsShape[2];
  const auto partialsType = options.partialsType;
  auto cache = options.cache ? options.cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cache = tempCache.get();
  }
  PlanningCacheImpl::Params params(dType,
                                   {inDimY, inDimX, inNumChans},
                                   weightsShape, stride, padding,
                                   isFractional, false, 0, 0, options);
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto match = plans.find(params);
    if (match != plans.end())
      return *match->second;
  }
  if (options.useWinograd) {
    if (options.winogradPatchSize != 4 ||
        strideY != 1 || strideX != 1 ||
        kernelSizeY != 3 || kernelSizeX != 3) {
      throw popstd::poplib_error("Attempt to force winograd convolution for "
                               "invalid parameters");

    }
    plan.useWinograd = true;
    plan.winogradPatchSize = options.winogradPatchSize;
    return plan;
  }

  std::tie(plan, cost) = popconv::createPlan(inDimY, inDimX, inNumChans,
                                             0, 0, 0,
                                             kernelSizeY, kernelSizeX,
                                             strideY, strideX,
                                             paddingY, paddingX,
                                             numChannels, batchSize, dType,
                                             partialsType,
                                             isFractional, false, options,
                                             costBounds, graph,
                                             cache);
  if (options.percentageCyclesExcessForMemOptim) {
    /* Set new bounds based on previous search */
    const double newCyclesBound =
        static_cast<double>(cost.cycles)
        * (100.0 + options.percentageCyclesExcessForMemOptim) / 100.0;

    CostBounds newCostBounds(static_cast<unsigned>(newCyclesBound), 0, false);

    std::tie(plan, cost) = popconv::createPlan(inDimY, inDimX, inNumChans,
                                               0, 0, 0,
                                               kernelSizeY, kernelSizeX,
                                               strideY, strideX,
                                               paddingY, paddingX,
                                               numChannels, batchSize, dType,
                                               partialsType,
                                               isFractional, false,
                                               options,
                                               newCostBounds, graph,
                                               cache);
  }
  if (!tempCache.get()) {
    auto &plans = cache->plans;
    auto pPlan = std::unique_ptr<Plan>(new Plan(std::move(plan)));
    auto res = plans.emplace(std::make_pair(params, std::move(pPlan)));
    return *res.first->second;
  }
  return plan;
}

Plan getWeightUpdatePlan(const poplar::Graph &graph,
                         const poplar::Tensor &activations,
                         const poplar::Tensor &deltas,
                         std::vector<std::size_t> weightsShape,
                         std::vector<unsigned> stride,
                         std::vector<unsigned> padding,
                         bool isFractional,
                         ConvOptions options) {
  assert (weightsShape.size() == 3);
  assert (stride.size() == 2);
  assert (padding.size() == 2);
  const auto dType = graph.getTensorElementType(activations);
  const auto batchSize = activations.dim(0);
  const auto inDimY = activations.dim(2);
  const auto inDimX = activations.dim(3);
  const auto inNumChans = activations.dim(1) * activations.dim(4);
  if (options.noLHSRearrangement) {
    auto plan = getPlan(graph, dType,  batchSize, inDimY, inDimX,
                        inNumChans, weightsShape, stride, padding,
                        isFractional, options);
    plan.useConvolutionInstructions = false;
    return plan;
  }
  Plan plan;
  Cost cost;
  CostBounds costBounds(0, 0);
  const auto strideY = stride[0];
  const auto strideX = stride[1];
  const auto paddingY = padding[0];
  const auto paddingX = padding[1];
  const auto kernelSizeY = weightsShape[0];
  const auto kernelSizeX = weightsShape[1];
  const auto numChannels = weightsShape[2];
  const auto partialsType = options.partialsType;
  const auto fwdPlan = getPlan(graph, dType, batchSize, inDimY, inDimX,
                               inNumChans, weightsShape, stride, padding,
                               isFractional, options);
  const auto actChansPerGroup = fwdPlan.inChansPerGroup;
  const auto deltasChansPerGroup = deltas.dim(4);
  auto cache = options.cache ? options.cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cache = tempCache.get();
  }
  PlanningCacheImpl::Params params(dType,
                                   {inDimY, inDimX, inNumChans},
                                   weightsShape, stride, padding,
                                   isFractional, true, actChansPerGroup,
                                   deltasChansPerGroup, options);

  std::tie(plan, cost) = popconv::createPlan(inDimY, inDimX, inNumChans,
                                             actChansPerGroup,
                                             deltasChansPerGroup,
                                             fwdPlan.partialChansPerGroup,
                                             kernelSizeY, kernelSizeX,
                                             strideY, strideX,
                                             paddingY, paddingX,
                                             numChannels, batchSize, dType,
                                             partialsType, isFractional,
                                             true, options,
                                             costBounds, graph, cache);
  return plan;
}

} // end namespace conv
