// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ConvModel.hpp"
#include "ExchangeEstimator.hpp"
#include "poplibs_support/popopsPerformanceEstimation.hpp"
#include <poputil/exceptions.hpp>

namespace poplin {

using namespace poplibs_support;

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

// mapping between ConvSizeVariables and the std::vector<T>
// that is passed to the callback for an m.call<T> constraint.
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
makeConvSize(const std::vector<unsigned> &values,
             const std::vector<unsigned> &fieldGrainSize,
             const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
             const unsigned outChansPerGroup) {
  const unsigned numFieldDims = fieldGrainSize.size();
  ConvSizeVariablesVector<unsigned> convSizeVarsVector(values, numFieldDims);

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

    auto weightsPerConvUnit = target.getWeightsPerConvUnit(floatActivations);

    const auto weightBytesPerConvUnit =
        weightsPerConvUnit * target.getTypeSize(params.inputType);

    auto convUnitCoeffLoadBytesPerCycle =
        target.getConvUnitCoeffLoadBytesPerCycle();
    if (!options.use128BitConvUnitLoad) {
      convUnitCoeffLoadBytesPerCycle /= 2;
    }

    assert(numConvUnitsRequired != 0);
    if (inChansPerGroup != weightsPerConvUnit) {
      auto numConvUnitsOnIpu =
          getNumConvUnits(floatActivations, floatPartials, target);
      assert(numConvUnitsOnIpu % numConvUnitsRequired == 0);
      weightsPerConvUnit /= numConvUnitsOnIpu / numConvUnitsRequired;
      assert(weightsPerConvUnit % inChansPerGroup == 0);
    }
    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;

    return m.call<unsigned>(
        convSizeVarsVector,
        [&target, fieldGrainSize, convGroupsPerGroup, inChansPerGroup,
         outChansPerGroup, partialType, params, transformedDims,
         transformedInputDilation, transformedOutputStride,
         convUnitWeightHeight, cache, floatActivations, weightBytesPerConvUnit,
         convUnitCoeffLoadBytesPerCycle, numConvUnitsRequired](
            const std::vector<unsigned> &values) -> popsolver::DataType {
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

            return popsolver::DataType{
                getConvPartial1x1SupervisorOuterLoopCycleEstimate(
                    innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
                    tileNumConvGroups, tileNumInGroups, tileNumOutGroups,
                    outChansPerGroup, weightBytesPerConvUnit,
                    numConvUnitsRequired, convUnitCoeffLoadBytesPerCycle,
                    floatActivations, floatPartials,
                    target.getNumWorkerContexts())};
          }
          const auto zeroCycles = cache->mEstimateZeroSupervisorCycles(
              product(convSize.fieldSize) * convSize.batchSize,
              tileNumOutGroups, tileNumConvGroups, outChansPerGroup,
              target.getDataPathWidth(), target.getNumWorkerContexts());

          const auto innerLoopCycles =
              cache->mGetConvPartialnx1InnerLoopCycleEstimate(
                  convSize.batchSize, convSize.fieldSize, convSize.kernelSize,
                  convUnitWeightHeight, outChansPerGroup,
                  weightBytesPerConvUnit, numConvUnitsRequired,
                  convUnitCoeffLoadBytesPerCycle, target.getNumWorkerContexts(),
                  floatActivations, floatPartials, transformedInputDilation,
                  transformedOutputStride);
          return popsolver::DataType{
              getConvPartialnx1SupervisorOuterLoopCycleEstimate(
                  innerLoopCycles, tileNumConvGroups, tileNumOutGroups,
                  tileNumInGroups, outChansPerGroup, numConvUnitsRequired,
                  target.getNumWorkerContexts(), floatActivations,
                  floatPartials) +
              zeroCycles};
        },
        debugName);
  }
  case Plan::Method::SLIC: {
    return m.call<unsigned>(
        convSizeVarsVector,
        [&target, params, fieldGrainSize, convGroupsPerGroup, inChansPerGroup,
         outChansPerGroup, transformedInputDilation, transformedOutputStride,
         numConvUnitsRequired, slicWindowWidth, floatActivations, floatPartials,
         cache](const auto &values) -> boost::optional<popsolver::DataType> {
          const auto convSize =
              makeConvSize(values, fieldGrainSize, convGroupsPerGroup,
                           inChansPerGroup, outChansPerGroup);

          assert(transformedOutputStride.back() <= 2);

          // current vertex requirements
          assert(inChansPerGroup == outChansPerGroup);
          assert(convGroupsPerGroup * inChansPerGroup == 4);

          if (ceildiv(convSize.inChanSize, inChansPerGroup) != 1 ||
              ceildiv(convSize.outChanSize, outChansPerGroup) != 1) {
            return boost::none;
          }

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
                  params.outputTransform.stride.back(),
                  /* implicitZeroing */ true, convSize.batchSize,
                  convSize.fieldSize, target.getNumWorkerContexts(),
                  numConvUnitsRequired, slicWindowWidth, floatActivations,
                  floatPartials);
          const auto innerLoopCycles =
              cache->mGetConvPartialSlicInnerLoopCycles(
                  params.outputTransform.stride.back(),
                  /* implicitZeroing */ false, convSize.batchSize,
                  convSize.fieldSize, target.getNumWorkerContexts(),
                  numConvUnitsRequired, slicWindowWidth, floatActivations,
                  floatPartials);
          const auto weightLoadCycles =
              getConvPartialSlicSupervisorWeightLoadCycleEstimate(
                  convGroupsPerGroup, inChansPerGroup,
                  target.getNumWorkerContexts(), slicWindowWidth);
          return popsolver::DataType{
              cache->mGetConvPartialSlicSupervisorOuterLoopCycleEstimate(
                  implicitZeroInnerLoopCycles, innerLoopCycles,
                  weightLoadCycles, tileNumConvGroups, numWeightBlocks,
                  numConvUnitsRequired, slicWindowWidth, floatActivations,
                  floatPartials)};
        });
  }
  case Plan::Method::MAC: {
    const auto outputStrideX = transformedInputDilation.back();
    return m.call<unsigned>(
        convSizeVarsVector,
        [&target, fieldGrainSize, inChansPerGroup, convGroupsPerGroup,
         outChansPerGroup, transformedInputDilation, cache, outputStrideX,
         floatActivations, floatPartials](
            const std::vector<unsigned> &values) -> popsolver::DataType {
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
              (numActiveOutRows * tileOutWidth), tileNumOutGroups,
              tileNumConvGroups, outChansPerGroup, target.getDataPathWidth(),
              target.getNumWorkerContexts());
          const auto innerLoopCycles =
              cache->mEstimateConvPartialHorizontalMacInnerLoopCycles(
                  numActiveOutRows, tileOutWidth, outputStrideX,
                  tileKernelElements / tileKernelWidth, tileKernelWidth,
                  target.getNumWorkerContexts(), floatActivations,
                  floatPartials, inChansPerGroup, outChansPerGroup,
                  target.getDataPathWidth());
          return popsolver::DataType{
              getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
                  innerLoopCycles, tileNumConvGroups, tileNumInGroups,
                  tileNumOutGroups, target.getNumWorkerContexts(),
                  floatActivations, floatPartials) +
              zeroCycles};
        },
        debugName);
  } break;
  case Plan::Method::OUTER_PRODUCT: {
    assert(inChansPerGroup == 1);
    const auto numContexts = target.getNumWorkerContexts();
    const auto outputIsFloat = params.outputType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    return m.call<unsigned>(
        convSizeVarsVector,
        [fieldGrainSize, numContexts, convGroupsPerGroup, outChansPerGroup,
         inChansPerGroup, floatActivations, outputIsFloat, dataPathWidth](
            const std::vector<unsigned> &values) -> popsolver::DataType {
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
          return popsolver::DataType{vertexRuntime * numContexts};
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

    if (inChansPerGroup != weightsPerConvUnit) {
      const auto numConvUnitsonIpu =
          getNumConvUnits(params.inputType == poplar::FLOAT,
                          partialType == poplar::FLOAT, target);
      assert(numConvUnitsRequired != 0);
      assert(numConvUnitsonIpu % numConvUnitsRequired == 0);
      weightsPerConvUnit /= numConvUnitsonIpu / numConvUnitsRequired;
      assert(weightsPerConvUnit % inChansPerGroup == 0);
    }
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

ExchangeEstimates<popsolver::Variable> addExchangeCycleEstimates(
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

  inputsPerLevel.clear();
  weightsPerLevel.clear();

  // The number of cycles for exchange is the sum of the cycles for the input,
  // weights and partials for each level in the hierarchy (not including the
  // tile level). These are stored in each vector.  The sum of each vector is
  // returned to give itemised results and help with analysis.
  std::vector<popsolver::Variable> inputExchangeCycles;
  std::vector<popsolver::Variable> weightExchangeCycles;
  std::vector<popsolver::Variable> reduceFirstStageExchangeCycles;
  std::vector<popsolver::Variable> reduceRemainingStagesExchangeCycles;
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
        auto inputFieldSize = m.call<unsigned>(
            {outputFieldSize, sizesNextLevel.kernelSize[dim]},
            [dim, params](
                const std::vector<unsigned> &values) -> popsolver::DataType {
              const auto outputFieldSize = values[0];
              const auto kernelSizeForThisDim = values[1];
              return popsolver::DataType{getMaxInputRangeSize(
                  outputFieldSize, dim, params, kernelSizeForThisDim)};
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

    // because we distribute the weights evenly across all tiles that require
    // them we can deduce that 1/Nth of the weights are already on the correct
    // tile. this needs to be calculated because each serial split will
    // introduce a certain amount of iterations where the data is exchanged onto
    // the tile and therefore the more splits the higher the cost. however, for
    // example, if the weights are split over a single tile we would expect a
    // zero exchange cost. we do this for both weights and inputs because of the
    // swap operands transformation.

    const auto tilesUsedByWeights =
        m.product({m.product(partitionVars[level].fieldSplit),
                   partitionVars[level].batchSplit});
    numberOfWeights =
        m.sub(numberOfWeights, m.floordiv(numberOfWeights, tilesUsedByWeights));

    const auto tilesUsedByInputElements =
        partitionVars[level].outChanSplit.parallel;
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

    inputExchangeCycles.push_back(exchangeEstimator.getInputElementCycles(
        numberOfInputElements, params.inputType, level));

    weightExchangeCycles.push_back(
        exchangeEstimator.getCycles(numberOfWeights, params.inputType, level));

    // We do the first stage of any reduction separately so that we
    // can prune the search space based on this from previous best
    // cycles and because the first stage exchange cycles are independent
    // of the reduction plan.
    //
    // Any further stages are dependent on the reduction plan and their
    // cycle cost is added through a call.
    reduceFirstStageExchangeCycles.push_back(exchangeEstimator.getCycles(
        numberOfPartialSums, types[level + 1].resultType, level));

    auto reduceDimSizes = partitionsNextLevel.kernelSplit;
    reduceDimSizes.push_back(partitionsNextLevel.inChanSplit.parallel);
    const auto reductionDepth =
        m.product(reduceDimSizes); // TODO: duplicate popsolver variable
    const auto resultType = types[level + 1].resultType;
    auto remainingExchangeCycles = m.call<unsigned>(
        {numberOfPartialSums, reductionDepth},
        [exchangeEstimator, resultType, level,
         &options](const std::vector<unsigned> &vars) -> popsolver::DataType {
          const auto numPartialSums = vars[0];
          const auto reductionDepth = vars[1];

          if (reductionDepth <= 1) {
            return popsolver::DataType{0};
          }

          unsigned remainingDepth = reductionDepth;
          unsigned outputSizeThisStage = numPartialSums;
          popsolver::DataType cycles{0};
          const auto reducePlan = getMultiStageReducePlan(
              reductionDepth, options.enableMultiStageReduce);
          bool firstStage = true;
          for (const auto d : reducePlan) {
            // We add first stage reduction exchange cycles separately above.
            if (!firstStage) {
              cycles += popsolver::DataType{exchangeEstimator.getCycles(
                  outputSizeThisStage, resultType, level)};
            }
            const auto depthThisStage = ceildiv(remainingDepth, d);
            outputSizeThisStage = ceildiv(outputSizeThisStage, depthThisStage);
            remainingDepth = ceildiv(remainingDepth, depthThisStage);
            firstStage = false;
          }
          // Final reduction
          if (remainingDepth > 1 && !firstStage) {
            cycles += popsolver::DataType{exchangeEstimator.getCycles(
                outputSizeThisStage, resultType, level)};
          }
          return cycles;
        },
        "partialSumExchangeCycleEstimate");
    reduceRemainingStagesExchangeCycles.push_back(remainingExchangeCycles);
  }
  ExchangeEstimates<popsolver::Variable> result;
  result.inputExchangeCycles = m.sum(inputExchangeCycles);
  result.weightExchangeCycles = m.sum(weightExchangeCycles);
  result.reduceFirstStageExchangeCycles = m.sum(reduceFirstStageExchangeCycles);
  result.reduceRemainingStagesExchangeCycles =
      m.sum(reduceRemainingStagesExchangeCycles);

  return result;
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
    reduceDimSizes.push_back(partitionVars[level].inChanSplit.parallel);
    const auto reductionDepth =
        m.product(reduceDimSizes); // TODO: duplicate popsolver variable
    outputsPerLevel.push_back(m.ceildiv(partialsPerTile, reductionDepth));
    bool floatPartials = types[level + 1].resultType == poplar::FLOAT;
    bool floatOutput = types[level].resultType == poplar::FLOAT;
    const auto dataPathWidth = target.getDataPathWidth();
    const auto numWorkers = target.getNumWorkerContexts();
    const auto partialsVectorWidth =
        target.getVectorWidth(floatPartials ? poplar::FLOAT : poplar::HALF);
    const auto outputVectorWidth =
        target.getVectorWidth(floatOutput ? poplar::FLOAT : poplar::HALF);
    const auto memoryElementOffsets = target.getMemoryElementOffsets();
    const auto bytesPerPartialsElement =
        target.getTypeSize(floatPartials ? poplar::FLOAT : poplar::HALF);
    const auto cycleEstimate = m.call<unsigned>(
        {outputsPerLevel.back(), reductionDepth,
         partitionVars[level].inChanSplit.serial},
        [floatOutput, floatPartials, numWorkers, dataPathWidth,
         partialsVectorWidth, outputVectorWidth, memoryElementOffsets,
         bytesPerPartialsElement, &options,
         cache](const std::vector<unsigned> &vars) -> popsolver::DataType {
          return popsolver::DataType{cache->mEstimateConvReduceCycles(
              vars[0], vars[1], vars[2], floatOutput, floatPartials, numWorkers,
              dataPathWidth, partialsVectorWidth, outputVectorWidth,
              memoryElementOffsets, bytesPerPartialsElement,
              options.enableMultiStageReduce, options.enableFastReduce,
              options.enableSingleInputReduce)};
        });
    cycleSumOperands.push_back(cycleEstimate);
    // Temporary memory for the reduction will be given by the number of
    // outputs on a tile
    const auto elementBytes = target.getTypeSize(types[level + 1].resultType);
    const auto tempBytesEstimate = m.call<unsigned>(
        {outputsPerLevel.back(), reductionDepth},
        [elementBytes,
         &options](const std::vector<unsigned> &vars) -> popsolver::DataType {
          const auto numOutputs = vars[0];
          const auto reductionDepth = vars[1];
          if (reductionDepth <= 1) {
            return popsolver::DataType{0};
          }

          const auto reducePlan = getMultiStageReducePlan(
              reductionDepth, options.enableMultiStageReduce);
          unsigned remainingDepth = reductionDepth;
          unsigned numOutputsThisStage = numOutputs * reductionDepth;
          popsolver::DataType maxTempBytes{0};
          for (const auto d : reducePlan) {
            const auto depthThisStage = ceildiv(remainingDepth, d);
            const auto tempBytesThisStage = numOutputsThisStage * elementBytes;
            maxTempBytes = std::max<popsolver::DataType>(
                maxTempBytes, popsolver::DataType{tempBytesThisStage});
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

// the number of inputs in the tile level of the hierarchy is how many
// inputs *after* broadcast, here we want to know how many there are before
// so take the number of inputs at the hierarchy above and evenly split them.
static popsolver::Variable
addInputsPerTile(popsolver::Model &m, const popsolver::Variable usedTiles,
                 const std::vector<popsolver::Variable> &inputsPerLevel,
                 const ConvParams &params) {
  assert(!inputsPerLevel.empty());
  const auto inputsPerIPU = [&] {
    // when there is only one IPU the "previous level" is actually the original
    // convolution parameters.
    if (inputsPerLevel.size() == 1) {
      // we don't need to take into account the kernel transforms here because
      // the transformation is applied after the dynamic slice, which is why
      // we want to calculate the number of inputs per tile.
      const auto numberOfInputs =
          product(params.inputFieldShape) * params.batchSize *
          params.inputChannelsPerConvGroup * params.numConvGroups;
      return m.addConstant(numberOfInputs);
    } else {
      return inputsPerLevel[inputsPerLevel.size() - 2];
    }
  }();

  return m.ceildiv(inputsPerIPU, usedTiles);
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
  const unsigned fieldGrainSizeProduct = product(partitionVars.fieldGrainSize);
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

bool isFullyConnected(Pass pass) {
  return pass == Pass::FC_INFERENCE_FWD || pass == Pass::FC_TRAINING_FWD ||
         pass == Pass::FC_TRAINING_BWD || pass == Pass::FC_TRAINING_WU;
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
  bool isFullyConnectedLayer = isFullyConnected(options.pass);
  bool expandDims = false;
  bool swapOperands = false;
  bool outChanFlattenDims = false;
  bool combineConvGroups = false;
  assert(transforms.size() >= 2);
  const auto ipuLevel = transforms.size() - 2;
  for (unsigned level = 0; level <= ipuLevel; ++level) {
    if (transforms[level].swapOperands)
      swapOperands = true;
    if (!transforms[level].expandDims.empty())
      expandDims = true;
    if (!transforms[level].outChanFlattenDims.empty())
      outChanFlattenDims = true;
    if (transforms[level].combineConvGroupsFactor > 1)
      combineConvGroups = true;
  }
  bool padInChannels = transformedOnceUnpaddedParams.inputChannelsPerConvGroup %
                           inChansPerGroup !=
                       0;
  bool padPartialChannels =
      transformedOnceUnpaddedParams.outputChannelsPerConvGroup %
          partialChansPerGroup !=
      0;
  bool rearrangeInput = isConvWeightUpdate || expandDims || swapOperands ||
                        combineConvGroups || padInChannels ||
                        options.pass == Pass::FC_TRAINING_WU ||
                        (options.pass == Pass::FC_TRAINING_BWD && !isJointPlan);
  bool rearrangeWeights =
      isConvWeightUpdate || expandDims || outChanFlattenDims || swapOperands ||
      combineConvGroups || padInChannels || padPartialChannels;
  const auto weightsPerConvUnit =
      target.getWeightsPerConvUnit(params.inputType == poplar::FLOAT);
  bool rearrangeOutput = (!isConvWeightUpdate && swapOperands) ||
                         (isConvWeightUpdate && !swapOperands) ||
                         outChanFlattenDims || combineConvGroups ||
                         padPartialChannels ||
                         (options.pass == Pass::FC_TRAINING_WU && !isJointPlan);
  ;
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
      auto inputFieldSize = m.call<unsigned>(
          {outputFieldSize, convSize.kernelSize[dim]},
          [dim, transformedOnceParams](
              const std::vector<unsigned> &values) -> popsolver::DataType {
            return popsolver::DataType{getMaxInputRangeSize(
                values[0], dim, transformedOnceParams, values[1])};
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
      partitionVars[ipuLevel].inChanSplit.parallel,
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
popsolver::Variable
addDynamicSliceEstimate(popsolver::Model &m, const poplar::Target &target,
                        const popsolver::Variable &elementsPerTile,
                        const popsolver::Variable &serialSplit,
                        const poplar::Type &inType) {
  // assume we have to slice an even amount of weights on each tile for each
  // each split.
  const auto sliceSize = m.ceildiv(elementsPerTile, serialSplit);

  const std::vector<popsolver::Variable> vars = {serialSplit, sliceSize};
  return m.call<unsigned>(
      vars,
      [target,
       inType](const std::vector<unsigned> &vars) -> popsolver::DataType {
        const auto serialSplit = vars[0];
        const auto sliceSize = vars[1];
        assert(serialSplit != 0);

        // when not splitting serially we require no dynamic slicing or
        // updating.
        if (serialSplit == 1) {
          return popsolver::DataType{0};
        }
        auto cycles =
            popops::getDynamicSlice1dEstimate(target, inType, sliceSize, 1);
        return popsolver::DataType{cycles};
      });
}

static popsolver::Variable
addDynamicUpdateEstimate(popsolver::Model &m, const poplar::Target &target,
                         const popsolver::Variable &outputsPerTile,
                         const PartitionVariables &tileSplits,
                         const std::vector<ConvTypes> &types) {
  const auto &outChanSerialSplit = tileSplits.outChanSplit.serial;
  const unsigned intraTileLevel = types.size() - 1;
  const auto outputsType = types[intraTileLevel].resultType;
  return addDynamicSliceEstimate(m, target, outputsPerTile, outChanSerialSplit,
                                 outputsType);
}

popsolver::Variable addCastEstimate(popsolver::Model &m,
                                    const poplar::Target &target,
                                    const popsolver::Variable outputsPerTile,
                                    const PartitionVariables &tileSplits,
                                    const std::vector<ConvTypes> &types) {
  assert(types.size() > 0);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto partialsType = types.back().resultType;
  const auto resultType = types[0].resultType;
  const auto partialsVectorWidth = target.getVectorWidth(partialsType);
  const auto resultVectorWidth = target.getVectorWidth(resultType);
  const auto &inChanSerialSplit = tileSplits.inChanSplit.serial;
  return m.call<unsigned>(
      {outputsPerTile, inChanSerialSplit},
      [numWorkers, partialsVectorWidth, resultVectorWidth](
          const std::vector<unsigned> &vars) -> popsolver::DataType {
        const auto outputsPerTile = vars[0];
        const auto inChanSerialSplit = vars[1];
        assert(inChanSerialSplit >= 1);
        if (inChanSerialSplit == 1) {
          return popsolver::DataType{0};
        }
        return popsolver::DataType{
            estimateCastCycles(outputsPerTile, partialsVectorWidth,
                               resultVectorWidth, numWorkers)};
      },
      "castCycles");
}

// estimation function for addInPlace accumulation of input-channel-serially
// split convolution partials
std::pair<popsolver::Variable, popsolver::Variable>
addInPlaceEstimate(popsolver::Model &m, const poplar::Target &target,
                   const popsolver::Variable &outputsPerTile,
                   const PartitionVariables &tileSplits,
                   const std::vector<ConvTypes> &types) {
  // currently the input channels are serially split only in the
  // intra-IPU level. TODO: T12878 Assert that this is the case.
  assert(types.size() > 0);
  const unsigned intraTileLevel = types.size() - 1;
  const auto outputsType = types[intraTileLevel].resultType;

  // Input channels serial splits do not cause a corresponding split in the
  // outputs. Hence the operation must be performed on the whole output
  const auto &inChanSerialSplit = tileSplits.inChanSplit.serial;
  const std::vector<popsolver::Variable> vars = {inChanSerialSplit,
                                                 outputsPerTile};
  auto cycles = m.call<unsigned>(
      vars,
      [target,
       outputsType](const std::vector<unsigned> &vars) -> popsolver::DataType {
        const auto &inChanSerialSplit = vars[0];
        const auto &outputsPerTile = vars[1];
        assert(inChanSerialSplit != 0);

        // when not splitting serially we require no inplace addition
        if (inChanSerialSplit == 1) {
          return popsolver::DataType{0};
        }
        return popsolver::DataType{
            popops::getBinaryOp1DInPlaceSupervisorEstimate(
                target, outputsType, popops::expr::BinaryOpType::ADD,
                outputsPerTile)};
      });

  // Estimate temp memory usage
  const auto partialType = types[intraTileLevel].resultType;
  const auto one = m.addConstant(1);
  const auto two = m.addConstant(2);
  auto isInChanSeriallySplit =
      m.min({m.floordiv(inChanSerialSplit, two), one}, "isInChanSeriallySplit");
  auto partialStorage = m.product(
      {outputsPerTile, m.addConstant(target.getTypeSize(partialType))},
      "addInPlaceTempBytes");
  auto tempBytes = m.product({isInChanSeriallySplit, partialStorage});
  return std::make_pair(cycles, tempBytes);
}

// estimation function for zero memory setting of output before addInPlace
// operations for every input channel serial split convolution
static popsolver::Variable
memsetZeroEstimate(popsolver::Model &m, const poplar::Target &target,
                   const popsolver::Variable &outputsPerTile,
                   const PartitionVariables &tileSplits,
                   const std::vector<ConvTypes> &types) {
  // currently the input channels are serially split only in the
  // intra-IPU level. TODO: T12878 Assert that this is the case.
  assert(types.size() > 0);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto intraTileLevel = types.size() - 1;
  const auto partialType = types[intraTileLevel].resultType;
  const auto vectorWidth = target.getVectorWidth(partialType);

  // Input channels serial splits do not cause a corresponding split in the
  // outputs. Hence the operation must be performed on the whole output
  const auto &inChanSerialSplit = tileSplits.inChanSplit.serial;
  const std::vector<popsolver::Variable> vars = {inChanSerialSplit,
                                                 outputsPerTile};
  return m.call<unsigned>(
      vars,
      [vectorWidth,
       numWorkers](const std::vector<unsigned> &vars) -> popsolver::DataType {
        const auto &inChanSerialSplit = vars[0];
        const auto &outputsPerTile = vars[1];

        assert(inChanSerialSplit != 0);
        // when not splitting serially we require no inplace addition
        if (inChanSerialSplit == 1) {
          return popsolver::DataType{0};
        }

        // rough cycles estimate of vertex overhead plus inner loop
        const unsigned cyclesPerVector = 1;
        const unsigned cyclesLoopOverhead = 0;
        const auto innerLoopCycles =
            cyclesPerVector * ceildiv(outputsPerTile, numWorkers * vectorWidth);
        return popsolver::DataType{(cyclesLoopOverhead + innerLoopCycles) *
                                   numWorkers};
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
  // isSeriallySplit is 1 if and only if any serial split (either
  // inChanSplit.serial or outChanSplit.serial) is greater than 1.
  const auto isSeriallySplit = m.addVariable(
      popsolver::DataType{0}, popsolver::DataType{1}, "isSeriallySplit");
  m.less(isSeriallySplit, m.product({tileSplits.inChanSplit.serial,
                                     tileSplits.outChanSplit.serial}));

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
    const boost::optional<Cost> &referenceCost, const ConvOptions &options,
    PlanningCacheImpl::CycleEstimationImpl *cache) {
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
    variables.push_back(vars.inChanSplit.parallel);
    variables.push_back(vars.inChanSplit.serial);
    variables.push_back(vars.convGroupSplit);
    variables.insert(variables.end(), vars.fieldSplit.begin(),
                     vars.fieldSplit.end());
    variables.insert(variables.end(), vars.kernelSplit.begin(),
                     vars.kernelSplit.end());
  };
  (void)m.call<popsolver::DataType>(variables,
                                    [](const auto &) -> popsolver::DataType {
                                      return popsolver::DataType{0U};
                                    });

  Estimates<popsolver::Variable> e;

  std::vector<popsolver::Variable> inputsPerLevel, weightsPerLevel;

  e.itemisedExchangeCycles = addExchangeCycleEstimates(
      m, partitionVars, convSize, transformedDims, exchangeEstimator,
      transformedOnceParams, options, types, inputsPerLevel, weightsPerLevel);

  std::tie(e.transformCycles, e.transformTempBytes) = addTransformCycleEstimate(
      m, untransformedParams, transformedOnceParams,
      transformedOnceUnpaddedParams, transforms, partitionVars,
      transformedConvSize, transformedDims, inChansPerGroup,
      partialChansPerGroup, types, isJointPlan, options, target);

  const auto &intraTileSplits = partitionVars.back();

  // create variables for the number of inputs and weights per tile before being
  // transformed and broadcast out. this is so we can calculate how much data
  // is dynamically sliced for serial convolutions. when calculating this we
  // assume the weights are distributed evenly.
  const auto weightsPerTile =
      addWeightsPerTile(m, usedTiles, weightsPerLevel, transformedOnceParams);
  const auto inputsPerTile =
      addInputsPerTile(m, usedTiles, inputsPerLevel, transformedOnceParams);

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

  const std::vector<popsolver::Variable> serialSplitFactors = {
      intraTileSplits.inChanSplit.serial, intraTileSplits.outChanSplit.serial};
  const auto serialSplits = m.product(serialSplitFactors);

  // Add a redundant inequality that relates the cycles required to calculate
  // the partial sums with the maximum number of MACs per cycle. Although this
  // constraint isn't necessary it provides an easy to calculate lower bound
  // on the number of cycles required that can be used to prune the search
  // space.
  const auto maxMACsPerCyclePerTile = getMaxMACsPerCyclePerTile(
      target, types.back().partialType, transformedOnceParams.inputType, method,
      slicWindowWidth);
  const auto totalMacs = cache->mGetNumberOfMACs(transformedOnceParams);
  m.lessOrEqual(popsolver::DataType{totalMacs / maxMACsPerCyclePerTile},
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
  // total by the amount of times we plan to execute this convolution.
  auto inputsDynamicSliceCycles = addDynamicSliceEstimate(
      m, target, inputsPerTile, intraTileSplits.inChanSplit.serial,
      transformedOnceParams.inputType);
  auto weightsDynamicSliceCycles = addDynamicSliceEstimate(
      m, target, weightsPerTile, serialSplits, transformedOnceParams.inputType);
  e.dynamicSliceCycles =
      m.sum({inputsDynamicSliceCycles, weightsDynamicSliceCycles});

  const auto &outputsPerTile = outputsPerLevel.back();
  e.dynamicUpdateCycles = addDynamicUpdateEstimate(m, target, outputsPerTile,
                                                   intraTileSplits, types);
  e.memsetZeroBeforeAddInPlace =
      memsetZeroEstimate(m, target, outputsPerTile, intraTileSplits, types);
  std::tie(e.addInPlaceCycles, e.addInPlaceTempBytes) =
      addInPlaceEstimate(m, target, outputsPerTile, intraTileSplits, types);

  // If input channel serial splits are used, casting is deferred until after
  // all serial splits have been processed.
  e.castCycles =
      addCastEstimate(m, target, outputsPerTile, intraTileSplits, types);

  e.totalExchangeCycles =
      m.sum({e.itemisedExchangeCycles.inputExchangeCycles,
             e.itemisedExchangeCycles.weightExchangeCycles,
             e.itemisedExchangeCycles.reduceFirstStageExchangeCycles,
             e.itemisedExchangeCycles.reduceRemainingStagesExchangeCycles});

  e.totalCycles =
      m.sum({e.dynamicSliceCycles, e.transformCycles, e.totalExchangeCycles,
             e.tileLevelTransformCycles, e.partialCalcCycles, e.reduceCycles,
             e.dynamicUpdateCycles, e.addInPlaceCycles});
  e.totalCycles = m.product({e.totalCycles, serialSplits});
  e.totalCycles = m.sum({e.memsetZeroBeforeAddInPlace, e.totalCycles,
                         e.rearrangeBeforeSliceCycles, e.castCycles});

  // take the total amount of temp bytes alive at the same time.
  e.totalTempBytes =
      m.sum({e.rearrangeBeforeSliceTempBytes,
             m.max({e.transformTempBytes,
                    m.sum({e.tileLevelTransformTempBytes, e.convTempBytes}),
                    e.reduceTempBytes,
                    e.rearrangeBeforeSliceTempDuringRearrangeBytes}),
             e.addInPlaceTempBytes});

  // calculate the positive cycle difference for each step in the cost model.
  if (referenceCost) {
    const auto posDiff = [&m](popsolver::Variable lhs,
                              popsolver::DataType rhs) {
      // can't use Model::sub here because that will invalidate the plan if the
      // answer is negative.
      return m.call<popsolver::DataType>(
          {lhs},
          [rhs](const std::vector<popsolver::DataType> &vs)
              -> popsolver::DataType {
            return popsolver::DataType{
                std::max<int64_t>(0, int64_t(*vs[0]) - *rhs)};
          });
    };

    const auto &c = *referenceCost;
    e.totalPerStepCycleDiff = m.sum(
        {posDiff(e.rearrangeBeforeSliceCycles, c.rearrangeBeforeSliceCycles),
         posDiff(e.memsetZeroBeforeAddInPlace, c.memsetZeroBeforeAddInPlace),
         posDiff(e.dynamicSliceCycles, c.dynamicSliceCycles),
         posDiff(e.transformCycles, c.transformCycles),

         // TODO: should this be using the itemised exchange estimates?
         posDiff(e.totalExchangeCycles, c.totalExchangeCycles),

         posDiff(e.tileLevelTransformCycles, c.tileLevelTransformCycles),
         posDiff(e.partialCalcCycles, c.partialCalcCycles),
         posDiff(e.reduceCycles, c.reduceCycles),
         posDiff(e.dynamicUpdateCycles, c.dynamicUpdateCycles),
         posDiff(e.addInPlaceCycles, c.addInPlaceCycles),
         posDiff(e.castCycles, c.castCycles)});
  } else {
    e.totalPerStepCycleDiff = m.addConstant(popsolver::DataType::max());
  }

  e.totalTiles = usedTiles;

  return e;
}

Plan::Method getFullyConnectedBwdMethod(Plan::Method fwdMethod) {
  if (fwdMethod == Plan::Method::OUTER_PRODUCT) {
    return Plan::Method::MAC;
  }
  return fwdMethod;
}

static Estimates<popsolver::Variable> addBwdEstimates(
    popsolver::Model &m, ConvParams bwdUntransformedParams,
    ConvParams bwdTransformedOnceParams,
    ConvParams bwdTransformedOnceUnpaddedParams,
    const unsigned numLevelsOfHierarchy,
    const std::vector<PartitionVariables> &partitionVars,
    const std::vector<ConvSizeVariables> &convSize,
    const std::vector<ConvTransform> &transforms, Plan::Method method,
    unsigned slicWindowWidth, unsigned numConvUnitsRequired,
    const popsolver::Variable usedTiles, const poplar::Target &target,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<ConvTypes> &types, const bool isJointPlan,
    const unsigned convGroupsPerGroup, const unsigned inChansPerGroup,
    const unsigned partialChansPerGroup,
    const boost::optional<Cost> &referenceCost, const ConvOptions &options,
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
    return {zero, zero, zero, zero};
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

  std::vector<std::unordered_set<unsigned>> transformedDims(
      numLevelsOfHierarchy);
  return addEstimates(
      m, bwdPartitionVars, bwdConvSize, bwdTransformedConvSize, usedTiles,
      transformedDims, target, perLevelExchangeBytesPerCycle,
      bwdUntransformedParams, bwdTransformedOnceParams,
      bwdTransformedOnceUnpaddedParams, isJointPlan, convGroupsPerGroup,
      bwdInChansPerGroup, partialChansPerGroup, types, transforms, bwdMethod,
      slicWindowWidth, numConvUnitsRequired,
      Plan::LinearizeTileOrder::FC_BWD_AS_CONV, referenceCost, options, cache);
}

Plan::Method getFullyConnectedWUMethod(const ConvParams &fwdParams,
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
    const unsigned partialChansPerGroup,
    const boost::optional<Cost> &referenceCost, const ConvOptions &options,
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
    return {zero, zero, zero, zero};
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
  return addEstimates(
      m, wuPartitionVars, wuConvSize, wuTransformedConvSize, usedTiles,
      transformedDims, target, perLevelExchangeBytesPerCycle,
      wuUntransformedParams, wuTransformedOnceParams,
      wuTransformedOnceUnpaddedParams, isJointPlan, convGroupsPerGroup,
      wuInChansPerGroup, wuPartialChansPerGroup, types, transforms, wuMethod,
      slicWindowWidth, numConvUnitsRequired, Plan::LinearizeTileOrder::FC_WU,
      referenceCost, options, cache);
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
bool canDeferDilation(const ConvParams &params, unsigned dim) {
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

ConvParams calculateSwappedParams(const ConvParams &params, bool swapOperands) {
  auto swappedParams = params;
  if (swapOperands) {
    poplin::swapOperands(swappedParams);
  }
  return swappedParams;
}

void expandDim(ConvParams &params, unsigned dim) {
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

ConvParams calculateExpandedParams(const ConvParams &params,
                                   const std::vector<unsigned> &expandDims) {
  auto expandedParams = params;
  for (const auto dim : expandDims) {
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

ConvParams
calculateFlattenedParams(const ConvParams &params,
                         const std::vector<unsigned> &outChanFlattenDims,
                         std::vector<unsigned> &flattenDims) {
  flattenDims.clear();
  auto flattenedParams = params;
  if (!outChanFlattenDims.empty()) {
    poplin::swapOperands(flattenedParams);
    for (const auto dim : outChanFlattenDims) {
      expandDim(flattenedParams, dim);
      // Flatten into the batch axis (this will become the output channel
      // axis when we swap back).
      flattenedParams.batchSize *= flattenedParams.inputFieldShape[dim];
      flattenedParams.inputFieldShape[dim] = 1;
    }
    poplin::swapOperands(flattenedParams);
  }
  // Flatten from the innermost out.

  if (flattenedParams.batchSize > 0) {
    flattenDims.push_back(0);
  }
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

ConvParams calculateGroupedParams(ConvParams groupedParams,
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

  std::vector<unsigned> ignoredFlattenedDims;
  const auto flattenedParams = calculateFlattenedParams(
      expandedParams, transform.outChanFlattenDims, ignoredFlattenedDims);

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
          thisPartition.get().get_optional<popsolver::DataType>(pathSuffix);
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
    constrainSplitVar("inChanSplit", p.inChanSplit);
    constrainVar("convGroupSplit", p.convGroupSplit);
    // All other PartitionVariables members are dependent on these splits.
  }
}

template <typename T> static inline std::string arrIndStr(T level) {
  return "[" + std::to_string(level) + "]";
}

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
  const auto val = options.planConstraints.get_optional<popsolver::DataType>(
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

// SLIC is only possible when the output has a stride of 1 or 2 in the  inner
// most dimension because this is implemented by striding the
// weights window across the input which is done by the SLIC vertex.
// Input dilation is also an issue because that is
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
    m.equal(m.addConstant(lvl1Params.inputTransform.flip[dim]),
            popsolver::DataType{0});

    // We don't handle kernel dilation, padding and flipping in the SLIC vertex
    // for now.
    m.equal(m.addConstant(lvl1Params.kernelTransform.dilation[dim]),
            popsolver::DataType{1});
    m.equal(m.addConstant(lvl1Params.kernelTransform.paddingLower[dim]),
            popsolver::DataType{0});
    m.equal(m.addConstant(lvl1Params.kernelTransform.paddingUpper[dim]),
            popsolver::DataType{0});
    m.equal(m.addConstant(lvl1Params.kernelTransform.flip[dim]),
            popsolver::DataType{0});

    if (dim == p.fieldGrainSize.size() - 1) {
      m.lessOrEqual(m.addConstant(lvl1Params.outputTransform.stride[dim]),
                    popsolver::DataType{2});
    }
  }

  m.equal(s.numInChanGrains, popsolver::DataType{1});
  m.equal(s.numOutChanGrains, popsolver::DataType{1});
}

// The Outer Product method can only be used if certain criteria are met (e.g.
// a batch size of 1 on any tile). See function implementation for a full list.
// The planner will not choose an Outer Product method unless all of these
// criteria are met.
static void addOuterProductConstaints(popsolver::Model &m,
                                      const PartitionVariables &p,
                                      const ConvSizeVariables &s,
                                      const ConvParams &lvl1Params) {
  m.equal(s.batchSize, popsolver::DataType{1});

  assert(lvl1Params.outputTransform.stride.size() == p.fieldGrainSize.size());
  assert(lvl1Params.inputTransform.dilation.size() == p.fieldGrainSize.size());
  assert(lvl1Params.inputTransform.flip.size() == p.fieldGrainSize.size());
  for (auto dim = 0U; dim < p.fieldGrainSize.size(); ++dim) {
    m.equal(s.kernelSize[dim], popsolver::DataType{1});
    m.equal(m.addConstant(lvl1Params.outputTransform.stride[dim]),
            popsolver::DataType{1});
    m.equal(m.addConstant(lvl1Params.inputTransform.dilation[dim]),
            popsolver::DataType{1});
    m.equal(m.addConstant(lvl1Params.inputTransform.flip[dim]),
            popsolver::DataType{0});
    m.equal(getInputChannelCount(m, p, s), popsolver::DataType{1});

    // Output size == (padded) input size (because kernelSize and stride are 1)
    m.equal(getInputFieldSize(m, p, s, dim), popsolver::DataType{1});
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
    splits.push_back(p.inChanSplit.parallel);
    splits.push_back(p.convGroupSplit);
    splits.insert(splits.end(), p.fieldSplit.begin(), p.fieldSplit.end());
    splits.insert(splits.end(), p.kernelSplit.begin(), p.kernelSplit.end());
    const auto levelSplit =
        m.product(splits, arrIndStr(level) + ".partition.total");
    m.lessOrEqual(levelSplit, popsolver::DataType{hierarchy[level]});
    perLevelSplits.push_back(levelSplit);
  }

  return m.product(std::move(perLevelSplits));
}

Estimates<popsolver::Variable> constructModel(
    const poplar::Target &target, const std::vector<ConvTransform> &transforms,
    const std::vector<ConvTypes> &types, const std::vector<unsigned> &hierarchy,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<unsigned> &fieldGrainSize,
    const ConvVertexType &convVertexType, const ConvParams &untransformedParams,
    bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
    const boost::optional<Plan> &referencePlan,
    const boost::optional<Cost> &referenceCost,
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
    const auto numGrains =
        ceildiv(transformedOnceParams.getOutputSize(dim), fieldGrainSize[dim]);

    convSize.back().numFieldGrains.push_back(
        m.addConstant(std::max(numGrains, 1UL),
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
          inputSize = m.call<unsigned>(
              {outputSize, transformedConvSize.back().kernelSize[dim]},
              [dim, transformedOnceParams](
                  const std::vector<unsigned> &values) -> popsolver::DataType {
                return DataType{getMaxInputRangeSize(
                    values[0], dim, transformedOnceParams, values[1])};
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
      // TODO: T10408 We do not support splitting the input channels serially
      // during a joint plan as that will become a serial field split
      // during the backward pass, which is not currently supported.
      if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
        p.inChanSplit.serial = m.addConstant(
            1, arrIndStr(level) + ".partition.inChanSplit.serial");
      } else {
        p.inChanSplit.serial =
            addPartitionConstant(m, options, level, "inChanSplit.serial", [&] {
              return m.addVariable(1, levelMaxSplit);
            });
      }
      p.outChanSplit.serial =
          addPartitionConstant(m, options, level, "outChanSplit.serial",
                               [&] { return m.addVariable(1, levelMaxSplit); });

      // we must avoid splitting the convolutions serially when it will
      // produce different sized convolutions as this is implemented as a
      // repeat loop of the same sub-convolution. we enforce this by
      // requiring that the serial split is a factor of the total number of
      // output channels.
      const auto initialOutputChansPerGroup =
          transformedViewParams.getNumOutputChansPerConvGroup();
      m.factorOf(popsolver::DataType{std::max(initialOutputChansPerGroup, 1ul)},
                 p.outChanSplit.serial);

      const auto initialInputChansPerConvGroup =
          transformedViewParams.getNumInputChansPerConvGroup();
      m.factorOf(
          popsolver::DataType{std::max(initialInputChansPerConvGroup, 1ul)},
          p.inChanSplit.serial);

      // Only support one kind of serial split at a time (for now)
      m.equal(m.min({p.inChanSplit.serial, p.outChanSplit.serial}),
              popsolver::DataType{1});
    } else {
      p.inChanSplit.serial =
          m.addConstant(1, arrIndStr(level) + ".partition.outChanSplit.serial");
      p.outChanSplit.serial =
          m.addConstant(1, arrIndStr(level) + ".partition.outChanSplit.serial");
    }

    if (referencePlan) {
      // TODO: this only needs to be "m.equal(total serial splits)", we don't
      // need to differentiate betweem input and output as they both get lowered
      // to a Repeat program that can be shared across convolutions.
      //
      // Ensure we match serial splits with the reference plan
      // This potentially causes factorisation problems which can make the plan
      // impossible immediately.
      const auto inReference = m.addConstant(
          referencePlan->partitions[level].inChanSplit.serial,
          "reference." + arrIndStr(level) + ".partition.inChanSplit.serial");
      const auto outReference = m.addConstant(
          referencePlan->partitions[level].outChanSplit.serial,
          "reference." + arrIndStr(level) + ".partition.outChanSplit.serial");
      m.equal(p.inChanSplit.serial, inReference);
      m.equal(p.outChanSplit.serial, outReference);
    }

    auto totalOutChanSplit =
        m.product({p.outChanSplit.parallel, p.outChanSplit.serial});
    m.lessOrEqual(totalOutChanSplit, prevConvSize.numOutChanGrains);

    p.inChanSplit.parallel = m.addVariable(
        1, levelMaxSplit, arrIndStr(level) + ".partition.inChanSplit.parallel");
    auto totalInChanSplit =
        m.product({p.inChanSplit.parallel, p.inChanSplit.serial});
    m.lessOrEqual(totalInChanSplit, prevConvSize.numInChanGrains);

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
    nextConvSize.numOutChanGrains = m.ceildivConstrainDivisor(
        prevConvSize.numOutChanGrains, totalOutChanSplit,
        arrIndStr(level + 1) + ".size.outChanGrains");
    nextConvSize.numInChanGrains = m.ceildivConstrainDivisor(
        prevConvSize.numInChanGrains, totalInChanSplit,
        arrIndStr(level + 1) + ".size.inChanGrains");

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
      Plan::LinearizeTileOrder::STANDARD, referenceCost, options, cache);

  if (isJointPlan) {
    assert(options.pass == Pass::FC_TRAINING_FWD);

    const auto bwd = addBwdEstimates(
        m, untransformedParams, transformedOnceParams,
        transformedOnceUnpaddedParams, numLevelsOfHierarchy, partitionVars,
        convSize, transforms, method, slicWindowWidth, numConvUnitsRequired,
        usedTiles, target, perLevelExchangeBytesPerCycle, types, isJointPlan,
        convGroupsPerGroup, inChansPerGroup, partialChansPerGroup,
        referenceCost, options, cache);

    const auto wu = addWuEstimates(
        m, untransformedParams, transformedOnceParams,
        transformedOnceUnpaddedParams, numLevelsOfHierarchy, partitionVars,
        convSize, transforms, method, slicWindowWidth, numConvUnitsRequired,
        usedTiles, target, numFieldDims, perLevelExchangeBytesPerCycle, types,
        isJointPlan, convGroupsPerGroup, inChansPerGroup, partialChansPerGroup,
        referenceCost, options, cache);

    if (objective.getTileTempMemoryBound() > popsolver::DataType{0}) {
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

    // report the total diff of all three phases.
    if (referenceCost) {
      e.totalPerStepCycleDiff =
          m.sum({e.totalPerStepCycleDiff, bwd.totalPerStepCycleDiff,
                 wu.totalPerStepCycleDiff},
                "totalPerStepCycleDiff");
    }

    // report the max amount of tiles used in all three phases.
    e.totalTiles = m.max({e.totalTiles, bwd.totalTiles, wu.totalTiles});
  }

  // if an explicit cycle or memory bound has been added to the objective then
  // enforce that. additionally, depending on the object type prune the
  // relevant variable based upon the best plan found so far.
  auto cyclesBound = objective.getCyclesBound();
  auto memoryBound = objective.getTileTempMemoryBound();
  popsolver::DataType perStepBound = popsolver::DataType::max();
  popsolver::DataType tilesBound = popsolver::DataType::max();

  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    cyclesBound = std::min(cyclesBound, bestCost.totalCycles);
    break;
  case PlanningObjective::MINIMIZE_COST_DIFF:
    perStepBound = std::min(perStepBound, bestCost.totalPerStepCycleDiff);

    if (bestCost.totalPerStepCycleDiff == popsolver::DataType{0}) {
      if (objective.getMinimizeForTiles()) {
        tilesBound = std::min(tilesBound, bestCost.totalTiles);
      } else {
        memoryBound = std::min(memoryBound, bestCost.totalTempBytes);
      }
    }
    break;
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    memoryBound = std::min(memoryBound, bestCost.totalTempBytes);
    break;
  case PlanningObjective::MINIMIZE_TILES:
    tilesBound = std::min(tilesBound, bestCost.totalTiles);
    break;
  }

  m.lessOrEqual(e.totalCycles, cyclesBound);
  m.lessOrEqual(e.totalTempBytes, memoryBound);
  m.lessOrEqual(e.totalPerStepCycleDiff, perStepBound);
  m.lessOrEqual(e.totalTiles, tilesBound);

  return e;
}

} // namespace poplin
