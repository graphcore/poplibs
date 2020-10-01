// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplinCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

#include <cassert>

using namespace poplar;

namespace poplin {

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialnx1)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad,
    unsigned numConvUnits) {
  // TODO: T12902 Add cost estimates for non-limited version.
  (void)useLimitedVer;
  CODELET_SCALAR_VAL(kernelOuterSizeM1, unsigned);
  CODELET_SCALAR_VAL(kernelInnerElementsM1, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(ampKernelHeightM1, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(zerosInfo, unsigned);

  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(weights);
  const auto kernelOuterSize = kernelOuterSizeM1 + 1;
  const auto kernelInnerElements = kernelInnerElementsM1 + 1;
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;
  const auto ampKernelHeight = ampKernelHeightM1 + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  const auto kernelSize = kernelOuterSize * kernelInnerElements;
  assert(kernelSize > 0);
  const auto usedContexts = worklists.size() / kernelSize;

  bool floatPartials = accumType == FLOAT;

  std::vector<unsigned> tZeroWorkList;
  tZeroWorkList.reserve(numWorkerContexts);
  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    tZeroWorkList.push_back((zerosInfo + numWorkerContexts - 1) /
                            numWorkerContexts);
  }

  uint64_t zeroCycles = getZeroSupervisorVertexCycleEstimate(
      tZeroWorkList, numOutGroups * numConvGroups, dataPathWidth,
      numWorkerContexts, floatPartials);
  if (numInGroups * inChansPerGroup == 0) {
    return convNx1Overhead() + zeroCycles;
  }
  workerPartitions.reserve(usedContexts);
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    workerPartitions.back().reserve(kernelSize);
    for (auto k = 0U; k != kernelSize; ++k) {
      workerPartitions.back().emplace_back();
      const auto &wl = worklists[k * usedContexts + context];
      workerPartitions.back().back().reserve(wl.size() / 3);
      for (auto wi = 0U; wi < wl.size(); wi += 3) {
        // The number of elements minus 3 is the second element in the work list
        int numFieldPos;
        if (useLimitedVer) {
          numFieldPos = static_cast<short>(wl[wi + 1]) + 3;
        } else {
          numFieldPos = static_cast<int>(wl[wi + 1]) + 3;
        }
        workerPartitions.back().back().push_back(numFieldPos);
      }
    }
  }
  bool floatWeights = fpType == FLOAT;
  const auto weightBytesPerConvUnit =
      target.getWeightsPerConvUnit(fpType == FLOAT) *
      target.getTypeSize(fpType);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }
  return zeroCycles + getConvPartialnx1SupervisorCycleEstimate(
                          workerPartitions, numConvGroups, numOutGroups,
                          numInGroups, kernelInnerElements, kernelOuterSize,
                          ampKernelHeight, inChansPerGroup, outChansPerGroup,
                          weightBytesPerConvUnit, numConvUnits,
                          convUnitCoeffLoadBytesPerCycle, numWorkerContexts,
                          floatWeights, floatPartials);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartial1x1Out)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad,
    unsigned numConvUnits) {
  // TODO: T12902 Add cost estimates for non-limited version.
  (void)useLimitedVer;
  CODELET_VECTOR_VALS(worklists, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  const auto numWorkerContexts = target.getNumWorkerContexts();
  CODELET_FIELD(weights);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);
  // find max work to bt done per worker
  std::vector<std::vector<unsigned>> workerPartitions;
  assert(worklists.size() / 3 <= target.getNumWorkerContexts());
  for (unsigned context = 0; context != target.getNumWorkerContexts();
       ++context) {
    workerPartitions.emplace_back();

    // The number of elements minus 3 is the second element in the work list
    int numFieldElems;
    if (useLimitedVer) {
      numFieldElems = static_cast<short>(worklists[3 * context + 1]) + 3;
    } else {
      numFieldElems = static_cast<int>(worklists[3 * context + 1]) + 3;
    }
    workerPartitions.back().push_back(numFieldElems);
  }
  bool floatWeights = fpType == FLOAT;
  const auto weightBytesPerConvUnit =
      target.getWeightsPerConvUnit(fpType == FLOAT) *
      target.getTypeSize(fpType);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }
  return getConvPartial1x1SupervisorCycleEstimate(
      workerPartitions, numConvGroups, numInGroups, numOutGroups,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights,
      accumType == FLOAT);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialHorizontalMac)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer) {
  // Non-limited versions of MAC have same cycle counts as the limited ones.
  (void)useLimitedVer;
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(kernelSizeM1, unsigned);
  CODELET_SCALAR_VAL(transformedOutStride, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(zerosInfo, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;
  const auto kernelSize = kernelSizeM1 + 1;
  const auto outStride = transformedOutStride / outChansPerGroup + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();

  std::vector<unsigned> tZeroWorkList;
  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    tZeroWorkList.push_back((zerosInfo + numWorkerContexts - 1) /
                            numWorkerContexts);
  }

  bool floatActivations = fpType == FLOAT;
  bool floatPartials = accumType == FLOAT;
  uint64_t zeroCycles = getZeroSupervisorVertexCycleEstimate(
      tZeroWorkList, numOutGroups * numConvGroups, dataPathWidth,
      numWorkerContexts, floatPartials);
  if (numInGroups * inChansPerGroup == 0) {
    return zeroCycles + convHorizontalMacOverhead(floatActivations);
  }

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  assert(kernelSize > 0);
  const auto usedContexts = worklists.size() / kernelSize;
  workerPartitions.reserve(usedContexts);
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    workerPartitions.back().reserve(kernelSize);
    for (auto k = 0U; k != kernelSize; ++k) {
      workerPartitions.back().emplace_back();
      const auto &wl = worklists[k * usedContexts + context];
      workerPartitions.back().back().reserve(wl.size() / 3);
      for (auto wi = 0U; wi < wl.size(); wi += 3) {
        auto numFieldPos = (wl[wi + 1] + outStride - 1) / outStride;
        workerPartitions.back().back().push_back(numFieldPos);
      }
    }
  }
  return zeroCycles + getConvPartialHorizontalMacSupervisorCycleEstimate(
                          workerPartitions, numConvGroups, numInGroups,
                          numOutGroups, kernelSize, inChansPerGroup,
                          outChansPerGroup, numWorkerContexts, floatActivations,
                          floatPartials);
}

// TODO: T12902 Add cost estimates for non-limited version?
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartial1x4SLIC)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned outStride, bool, /* useShortTypes */
    unsigned numConvUnits) {
  CODELET_SCALAR_VAL(mode, unsigned char);
  CODELET_SCALAR_VAL(numSubKernelsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupGroupsM1, unsigned);
  CODELET_FIELD(in);
  CODELET_FIELD(out);
  CODELET_FIELD(weights);
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  assert(fpType == HALF);

  const auto numWorkerContexts = target.getNumWorkerContexts();

  const auto numSubKernels = numSubKernelsM1 + 1;
  const auto numConvGroupGroups = numConvGroupGroupsM1 + 1;
  assert(in.size() == numConvGroupGroups);
  assert(weights.size() == numConvGroupGroups * numSubKernels);
  assert(out.size() == numConvGroupGroups);

  const auto chansPerGroup = 1u << mode;
  const auto convGroupsPerGroup = 4u / chansPerGroup;

  std::vector<std::vector<unsigned>> workerPartitions(numWorkerContexts);
  for (unsigned context = 0; context < numWorkerContexts; ++context) {
    const auto &wl = worklists[context];
    workerPartitions[context].reserve(wl.size() / 3);
    for (unsigned wi = 0; wi < wl.size(); wi += 3) {
      workerPartitions[context].push_back(wl[wi + 2]);
    }
  }
#if !defined(NDEBUG)
  // Verify the assumption that partitions for different sub-kernels are
  // for the same amount of work, just with different offsets.
  for (unsigned subKernel = 1; subKernel < numSubKernels; ++subKernel) {
    for (unsigned context = 0; context < numWorkerContexts; ++context) {
      const auto &wl = worklists[subKernel * numWorkerContexts + context];
      for (unsigned wi = 0; wi < wl.size() / 3; ++wi) {
        assert(wl[wi * 3 + 2] == workerPartitions[context][wi]);
      }
    }
  }
#endif // !defined(NDEBUG)

  const unsigned slicWindowWidth = 4u;
  const bool floatActivations = fpType == FLOAT;
  const bool floatPartials = accumType == FLOAT;

  const auto implicitZeroingInnerCycles =
      getConvPartialSlicSupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvUnits, slicWindowWidth,
          floatActivations, floatPartials, outStride,
          /* implicitZeroing */ true);
  const auto innerCycles = getConvPartialSlicSupervisorInnerLoopCycleEstimate(
      workerPartitions, numWorkerContexts, numConvUnits, slicWindowWidth,
      floatActivations, floatPartials, outStride, /* implicitZeroing */ false);
  const auto weightLoadCycles =
      getConvPartialSlicSupervisorWeightLoadCycleEstimate(
          convGroupsPerGroup, chansPerGroup, numWorkerContexts,
          slicWindowWidth);
  const auto cycles = getConvPartialSlicSupervisorOuterLoopCycleEstimate(
      implicitZeroingInnerCycles, innerCycles, weightLoadCycles,
      numConvGroupGroups, numSubKernels, numConvUnits, slicWindowWidth,
      floatActivations, floatPartials);
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(WgdDataTransform)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    unsigned patchSizeX, unsigned patchSizeY, unsigned kernelX,
    unsigned kernelY) {
  CODELET_FIELD(dIn);

  const bool isFloat = fpType == FLOAT;
  const unsigned numInpRows = patchSizeX;
  const unsigned numInpCols = patchSizeY;

  const unsigned nPatches = dIn.size() / (numInpCols * numInpRows);

  return getWgdDataTransformCycles(nPatches * dIn[0].size(), isFloat);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdPartials)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &fpType) {
  CODELET_FIELD(dTf);
  CODELET_FIELD(wTf);
  CODELET_FIELD(partials);
  CODELET_SCALAR_VAL(numConvUnits, unsigned);
  CODELET_SCALAR_VAL(weightsPerConvUnit, unsigned);
  CODELET_SCALAR_VAL(convUnitCoeffLoadBytesPerCycle, unsigned);
  const auto numWorkers = target.getNumWorkerContexts();

  const bool isFloat = fpType == FLOAT;
  const unsigned outChanDepth = partials[0].size();
  const unsigned inpChanDepth = dTf[0].size();
  const unsigned comPencils = partials.size();
  const unsigned numInpGroups = wTf.size();

  return getWgdAccumCycles(numInpGroups, comPencils, inpChanDepth, outChanDepth,
                           numWorkers, numConvUnits, weightsPerConvUnit,
                           convUnitCoeffLoadBytesPerCycle, isFloat);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdReduce)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &fpType,
                                     unsigned patchSizeX, unsigned patchSizeY) {
  CODELET_FIELD(inPartial);
  CODELET_FIELD(outPartial);

  const bool isFloat = fpType == FLOAT;

  const unsigned numElems = outPartial.size();
  const unsigned numOutChans = outPartial[0].size();
  const unsigned numInpChans = inPartial.size() / numElems;

  return getWgdReduceCycles(numElems * numOutChans, numInpChans, isFloat);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(WgdInverseTransform)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    unsigned patchSizeX, unsigned patchSizeY, unsigned kernelX,
    unsigned kernelY) {
  CODELET_FIELD(dTf);
  CODELET_FIELD(dOut);

  const bool isFloat = fpType == FLOAT;
  const unsigned numInCols = patchSizeY;
  const unsigned numInRows = patchSizeX;

  const unsigned nGroups = dTf.size() / (numInCols * numInRows);
  const unsigned depthDim = dOut[0].size();

  return getWgdInvTransformCycles(nGroups * depthDim, isFloat);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdConvComplete)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &fpType) {
  CODELET_FIELD(dIn);

  const bool isFloat = fpType == FLOAT;
  const unsigned nGroups = dIn.size();
  const unsigned vecLen = dIn[0].size();
  return getWgdCompleteCycles(vecLen * nGroups, isFloat);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(InverseStdDeviation)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &meanType, const Type &powerType, const Type &outType,
    bool stableAlgo) {
  CODELET_FIELD(mean);
  const auto dataPathWidth = target.getDataPathWidth();
  CODELET_FIELD(power);
  CODELET_FIELD(iStdDev);
  assert(mean.size() == power.size());
  assert(mean.size() == iStdDev.size());
  uint64_t cycles = 6;
  for (unsigned i = 0; i < mean.size(); ++i) {
    assert(mean[i].size() == power[i].size());
    assert(mean[i].size() == iStdDev[i].size());
    unsigned numElem = mean[i].size();
    // always use float as we want float intermediates
    unsigned vectorWidth = dataPathWidth / 32;
    // mul, add, sub done as vectors of vectorWidth.
    // invsqrt is scalar
    unsigned cyclesPerVector = 3 + 1 * vectorWidth;
    unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;
    cycles += 4 + cyclesPerVector * numVectors;
  }
  return cycles;
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(OuterProduct)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(chansPerGroup, unsigned);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = type == FLOAT;
  const auto width = in.size();
  const auto numChans = weights.size();
#ifndef NDEBUG
  const auto numChanGroups = out.size();
  assert(numChans % numChanGroups == 0);
#endif

  return getOuterProductCycleEstimate(isFloat, width, numChans, chansPerGroup,
                                      dataPathWidth);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ReduceAdd)(
    const VertexIntrospector &vertex, const Target &target, const Type &outType,
    const Type &partialsType, bool singleInput, bool constrainPartials) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(numPartials, unsigned short);
  const auto dataPathWidth = target.getDataPathWidth();

  return getReduceCycleEstimate(out.size(), numPartials, dataPathWidth,
                                outType == FLOAT, partialsType == FLOAT,
                                singleInput, constrainPartials,
                                target.getNumWorkerContexts());
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(poplin, OuterProduct, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, OuterProduct, HALF),

      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, FLOAT, FLOAT, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, FLOAT, FLOAT, HALF,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, HALF, FLOAT, HALF,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, HALF, HALF, HALF,
                            true),

      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, FLOAT, FLOAT, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, FLOAT, FLOAT, HALF,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, HALF, FLOAT, HALF,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation, HALF, HALF, HALF,
                            false),

      CYCLE_ESTIMATOR_ENTRY(poplin, WgdConvComplete, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, WgdConvComplete, HALF),

      CYCLE_ESTIMATOR_ENTRY(poplin, WgdInverseTransform, FLOAT, 4, 4, 3, 3),
      CYCLE_ESTIMATOR_ENTRY(poplin, WgdInverseTransform, HALF, 4, 4, 3, 3),

      CYCLE_ESTIMATOR_ENTRY(poplin, WgdReduce, FLOAT, 4, 4),
      CYCLE_ESTIMATOR_ENTRY(poplin, WgdReduce, HALF, 4, 4),

      CYCLE_ESTIMATOR_ENTRY(poplin, WgdPartials, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, WgdPartials, HALF),

      CYCLE_ESTIMATOR_ENTRY(poplin, WgdDataTransform, FLOAT, 4, 4, 3, 3),
      CYCLE_ESTIMATOR_ENTRY(poplin, WgdDataTransform, HALF, 4, 4, 3, 3),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, FLOAT, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, HALF, FLOAT,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, HALF, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, FLOAT, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, HALF, FLOAT,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac, HALF, HALF,
                            false),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, true, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true,
                            false, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false,
                            false, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, false,
                            false, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            false, 8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, true, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, false, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            true, 8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, false, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, false,
                            8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, true, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, true, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, true, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, true,
                            8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false,
                            false, 4),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, true,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true, true,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, true,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false, true,
                            4),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, false, 4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, false,
                            4),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, true, 4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, true, 4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, true, 4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, true,
                            4),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true,
                            false, 16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            false, 16),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            true, 16),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, false,
                            16),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, true, 16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, true,
                            16),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, 1, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, 1, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, 2, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, 2, false,
                            8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 1, true, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 1, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 2, true, 8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 2, false,
                            8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 1, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 1, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 2, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, HALF, 2, false,
                            16),

      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF, true, false),

      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF, true, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, HALF, false, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF, false, false)};
}

} // end namespace poplin
