// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplinCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

#include <cassert>

using namespace poplar;

namespace poplin {

static unsigned getNumConvUnits(const Type &fpType, const Type &accumType,
                                const poplar::Target &target) {
  if (fpType == FLOAT) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  }
  assert(fpType == HALF);
  if (accumType == FLOAT)
    return target.getFp16InFp32OutConvUnitsPerTile();
  assert(accumType == HALF);
  return target.getFp16InFp16OutConvUnitsPerTile();
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialnx1)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad) {
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
  const auto numConvUnits = getNumConvUnits(fpType, accumType, target);

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  const auto kernelSize = kernelOuterSize * kernelInnerElements;
  assert(kernelSize > 0);
  const auto usedContexts = worklists.size() / kernelSize;

  bool floatPartials = accumType == FLOAT;

  std::vector<unsigned> tZeroWorkList;
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
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    for (auto k = 0U; k != kernelSize; ++k) {
      workerPartitions.back().emplace_back();
      const auto &wl = worklists[k * usedContexts + context];
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
  const auto convUnitInputLoadElemsPerCycle =
      target.getConvUnitInputLoadElemsPerCycle(fpType == FLOAT);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad)
    convUnitCoeffLoadBytesPerCycle /= 2;
  return zeroCycles + getConvPartialnx1SupervisorCycleEstimate(
                          workerPartitions, numConvGroups, numOutGroups,
                          numInGroups, kernelInnerElements, kernelOuterSize,
                          ampKernelHeight, inChansPerGroup, outChansPerGroup,
                          convUnitInputLoadElemsPerCycle, numConvUnits,
                          convUnitCoeffLoadBytesPerCycle, numWorkerContexts,
                          floatWeights, floatPartials);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartial1x1Out)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad) {
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
  const auto numConvUnits = getNumConvUnits(fpType, accumType, target);
  const auto convUnitInputLoadElemsPerCycle =
      target.getConvUnitInputLoadElemsPerCycle(fpType == FLOAT);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad)
    convUnitCoeffLoadBytesPerCycle /= 2;
  return getConvPartial1x1SupervisorCycleEstimate(
      workerPartitions, numConvGroups, numInGroups, numOutGroups,
      outChansPerGroup, convUnitInputLoadElemsPerCycle, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights,
      accumType == FLOAT);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialHorizontalMac)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer) {
  // TODO: T12902 Add cost estimates for non-limited version.
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
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    for (auto k = 0U; k != kernelSize; ++k) {
      workerPartitions.back().emplace_back();
      const auto &wl = worklists[k * usedContexts + context];
      for (auto wi = 0U; wi < wl.size(); wi += 3) {
        auto numFieldPos = (wl[wi + 1] + outStride - 1) / outStride;
        workerPartitions.back().back().push_back(numFieldPos);
      }
    }
  }
  return zeroCycles + getConvPartialHorizontalMacSupervisorCycleEstimate(
                          workerPartitions, numConvGroups, numInGroups,
                          numOutGroups, kernelSize, inChansPerGroup,
                          outChansPerGroup, numWorkerContexts,
                          floatActivations);
}

// TODO: T12902 Add cost estimates for non-limited version?
std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(ConvPartial1x4SLIC)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool /* useShortTypes */) {
  CODELET_SCALAR_VAL(mode, unsigned char);
  CODELET_SCALAR_VAL(numSubKernelsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupGroupsM1, unsigned);
  CODELET_FIELD(in);
  CODELET_FIELD(out);
  CODELET_FIELD(weights);
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  assert(fpType == HALF);
  assert(accumType == FLOAT);

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
      getConvPartialSlicSupervisorCycleInnerLoopEstimate(
          workerPartitions, numWorkerContexts, slicWindowWidth,
          floatActivations, floatPartials, /* implicitZeroing */ true);
  const auto innerCycles = getConvPartialSlicSupervisorCycleInnerLoopEstimate(
      workerPartitions, numWorkerContexts, slicWindowWidth, floatActivations,
      floatPartials, /* implicitZeroing */ false);
  const auto weightLoadCycles =
      getConvPartialSlicSupervisorCycleWeightLoadEstimate(
          convGroupsPerGroup, chansPerGroup, numWorkerContexts,
          slicWindowWidth);
  const auto cycles = getConvPartialSlicSupervisorCycleOuterLoopEstimate(
      implicitZeroingInnerCycles, innerCycles, weightLoadCycles,
      numConvGroupGroups, numSubKernels, slicWindowWidth, floatActivations,
      floatPartials);
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Transpose2d)(const VertexIntrospector &vertex,
                                       const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRows, unsigned);
  CODELET_SCALAR_VAL(numSrcColumns, unsigned);

  const bool isFloat = type == FLOAT;
  const auto matrices = dst.size();
  std::uint64_t cycles;

// TODO T14719: Derive this from IPUArchInfo
#define CSR_W_REPEAT_COUNT__VALUE__MASK 0x0FFF
  auto const hardwareRptCountConstraint = CSR_W_REPEAT_COUNT__VALUE__MASK + 1;

  if (isFloat) {
    if (((numSrcRows & 1) == 0) && ((numSrcColumns & 1) == 0) &&
        (numSrcColumns / 2 < hardwareRptCountConstraint) &&
        (numSrcRows * (numSrcColumns - 2) / 2 < 512)) { // Largest stride used
      // Float, fast path estimates
      cycles = 25 + matrices * (11 + (numSrcRows / 2) *
                                         (6 + 3 * (numSrcColumns / 2 - 1)));
    } else {
      // Float, slow path estimates based on numSrcRows being even
      cycles = 13 + matrices * (8 + numSrcColumns * (5 + (numSrcRows * 4) / 2));
    }
  } else {
    if (((numSrcRows & 3) == 0) && ((numSrcColumns & 3) == 0) &&
        (numSrcColumns >= 8) &&
        (numSrcColumns / 4 < hardwareRptCountConstraint) &&
        (1 + 3 * (numSrcColumns / 4) < 512)) { // Largest stride used
      // Half, fast path estimates, with >=8 input columns
      cycles = 37 + matrices * (12 + (numSrcRows / 4) *
                                         (15 + 4 * (numSrcColumns / 4 - 2)));
    } else if (((numSrcRows & 3) == 0) && (numSrcColumns == 4) &&
               (numSrcRows / 4 < hardwareRptCountConstraint) &&
               (1 + 3 * (numSrcRows / 4) < 512)) { // Largest stride used
      // Half, fast path estimates, 4x4 or Nx4 cases
      if (numSrcRows == 4)
        cycles = 32 + 15 * matrices;
      else
        cycles = 28 + matrices * (17 + (20 + 4 * (numSrcRows / 4 - 2)));
    } else {
      // Half, slow path estimates based on numSrcRows being even
      cycles = 15 + matrices * (8 + numSrcColumns * (5 + (numSrcRows * 5) / 2));
    }
  }
  return cycles;
}

// Cycle estimation for the "Transpose" worker (half, fast version)
static std::uint64_t TransposeWorkerCycles(unsigned short numSrcRowsD4,
                                           unsigned short numSrcColumnsD4,
                                           unsigned short numMatrices) {
  std::uint64_t cycles;
  if (numSrcRowsD4 == 1 && numSrcColumnsD4 == 1) {
    if (numMatrices == 1)
      cycles = 19 + 12;
    else
      cycles = 19 + 20 + (numMatrices - 2) * 4;
  } else if (numSrcColumnsD4 == 1) {
    cycles = 29 + numMatrices * (15 + (20 + 4 * (numSrcRowsD4 - 2)));
  } else {
    cycles = 31 + numMatrices *
                      (18 + numSrcRowsD4 * (12 + 4 * (numSrcColumnsD4 - 2)));
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Transpose)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRowsD4, unsigned short);
  CODELET_SCALAR_VAL(numSrcColumnsD4, unsigned short);
  CODELET_SCALAR_VAL(numTranspositionsM1, unsigned short);

  const unsigned matrices = numTranspositionsM1 + 1;

  // only half supported
  assert(type == HALF);

  return TransposeWorkerCycles(numSrcRowsD4, numSrcColumnsD4, matrices);
}

std::uint64_t MAKE_CYCLE_ESTIMATOR_NAME(TransposeSupervisor)(
    const VertexIntrospector &vertex, const Target &target, const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcRowsD4, unsigned short);
  CODELET_SCALAR_VAL(numSrcColumnsD4, unsigned short);
  CODELET_SCALAR_VAL(numTranspositions, unsigned short);

  // only half type supported
  assert(type == HALF);

  // This supervisor vertex will start 6 workers: 'workerCount' workers will
  // do 'numTranspositions' matrices, and (6-workerCount) will do
  // one less matrices (numTranspositions-1). We compute the cycles for
  // the slowest ones (transposing 'numTranspositions' matrices).
  // We also add the additional cycles executed, compared to the 'plain'
  // "Transpose" codelet.
  std::uint64_t maxCycles =
      TransposeWorkerCycles(numSrcRowsD4, numSrcColumnsD4, numTranspositions) +
      12 - 2;

  // Add 7 for the supervisor code
  return 7 + 6 * maxCycles;
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAdd)(const VertexIntrospector &vertex,
                                     const Target &target, const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  return getReduceCycleEstimate(out.size(), partials.size(), dataPathWidth,
                                outType == FLOAT, partialsType == FLOAT,
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

      CYCLE_ESTIMATOR_ENTRY(poplin, Transpose2d, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, Transpose2d, HALF),

      CYCLE_ESTIMATOR_ENTRY(poplin, Transpose, HALF),
      CYCLE_ESTIMATOR_ENTRY(poplin, TransposeSupervisor, HALF),

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

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, false,
                            false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            false),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, true,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, false,
                            true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, FLOAT, false,
                            true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, false),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, false),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x4SLIC, HALF, FLOAT, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, FLOAT),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, HALF),
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF)};
};

} // end namespace poplin
