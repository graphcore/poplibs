// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplinCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/FlopEstimation.hpp"
#include "poplibs_support/forceInterleavedEstimates.hpp"

#include <cassert>

#define DIV_UP(a, b) (a + b - 1) / b

using namespace poplar;
using namespace poplibs_support;

namespace poplin {

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(TriangularInverse)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &type, bool lower) {
  CODELET_SCALAR_VAL(dim, unsigned);

  std::uint64_t iLoop = dim;
  std::uint64_t jLoop = dim * (dim - 1) / 2;
  std::uint64_t kLoop = (dim + 1) * dim * (dim - 1) / 6;

  std::uint64_t flops = iLoop * flopsForDiv() + jLoop * flopsForMultiply() +
                        kLoop * flopsForMAC();

  bool half = type == poplar::HALF;
  std::uint64_t cycles;
  if (half) {
    cycles = lower ? 7 * dim * dim * dim + 387 * dim * dim / 2 + 272 * dim - 120
                   : 7 * dim * dim * dim + 123 * dim * dim + 269 * dim - 24;
  } else {
    cycles = lower ? 6 * dim * dim * dim + 264 * dim * dim + 78 * dim - 120
                   : 6 * dim * dim * dim + 51 * dim * dim + 213 * dim + 12;
  }

  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cholesky)(const VertexIntrospector &vertex,
                                   const Target &target, const Type &type,
                                   bool lower) {
  CODELET_SCALAR_VAL(dim, unsigned);

  std::uint64_t iLoop = dim;
  std::uint64_t kLoop = (dim + 1) * dim / 2;
  std::uint64_t jLoop = (dim + 1) * dim * (dim - 1) / 6;

  std::uint64_t flops = iLoop * flopsForSqrt() +
                        kLoop * (flopsForAdd() + flopsForDiv()) +
                        jLoop * flopsForMAC();

  bool half = type == poplar::HALF;
  std::uint64_t cycles;
  if (half) {
    cycles = dim * dim * dim / 2 + 795 * dim * dim / 4 + 191 * dim / 2 + 426;
  } else {
    cycles = dim * dim * dim + 102 * dim * dim + 101 * dim + 102;
  }
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartialnx1)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad,
    unsigned numConvUnits, unsigned convInputLoadElems,
    bool /* disableSRForAMPVertices*/) {
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
  unsigned totalFieldPos = 0;
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
        totalFieldPos += numFieldPos;
        workerPartitions.back().back().push_back(numFieldPos);
      }
    }
  }
  const auto convChainLength = target.getConvUnitMaxPipelineDepth(accumType);
  const auto weightsPerConvUnit = convChainLength * convInputLoadElems;
  const auto weightBytesPerConvUnit =
      weightsPerConvUnit * target.getTypeSize(fpType);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }
  std::uint64_t flops = static_cast<uint64_t>(numConvGroups) * numOutGroups *
                        numInGroups * inChansPerGroup * outChansPerGroup *
                        totalFieldPos * ampKernelHeight * flopsForMAC();

  std::uint64_t cycles =
      zeroCycles + getConvPartialnx1SupervisorCycleEstimate(
                       workerPartitions, numConvGroups, numOutGroups,
                       numInGroups, kernelInnerElements, kernelOuterSize,
                       ampKernelHeight, inChansPerGroup, outChansPerGroup,
                       weightBytesPerConvUnit, numConvUnits,
                       convUnitCoeffLoadBytesPerCycle, numWorkerContexts,
                       fpType, floatPartials);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartial1x1Out)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, bool use128BitConvUnitLoad,
    unsigned numConvUnits, unsigned convInputLoadElems,
    bool /*disableSRForAMPVertices*/) {
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
  unsigned totalFieldPos = 0;
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
    totalFieldPos += numFieldElems;
    workerPartitions.back().push_back(numFieldElems);
  }
  const auto convChainLength = target.getConvUnitMaxPipelineDepth(accumType);
  const auto weightsPerConvUnit = convChainLength * convInputLoadElems;
  const auto weightBytesPerConvUnit =
      weightsPerConvUnit * target.getTypeSize(fpType);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }

  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroups) *
                        numInGroups * numOutGroups * outChansPerGroup *
                        weightsPerConvUnit * totalFieldPos * flopsForMAC();

  std::uint64_t cycles = getConvPartial1x1SupervisorCycleEstimate(
      workerPartitions, numConvGroups, numInGroups, numOutGroups,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, fpType,
      accumType == FLOAT);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartialHorizontalMac)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer) {
  // Non-limited versions of HMAC have same cycle counts as the limited ones.
  (void)useLimitedVer;
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(kernelSizeM1, unsigned);
  CODELET_SCALAR_VAL(transformedOutStride, int);
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
  const auto actsVectorWidth = target.getVectorWidth(fpType);

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
  unsigned totalFieldPos = 0;
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
        totalFieldPos += numFieldPos;
      }
    }
  }
  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroups) *
                        numInGroups * numOutGroups * inChansPerGroup *
                        outChansPerGroup * totalFieldPos * flopsForMAC();
  std::uint64_t cycles =
      zeroCycles + getConvPartialHorizontalMacSupervisorCycleEstimate(
                       workerPartitions, numConvGroups, numInGroups,
                       numOutGroups, kernelSize, inChansPerGroup,
                       outChansPerGroup, numWorkerContexts, actsVectorWidth,
                       floatActivations, floatPartials);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartialVerticalMac)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, bool useLimitedVer, unsigned convGroupsPerGroup) {
  (void)useLimitedVer;
  CODELET_VECTOR_2D_VALS(worklists, unsigned short);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(zerosInfo, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  const unsigned inChansPerGroup = 1;
  const unsigned outChansPerGroup = 1;
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = 1;

  assert(weights.size() == numInGroups * numConvGroups);
  assert(out.size() == numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);

  const auto numWorkerContexts = target.getNumWorkerContexts();
  bool floatActivations = fpType == FLOAT;
  bool floatPartials = accumType == FLOAT;
  std::vector<std::vector<unsigned>> workerPartitions;
  auto worklistSizeMax = std::max_element(worklists.begin(), worklists.end(),
                                          [](auto const &lhs, auto const &rhs) {
                                            return lhs.size() < rhs.size();
                                          })
                             ->size();
  assert(worklistSizeMax > 0);

  const auto usedContexts = worklists.size();
  workerPartitions.reserve(usedContexts);
  unsigned totalFieldPos = 0;
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    const auto &wl = worklists[context];
    workerPartitions.back().reserve(wl.size() / 3);
    for (auto wi = 0U; wi < wl.size(); wi += 4) {
      auto numFieldPos = wl[wi + 3];
      workerPartitions.back().push_back(numFieldPos);
      totalFieldPos += numFieldPos + 1;
    }
  }
  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroups) *
                            numInGroups * convGroupsPerGroup * totalFieldPos *
                            flopsForMAC() +
                        // reduction across workers
                        zerosInfo * (numWorkerContexts - 1) * flopsForAdd();
  std::uint64_t cycles = getConvPartialVerticalMacSupervisorCycleEstimate(
      workerPartitions, numConvGroups, numInGroups, numOutGroups,
      worklistSizeMax, zerosInfo, inChansPerGroup, outChansPerGroup,
      convGroupsPerGroup, numWorkerContexts, floatActivations, floatPartials);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

// TODO: T12902 Add cost estimates for non-limited version?
VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartial1xNSLIC)(
    const VertexIntrospector &vertex, const Target &target,
    const Type &actsType, const Type &accumType, unsigned outStride,
    bool, /* useShortTypes */
    unsigned windowWidth, unsigned numConvChains,
    unsigned convGroupsPerGroupVertexType, bool /* disableSRForAMPVertices*/) {
  CODELET_SCALAR_VAL(chansPerGroupLog2, unsigned char);
  CODELET_SCALAR_VAL(numSubKernelsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupGroupsM1, unsigned);
  CODELET_FIELD(in);
  CODELET_FIELD(out);
  CODELET_FIELD(weights);
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  assert(actsType == HALF || actsType == QUARTER);

  const auto numWorkerContexts = target.getNumWorkerContexts();

  const auto numSubKernels = numSubKernelsM1 + 1;
  const auto numConvGroupGroups = numConvGroupGroupsM1 + 1;
  assert(in.size() == numConvGroupGroups);
  assert(weights.size() == numConvGroupGroups * numSubKernels);
  assert(out.size() == numConvGroupGroups);

  const unsigned chansPerGroup = 1u << chansPerGroupLog2;
  const unsigned convGroupsPerGroup =
      convGroupsPerGroupVertexType >> chansPerGroupLog2;

  std::vector<std::vector<unsigned>> workerPartitions(numWorkerContexts);
  unsigned totalFieldElems = 0;
  for (unsigned context = 0; context < numWorkerContexts; ++context) {
    const auto &wl = worklists[context];
    workerPartitions[context].reserve(wl.size() / 3);
    for (unsigned wi = 0; wi < wl.size(); wi += 3) {
      const auto fieldElems = wl[wi + 2];
      workerPartitions[context].push_back(fieldElems);
      totalFieldElems += fieldElems;
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

  const bool floatPartials = accumType == FLOAT;

  const auto implicitZeroingInnerCycles =
      getConvPartialSlicSupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvChains, windowWidth,
          convGroupsPerGroup, actsType, floatPartials, outStride,
          /* implicitZeroing */ true);
  const auto innerCycles = getConvPartialSlicSupervisorInnerLoopCycleEstimate(
      workerPartitions, numWorkerContexts, numConvChains, windowWidth,
      convGroupsPerGroup, actsType, floatPartials, outStride,
      /* implicitZeroing */ false);
  const auto weightLoadCycles =
      getConvPartialSlicSupervisorWeightLoadCycleEstimate(
          convGroupsPerGroup, chansPerGroup, numWorkerContexts, windowWidth,
          actsType);
  const auto cycles = getConvPartialSlicSupervisorOuterLoopCycleEstimate(
      implicitZeroingInnerCycles, innerCycles, weightLoadCycles,
      numConvGroupGroups, numSubKernels, numConvChains, windowWidth,
      convGroupsPerGroup, actsType, floatPartials, numWorkerContexts);

  // total field elements are for a single sub-kernel element
  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroupGroups) *
                        numSubKernels * windowWidth * convGroupsPerGroup *
                        chansPerGroup * chansPerGroup * totalFieldElems *
                        flopsForMAC();
  return {cycles, convertToTypeFlops(flops, actsType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(WgdDataTransform)(
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    unsigned patchSizeX, unsigned patchSizeY, unsigned kernelX,
    unsigned kernelY) {
  CODELET_FIELD(dIn);

  const bool isFloat = fpType == FLOAT;
  const unsigned numInpRows = patchSizeX;
  const unsigned numInpCols = patchSizeY;

  const unsigned nPatches = dIn.size() / (numInpCols * numInpRows);
  std::uint64_t flops = (static_cast<std::uint64_t>(nPatches) * dIn[0].size() *
                         4 * (patchSizeX + patchSizeY)) *
                        flopsForAdd();
  std::uint64_t cycles =
      getWgdDataTransformCycles(nPatches * dIn[0].size(), isFloat);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(WgdPartials)(const VertexIntrospector &vertex,
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
  std::uint64_t flops = static_cast<std::uint64_t>(numInpGroups) * comPencils *
                        outChanDepth * inpChanDepth * flopsForMAC();

  std::uint64_t cycles =
      getWgdAccumCycles(numInpGroups, comPencils, inpChanDepth, outChanDepth,
                        numWorkers, numConvUnits, weightsPerConvUnit,
                        convUnitCoeffLoadBytesPerCycle, isFloat);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(WgdReduce)(const VertexIntrospector &vertex,
                                    const Target &target, const Type &fpType,
                                    unsigned patchSizeX, unsigned patchSizeY) {
  CODELET_FIELD(inPartial);
  CODELET_FIELD(outPartial);

  const bool isFloat = fpType == FLOAT;

  const unsigned numElems = outPartial.size();
  const unsigned numOutChans = outPartial[0].size();
  const unsigned numInpChans = inPartial.size() / numElems;
  std::uint64_t flops = numElems * numOutChans * numInpChans;
  std::uint64_t cycles =
      getWgdReduceCycles(numElems * numOutChans, numInpChans, isFloat) *
      flopsForAdd();
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(WgdInverseTransform)(
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
  std::uint64_t flops =
      static_cast<std::uint64_t>(nGroups) * depthDim * 24 * flopsForAdd();
  std::uint64_t cycles = getWgdInvTransformCycles(nGroups * depthDim, isFloat);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(WgdConvComplete)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &fpType) {
  CODELET_FIELD(dIn);

  const bool isFloat = fpType == FLOAT;
  const unsigned nGroups = dIn.size();
  const unsigned vecLen = dIn[0].size();
  std::uint64_t cycles = getWgdCompleteCycles(vecLen * nGroups, isFloat);
  return {cycles, 0UL};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(InverseStdDeviation)(
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
  std::uint64_t flops = 0;
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
    // addition by eps, scale by scaleFactor, division and inverse sqrt
    flops += 4 * mean[i].size();
  }
  return {cycles, convertToTypeFlops(flops, outType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(OuterProduct)(const VertexIntrospector &vertex,
                                       const Target &target, const Type &type) {
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
  std::uint64_t flops = static_cast<std::uint64_t>(out.size()) * chansPerGroup *
                        width * flopsForMultiply();
  auto cycles = getOuterProductCycleEstimate(target, isFloat, width, numChans,
                                             chansPerGroup, dataPathWidth);
  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ReduceAdd)(
    const VertexIntrospector &vertex, const Target &target, const Type &outType,
    const Type &partialsType, bool singleInput, bool constrainPartials) {
  CODELET_FIELD(out);
  CODELET_SCALAR_VAL(numPartials, unsigned short);
  CODELET_SCALAR_VAL(numElems, unsigned short);
  const auto dataPathWidth = target.getDataPathWidth();

  std::uint64_t flops =
      static_cast<std::uint64_t>(numElems) * (numPartials - 1) * flopsForAdd();

  if (getForceInterleavedEstimates()) {
    constrainPartials = true;
  }

  auto cycles = getReduceCycleEstimate(out.size(), numPartials, dataPathWidth,
                                       outType == FLOAT, partialsType == FLOAT,
                                       singleInput, constrainPartials,
                                       target.getNumWorkerContexts());
  return {cycles, convertToTypeFlops(flops, partialsType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(PartialSquareElements)(
    const VertexIntrospector &vertex, const Target &target) {
  CODELET_FIELD(rowToProcess);
  CODELET_SCALAR_VAL(offset, unsigned);
  CODELET_FIELD(padding);

  constexpr unsigned nWorkers = 6;
  const auto nelems = rowToProcess.size();
  const unsigned padd = 16; // avg of default padding
  const unsigned skipped =
      std::max(std::min((int)(padd - offset), (int)nelems), 0);
  const auto toCompute = nelems - skipped;
  const auto skippedPerWorker = DIV_UP(skipped, nWorkers);
  const auto toComputePerWorker = DIV_UP(toCompute, nWorkers);

  std::uint64_t cycles =
      nWorkers * (34 + 2 * skippedPerWorker + 6 + toComputePerWorker * 5 + 1);
  std::uint64_t flops =
      static_cast<std::uint64_t>(toCompute) * flopsForMultiply();

  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Householder)(const VertexIntrospector &vertex,
                                      const Target &target) {
  CODELET_FIELD(v);
  CODELET_FIELD(dotProduct);
  CODELET_SCALAR_VAL(offset, unsigned);
  CODELET_FIELD(padding);
  CODELET_FIELD(diagonalValue);

  constexpr unsigned nWorkers = 6;
  const auto nelems = v.size();
  const unsigned padd = 16; // avg of default padding
  const unsigned skipped = std::max((int)(padd - offset), 0);
  const auto toCompute = nelems - skipped;
  const auto skippedPerWorker = DIV_UP(skipped, nWorkers);
  const auto toComputePerWorker = DIV_UP(toCompute, nWorkers);

  std::uint64_t cycles =
      nWorkers * (37 + 4 * skippedPerWorker + 9 + toComputePerWorker * 5 + 1);
  std::uint64_t flops = static_cast<std::uint64_t>(toCompute) * flopsForDiv() +
                        3 * flopsForSqrt();

  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Update)(const VertexIntrospector &vertex,
                                 const Target &target) {
  CODELET_FIELD(v);
  CODELET_FIELD(AQRows);
  CODELET_FIELD(padding);

  constexpr unsigned nWorkers = 6;
  const auto nelems = v.size();
  constexpr unsigned padd = 16; // avg of default padding
  const auto toCompute = nelems - padd;
  const auto toComputePerWorker = DIV_UP(nelems - padd, nWorkers);
  const auto rowsPerWorker = DIV_UP(AQRows.size(), nWorkers);

  std::uint64_t cycles = 72 +
                         nWorkers *
                             ((41 + 3 * toComputePerWorker + 3) +
                              (29 + 4 * toComputePerWorker + 3)) *
                             rowsPerWorker +
                         8;
  std::uint64_t flops = static_cast<std::uint64_t>(toCompute) * 2 *
                        (flopsForDiv() + flopsForAdd());

  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(RowCopy)(const VertexIntrospector &vertex,
                                  const Target &target) {
  CODELET_FIELD(copiedRow);
  CODELET_FIELD(diagonalValueVector);
  CODELET_FIELD(A);
  CODELET_FIELD(padding);

  constexpr unsigned nWorkers = 6;
  const auto nelemsPerWorker = DIV_UP(A.size(), nWorkers);

  std::uint64_t cycles = nWorkers * (16 + 7 * nelemsPerWorker + 16);
  std::uint64_t flops = 0;

  return {cycles, flops};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(TriangularSolve)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type, bool lower) {
  CODELET_FIELD(a);
  CODELET_FIELD(b);
  CODELET_FIELD(x);
  CODELET_SCALAR_VAL(an, unsigned short);

  bool half = type == poplar::HALF;
  std::uint64_t cycles;
  if (half) {
    cycles = lower ? 3 * an * an / 4 + 318 * an - 150
                   : 135 * an * an / 16 + 1089 * an / 4 + 12;
  } else {
    cycles = lower ? 3 * an * an / 2 + 180 * an - 126
                   : 27 * an * an / 4 + 327 * an / 2 + 54;
  }
  std::uint64_t flops =
      static_cast<std::uint64_t>(an * (an - 1) / 2) * flopsForMultiply() +
      an * flopsForAdd();

  return {cycles, convertToTypeFlops(flops, type)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(TriangularSolveMultiWorker)(
    const VertexIntrospector &vertex, const Target &target, const Type &type,
    bool lower) {
  CODELET_FIELD(a);
  CODELET_FIELD(b);
  CODELET_FIELD(x);
  CODELET_SCALAR_VAL(an, unsigned short);

  const auto numWorkers = target.getNumWorkerContexts();
  const bool half = type == poplar::HALF;

  const unsigned supervisorOutsideLoop = lower ? 21 : 42;
  const unsigned supervisorSingleIter = lower ? 9 : 6;
  constexpr unsigned columnsPerWorker = 4;
  const unsigned supervisorNumIters = an / columnsPerWorker;

  const unsigned numWorkerCalls = supervisorNumIters;
  unsigned workerLoopSingleIter, workerOutsideLoop;
  if (lower) {
    workerLoopSingleIter = half ? 17 : 11;
    workerOutsideLoop = half ? 112 : 74;
  } else {
    workerLoopSingleIter = half ? 17 : 10;
    workerOutsideLoop = half ? 145 : 99;
  }

  const unsigned numElemsPerIter = half ? 2 : 1;
  constexpr unsigned workerElemsBeforeLoop = 4;
  const double workerAvgIters = static_cast<float>(an - workerElemsBeforeLoop) /
                                (numWorkers * numElemsPerIter) / 2;
  const double workerAvgLoopCycles = workerLoopSingleIter * workerAvgIters;

  const std::uint64_t workersCycles =
      (workerAvgLoopCycles + workerOutsideLoop) * numWorkerCalls * numWorkers;
  const std::uint64_t cycles = workersCycles +
                               supervisorSingleIter * supervisorNumIters +
                               supervisorOutsideLoop;

  std::uint64_t flops =
      static_cast<std::uint64_t>(an * (an - 1) / 2) * flopsForMultiply() +
      an * flopsForAdd();

  return {cycles, convertToTypeFlops(flops, type)};
}

poputil::internal::PerfEstimatorTable makePerfFunctionTable() {
  return {
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularInverse, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularInverse, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularInverse, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularInverse, HALF, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, Cholesky, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, Cholesky, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, Cholesky, HALF, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, Cholesky, HALF, true),

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

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, true,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, false,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, true,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, false,
                            4),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, false,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, false,
                            8),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, false,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, true,
                            16),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, HALF, false,
                            16),

#define ENTRY_DISABLE_SR(name, inType, parType, limited, weights128,           \
                         convUnits, inputElems)                                \
  CYCLE_ESTIMATOR_ENTRY(poplin, name, inType, parType, limited, weights128,    \
                        convUnits, inputElems, true),                          \
      CYCLE_ESTIMATOR_ENTRY(poplin, name, inType, parType, limited,            \
                            weights128, convUnits, inputElems, false)

#define ENTRY_LIMITED(name, inType, parType, weights128, convUnits,            \
                      inputElems)                                              \
  ENTRY_DISABLE_SR(name, inType, parType, true, weights128, convUnits,         \
                   inputElems),                                                \
      ENTRY_DISABLE_SR(name, inType, parType, false, weights128, convUnits,    \
                       inputElems)

#define ENTRY_WEIGHTS_128(name, inType, parType, convUnits, inputElems)        \
  ENTRY_LIMITED(name, inType, parType, true, convUnits, inputElems),           \
      ENTRY_LIMITED(name, inType, parType, false, convUnits, inputElems)

      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, HALF, 8, 4),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, FLOAT, 8, 4),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, FLOAT, FLOAT, 8, 2),

      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, HALF, 16, 4),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, FLOAT, FLOAT, 16, 2),

      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, HALF, 8, 8),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, FLOAT, 8, 8),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, FLOAT, FLOAT, 8, 4),

      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, HALF, 16, 8),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, HALF, FLOAT, 16, 8),
      ENTRY_WEIGHTS_128(ConvPartial1x1Out, FLOAT, FLOAT, 16, 4),

      ENTRY_WEIGHTS_128(ConvPartial1x1Out, QUARTER, HALF, 16, 8),

      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, HALF, 8, 4),
      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, FLOAT, 8, 4),
      ENTRY_WEIGHTS_128(ConvPartialnx1, FLOAT, FLOAT, 8, 2),

      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, HALF, 16, 4),
      ENTRY_WEIGHTS_128(ConvPartialnx1, FLOAT, FLOAT, 16, 2),

      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, HALF, 8, 8),
      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, FLOAT, 8, 8),
      ENTRY_WEIGHTS_128(ConvPartialnx1, FLOAT, FLOAT, 8, 4),

      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, HALF, 16, 8),
      ENTRY_WEIGHTS_128(ConvPartialnx1, HALF, FLOAT, 16, 8),
      ENTRY_WEIGHTS_128(ConvPartialnx1, FLOAT, FLOAT, 16, 4),

      ENTRY_WEIGHTS_128(ConvPartialnx1, QUARTER, HALF, 16, 8),

#undef ENTRY_WEIGHTS_128
#undef ENTRY_LIMITED
#undef ENTRY_DISABLE_SR

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, true, 4,
                            2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, false,
                            4, 2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, true, 4,
                            2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, false,
                            4, 2, 4, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            2, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            2, 4, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 4, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 4, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 8, true),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 16, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 16, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 16, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 16, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, true, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, false,
                            4, 2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, true, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, false,
                            4, 2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            2, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 4, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4, 16, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4, 16, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4, 16, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4, 16, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 1, true,
                            4, 4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 1, false,
                            4, 4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 2, true,
                            4, 4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 2, false,
                            4, 4, 8, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 1, true,
                            4, 4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 1, false,
                            4, 4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 2, true,
                            4, 4, 8, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, QUARTER, HALF, 2, false,
                            4, 4, 8, false),

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
      CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF, false, false),

      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolve, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolve, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolve, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolve, HALF, false),

      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolveMultiWorker, FLOAT, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolveMultiWorker, FLOAT, false),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolveMultiWorker, HALF, true),
      CYCLE_ESTIMATOR_ENTRY(poplin, TriangularSolveMultiWorker, HALF, false),

      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poplin::experimental,
                                     PartialSquareElements),
      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poplin::experimental, Householder),
      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poplin::experimental, Update),
      CYCLE_ESTIMATOR_ENTRY_NOPARAMS(poplin::experimental, RowCopy),

  };
}

} // end namespace poplin
