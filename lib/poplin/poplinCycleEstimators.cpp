// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplinCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/FlopEstimation.hpp"
#include "poplibs_support/forceInterleavedEstimates.hpp"

#include <cassert>

using namespace poplar;
using namespace poplibs_support;

namespace poplin {

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(TriangularInverse)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &floatType, bool lower) {
  CODELET_SCALAR_VAL(dim, unsigned);

  std::uint64_t n_i_loop = dim;
  std::uint64_t n_j_loop = dim * (dim - 1) / 2;
  std::uint64_t n_k_loop = (dim + 1) * dim * (dim - 1) / 6;

  std::uint64_t flops = n_i_loop * flopsForDiv() +
                        n_j_loop * flopsForMultiply() +
                        n_k_loop * flopsForMAC();
  std::uint64_t cycles = 28 + n_i_loop * 63 + n_j_loop * 17 + n_k_loop * 6;

  return {cycles, convertToTypeFlops(flops, floatType)};
}

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(Cholesky)(const VertexIntrospector &vertex,
                                   const Target &target, const Type &floatType,
                                   bool lower) {
  CODELET_SCALAR_VAL(dim, unsigned);

  std::uint64_t n_i_loop = dim;
  std::uint64_t n_k_loop = (dim + 1) * dim / 2;
  std::uint64_t n_j_loop = (dim + 1) * dim * (dim - 1) / 6;

  std::uint64_t flops = n_i_loop * flopsForSqrt() +
                        n_k_loop * (flopsForAdd() + flopsForDiv()) +
                        n_j_loop * flopsForMAC();
  std::uint64_t cycles = 1 + n_i_loop * 11 + n_k_loop * 18 + n_j_loop * 5;

  return {cycles, convertToTypeFlops(flops, floatType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartialnx1)(
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
  bool floatWeights = fpType == FLOAT;
  const auto weightBytesPerConvUnit =
      target.getWeightsPerConvUnit(fpType == FLOAT) *
      target.getTypeSize(fpType);
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
                       floatWeights, floatPartials);
  return {cycles, convertToTypeFlops(flops, fpType)};
}

VertexPerfEstimate MAKE_PERF_ESTIMATOR_NAME(ConvPartial1x1Out)(
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
  bool floatWeights = fpType == FLOAT;
  const auto weightBytesPerConvUnit =
      target.getWeightsPerConvUnit(fpType == FLOAT) *
      target.getTypeSize(fpType);
  auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  if (!use128BitConvUnitLoad) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }

  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroups) *
                        numInGroups * numOutGroups * outChansPerGroup *
                        target.getWeightsPerConvUnit(fpType == FLOAT) *
                        totalFieldPos * flopsForMAC();

  std::uint64_t cycles = getConvPartial1x1SupervisorCycleEstimate(
      workerPartitions, numConvGroups, numInGroups, numOutGroups,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights,
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
                       outChansPerGroup, numWorkerContexts, floatActivations,
                       floatPartials);
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
    const VertexIntrospector &vertex, const Target &target, const Type &fpType,
    const Type &accumType, unsigned outStride, bool, /* useShortTypes */
    unsigned windowWidth, unsigned numConvChains) {
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

  const bool floatActivations = fpType == FLOAT;
  const bool floatPartials = accumType == FLOAT;

  const auto implicitZeroingInnerCycles =
      getConvPartialSlicSupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvChains, windowWidth,
          floatActivations, floatPartials, outStride,
          /* implicitZeroing */ true);
  const auto innerCycles = getConvPartialSlicSupervisorInnerLoopCycleEstimate(
      workerPartitions, numWorkerContexts, numConvChains, windowWidth,
      floatActivations, floatPartials, outStride, /* implicitZeroing */ false);
  const auto weightLoadCycles =
      getConvPartialSlicSupervisorWeightLoadCycleEstimate(
          convGroupsPerGroup, chansPerGroup, numWorkerContexts, windowWidth);
  const auto cycles = getConvPartialSlicSupervisorOuterLoopCycleEstimate(
      implicitZeroingInnerCycles, innerCycles, weightLoadCycles,
      numConvGroupGroups, numSubKernels, numConvChains, windowWidth,
      floatActivations, floatPartials);

  // total field elements are for a single sub-kernel element
  std::uint64_t flops = static_cast<std::uint64_t>(numConvGroupGroups) *
                        numSubKernels * windowWidth * convGroupsPerGroup *
                        chansPerGroup * chansPerGroup * totalFieldElems *
                        flopsForMAC();
  return {cycles, convertToTypeFlops(flops, fpType)};
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
  auto cycles = getOuterProductCycleEstimate(isFloat, width, numChans,
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

VertexPerfEstimate
MAKE_PERF_ESTIMATOR_NAME(TriangularSolve)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type, bool lower) {
  CODELET_FIELD(a);
  CODELET_FIELD(b);
  CODELET_FIELD(x);
  CODELET_SCALAR_VAL(an, unsigned short);

  std::uint64_t cycles =
      0x22 /*prologue*/ + 4 * an /* before dot product */ +
      (8 - 2) * (an - 1) * an /
          2 /*dot loop 1..an, minus 2 co-issued instruction*/
      + (26 - 1) * an /*outer loop epilogue, minus 1 co-issued instruction*/ +
      7 /*epilogue*/
      ;
  std::uint64_t flops =
      static_cast<std::uint64_t>(an * (an - 1) / 2) * flopsForMultiply() +
      an * flopsForAdd();

  return {cycles, convertToTypeFlops(flops, type)};
}

poplibs::PerfEstimatorTable makePerfFunctionTable() {
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
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, true,
                            8),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialVerticalMac, HALF, FLOAT, false,
                            8),

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

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, true, 4,
                            2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 1, false,
                            4, 2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, true, 4,
                            2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, FLOAT, 2, false,
                            4, 2),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            2),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            2),

      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, true, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 1, false, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, true, 4,
                            4),
      CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1xNSLIC, HALF, HALF, 2, false, 4,
                            4),

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
  };
}

} // end namespace poplin
