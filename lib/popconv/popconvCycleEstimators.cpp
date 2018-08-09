#include "popconvCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popconv {

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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialnx1)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &fpType,
                                          const Type &accumType,
                                          bool useLimitedVer) {
  // TODO: cost fo non-limited version not estimated
  (void) useLimitedVer;
  CODELET_SCALAR_VAL(kernelOuterSizeM1, unsigned);
  CODELET_SCALAR_VAL(kernelInnerElementsM1, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroupsM1, unsigned);
  CODELET_SCALAR_VAL(ampKernelHeightM1, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);

  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_VECTOR_VALS(zeroWorklist, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(weights);
  const auto kernelOuterSize = kernelOuterSizeM1 + 1;
  const auto kernelInnerElements = kernelInnerElementsM1 + 1;
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;
  const auto numInGroups = numInGroupsM1 + 1;
  const auto ampKernelHeight = ampKernelHeightM1 + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(zeroWorklist.size() % 2 == 0);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto numConvUnits = getNumConvUnits(fpType, accumType, target);

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  const auto kernelSize = kernelOuterSize * kernelInnerElements;
  assert(kernelSize > 0);
  const auto usedContexts = worklists.size() / kernelSize;

  std::vector<unsigned> tZeroWorkList;
  for (unsigned i = 0; i != zeroWorklist.size() / 2; ++i) {
    tZeroWorkList.push_back(zeroWorklist[2 * i + 1]);
  }
  bool floatPartials = accumType == FLOAT;
  uint64_t zeroCycles =
    getZeroSupervisorVertexCycleEstimate(tZeroWorkList,
                                         numOutGroups * numConvGroups,
                                         dataPathWidth,
                                         numWorkerContexts,
                                         floatPartials);
  for (unsigned context = 0; context < usedContexts; ++context) {
    workerPartitions.emplace_back();
    for (auto k = 0U; k != kernelSize; ++k) {
      workerPartitions.back().emplace_back();
      const auto &wl = worklists[k * usedContexts + context];
      for (auto wi = 0U; wi < wl.size(); wi += 3) {
        auto numFieldPos = wl[wi + 1];
        workerPartitions.back().back().push_back(numFieldPos);
      }
    }
  }
  bool floatWeights = fpType == FLOAT;
  const auto convUnitInputLoadElemsPerCycle =
    target.getConvUnitInputLoadElemsPerCycle(fpType == FLOAT);
  const auto convUnitCoeffLoadBytesPerCycle =
      target.getConvUnitCoeffLoadBytesPerCycle();
  return zeroCycles +
    getConvPartialnx1SupervisorCycleEstimate(workerPartitions,
                                             numConvGroups,
                                             numOutGroups,
                                             numInGroups,
                                             kernelInnerElements,
                                             kernelOuterSize,
                                             ampKernelHeight,
                                             inChansPerGroup,
                                             outChansPerGroup,
                                             convUnitInputLoadElemsPerCycle,
                                             numConvUnits,
                                             convUnitCoeffLoadBytesPerCycle,
                                             numWorkerContexts,
                                             floatWeights);
}


std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvPartial1x1Out)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &fpType,
                                             const Type &accumType,
                                             bool useLimitedVer) {
  // TODO: cost for non-limited version not estimated
  (void) useLimitedVer;
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  const auto numWorkerContexts = target.getNumWorkerContexts();
  CODELET_FIELD(weights);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numInGroups = numInGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);
  // find max work to bt done per worker
  std::vector<std::vector<unsigned>> workerPartitions;
  const auto usedContexts = worklists.size();
  for (unsigned context = 0; context != usedContexts; ++context) {
    workerPartitions.emplace_back();
    const auto &wl = worklists[context];
    assert(wl.size() % 3 == 0);
    for (unsigned wi = 0; wi != wl.size(); wi += 3) {
      workerPartitions.back().push_back(wl[wi + 1]);
    }
  }
  bool floatWeights = fpType == FLOAT;
  const auto numConvUnits = getNumConvUnits(fpType, accumType, target);
  const auto convUnitInputLoadElemsPerCycle =
      target.getConvUnitInputLoadElemsPerCycle(fpType == FLOAT);
  const auto convUnitCoeffLoadBytesPerCycle =
                        target.getConvUnitCoeffLoadBytesPerCycle();
  return
    getConvPartial1x1SupervisorCycleEstimate(workerPartitions,
                                             numConvGroups,
                                             numInGroups,
                                             numOutGroups,
                                             outChansPerGroup,
                                             convUnitInputLoadElemsPerCycle,
                                             numConvUnits,
                                             convUnitCoeffLoadBytesPerCycle,
                                             numWorkerContexts,
                                             floatWeights);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialHorizontalMac)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &fpType,
    const Type &accumType,
    bool useLimitedVer) {
  // TODO: cost for non-limited version not estimated
  (void) useLimitedVer;
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_VECTOR_VALS(zeroWorklist, unsigned);
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numConvGroupsM1, unsigned);
  CODELET_SCALAR_VAL(kernelSizeM1, unsigned);
  CODELET_SCALAR_VAL(transformedOutStride, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  const auto numConvGroups = numConvGroupsM1 + 1;
  const auto numOutGroups = numOutGroupsM1 + 1;
  const auto numInGroups = numInGroupsM1 + 1;
  const auto kernelSize = kernelSizeM1 + 1;
  const auto outStride = transformedOutStride / outChansPerGroup + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);
  assert(zeroWorklist.size() % 2 == 0);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();

  bool floatActivations = fpType == FLOAT;
  std::vector<unsigned> tZeroWorkList;
  for (unsigned i = 0; i != zeroWorklist.size() / 2; ++i) {
    tZeroWorkList.push_back(zeroWorklist[2 * i + 1]);
  }
  bool floatPartials = accumType == FLOAT;
  uint64_t zeroCycles =
    getZeroSupervisorVertexCycleEstimate(tZeroWorkList,
                                         numOutGroups * numConvGroups,
                                         dataPathWidth,
                                         numWorkerContexts,
                                         floatPartials);

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
  return zeroCycles +
    getConvPartialHorizontalMacSupervisorCycleEstimate(
        workerPartitions,
        numConvGroups,
        numInGroups,
        numOutGroups,
        kernelSize,
        inChansPerGroup,
        outChansPerGroup,
        dataPathWidth,
        numWorkerContexts,
        floatActivations);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdDataTransform)(const VertexIntrospector &vertex,
                                            const Target &target,
                                            const Type &fpType,
                                            unsigned patchSizeX,
                                            unsigned patchSizeY,
                                            unsigned kernelX,
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
  const auto  numWorkers = target.getNumWorkerContexts();

  const bool isFloat = fpType == FLOAT;
  const unsigned outChanDepth = partials[0].size();
  const unsigned inpChanDepth = dTf[0].size();
  const unsigned comPencils = partials.size();
  const unsigned numInpGroups = wTf.size();

  return getWgdAccumCycles(
                    numInpGroups,
                    comPencils,
                    inpChanDepth,
                    outChanDepth,
                    numWorkers,
                    numConvUnits,
                    weightsPerConvUnit,
                    convUnitCoeffLoadBytesPerCycle,
                    isFloat);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdReduce)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &fpType,
                                     unsigned patchSizeX,
                                     unsigned patchSizeY) {
  CODELET_FIELD(inPartial);
  CODELET_FIELD(outPartial);

  const bool isFloat = fpType == FLOAT;

  const unsigned numElems = outPartial.size();
  const unsigned numOutChans = outPartial[0].size();
  const unsigned numInpChans = inPartial.size() / numElems;

  return getWgdReduceCycles(
                 numElems * numOutChans,
                 numInpChans,
                 isFloat
                 );
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(WgdInverseTransform)(const VertexIntrospector &vertex,
                                               const Target &target,
                                               const Type &fpType,
                                               unsigned patchSizeX,
                                               unsigned patchSizeY,
                                               unsigned kernelX,
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
  return getWgdCompleteCycles(
                             vecLen * nGroups,
                             isFloat);
}


std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(Transpose2d)(const VertexIntrospector &vertex,
                                       const Target &target,
                                       const Type &type) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcColumns, unsigned);

  const bool isFloat = type == FLOAT;
  std::uint64_t cycles = 2; // Run instruction.
  if (isFloat)
    cycles += 6;  // Vertex overhead.
  else
    cycles += 7;
  const auto numTranspositions = src.size();
  assert(src.size() == dst.size());
#ifndef NDEBUG
  for (unsigned i = 0; i != numTranspositions; ++i) {
    assert(src[i].size() == dst[i].size());
    const auto numElements = src[i].size();
    assert(numElements % numSrcColumns == 0);
  }
#endif
  for (unsigned i = 0; i != numTranspositions; ++i) {
    const auto numElements = src[i].size();
    cycles += 2; // Load src and dst pointers.
    if (isFloat) {
      cycles += 1; // 1 cycle latency before first value is written to memory.
      cycles += numElements;
    } else {
      // Cycle count based on the transpose16x16 microbenchmark which takes
      // 75 cycles per 16x16 block, reading each 4xn in turn. Any nx4 block
      // where n is 1 or even will have a similar cost until the offset
      // between rows exceeds the allowable triple addressing offset
      assert(numElements % numSrcColumns == 0);
      if (numSrcColumns % 4 == 0 && numElements % 16 == 0) {
        const auto num4x4Blocks = numElements / (4 * 4);
        cycles += 11 + num4x4Blocks * 4;
      } else {
        // Cycle count taken from transpose16x8 microbenchmark.
        const auto numSrcRows = numElements / numSrcColumns;
        const auto middleIterations = (numSrcColumns + 3) / 4;
        const auto innerIterations = (numSrcRows + 1) / 2;
        cycles += 3 + middleIterations * (3 + innerIterations * 6);
      }
    }
  }
  return cycles;
}

// Exact worker cycle count for popconv_AddToChannel__float_core
std::uint64_t addToChannelCoreCycles_float(unsigned addendLen,
                                           unsigned blockCount) {
  std::uint64_t cycles = 1; // return

  ++cycles; // brz

  if (blockCount == 0)
    return cycles;

  ++cycles; // acts_loop_count = blockCount - 1

  for (unsigned i = 0; i < addendLen; ++i) {
    cycles += 4; // start of loop
    cycles += 3 * blockCount; // rpt loop
    ++cycles; // brnzdec
  }
  return cycles;
}

std::uint64_t addToChannelCoreCycles_half_scalar(unsigned addendLen,
                                                 unsigned blockCount) {
  std::uint64_t cycles = 3; // pre-loop
  // Aligned loop bodies take 7 cycles, misaligned take 9, but they are
  // equally numerous so it averages to 8.
  cycles += addendLen * (4 + blockCount * 8);
  return cycles;
}
std::uint64_t addToChannelCoreCycles_half_multiple_of_8(unsigned addendLen,
                                                        unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop
  cycles += (addendLen/8) * (
    8 + // pre-rpt
    2 * (blockCount - 1) + // rpt body
    // post-rpt
    4
  );
  return cycles;
}
std::uint64_t addToChannelCoreCycles_half_multiple_of_4(unsigned addendLen,
                                                        unsigned blockCount) {
  std::uint64_t cycles = 6; // pre-loop
  cycles += (addendLen/4) * (
    7 + // pre-rpt
    2 * (blockCount/2 - 1) + // rpt body
    // post-rpt. The code actually depends on whether or not blockCount
    // was odd but it takes the same number of cycles in both cases.
    6
  );
  return cycles;
}

// Exact worker cycle count for popconv_AddToChannel__half_core
std::uint64_t addToChannelCoreCycles_half(unsigned addendLen,
                                          unsigned blockCount) {
  std::uint64_t cycles = 1; // return

  cycles += 2; // cmpult > 2048, brz
  if (addendLen > 2048) {
    return cycles + addToChannelCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 8 == 0) {
    return cycles + addToChannelCoreCycles_half_multiple_of_8(addendLen,
                                                              blockCount);
  }

  cycles += 2; // cmpult, brnz
  if (blockCount < 2) {
    return cycles + addToChannelCoreCycles_half_scalar(addendLen, blockCount);
  }

  cycles += 2; // and, brz
  if (addendLen % 4 == 0) {
    return cycles + addToChannelCoreCycles_half_multiple_of_4(addendLen,
                                                              blockCount);
  }
  return cycles + addToChannelCoreCycles_half_scalar(addendLen, blockCount);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddToChannel2D)(
                                        const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_FIELD(addendLen);
  CODELET_FIELD(actsBlockCount);

  std::uint64_t numCycles = 7; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    numCycles += 6; // loop overhead.

    auto coreFunc = type == HALF ? addToChannelCoreCycles_half
                                 : addToChannelCoreCycles_float;

    numCycles += coreFunc(addendLen.getInitialValue<uint16_t>(target, i),
                          actsBlockCount.getInitialValue<uint16_t>(target, i));
  }

  // Exit
  numCycles += 1;

  return numCycles;
}


std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AddToChannel2D)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &type) {
  // ScaledAddToChannel2D and AddToChannel2D use nearly the same code. There is
  // an additional branch in the supervisor part though.
  return getCyclesEstimateForScaledAddToChannel2D(vertex, target, type) + 1;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddToChannel)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  CODELET_SCALAR_VAL(actsBlockCountPacked, uint16_t);

  const auto numWorkerContexts = target.getNumWorkerContexts();

  // These numbers may not be exact (e.g. the remainder of
  // actsBlockCountPacked is ignored).

  // Supervisor overhead.
  std::uint64_t numCycles = 1 + 6 + 1 + 6;

  auto approxBlocksPerWorker = actsBlockCountPacked >> 3;


  // Worker overhead.
  numCycles += numWorkerContexts * (type == HALF ? 26 : 19);

  auto coreFunc = type == HALF ? addToChannelCoreCycles_half
                               : addToChannelCoreCycles_float;

  numCycles += numWorkerContexts * coreFunc(addend.size(),
                                            approxBlocksPerWorker);

  // Exit
  numCycles += 1;

  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AddToChannel)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {

  // ScaledAddToChannel and AddToChannel use nearly the same code. There is
  // an additional branch in the supervisor part though.
  return getCyclesEstimateForScaledAddToChannel(vertex, target, type) + 1;
}

static std::uint64_t
channelMulCycles(unsigned elemsPerWorker, unsigned chansPerGroup,
                 unsigned dataPathWidth, const Type &type) {
  const bool isFloat = type == FLOAT;
  const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  std::uint64_t numCycles = 3; // Load scale and act pointers.

  // multiply scale by acts using mul + dual load + store.
  numCycles += (elemsPerWorker * chansPerGroup + vectorWidth - 1) / vectorWidth;
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ChannelMul2D)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(actsIn);
  CODELET_FIELD(actsOut);
  CODELET_FIELD(scale);
  const auto dataPathWidth = target.getDataPathWidth();
  unsigned n = actsIn.size();
  unsigned numCycles = 5;
  assert(actsIn.size() == actsOut.size());
  assert(scale.size() == n);
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = scale[i].size();
    assert(actsIn[i].size() % chansPerGroup == 0);
    assert(actsOut[i].size() % chansPerGroup == 0);
    numCycles +=
        channelMulCycles(actsIn[i].size() / chansPerGroup, chansPerGroup,
                         dataPathWidth, type);
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ChannelMul)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  CODELET_FIELD(actsIn);
  CODELET_FIELD(actsOut);
  CODELET_FIELD(scale);
  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  assert(actsIn.size() == actsOut.size());
  unsigned chansPerGroup = scale.size();
  assert(actsIn.size() % chansPerGroup == 0);
  assert(actsOut.size() % chansPerGroup == 0);
  unsigned len = actsIn.size() / chansPerGroup;
  unsigned elemsPerWorker = (len + numWorkerContexts - 1) / numWorkerContexts;
  std::uint64_t numCycles =
      10 + channelMulCycles(elemsPerWorker, chansPerGroup, dataPathWidth, type);
  return numCycles * numWorkerContexts + 10;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(InverseStdDeviation)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &meanType,
    const Type &powerType,
    const Type &outType) {
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(OuterProduct)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  CODELET_FIELD(out);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = type == FLOAT;
  const auto width = in.size();
  const auto numChans = weights.size();
  const auto numChanGroups = out.size();
  assert(numChans % numChanGroups == 0);
  const auto chansPerGroup = numChans / numChanGroups;
  return getOuterProductCycleEstimate(isFloat, width, numChans, chansPerGroup,
                                      dataPathWidth);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAdd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return getReduceCycleEstimate(outSizes,
                                partials.size(),
                                dataPathWidth,
                                false, false,
                                outType == FLOAT,
                                partialsType == FLOAT);
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return
  {
    CYCLE_ESTIMATOR_ENTRY(popconv, OuterProduct, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, OuterProduct, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                   FLOAT, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                   FLOAT, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                   HALF, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                   HALF, HALF, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, Transpose2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, Transpose2d, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, WgdConvComplete, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, WgdConvComplete, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, WgdInverseTransform,
                                   FLOAT, 4, 4, 3, 3),
    CYCLE_ESTIMATOR_ENTRY(popconv, WgdInverseTransform,
                                   HALF, 4, 4, 3, 3),

    CYCLE_ESTIMATOR_ENTRY(popconv, WgdReduce, FLOAT, 4, 4),
    CYCLE_ESTIMATOR_ENTRY(popconv, WgdReduce, HALF, 4, 4),

    CYCLE_ESTIMATOR_ENTRY(popconv, WgdPartials, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, WgdPartials, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, WgdDataTransform,
                                   FLOAT, 4, 4, 3, 3),
    CYCLE_ESTIMATOR_ENTRY(popconv, WgdDataTransform,
                                   HALF, 4, 4, 3, 3),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                   FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                   HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                   HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                       FLOAT, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                       HALF, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                       HALF, HALF, false),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, FLOAT, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                   FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                   HALF, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                   FLOAT, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                   FLOAT, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, FLOAT, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, HALF, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(popconv, ReduceAdd, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ReduceAdd, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ReduceAdd, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, ReduceAdd, HALF, HALF)
  };
};

} // end namespace popconv
