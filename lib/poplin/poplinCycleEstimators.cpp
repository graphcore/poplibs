#include "poplinCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"
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
  CODELET_SCALAR_VAL(zerosInfo, unsigned);

  CODELET_VECTOR_2D_VALS(worklists, unsigned);
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

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto numConvUnits = getNumConvUnits(fpType, accumType, target);

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  const auto kernelSize = kernelOuterSize * kernelInnerElements;
  assert(kernelSize > 0);
  const auto usedContexts = worklists.size() / kernelSize;

  bool floatPartials = accumType == FLOAT;
  const auto outBytesPerAtom = target.getTypeSize(accumType);

  std::vector<unsigned> tZeroWorkList;
  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    tZeroWorkList.push_back((zerosInfo + numWorkerContexts - 1) /
                            numWorkerContexts);
  }

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
  CODELET_SCALAR_VAL(numOutGroupsM1, unsigned);
  CODELET_SCALAR_VAL(numInGroupsM1, unsigned);
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
  const auto numInGroups = numInGroupsM1 + 1;
  const auto kernelSize = kernelSizeM1 + 1;
  const auto outStride = transformedOutStride / outChansPerGroup + 1;

  assert(numConvGroups * numOutGroups * numInGroups == weights.size());
  assert(out.size() == numOutGroups * numConvGroups);
  assert(in.size() == numInGroups * numConvGroups);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();

  std::vector<unsigned> tZeroWorkList;
  const auto outBytesPerAtom = target.getTypeSize(accumType);
  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    tZeroWorkList.push_back((zerosInfo + numWorkerContexts -1)
                             / numWorkerContexts);
  }

  bool floatActivations = fpType == FLOAT;
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
  CODELET_SCALAR_VAL(numSrcRows, unsigned);
  CODELET_SCALAR_VAL(numSrcColumns, unsigned);

  const bool isFloat = type == FLOAT;
  const auto matrices = dst.size();
  std::uint64_t cycles;

  if(isFloat) {
    if( ((numSrcRows & 1) == 0) &&
        ((numSrcColumns & 1 ) == 0) &&
        (numSrcColumns/2 < 0x1000 ) &&      // hardware RPT count constraint
        (numSrcRows * (numSrcColumns-2)/2 < 512) ) {  // Largest stride used
        // Float, fast path estimates
        cycles = 25 + matrices *
                  (11 + (numSrcRows/2 ) * ( 6 + 3 * (numSrcColumns/2 -1)));
    }
    else {
        // Float, slow path estimates based on numSrcRows being even
        cycles = 13 + matrices *
                  (8 + numSrcColumns * ( 5 + (numSrcRows * 4)/2));
    }
  }
  else {
    if( ((numSrcRows & 3) == 0) &&
        ((numSrcColumns & 3 ) == 0)  &&
        (numSrcColumns >= 8) &&
        (numSrcColumns/4 < 0x1000 ) &&        // hardware RPT count constraint
        (1 + 3 * (numSrcColumns/4) < 512) ) {  // Largest stride used
        // Half, fast path estimates
        cycles = 34 + matrices *
                  (11 + (numSrcRows/4 ) * ( 15 + 4 *(numSrcColumns/4 -2)));
    }
    else {
        // Half, slow path estimates based on numSrcRows being even
        cycles = 15 + matrices *
                  (8 + numSrcColumns * ( 5 + (numSrcRows * 5)/2));
    }
  }
  return cycles;
}

// Exact worker cycle count for poplin_AddToChannel__float_core
std::uint64_t addToChannelCoreCycles_float(unsigned addendLen,
                                           unsigned blockCount) {
  std::uint64_t cycles = 1; // return

  ++cycles; // brz

  if (blockCount == 0)
    return cycles;

  cycles += 5; // before loop

  for (unsigned i = 0; i < addendLen; ++i) {
    cycles += 2; // start of loop
    cycles += 2 * blockCount; // rpt loop
    cycles += 4; // end of loop
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

// Exact worker cycle count for poplin_AddToChannel__half_core
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


// Exact worker cycle count for popconv_ChannelMul__float_core
std::uint64_t channelMulCoreCycles_float(unsigned scaleLen,
                                         unsigned blockCount) {
  std::uint64_t cycles = 1; // return

  ++cycles; // brz

  if (blockCount == 0)
    return cycles;

  cycles += 5; // before loop

  for (unsigned i = 0; i < scaleLen; ++i) {
    cycles += 2; // start of loop
    cycles += 2 * blockCount; // rpt loop
    cycles += 5; // end of loop
  }
  return cycles;
}

std::uint64_t channelMulCoreCycles_half_scalar(unsigned scaleLen,
                                               unsigned blockCount) {
  std::uint64_t cycles = 4; // pre-loop
  // Aligned loop bodies take 7 cycles, misaligned take 9, but they are
  // equally numerous so it averages to 8.
  cycles += scaleLen * (5 + blockCount * 8);
  return cycles;
}

std::uint64_t channelMulCoreCycles_half_multiple_of_4(unsigned scaleLen,
                                                      unsigned blockCount) {
  std::uint64_t cycles = 5; // pre-loop
  cycles += (scaleLen/4) * (
    5 + // pre-rpt
    2 * (blockCount/2 - 1) + // rpt body
    // post-rpt. The code actually depends on whether or not blockCount
    // was odd but it takes the same number of cycles in both cases.
    7
  );
  return cycles;
}

// Exact worker cycle count for popconv_ChannelMul__half_core
std::uint64_t channelMulCoreCycles_half(unsigned scaleLen,
                                        unsigned blockCount) {
  std::uint64_t cycles = 1; // return

  cycles += 2; // cmpult > 2044, brz
  if (scaleLen > 2044) {
    return cycles + addToChannelCoreCycles_half_scalar(scaleLen, blockCount);
  }

  cycles += 2; // cmpult, brnz
  if (blockCount < 2) {
    return cycles + addToChannelCoreCycles_half_scalar(scaleLen, blockCount);
  }

  cycles += 2; // and, brz
  if (scaleLen % 4 == 0) {
    return cycles + addToChannelCoreCycles_half_multiple_of_4(scaleLen,
                                                              blockCount);
  }
  return cycles + addToChannelCoreCycles_half_scalar(scaleLen, blockCount);
}



std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ChannelMul2D)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_SCALAR_VAL(n, uint32_t);
  CODELET_FIELD(scaleLen);
  CODELET_FIELD(actsBlockCount);

  std::uint64_t numCycles = 7; // pre-loop

  for (unsigned i = 0; i != n; ++i) {
    numCycles += type == HALF ? 13 : 10; // loop overhead.

    auto coreFunc = type == HALF ? addToChannelCoreCycles_half
                                 : addToChannelCoreCycles_float;

    numCycles += coreFunc(scaleLen.getInitialValue<uint16_t>(target, i),
                          actsBlockCount.getInitialValue<uint16_t>(target, i));
  }

  // Exit
  numCycles += 1;

  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ChannelMul)(const VertexIntrospector &vertex,
                                      const Target &target,
                                      const Type &type) {
  CODELET_FIELD(scale);
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

  numCycles += numWorkerContexts * coreFunc(scale.size(),
                                            approxBlocksPerWorker);

  // Exit
  numCycles += 1;

  return numCycles;
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
  CODELET_SCALAR_VAL(chansPerGroup,unsigned);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = type == FLOAT;
  const auto width = in.size();
  const auto numChans = weights.size();
  const auto numChanGroups = out.size();
  assert(numChans % numChanGroups == 0);

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


  return getReduceCycleEstimate(out.size(),
                                partials.size(),
                                dataPathWidth,
                                outType == FLOAT,
                                partialsType == FLOAT,
                                target.getNumWorkerContexts());
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return
  {
    CYCLE_ESTIMATOR_ENTRY(poplin, OuterProduct, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, OuterProduct, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation,
                                   FLOAT, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation,
                                   FLOAT, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation,
                                   HALF, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, InverseStdDeviation,
                                   HALF, HALF, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, ChannelMul, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ChannelMul, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, ChannelMul2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ChannelMul2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, ScaledAddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ScaledAddToChannel, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, ScaledAddToChannel2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ScaledAddToChannel2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, AddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, AddToChannel, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, AddToChannel2D, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, AddToChannel2D, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, Transpose2d, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, Transpose2d, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, WgdConvComplete, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, WgdConvComplete, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, WgdInverseTransform,
                                   FLOAT, 4, 4, 3, 3),
    CYCLE_ESTIMATOR_ENTRY(poplin, WgdInverseTransform,
                                   HALF, 4, 4, 3, 3),

    CYCLE_ESTIMATOR_ENTRY(poplin, WgdReduce, FLOAT, 4, 4),
    CYCLE_ESTIMATOR_ENTRY(poplin, WgdReduce, HALF, 4, 4),

    CYCLE_ESTIMATOR_ENTRY(poplin, WgdPartials, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, WgdPartials, HALF),

    CYCLE_ESTIMATOR_ENTRY(poplin, WgdDataTransform,
                                   FLOAT, 4, 4, 3, 3),
    CYCLE_ESTIMATOR_ENTRY(poplin, WgdDataTransform,
                                   HALF, 4, 4, 3, 3),

    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                   FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                   HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                   HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                       FLOAT, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                       HALF, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialHorizontalMac,
                                       HALF, HALF, false),

    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, FLOAT, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out,
                                   FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out,
                                   HALF, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out,
                                   FLOAT, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartial1x1Out,
                                   FLOAT, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, true),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, FLOAT, FLOAT, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, HALF, false),
    CYCLE_ESTIMATOR_ENTRY(poplin, ConvPartialnx1, HALF, FLOAT, false),

    CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(poplin, ReduceAdd, HALF, HALF)
  };
};

} // end namespace poplin
