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
MAKE_CYCLE_ESTIMATOR_NAME(ConvChanReduce2)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &fpType) {
  CODELET_FIELD(out);
  CODELET_VECTOR_VALS(numInputsPerOutput, unsigned);
  auto numBiases = out.size();
  uint64_t cycles = 10;

  for (unsigned bias = 0; bias < numBiases; ++bias) {
    cycles += numInputsPerOutput[bias];
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvChanReduceAcc)(const VertexIntrospector &vertex,
                                             const Target &target,
                                             const Type &inType,
                                             const Type &outType) {
  CODELET_FIELD(in);
  return 15 + in.size();
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
  CODELET_SCALAR_VAL(numOutGroups, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroups, unsigned);
  CODELET_SCALAR_VAL(kernelSize, unsigned);
  CODELET_SCALAR_VAL(outStride, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
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

static std::uint64_t
addToChannelCycles(unsigned elemsPerWorker, unsigned chansPerGroup,
                   unsigned dataPathWidth, const Type &type, bool scaledAdd) {
  const bool isFloat = type == FLOAT;
  const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  // Load bias and act pointers.
  std::uint64_t numCycles = 2;
  if (scaledAdd) // to load CSR
    numCycles += 1;
  // Add addend to acts using add + dual load + store
  // Add addend to acts using axpby + dual load + store for scaledAdd = 1
  numCycles += (elemsPerWorker * chansPerGroup + vectorWidth - 1) / vectorWidth;
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AddToChannel)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  std::uint64_t numCycles = 10;
  unsigned chansPerGroup = addend.size();
  assert(acts.size() % chansPerGroup == 0);
  const auto elemsPerWorker =
      (acts.size() / chansPerGroup + numWorkerContexts - 1) / numWorkerContexts;
  numCycles += addToChannelCycles(elemsPerWorker, chansPerGroup, dataPathWidth,
                                  type, false);
  return numCycles * numWorkerContexts + 10;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AddToChannel2D)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();
  unsigned n = acts.size();
  std::uint64_t numCycles = 5;
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = addend[i].size();
    assert(acts[i].size() % chansPerGroup == 0);
    numCycles += addToChannelCycles(acts[i].size() / chansPerGroup,
                                    chansPerGroup, dataPathWidth, type, false);
    numCycles += 1; // branch.
  }
  return numCycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddToChannel)(const VertexIntrospector &vertex,
                                              const Target &target,
                                              const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();
  std::uint64_t numCycles = 10;
  unsigned chansPerGroup = addend.size();
  assert(acts.size() % chansPerGroup == 0);
  const auto elemsPerWorker =
      (acts.size() / chansPerGroup + numWorkerContexts - 1) / numWorkerContexts;
  numCycles += addToChannelCycles(elemsPerWorker, chansPerGroup,
                                               dataPathWidth, type, true);
  return numCycles * numWorkerContexts + 10;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ScaledAddToChannel2D)(
                                        const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();
  unsigned n = acts.size();
  std::uint64_t numCycles = 5;
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = addend[i].size();
    assert(acts[i].size() % chansPerGroup == 0);
    numCycles += addToChannelCycles(acts[i].size() / chansPerGroup,
                                    chansPerGroup, dataPathWidth, type, true);
    numCycles += 1; // branch.
  }
  return numCycles;
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
MAKE_CYCLE_ESTIMATOR_NAME(ConvChanReduce)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &inType,
                                          const Type &outType) {
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_SCALAR_VAL(useDoubleDataPathInstr, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = inType == FLOAT;
  // factor of 2 for instructions that allow double the datapath width
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  if (useDoubleDataPathInstr) {
    vectorWidth *= 2;
  }
  unsigned numVectors = (out.size() + vectorWidth - 1) / vectorWidth;

  uint64_t cycles = 11; // overhead from benchmark including 2 cycles for run
  cycles += 7 * numVectors;
  for (unsigned d = 0; d < in.size(); d++) {
    cycles += 5;
    auto samplesPerEst = in[d].size() / out.size();
    cycles += numVectors * (3 + samplesPerEst);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvChanReduceSquare)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &inType,
    const Type &outType) {
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_SCALAR_VAL(useDoubleDataPathInstr, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = inType == FLOAT;
  // factor of 2 for instructions that allow double the datapath width
  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  if (useDoubleDataPathInstr) {
    vectorWidth *= 2;
  }
  unsigned numVectors = (out.size() + vectorWidth - 1) / vectorWidth;

  uint64_t cycles = 11; // overhead from benchmark including 2 cycles for run
  cycles += 7 * numVectors;
  for (unsigned d = 0; d < in.size(); d++) {
    cycles += 5;
    auto samplesPerEst = in[d].size() / out.size();
    cycles += numVectors * (3 + samplesPerEst);
  }
  return cycles;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvChanReduceAndScale)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &inType,
    const Type &outType) {
  CODELET_FIELD(in);
  return 15 + in.size();
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

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, FLOAT,
                                   FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, FLOAT, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, HALF, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceSquare, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceSquare, HALF, FLOAT),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, HALF, HALF),

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

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, FLOAT, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce2, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce2, HALF),

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
