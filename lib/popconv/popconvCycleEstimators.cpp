#include "popconvCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popconv {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialnx1)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &fpType,
                                          const Type &accumType,
                                          bool useDeltasForEdges) {
  CODELET_SCALAR_VAL(kernelOuterSize, unsigned);
  CODELET_SCALAR_VAL(kernelInnerElements, unsigned);
  CODELET_SCALAR_VAL(numOutGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroups, unsigned);
  CODELET_SCALAR_VAL(outStride, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(ampKernelHeight, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(convUnitInputLoadElemsPerCycle, unsigned);
  CODELET_SCALAR_VAL(convUnitCoeffLoadBytesPerCycle, unsigned);

  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_VECTOR_VALS(zeroWorklist, unsigned);

  const auto dataPathWidth = target.getDataPathWidth();
  const auto numWorkerContexts = target.getNumWorkerContexts();

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  const auto kernelSize = kernelOuterSize * kernelInnerElements;
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
                                         floatPartials,
                                         useDeltasForEdges);
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
  bool floatWeights = fpType == FLOAT;
  return zeroCycles +
    getConvPartialnx1SupervisorCycleEstimate(workerPartitions,
                                             numConvGroups,
                                             numOutGroups,
                                             numInGroups,
                                             kernelSize,
                                             ampKernelHeight,
                                             inChansPerGroup,
                                             convUnitInputLoadElemsPerCycle,
                                             outChansPerGroup,
                                             convUnitCoeffLoadBytesPerCycle,
                                             numWorkerContexts,
                                             floatWeights,
                                             useDeltasForEdges);
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
                                             bool useDeltasForEdges) {
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_SCALAR_VAL(numConvGroups, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numOutGroups, unsigned);
  CODELET_SCALAR_VAL(convUnitInputLoadElemsPerCycle, unsigned);
  CODELET_SCALAR_VAL(outChansPerGroup, unsigned);
  CODELET_SCALAR_VAL(convUnitCoeffLoadBytesPerCycle, unsigned);
  const auto numWorkerContexts = target.getNumWorkerContexts();

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
  return
    getConvPartial1x1SupervisorCycleEstimate(workerPartitions,
                                             numConvGroups,
                                             numInGroups,
                                             numOutGroups,
                                             convUnitInputLoadElemsPerCycle,
                                             outChansPerGroup,
                                             convUnitCoeffLoadBytesPerCycle,
                                             numWorkerContexts,
                                             floatWeights,
                                             useDeltasForEdges
                                             );
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ConvPartialHorizontalMac)(
    const VertexIntrospector &vertex,
    const Target &target,
    const Type &fpType,
    const Type &accumType) {
  CODELET_VECTOR_2D_VALS(worklists, unsigned);
  CODELET_VECTOR_VALS(zeroWorklist, unsigned);
  CODELET_SCALAR_VAL(numOutGroups, unsigned);
  CODELET_SCALAR_VAL(numInGroups, unsigned);
  CODELET_SCALAR_VAL(numConvGroups, unsigned);
  CODELET_SCALAR_VAL(kernelSize, unsigned);
  CODELET_SCALAR_VAL(outStride, unsigned);
  CODELET_SCALAR_VAL(inChansPerGroup, unsigned);

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
                                         floatPartials,
                                         false);

  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
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

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(AddToChannel)(const VertexIntrospector &vertex,
                                        const Target &target,
                                        const Type &type) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();

  const bool isFloat = type == FLOAT;
  const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  unsigned n = acts.size();
  unsigned numCycles = 5;
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = addend[i].size();
    assert(acts[i].size() % chansPerGroup == 0);
    unsigned len = acts[i].size() / chansPerGroup;
    numCycles += 2; // Load bias and act pointers.
    numCycles += 1; // Warmup.
    // Add biases to acts using add + dual load + store.
    numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
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

  const bool isFloat = type == FLOAT;
  const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  unsigned n = acts.size();
  unsigned numCycles = 6;
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = addend[i].size();
    assert(acts[i].size() % chansPerGroup == 0);
    unsigned len = acts[i].size() / chansPerGroup;
    numCycles += 2; // Load addend and act pointers.
    numCycles += 1; // Warmup.
    // Add addend to acts using axpby + dual load + store.
    numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
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

  const bool isFloat = type == FLOAT;
  const auto vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  unsigned n = actsIn.size();
  unsigned numCycles = 5;
  for (unsigned i = 0; i != n; ++i) {
    unsigned chansPerGroup = scale[i].size();
    assert(actsIn[i].size() % chansPerGroup == 0);
    assert(actsOut[i].size() % chansPerGroup == 0);
    unsigned len = actsIn[i].size() / chansPerGroup;
    numCycles += 3; // Load scale and act pointers.
    numCycles += 1; // Warmup.
    // multiply scale by acts using mul + dual load + store.
    numCycles += (len * chansPerGroup + vectorWidth - 1) / vectorWidth;
  }
  return numCycles;
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

  uint64_t cycles = 6;
  for (unsigned i = 0; i < mean.size(); ++i) {
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

    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, HALF),

    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, HALF),

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
                                   FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                   HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                   HALF, HALF),

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
    CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, HALF, FLOAT, false)
  };
};

} // end namespace popconv
