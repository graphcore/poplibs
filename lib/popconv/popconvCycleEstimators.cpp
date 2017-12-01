#include "popconvCycleEstimators.hpp"
#include <poplar/HalfFloat.hpp>
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popconv {

template <class FPType, class AccumType, bool useDeltasForEdges>
MAKE_CYCLE_ESTIMATOR(ConvPartialnx1, vertex, target) {
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
  constexpr bool floatPartials = std::is_same<AccumType, float>::value;
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
  constexpr bool floatWeights = std::is_same<FPType, float>::value;
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

template <class FPType>
MAKE_CYCLE_ESTIMATOR(ConvChanReduce2, vertex, target) {
  CODELET_FIELD(out);
  CODELET_VECTOR_VALS(numInputsPerOutput, unsigned);
  auto numBiases = out.size();
  uint64_t cycles = 10;

  for (unsigned bias = 0; bias < numBiases; ++bias) {
    cycles += numInputsPerOutput[bias];
  }
  return cycles;
}

template <typename InType, typename OutType>
MAKE_CYCLE_ESTIMATOR(ConvChanReduceAcc, vertex, target) {
  CODELET_FIELD(in);
  return 15 + in.size();
}

template <class FPType, class AccumType, bool useDeltasForEdges>
MAKE_CYCLE_ESTIMATOR(ConvPartial1x1Out, vertex, target) {
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
  constexpr bool floatWeights = std::is_same<FPType, float>::value;
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

template <class FPType, class AccumType>
MAKE_CYCLE_ESTIMATOR(ConvPartialHorizontalMac, vertex, target) {
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

  constexpr bool floatActivations = std::is_same<FPType, float>::value;
  std::vector<unsigned> tZeroWorkList;
  for (unsigned i = 0; i != zeroWorklist.size() / 2; ++i) {
    tZeroWorkList.push_back(zeroWorklist[2 * i + 1]);
  }
  constexpr bool floatPartials = std::is_same<AccumType, float>::value;
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

template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
MAKE_CYCLE_ESTIMATOR(WgdDataTransform, vertex, target) {
  CODELET_FIELD(dIn);

  constexpr bool isFloat = std::is_same<FPType, float>::value;
  const unsigned numInpRows = patchSizeX;
  const unsigned numInpCols = patchSizeY;

  const unsigned nPatches = dIn.size() / (numInpCols * numInpRows);

  return getWgdDataTransformCycles(nPatches * dIn[0].size(), isFloat);
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(WgdPartials, vertex, target) {
  CODELET_FIELD(dTf);
  CODELET_FIELD(wTf);
  CODELET_FIELD(partials);
  CODELET_SCALAR_VAL(numConvUnits, unsigned);
  CODELET_SCALAR_VAL(weightsPerConvUnit, unsigned);
  CODELET_SCALAR_VAL(convUnitCoeffLoadBytesPerCycle, unsigned);
  const auto  numWorkers = target.getNumWorkerContexts();

  constexpr bool isFloat = std::is_same<FPType, float>::value;
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

template <class FPType, unsigned patchSizeX, unsigned patchSizeY>
MAKE_CYCLE_ESTIMATOR(WgdReduce, vertex, target) {
  CODELET_FIELD(inPartial);
  CODELET_FIELD(outPartial);

  constexpr bool isFloat = std::is_same<FPType, float>::value;

  const unsigned numElems = outPartial.size();
  const unsigned numOutChans = outPartial[0].size();
  const unsigned numInpChans = inPartial.size() / numElems;

  return getWgdReduceCycles(
                 numElems * numOutChans,
                 numInpChans,
                 isFloat
                 );
}

template <class FPType, unsigned patchSizeX, unsigned patchSizeY,
          unsigned kernelX, unsigned kernelY>
MAKE_CYCLE_ESTIMATOR(WgdInverseTransform, vertex, target) {
  CODELET_FIELD(dTf);
  CODELET_FIELD(dOut);

  constexpr bool isFloat = std::is_same<FPType, float>::value;
  const unsigned numInCols = patchSizeY;
  const unsigned numInRows = patchSizeX;

  const unsigned nGroups = dTf.size() / (numInCols * numInRows);
  const unsigned depthDim = dOut[0].size();

  return getWgdInvTransformCycles(nGroups * depthDim, isFloat);
}

template <class FPType>
MAKE_CYCLE_ESTIMATOR(WgdConvComplete, vertex, target) {
  CODELET_FIELD(dIn);

  constexpr bool isFloat = std::is_same<FPType, float>::value;
  const unsigned nGroups = dIn.size();
  const unsigned vecLen = dIn[0].size();
  return getWgdCompleteCycles(
                             vecLen * nGroups,
                             isFloat);
}

template <typename T>
MAKE_CYCLE_ESTIMATOR(Transpose2D, vertex, target) {
  CODELET_FIELD(src);
  CODELET_FIELD(dst);
  CODELET_SCALAR_VAL(numSrcColumns, unsigned);

  constexpr bool isFloat = std::is_same<T, float>::value;
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

template <typename FPType>
MAKE_CYCLE_ESTIMATOR(AddToChannel, vertex, target) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<FPType, float>::value;
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

template <typename FPType>
MAKE_CYCLE_ESTIMATOR(ScaledAddToChannel, vertex, target) {
  CODELET_FIELD(acts);
  CODELET_FIELD(addend);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<FPType, float>::value;
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

template <typename FPType>
MAKE_CYCLE_ESTIMATOR(ChannelMul, vertex, target) {
  CODELET_FIELD(actsIn);
  CODELET_FIELD(actsOut);
  CODELET_FIELD(scale);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<FPType, float>::value;
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

template <typename InType, typename OutType>
MAKE_CYCLE_ESTIMATOR(ConvChanReduce, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_SCALAR_VAL(useDoubleDataPathInstr, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<InType, float>::value;
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

template <typename InType, typename OutType>
MAKE_CYCLE_ESTIMATOR(ConvChanReduceSquare, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(in);
  CODELET_SCALAR_VAL(useDoubleDataPathInstr, bool);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<InType, float>::value;
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

template <typename InType, typename OutType>
MAKE_CYCLE_ESTIMATOR(ConvChanReduceAndScale, vertex, target) {
  CODELET_FIELD(in);
  return 15 + in.size();
}

template <class MeanType, class PowerType, class OutType>
MAKE_CYCLE_ESTIMATOR(InverseStdDeviation, vertex, target) {
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

template <class T>
MAKE_CYCLE_ESTIMATOR(OuterProduct, vertex, target) {
  CODELET_FIELD(in);
  CODELET_FIELD(weights);
  CODELET_FIELD(out);
  const auto dataPathWidth = target.getDataPathWidth();

  constexpr bool isFloat = std::is_same<T, float>::value;
  const auto width = in.size();
  const auto numChans = weights.size();
  const auto numChanGroups = out.size();
  assert(numChans % numChanGroups == 0);
  const auto chansPerGroup = numChans / numChanGroups;
  return getOuterProductCycleEstimate(isFloat, width, numChans, chansPerGroup,
                                      dataPathWidth);
}

poplibs::CycleEstimatorTable cyclesFunctionTable = {
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, OuterProduct, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, OuterProduct, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                 float, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                 float, float, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                 half, float, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, InverseStdDeviation,
                                 half, half, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, float, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAndScale, half, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceSquare, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceSquare, half, float),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce, half, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ChannelMul, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ScaledAddToChannel, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, AddToChannel, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, Transpose2D, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, Transpose2D, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdConvComplete, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdConvComplete, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdInverseTransform,
                                 float, 4, 4, 3, 3),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdInverseTransform,
                                 half, 4, 4, 3, 3),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdReduce, float, 4, 4),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdReduce, half, 4, 4),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdPartials, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdPartials, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdDataTransform,
                                 float, 4, 4, 3, 3),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, WgdDataTransform,
                                 half, 4, 4, 3, 3),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                 float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                 half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialHorizontalMac,
                                 half, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, half, half, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, half, float, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, float, half, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                 float, float, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out, half, half, false),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                 half, float, false),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                 float, half, false),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartial1x1Out,
                                 float, float, false),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduceAcc, float, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce2, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvChanReduce2, half),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, float, float, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, half, half, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, half, float, true),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, float, float, false),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, half, half, false),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popconv, ConvPartialnx1, half, float, false)
};

} // end namespace popconv
