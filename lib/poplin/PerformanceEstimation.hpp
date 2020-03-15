// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

inline static std::uint64_t convHorizontalMacOverhead(bool floatActivations) {
  return floatActivations ? 58 : 63;
}

inline static std::uint64_t convNx1Overhead() {
#if WORKER_REG_STATE_RETAINED
  return 103;
#else
  return 101;
#endif
}

// Number of worker cycle savings if state retention is used.
// The first entry is the total savings and the second is
// because of retention of state related to input channel processing.
inline static std::pair<std::uint64_t, std::uint64_t>
conv1x1WorkerRetentionSavings(bool floatActivations, bool floatPartials) {
#if WORKER_REG_STATE_RETAINED
  if (floatActivations == false && floatPartials == true) {
    return std::make_pair(14, 3);
  } else {
    return std::make_pair(0, 0);
  }
#else
  (void)floatActivations;
  (void)floatPartials;
  return std::make_pair(0, 0);
#endif
}

inline static std::uint64_t
convnx1WorkerRetentionSavings(bool /*floatActivations */,
                              bool /*floatPartials */) {
#if WORKER_REG_STATE_RETAINED
  return 6;
#else
  return 0;
#endif
}

inline static std::uint64_t zeroPartialsRetentionSavings(bool floatPartials) {
#if WORKER_REG_STATE_RETAINED
  return floatPartials ? 9 : 10;
#else
  return 0;
#endif
}

inline std::uint64_t getDenseDotProductCycles(bool isFloat, unsigned size) {
  if (isFloat) {
    if ((size % 2) == 0)
      return 4 + size;
    else
      return 4 + (2 * size);
  }

  if ((size % 4) == 0)
    return 6 + size / 4;
  else
    return 4 + size;
}

template <class InputIterator>
bool allEqual(InputIterator begin, InputIterator end) {
  if (begin == end)
    return true;
  const auto &first = *begin;
  for (auto it = begin + 1; it != end; ++it) {
    if (*it != first)
      return false;
  }
  return true;
}

inline std::uint64_t getConvPartialHorizontalMacCycleEstimate(
    bool isFloat, unsigned numInChans, unsigned numOutChans,
    const std::vector<unsigned> &convSizes) {
  uint64_t cycles = 16;
  for (auto convSize : convSizes) {
    if (convSize == 0) {
      cycles += 7;
    } else {
      cycles += 19;
      cycles +=
          convSize *
          (7 + numOutChans * getDenseDotProductCycles(isFloat, numInChans));
    }
  }
  return cycles;
}

inline std::uint64_t
getZeroSupervisorVertexCycleEstimate(const std::vector<unsigned> &worklist,
                                     unsigned numGroups, unsigned dataPathWidth,
                                     unsigned numWorkerContexts, bool isFloat) {
  const unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);

  std::uint64_t maxWorkerCyclesZero = 0;
  for (unsigned context = 0; context != worklist.size(); ++context) {
    uint64_t numVectors = (worklist[context] + vectorWidth - 1) / vectorWidth;
    maxWorkerCyclesZero = std::max(maxWorkerCyclesZero,
                                   numVectors + (isFloat ? 14 : 15) -
                                       zeroPartialsRetentionSavings(isFloat));
  }
  uint64_t zeroCycles = maxWorkerCyclesZero * numWorkerContexts * numGroups;
  return zeroCycles;
}

inline std::uint64_t
getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelSize, unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup, unsigned numWorkerContexts, bool isFloat) {
  unsigned usedContexts = workerPartitions.size();
  uint64_t cycles = 0;
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  for (auto context = 0U; context != usedContexts; ++context) {
    uint64_t thisWorkerCycles = 0;
    for (auto k = 0U; k != kernelSize; ++k) {
      thisWorkerCycles += getConvPartialHorizontalMacCycleEstimate(
          isFloat, numInChansPerGroup, numOutChansPerGroup,
          workerPartitions[context][k]);
    }
    const unsigned workerNonLoopOverhead = 16;
    thisWorkerCycles += workerNonLoopOverhead;
    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }
  cycles += std::max(maxWorkerCycles, minWorkerCycles);
  return cycles;
}

inline std::uint64_t
getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCycles, unsigned numConvGroups, unsigned numInGroups,
    unsigned numOutGroups, bool isFloat) {
  uint64_t cycles = innerLoopCycles;
  return convHorizontalMacOverhead(isFloat) +
         numConvGroups *
             (23 + numInGroups * (15 + numOutGroups * (10 + cycles)));
}

inline std::uint64_t getConvPartialHorizontalMacSupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned kernelSize, unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup, unsigned numWorkerContexts, bool isFloat) {
  auto cycles = getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
      workerPartitions, kernelSize, numInChansPerGroup, numOutChansPerGroup,
      numWorkerContexts, isFloat);
  return getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
      cycles, numConvGroups, numInGroups, numOutGroups, isFloat);
}

inline std::uint64_t getConvPartial1x1SupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvUnits, bool outputZeroing,
    bool floatActivations, bool floatPartials) {
  // Core loop cycles for 16x8 AMP vertex
  auto coreCycles = floatActivations ? 8 : 4;
  // Core loop cycles for 8x4 AMP vertex
  if (numConvUnits == 4) {
    coreCycles /= 2;
  }

  auto retentionSavings =
      conv1x1WorkerRetentionSavings(floatActivations, floatPartials);
  unsigned usedContexts = workerPartitions.size();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  unsigned zeroCyclesPerGroup = floatPartials ? 4 : 2;
  for (const auto &worker : workerPartitions) {
    // 1x1 vertex doesn't support more than one worklist item per worker.
    assert(worker.size() <= 1);

    uint64_t thisWorkerCycles = 0;
    if (!worker.empty()) {
      const auto numElems = worker.front();
      switch (numElems) {
      case 0:
        if (floatActivations) {
          thisWorkerCycles += 27;
        } else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 25 : 28);
          } else {
            thisWorkerCycles += 27;
          }
        }
        break;
      case 1:
        if (floatActivations)
          thisWorkerCycles += 49 + (3 + 4) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 38 : 42);
          } else {
            thisWorkerCycles += 42 + (3 + zeroCyclesPerGroup) * outputZeroing;
          }
        }
        break;
      case 2:
        if (floatActivations)
          thisWorkerCycles += 49 + (3 + 4 * 2) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 40 : 44);
          } else {
            thisWorkerCycles +=
                43 + (3 + zeroCyclesPerGroup * 2) * outputZeroing;
          }
        }
        break;
      default:
        if (floatActivations)
          thisWorkerCycles += 49 + (3 + 4 * numElems) * outputZeroing +
                              (numElems - 3) * coreCycles;
        else {
          if (floatPartials) {
            thisWorkerCycles +=
                (outputZeroing ? 40 : 43) + (numElems - 3) * coreCycles;
          } else {
            thisWorkerCycles +=
                44 + (3 + zeroCyclesPerGroup * numElems) * outputZeroing +
                (numElems - 3) * coreCycles;
          }
        }
      }
      thisWorkerCycles -= retentionSavings.first;
    }

    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }

  // tag cost to worker with min cycles
  maxWorkerCycles = std::max(maxWorkerCycles, minWorkerCycles + 14);

  return maxWorkerCycles;
}

inline std::uint64_t getConvPartial1x1SupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing, unsigned numConvGroups,
    unsigned numInGroups, unsigned numOutGroups, unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, bool floatActivations,
    bool floatPartials, unsigned numWorkerContexts) {
  const auto outputPassesPerGroup =
      (outChansPerGroup + numConvUnits - 1) / numConvUnits;

  const auto retentionSavings =
      conv1x1WorkerRetentionSavings(floatActivations, floatPartials);
  auto numInputLoadsInnerLoop = numConvUnits / 2;
  const auto numLoads =
      convUnitInputLoadElemsPerCycle * numInputLoadsInnerLoop * numConvUnits *
      (floatActivations ? 4 : 2) / convUnitCoeffLoadBytesPerCycle;
  const uint64_t supervisorNonloopOverhead = 50;
#if WORKER_REG_STATE_RETAINED
  const unsigned outPassesOverhead = 7;
  const unsigned excessInChanOverhead = 1;
#else
  const unsigned excessInChanOverhead = 0;
  const unsigned outPassesOverhead = 6;
#endif
  return supervisorNonloopOverhead +
         numWorkerContexts *
             (retentionSavings.first +
              retentionSavings.second * (numInGroups * numConvGroups - 1)) +
         numConvGroups *
             (12 +
              (numInGroups - 1) *
                  (15 + excessInChanOverhead +
                   numOutGroups * (19 + outputPassesPerGroup *
                                            (6 + numLoads +
                                             innerLoopCyclesWithoutZeroing))) +
              (10 + excessInChanOverhead +
               numOutGroups *
                   (19 + outputPassesPerGroup * (outPassesOverhead + numLoads +
                                                 innerLoopCyclesWithZeroing))));
}

inline std::uint64_t getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned outChansPerGroup, unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {
  auto innerLoopCyclesWithZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvUnits, true,
          floatActivations, floatPartials);
  auto innerLoopCyclesWithoutZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvUnits, false,
          floatActivations, floatPartials);

  return getConvPartial1x1SupervisorOuterLoopCycleEstimate(
      innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing, numConvGroups,
      numInGroups, numOutGroups, outChansPerGroup,
      convUnitInputLoadElemsPerCycle, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, floatActivations, floatPartials,
      numWorkerContexts);
}

inline std::uint64_t getConvPartialnx1SupervisorCycleOuterLoopEstimate(
    std::uint64_t innerLoopCycles, unsigned numConvGroups,
    unsigned numOutGroups, unsigned numInGroups, unsigned outChansPerGroup,
    unsigned numConvUnits, unsigned numWorkerContexts, bool floatActivations,
    bool floatPartials) {
  uint64_t cycles = innerLoopCycles;
  return // Other constant supervisor code cycles
      convNx1Overhead() +
      // First iteration does not save cycles to calculate state which
      // will then be retained
      numWorkerContexts *
          convnx1WorkerRetentionSavings(floatActivations, floatPartials) +
      numWorkerContexts * zeroPartialsRetentionSavings(floatPartials) +
      // Supervisor code loop to zero partials. brnzdec loops mean
      // 6-cycle stall for all but last iteration.
      numConvGroups * (numOutGroups * (16 + WORKER_REG_STATE_RETAINED ? 1 : 0) +
                       (numOutGroups - 1) * 6 + 1) +
      (numConvGroups - 1) * 6 + 1 +
      // Supervisor code loop over conv/in/out groups
      numConvGroups * (16 + numInGroups * (14 + numOutGroups * (14 + cycles)));
}

inline std::uint64_t getConvPartialnx1SupervisorCycleInnerLoopEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {
  // Core loop cycles for vertex will all engines in use
  auto coreCycles = floatActivations ? 8 : 4;
  auto numInputLoadsInnerLoop = 4;
  // when using half of AMP engines need to reduce core cycles as well
  if (numConvUnits == 4) {
    coreCycles /= 2;
    numInputLoadsInnerLoop /= 2;
  }

  const auto retentionSavings =
      convnx1WorkerRetentionSavings(floatActivations, floatPartials);
  unsigned usedContexts = workerPartitions.size();
  unsigned numOutChanPasses = outChansPerGroup / numConvUnits;
  // TODO: T12901 Update for float input when assembler code is written.
  if (filterHeight == 4 && convUnitCoeffLoadBytesPerCycle >= 8)
    convUnitCoeffLoadBytesPerCycle = 8;
  const auto numLoads =
      convUnitInputLoadElemsPerCycle * // 2 for floats and 4 for halves
      numInputLoadsInnerLoop *         // number of input channels
      numConvUnits *                   // num of out chans = num of conv units
      (floatActivations ? 4 : 2) /     // convert channels to bytes
      convUnitCoeffLoadBytesPerCycle;
  // innermostLoopCycles is the cycles in the innermost supervisor loop
  uint64_t innermostLoopCycles = numLoads;

  // additional load cycles dependent on filterHeight
  switch (filterHeight) {
  case 4:
    innermostLoopCycles += 60;
    break;
  case 2:
    innermostLoopCycles += 46;
    break;
  case 1:
    innermostLoopCycles += 15;
    break;
  default:
    // non-limited version will pick this up and we don't estimate unlimited
    // version correctly
    innermostLoopCycles += 20 * filterHeight;
  }

#if WORKER_REG_STATE_RETAINED
  innermostLoopCycles += 3;
#endif

  uint64_t innerLoopCycles = 0;
  for (auto ky = 0U; ky != kernelOuterElems; ++ky) {
    innerLoopCycles += 14;
    for (auto kx = 0U; kx != kernelInnerElems; ++kx) {
      // remove cycles for branch in outChanPasses loop for last iteration
      innerLoopCycles += 17 - 5;
      const unsigned extraCycles = floatPartials ? 0 : 1;
      for (auto ocp = 0U; ocp != numOutChanPasses; ++ocp) {
        uint64_t maxWorkerCycles = 0;
        uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                       ? 0
                                       : std::numeric_limits<uint64_t>::max();
        for (auto context = 0U; context != usedContexts; ++context) {
          uint64_t thisWorkerCycles = 19 + extraCycles;
          const auto k = ky * kernelInnerElems + kx;
          for (auto &numElems : workerPartitions[context][k]) {
            switch (numElems) {
            case 0:
              thisWorkerCycles += 17;
              break;
            case 1:
              thisWorkerCycles += (floatActivations ? 33 : 29);
              break;
            case 2:
              thisWorkerCycles += (floatActivations ? 44 : 33);
              break;
            default:
              if (floatActivations)
                thisWorkerCycles += 45 + (numElems - 3) * coreCycles;
              else
                thisWorkerCycles += 34 + (numElems - 3) * coreCycles;
            }
            thisWorkerCycles -= retentionSavings;
          }
          maxWorkerCycles =
              std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
          minWorkerCycles =
              std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
        }
        innerLoopCycles += innermostLoopCycles +
                           std::max(maxWorkerCycles, minWorkerCycles + 9);
      }
    }
  }
  return innerLoopCycles;
}

inline std::uint64_t getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups, unsigned numOutGroups, unsigned numInGroups,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned inChansPerGroup, unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials) {
  auto innerLoopCycles = getConvPartialnx1SupervisorCycleInnerLoopEstimate(
      workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, convUnitInputLoadElemsPerCycle, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatActivations,
      floatPartials);
  return getConvPartialnx1SupervisorCycleOuterLoopEstimate(
      innerLoopCycles, numConvGroups, numOutGroups, numInGroups,
      outChansPerGroup, numConvUnits, numWorkerContexts, floatActivations,
      floatPartials);
}

inline std::uint64_t getConvPartialSlicSupervisorCycleOuterLoopEstimate(
    std::uint64_t zeroCycles, std::uint64_t innerLoopCycles,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned slicWindowWidth, unsigned convUnitInputLoadElemsPerCycle,
    unsigned numWeightBlocks, unsigned convUnitCoeffLoadBytesPerCycle,
    bool floatActivations, bool floatPartials) {
  // TODO: we currently only target SLIC for half->float which loads the weights
  // using the ld64putcs instruction.
  assert(!floatActivations);
  assert(floatPartials);
  (void)convUnitInputLoadElemsPerCycle;
  (void)convUnitCoeffLoadBytesPerCycle;

  // TODO: rougly based off of the ConvPartial1x1 codelet, when the SLIC
  // codelets have been written in assembly this estimate should be updated.
  // also, any unavoidable register bubbles have not been accounted for.
  std::uint64_t cycles = 0;

  // store $m9 and $m10 on the stack.
  cycles += 4;

  // load in, weights and worklists pointers and expand from scaled pointers
  // (setzi mem_base ; add ; shl) then store on the stack.
  cycles += 1 + 3 * 4;

  // load and store in chans per group and out chans per group.
  cycles += 4;

  // load num conv groups, in groups, num out groups, num weight blocks, num out
  // elems and first set of weights.
  cycles += 7;

  // setzi and runall the zero worker.
  cycles += 7 + zeroCycles;

  // setzi the slic kernel.
  cycles += 1;

  // for each weight block.
  std::uint64_t weightBlockLoop = 0;

  // load a quarter of the CWEI registers
  weightBlockLoop += slicWindowWidth * 2;

  // runall
  weightBlockLoop += 7 + innerLoopCycles;

  // move weights pointer on and brnzdec
  weightBlockLoop += 2;

  // for each out chan group
  std::uint64_t outChanGroupLoop = 0;

  // load next weights pointer, expand. load next out pointer, expand. store
  // out chan pointer to vertex state.
  outChanGroupLoop += 5;

  // perform weight block loop then brnzdec
  outChanGroupLoop += weightBlockLoop * numWeightBlocks + 1;

  // for each in chan group + brnzdec.
  std::uint64_t inChanGroupLoop = outChanGroupLoop * numOutGroups + 1;

  // for each conv group + brnzdec.
  std::uint64_t convGroupLoop = inChanGroupLoop * numInGroups + 1;

  cycles += convGroupLoop * numConvGroups;

  // restore m9, m10, sp and finally br $lr.
  cycles += 4;

  return cycles;
}

inline std::uint64_t getConvPartialSlicSupervisorCycleInnerLoopEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned slicWindowWidth, bool floatActivations,
    bool floatPartials) {
  // TODO: we currently only target SLIC for half->float.
  assert(!floatActivations);
  assert(floatPartials);

  // TODO: roughly based off of the ConvPartial1x1 codelet, when the SLIC
  // codelets have been written in assembly this estimate should be updated.
  std::uint64_t maxWorkerCycles = 0;

  // worker partitions is indexed by [worker][partitions].
  // TODO: should there only be a single partition? like the ConvPartial1x1.
  for (const auto &worker : workerPartitions) {
    std::uint64_t cycles = 0;

    // extract the worker id and turn it into a 64-bit offset (get ; and ; mul)
    cycles += 3;

    // load the worklist, in, out, in and out chans per group vertex state.
    cycles += 5;

    // extract the in offset, out offset and num elems from the worklist. scale
    // the offsets up by the number of channels per group.
    cycles += 5;

    // move the in and out pointers on by the offset.
    cycles += 2;

    // partition loop.
    for (const auto numElems : worker) {
      // warm-up and cool-down based off of the table in section 3.7.4.3.35 of
      // the arch man.

      // warm-up
      cycles += slicWindowWidth - 1;

      // rpt + tapack
      cycles += 2;

      // main inner loop, the rpt loop body is a single bundle (ld2xst ; slic)
      cycles += numElems - (slicWindowWidth - 1);

      // cool-down
      cycles += slicWindowWidth - 1;

      // brnzdec num partitions
      cycles += 1;
    }

    // exitz
    cycles += 1;

    maxWorkerCycles = std::max(maxWorkerCycles, cycles);
  }

  // transform worker cycles into supervisor cycles.
  return maxWorkerCycles * numWorkerContexts;
}

inline std::uint64_t getMatMul2CycleEstimate(unsigned size) {
  // Inner loop is dominated by loads (load pointer, load 64bits, load 16
  // bits). This could be improved if we uses strided loads instead of
  // pointers.
  return 5 + size * 3;
}

inline uint64_t getWgdDataTransformCycles(unsigned numChannels, bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 13 + 56 * ((numChannels + chansPerOp - 1) / chansPerOp);
}

inline uint64_t getWgdKernelTransformCycles(unsigned numChannels,
                                            bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 2 + 35 * ((numChannels + chansPerOp - 1) / chansPerOp);
}

inline uint64_t getWgdInvTransformCycles(unsigned numChannels, bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 15 + 30 * ((numChannels + chansPerOp - 1) / chansPerOp);
}

/**
 * The accumulator operates on pencils which are of depth "pencilDepth".
 * An inner product of a coefficient vector and data vector is computed.
 * "comPencils" gives the number of pencils which share a common coefficient
 * vector. "numPencils" gives a set of pencils which share common coefficients
 */
inline uint64_t getWgdAccumCycles(unsigned numPencils, unsigned comPencils,
                                  unsigned pencilDepth, unsigned outDepth,
                                  unsigned numWorkers, unsigned numConvUnits,
                                  unsigned weightsPerConvUnit,
                                  unsigned convUnitCoeffLoadBytesPerCycle,
                                  bool isFloat) {

  unsigned numCoeffSets = (outDepth + numConvUnits - 1) / numConvUnits;
  numCoeffSets *= (pencilDepth + weightsPerConvUnit - 1) / weightsPerConvUnit;
  numCoeffSets *= numPencils;
  const auto coeffLoadCycles = numConvUnits * weightsPerConvUnit *
                               (isFloat ? 2 : 4) /
                               convUnitCoeffLoadBytesPerCycle;
  const auto overhead = 4;

  const auto numPencilsPerWorker = (comPencils + numWorkers - 1) / numWorkers;
  return (overhead + coeffLoadCycles + numPencilsPerWorker * numWorkers * 4) *
         numCoeffSets;
}

inline uint64_t getWgdReduceCycles(unsigned numPencils, unsigned depth,
                                   bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 5 + ((numPencils * depth + chansPerOp - 1) / chansPerOp);
}

inline uint64_t getWgdCompleteCycles(unsigned numChannels, bool isFloat) {
  unsigned divFactor = isFloat ? 2 : 4;

  return 5 + numChannels / divFactor;
}

inline std::uint64_t getOuterProductCycleEstimate(bool isFloat, unsigned width,
                                                  unsigned numChannels,
                                                  unsigned chansPerGroup,
                                                  unsigned dataPathWidth) {
  assert(numChannels % chansPerGroup == 0);
  const auto numChanGroups = numChannels / chansPerGroup;

// TODO T14719: Derive this from IPUArchInfo
#define CSR_W_REPEAT_COUNT__VALUE__MASK 0x0FFF
  auto const hardwareRptCountConstraint = CSR_W_REPEAT_COUNT__VALUE__MASK + 1;

  int cycles;
  // Conditions for executing a fast or slow path, replicated from the assembly
  // implementation
  if (isFloat) {
    if ((chansPerGroup >= 6) &&       // Min size of unrolled loop
        ((chansPerGroup & 1) == 0) && // Loop processes 2 at once
        ((chansPerGroup / 2 - 3) < hardwareRptCountConstraint) &&
        ((chansPerGroup / 2 + 1) < 512)) { // Stride size constraint

      // Float, Fast path cycle estimates
      cycles =
          25 + numChanGroups * (11 + width * (6 + (chansPerGroup - 6) / 2));
    } else {
      // Float, Slow path cycle estimates
      cycles = 25 + numChanGroups * (11 + width * (10 + chansPerGroup * 2));
    }
  } else {
    if ((chansPerGroup >= 12) &&      // Min size of unrolled loop
        ((chansPerGroup & 3) == 0) && // Loop processes 2 at once
        ((chansPerGroup / 4 - 3) < hardwareRptCountConstraint) &&
        ((chansPerGroup / 4 + 1) < 512)) { // Stride size constraint

      // Half, Fast path cycle estimates
      cycles =
          25 + numChanGroups * (10 + width * (6 + (chansPerGroup - 12) / 4));
    } else {
      // Half, Slow path cycle estimates
      cycles =
          25 + numChanGroups * (10 + width * (10 + (chansPerGroup * 5) / 2));
    }
  }
  return cycles;
}

inline uint64_t getReduceCycleEstimate(unsigned outSize, unsigned partialsSize,
                                       unsigned dataPathWidth,
                                       bool isOutTypeFloat,
                                       bool isPartialsFloat,
                                       unsigned numWorkers) {
  unsigned cycles = 0;

  // Supervisor vertex, and new implementation
  if (isPartialsFloat) {
    cycles = 32;
    // Float - workers process 4 at once, and account for remainder loops
    auto loops = outSize / 4;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers = loops / numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 3)
        cycles += (2 * partialsSize + 13);
      else
        loopsDividedBetweenWorkers++;
    }

    if (isOutTypeFloat)
      cycles += (3 * partialsSize + 7) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 6) * loopsDividedBetweenWorkers;
  } else {
    cycles = 32;
    // Half - workers process 8 at once, and account for remainder loops
    auto loops = outSize / 8;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    if (outSize & 4)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers = loops / numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 7)
        cycles += (2 * partialsSize + 11);
      else
        loopsDividedBetweenWorkers++;
    }

    if (isOutTypeFloat)
      cycles += (3 * partialsSize + 9) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 8) * loopsDividedBetweenWorkers;
  }
  cycles = cycles * numWorkers;

  return cycles;
}

#endif // _performance_estimation_h_
