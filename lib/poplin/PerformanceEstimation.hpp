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
    return std::make_pair(15, 3);
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
    unsigned numOutGroups, unsigned numWorkers, bool isFloat) {
  uint64_t cycles = innerLoopCycles;
  return convHorizontalMacOverhead(isFloat) +
         numWorkers * zeroPartialsRetentionSavings(/* floatPartials */ true) +
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
      cycles, numConvGroups, numInGroups, numOutGroups, numWorkerContexts,
      isFloat);
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
          thisWorkerCycles += 28;
        } else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 26 : 29);
          } else {
            thisWorkerCycles += 28;
          }
        }
        break;
      case 1:
        if (floatActivations)
          thisWorkerCycles += 50 + (2 + 8) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 39 : 43);
          } else {
            thisWorkerCycles += 43 + (2 + zeroCyclesPerGroup) * outputZeroing;
          }
        }
        break;
      case 2:
        if (floatActivations)
          thisWorkerCycles += 50 + (2 + 8 * 2) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += (outputZeroing ? 41 : 45);
          } else {
            thisWorkerCycles +=
                44 + (2 + zeroCyclesPerGroup * 2) * outputZeroing;
          }
        }
        break;
      default:
        if (floatActivations)
          thisWorkerCycles += 50 + (2 + 8 * numElems) * outputZeroing +
                              (numElems - 3) * coreCycles;
        else {
          if (floatPartials) {
            thisWorkerCycles +=
                (outputZeroing ? 41 : 44) + (numElems - 3) * coreCycles;
          } else {
            thisWorkerCycles +=
                45 + (2 + zeroCyclesPerGroup * numElems) * outputZeroing +
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
              thisWorkerCycles += 18;
              break;
            case 1:
              thisWorkerCycles += (floatActivations ? 34 : 30);
              break;
            case 2:
              thisWorkerCycles += (floatActivations ? 45 : 34);
              break;
            default:
              if (floatActivations)
                thisWorkerCycles += 46 + (numElems - 3) * coreCycles;
              else
                thisWorkerCycles += 35 + (numElems - 3) * coreCycles;
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

inline std::uint64_t getConvPartialSlicSupervisorCycleWeightLoadEstimate(
    unsigned convGroupsPerGroup, unsigned chansPerGroup,
    unsigned numWorkerContexts, unsigned slicWindowWidth) {
  assert(slicWindowWidth == 4u);
  assert(chansPerGroup == 4u / convGroupsPerGroup);
  std::uint64_t cycles = 0;
  if (convGroupsPerGroup == 1) {
    cycles += (6 + // brnzdec
               6 + // put CCCSLOAD
               6); // bri
  } else {
    assert(convGroupsPerGroup == 4 || convGroupsPerGroup == 2);
    const std::uint64_t workerLoadWeightsCycles =
        (convGroupsPerGroup == 4) ? 10 : 14;
    cycles += (9 + // brnzdec, put CCCSLOAD pointer (stall), store weights
                   // pointer for rearrangement.
               6 + // runall
               // Rearrange weights in workers
               (workerLoadWeightsCycles * numWorkerContexts) + 6); // sync
  }
  cycles += 16; // 16 * ld64putcs
  return cycles;
}

inline std::uint64_t getConvPartialSlicSupervisorCycleOuterLoopEstimate(
    std::uint64_t implicitZeroingInnerLoopCycles, std::uint64_t innerLoopCycles,
    std::uint64_t weightLoadCycles, unsigned numConvGroupGroups,
    unsigned numSubKernels, unsigned slicWindowWidth, bool floatActivations,
    bool floatPartials) {
  // TODO: we currently only target SLIC for half->float
  // TODO: we currently only target a kernel width of 4.
  assert(!floatActivations);
  assert(floatPartials);
  assert(slicWindowWidth == 4);
  assert(numConvGroupGroups >= 1);
  assert(numSubKernels >= 1);

  const std::uint64_t supervisorPreambleCycles = 28;
  const std::uint64_t supervisorConvGroupGroupsBodyCycles = 15;
  const std::uint64_t supervisorConvGroupGroupsLoopCycles =
      supervisorConvGroupGroupsBodyCycles * numConvGroupGroups +
      6 * (numConvGroupGroups - 1) +
      1; // 6 cycles brnzdec stall for all but last conv group group
  const std::uint64_t supervisorSubKernelBodyCycles =
      weightLoadCycles + 3 + // deal with whether to swap output pointers or not
      2 +                    // store new worklist pointer and increment
      2 +                    // or, store implicit zero/stride
      6 +                    // runall
      6 +                    // sync
      1;                     // load new weights pointer

  const std::uint64_t supervisorSubKernelLoopCycles =
      supervisorSubKernelBodyCycles * numSubKernels + 6 * (numSubKernels - 1) +
      1; // brnzdec is 6 cycles in all but the last iteration.

  const std::uint64_t cycles =
      supervisorPreambleCycles + supervisorConvGroupGroupsLoopCycles +
      supervisorSubKernelLoopCycles +
      // Workers make one pass for the first sub-kernel implicitly zeroing
      // partials, and the remainder of the sub-kernels not implicitly zeroing.
      (numConvGroupGroups * implicitZeroingInnerLoopCycles +
       numConvGroupGroups * (numSubKernels - 1) * innerLoopCycles);

  return cycles;
}

// This gives us the number of cycles in terms of supervisor cycles
// for all workers to process a single conv group/sub-kernel. There is
// a strong assumption that the amount of work is always the same between
// sub-kernels.
inline std::uint64_t getConvPartialSlicSupervisorCycleInnerLoopEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned slicWindowWidth, bool floatActivations,
    bool floatPartials, unsigned outputStride, bool implicitZeroing) {
  // TODO: we currently only target SLIC for half->float.
  // TODO: we currently only target kernel width of 4.
  assert(!floatActivations);
  assert(floatPartials);
  assert(slicWindowWidth == 4);

  std::uint64_t maxWorkerCycles = 0;

  const std::uint64_t workerProcessGroupPreambleCycles =
      2 + // Get worker ID
      3 + // Load and maybe switch output pointers
      1 + // Load input pointer
      2 + // Load worklist DeltaN for worker
      4 + // Unpack DeltaN
      2 + // Load base pointer for DeltaN and add to form final worklist pointer
      2 + // Divide number of work items in the list by 3
      1;  // Load implicit zero flag + strides from stack
  // worker partitions is indexed by [worker][partitions].
  for (const auto &worker : workerPartitions) {
    std::uint64_t workerCycles = workerProcessGroupPreambleCycles;

    for (const auto &numFieldElems : worker) {
      workerCycles += 10; // Pre-amble, brnzdec
      if (implicitZeroing) {
        workerCycles += 1; // Extra branch to exit
      }
      std::uint64_t rowCycles = 0;
      if (outputStride == 1) {
        if (numFieldElems < 5) {
          if (implicitZeroing) {
            rowCycles += 10 + (numFieldElems > 1 ? numFieldElems : 0) + 3;
          } else {
            rowCycles += 7;
            if (numFieldElems == 1) {
              rowCycles += 6;
            } else {
              rowCycles += 1 + (numFieldElems - 1) + 2 + (4 - numFieldElems) +
                           2 + (3 - (4 - numFieldElems)) + 3;
            }
          }
        } else {
          // Cycles for > 5 field elements matches for implicit
          // zeroing vs. normal
          rowCycles += 15 + (numFieldElems - 5);
        }
      } else {
        // outputStride == 2
        if (numFieldElems < 3) {
          // Cycles for > 3 field elements matches for implicit
          // zeroing vs. normal
          rowCycles += 7 + (numFieldElems == 1 ? 3 : 5) + 3;
        } else {
          // Cycles for < 3 field elements matches for implicit
          // zeroing vs. normal
          rowCycles += 15 + 2 * (numFieldElems - 3);
        }
      }
      // 2 passes over input data
      workerCycles += 3 + rowCycles * 2;
    }

    maxWorkerCycles = std::max(maxWorkerCycles, workerCycles);
  }

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
