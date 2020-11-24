// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "ConvReducePlan.hpp"
#include "ConvUtilInternal.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <poplar/Target.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/ConvUtil.hpp>
#include <utility>
#include <vector>

using namespace poplibs_support;

inline static std::uint64_t convHorizontalMacOverhead(bool floatActivations) {
  return floatActivations ? 58 : 63;
}

inline static std::uint64_t convNx1Overhead() { return 103; }

// Number of worker cycle savings if state retention is used.
// The first entry is the total savings and the second is
// because of retention of state related to input channel processing.
inline static std::pair<std::uint64_t, std::uint64_t>
conv1x1WorkerRetentionSavings(bool floatActivations, bool floatPartials) {
  if (floatActivations == false && floatPartials == true) {
    return std::make_pair(10, 2);
  } else {
    return std::make_pair(0, 0);
  }
}

inline static std::uint64_t
convnx1WorkerRetentionSavings(bool /*floatActivations */,
                              bool /*floatPartials */) {
  return 6;
}

inline static std::uint64_t zeroPartialsRetentionSavings(bool floatPartials) {
  return floatPartials ? 9 : 10;
}

inline std::uint64_t getDenseDotProductCycles(bool floatActivations,
                                              bool floatPartials,
                                              unsigned size) {
  const auto innerCycles = 1 + // rpt
                           2 + // loop wind down
                           3 + // sum with previous partials (load, acc, store)
                           1;  // branch

  // float activations and float partials
  if (floatActivations) {
    if ((size % 2) == 0) {
      return innerCycles + size;
    } else {
      return innerCycles + (2 * size);
    }
  }

  // half activations and float partials
  if (floatPartials) {
    if ((size % 4) == 0) {
      return innerCycles + size / 4;
    } else {
      return innerCycles + size;
    }
  }

  // half activations and half partials
  if ((size % 4) == 0) {
    const auto innerCyclesv4 =
        2 * (1 + 2) + // rpt + loop wind down(macros)
        1 +           // f2h conversion (packing) (1)
        3 +           // sum with previous partials (load, acc, store)
        1;            // branch
    return innerCyclesv4 + size / 4;
  } else {
    const auto innerCyclesv2 =
        2 +           // weights load
        2 * (1 + 2) + // rpt + loop wind down
        3 + // results combine, sum with previous partials (load, acc, store)
        1;  // branch
    return innerCyclesv2 + size;
  }
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
    bool floatActivations, bool floatPartials, unsigned numInChans,
    unsigned numOutChans, const std::vector<unsigned> &convSizes) {
  uint64_t cycles = 16;
  for (auto convSize : convSizes) {
    if (convSize == 0) {
      cycles += 7;
    } else {
      if (!floatPartials) {
        numOutChans /= 2; // Processing two channels inside inner loop
      }
      cycles += 19;
      cycles += convSize * (7 + numOutChans * getDenseDotProductCycles(
                                                  floatActivations,
                                                  floatPartials, numInChans));
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
    unsigned numOutChansPerGroup, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials) {
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
          floatActivations, floatPartials, numInChansPerGroup,
          numOutChansPerGroup, workerPartitions[context][k]);
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
    unsigned numOutGroups, unsigned numWorkers, bool floatActivations,
    bool floatPartials) {
  uint64_t cycles = innerLoopCycles;
  return convHorizontalMacOverhead(floatActivations) +
         numWorkers * zeroPartialsRetentionSavings(floatPartials) +
         numConvGroups *
             (23 + numInGroups * (15 + numOutGroups * (10 + cycles)));
}

inline std::uint64_t getConvPartialHorizontalMacSupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned kernelSize, unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials) {
  auto innerLoopCycles =
      getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
          workerPartitions, kernelSize, numInChansPerGroup, numOutChansPerGroup,
          numWorkerContexts, floatActivations, floatPartials);
  auto cycles = getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
      innerLoopCycles, numConvGroups, numInGroups, numOutGroups,
      numWorkerContexts, floatActivations, floatPartials);
  return cycles;
}

inline std::uint64_t getVerticalMacDotProductCycles(bool floatActivations,
                                                    bool floatPartials,
                                                    unsigned size,
                                                    unsigned numChannels) {
  assert(!floatActivations && floatPartials);
  const auto innerCyclesOverhead = 5;
  return innerCyclesOverhead + (2 * (size - 1));
}

inline std::uint64_t getConvPartialVerticalMacCycleEstimate(
    bool floatActivations, bool floatPartials, unsigned convGroupsPerGroup,
    const std::vector<unsigned> &convSizes) {
  uint64_t cycles =
      6; // 10 (reload state)  + 1 (bri) - 5 (cmpn, brz, Store-Acc);
  for (auto convSize : convSizes) {
    cycles += 10;
    cycles += 5; // Cycles to store accumulators and then reload. Note that
                 // this is an overestimate since these cycles should only be
                 // incurred  when the output differs from that of the
                 // previous worklist.
    auto dotProdCycles = getVerticalMacDotProductCycles(
        floatActivations, floatPartials, convSize, convGroupsPerGroup);
    cycles += dotProdCycles;
  }
  return cycles;
}

inline std::uint64_t
getConvPartialVerticalReductionCycleEstimate(unsigned numElems,
                                             unsigned numWorkers) {
  const auto cyclesPerRpt = 2;
  return 10 + ((9 + (cyclesPerRpt * (numWorkers - 1))) * numElems / 4);
}

inline std::uint64_t getConvPartialVerticalMacSupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned kernelHeight, unsigned convGroupsPerGroup,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {
  unsigned usedContexts = workerPartitions.size();
  uint64_t cycles = 0;
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  for (auto context = 0U; context != usedContexts; ++context) {
    uint64_t thisWorkerCycles = getConvPartialVerticalMacCycleEstimate(
        floatActivations, floatPartials, convGroupsPerGroup,
        workerPartitions[context]);
    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }
  cycles += std::max(maxWorkerCycles, minWorkerCycles);
  return cycles;
}

// VMAC Zeroing is done on all the partials for each worker. Hence this
// function does not require the number of workers to be passed as an argument.
inline std::uint64_t
getConvPartialVerticalMacSupervisorZeroInnerLoopCycleEstimate(
    unsigned numOutElems) {
  return 4 + numOutElems;
}

inline std::uint64_t
getConvPartialVerticalMacSupervisorReductionInnerLoopCycleEstimate(
    unsigned numOutElems, unsigned numWorkerContexts) {
  auto numElemsPerWorker =
      (numOutElems + numWorkerContexts - 1) / numWorkerContexts;
  uint64_t cycles = getConvPartialVerticalReductionCycleEstimate(
      numElemsPerWorker, numWorkerContexts);
  return cycles;
}

inline std::uint64_t getConvPartialVerticalMacSupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCycles, std::uint64_t zeroInitInnerCycles,
    std::uint64_t reductionInnerCycles, unsigned numConvGroups,
    unsigned numInGroups) {
  const auto supOverheadCycles = 61;
  const auto wkrCoreVMACInit = 13;
  const auto wkrStateRetentionInit = 26;
  auto outerLoopCycles = 32 + numInGroups * (24 + innerLoopCycles);
  return supOverheadCycles + wkrCoreVMACInit + wkrStateRetentionInit +
         (zeroInitInnerCycles + outerLoopCycles + reductionInnerCycles) *
             numConvGroups;
}

inline std::uint64_t getConvPartialVerticalMacSupervisorCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned kernelHeight, unsigned numOutElems, unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup, unsigned convGroupsPerGroup,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {
  assert(numOutGroups == 1);
  assert(numInChansPerGroup == 1);
  assert(numOutChansPerGroup == 1);
  uint64_t cycles = 0;
  auto innerLoopCycles =
      getConvPartialVerticalMacSupervisorInnerLoopCycleEstimate(
          workerPartitions, kernelHeight, convGroupsPerGroup, numWorkerContexts,
          floatActivations, floatPartials);
  auto zeroInitCycles =
      getConvPartialVerticalMacSupervisorZeroInnerLoopCycleEstimate(
          numOutElems);
  auto reductionCycles =
      getConvPartialVerticalMacSupervisorReductionInnerLoopCycleEstimate(
          numOutElems, numWorkerContexts);
  cycles += getConvPartialVerticalMacSupervisorOuterLoopCycleEstimate(
      innerLoopCycles, zeroInitCycles, reductionCycles, numConvGroups,
      numInGroups);
  return cycles;
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
          thisWorkerCycles += 24;
        } else {
          if (floatPartials) {
            thisWorkerCycles += 20 + (outputZeroing ? 2 : 4);
          } else {
            thisWorkerCycles += 24;
          }
        }
        break;
      case 1:
        if (floatActivations)
          thisWorkerCycles += 47 + (2 + zeroCyclesPerGroup) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += 31 + (outputZeroing ? 2 : 4);
          } else {
            thisWorkerCycles += 39 + (2 + zeroCyclesPerGroup) * outputZeroing;
          }
        }
        break;
      case 2:
        if (floatActivations)
          thisWorkerCycles += 46 + (2 + zeroCyclesPerGroup * 2) * outputZeroing;
        else {
          if (floatPartials) {
            thisWorkerCycles += 40 + (outputZeroing ? 2 : 4);
          } else {
            thisWorkerCycles +=
                40 + (2 + zeroCyclesPerGroup * 2) * outputZeroing;
          }
        }
        break;
      default:
        if (floatActivations)
          thisWorkerCycles +=
              46 + (2 + zeroCyclesPerGroup * numElems) * outputZeroing +
              (numElems - 3) * coreCycles;
        else {
          if (floatPartials) {
            thisWorkerCycles +=
                (outputZeroing ? 38 : 40) + (numElems - 3) * coreCycles;
          } else {
            thisWorkerCycles +=
                41 + (2 + zeroCyclesPerGroup * numElems) * outputZeroing +
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

inline unsigned getConvPartialAmpSupervisorWeightLoadCycleEstimate(
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, bool floatActivations,
    unsigned filterHeight) {

  // Nx1 specific - due to data shuffling can't use ld128 for filter height
  // equal to 4 so it's always uses ld64. ld128 allows to us to
  // load 16 bytes per cycle hence convUnitCoeffLoadBytesPerCycle needs to be
  // halved
  if (filterHeight == 4 && convUnitCoeffLoadBytesPerCycle > 8) {
    convUnitCoeffLoadBytesPerCycle /= 2;
  }
  const unsigned weightsLoadCycles =
      (numConvUnits * weightBytesPerConvUnit) / convUnitCoeffLoadBytesPerCycle;
  return weightsLoadCycles;
}

inline std::uint64_t getConvPartial1x1SupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing, unsigned numConvGroups,
    unsigned numInGroups, unsigned numOutGroups, unsigned outChansPerGroup,
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, bool floatActivations,
    bool floatPartials, unsigned numWorkerContexts) {
  const auto outputPassesPerGroup =
      (outChansPerGroup + numConvUnits - 1) / numConvUnits;

  const auto retentionSavings =
      conv1x1WorkerRetentionSavings(floatActivations, floatPartials);

  // Filter height is not applicable to 1x1 vertex so set it to 1
  const auto numLoads = getConvPartialAmpSupervisorWeightLoadCycleEstimate(
      weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
      floatActivations, 1);

  const uint64_t supervisorNonloopOverhead = 50;
  const unsigned outPassesOverhead = 7;
  const unsigned excessInChanOverhead = 1;
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
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
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
      numInGroups, numOutGroups, outChansPerGroup, weightBytesPerConvUnit,
      numConvUnits, convUnitCoeffLoadBytesPerCycle, floatActivations,
      floatPartials, numWorkerContexts);
}

inline std::uint64_t getConvPartialnx1SupervisorOuterLoopCycleEstimate(
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
      numConvGroups * (numOutGroups * 17 + (numOutGroups - 1) * 6 + 1) +
      (numConvGroups - 1) * 6 + 1 +
      // Supervisor code loop over conv/in/out groups
      numConvGroups * (16 + numInGroups * (14 + numOutGroups * (14 + cycles)));
}

inline std::uint64_t getConvPartialnx1SupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {
  // Core loop cycles for vertex will all engines in use
  auto coreCycles = floatActivations ? 8 : 4;
  // when using half of AMP engines need to reduce core cycles as well
  if (numConvUnits == 4) {
    coreCycles /= 2;
  }

  const auto retentionSavings =
      convnx1WorkerRetentionSavings(floatActivations, floatPartials);
  unsigned usedContexts = workerPartitions.size();
  unsigned numOutChanPasses = outChansPerGroup / numConvUnits;

  // innermostLoopCycles is the cycles in the innermost supervisor loop
  uint64_t innermostLoopCycles =
      getConvPartialAmpSupervisorWeightLoadCycleEstimate(
          weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
          floatActivations, filterHeight);

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

  innermostLoopCycles += 3;

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
          uint64_t thisWorkerCycles = 17 + extraCycles;
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
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials) {
  auto innerLoopCycles = getConvPartialnx1SupervisorInnerLoopCycleEstimate(
      workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatActivations,
      floatPartials);
  return getConvPartialnx1SupervisorOuterLoopCycleEstimate(
      innerLoopCycles, numConvGroups, numOutGroups, numInGroups,
      outChansPerGroup, numConvUnits, numWorkerContexts, floatActivations,
      floatPartials);
}

inline std::uint64_t getConvPartialSlicSupervisorWeightLoadCycleEstimate(
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
        (convGroupsPerGroup == 4) ? 10 : 12;
    cycles += (9 + // brnzdec, put CCCSLOAD pointer (stall), store weights
                   // pointer for rearrangement.
               6 + // runall
               // Rearrange weights in workers
               (workerLoadWeightsCycles * numWorkerContexts) + 6); // sync
  }
  cycles += 16; // 16 * ld64putcs
  return cycles;
}

inline std::uint64_t getConvPartialSlicSupervisorOuterLoopCycleEstimate(
    std::uint64_t implicitZeroingInnerLoopCycles, std::uint64_t innerLoopCycles,
    std::uint64_t weightLoadCycles, unsigned numConvGroupGroups,
    unsigned numSubKernels, unsigned numConvUnits, unsigned slicWindowWidth,
    bool floatActivations, bool floatPartials) {

  // TODO: we currently only target a kernel width of 4.
  assert(!floatActivations);
  assert(slicWindowWidth == 4);
  assert(numConvGroupGroups >= 1);
  assert(numSubKernels >= 1);

  // Similar, but different function for the 8 convUnits, half partials case
  const bool half8Conv = (numConvUnits == 8 && floatPartials == false);

  const std::uint64_t supervisorPreambleCycles = half8Conv ? 25 : 28;
  const std::uint64_t supervisorConvGroupGroupsBodyCycles = half8Conv ? 12 : 15;
  const std::uint64_t supervisorConvGroupGroupsLoopCycles =
      supervisorConvGroupGroupsBodyCycles * numConvGroupGroups +
      6 * (numConvGroupGroups - 1) +
      1; // 6 cycles brnzdec stall for all but last conv group group
  const std::uint64_t supervisorSubKernelBodyCycles =
      weightLoadCycles +
      (half8Conv ? 0 : 3) + // deal with whether to swap output pointers or not
      2 +                   // store new worklist pointer and increment
      (half8Conv ? 0 : 1) + // or, store implicit zero/stride
      6 +                   // runall
      6 +                   // sync
      1;                    // load new weights pointer

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
inline std::uint64_t getConvPartialSlicSupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvUnits, unsigned slicWindowWidth,
    bool floatActivations, bool floatPartials, unsigned outputStride,
    bool implicitZeroing) {
  // TODO: we currently only target kernel width of 4.
  assert(!floatActivations);
  assert(slicWindowWidth == 4);

  const unsigned inputDataPasses = numConvUnits == 16 ? 1 : 2;
  // Similar, but different function for the 8 convUnits, half partials case
  const bool half8Conv = (numConvUnits == 8 && floatPartials == false);
  const unsigned loopDecisionThreshold = (half8Conv ? 6 : 5);

  std::uint64_t maxWorkerCycles = 0;

  const std::uint64_t workerProcessGroupPreambleCycles =
      2 +                   // Get worker ID
      (half8Conv ? 2 : 3) + // Load and maybe switch output pointers
      1 +                   // Load input pointer
      2 +                   // Load worklist DeltaN for worker
      4 +                   // Unpack DeltaN
      2 + // Load base pointer for DeltaN and add to form final worklist pointer
      2 + // Divide number of work items in the list by 3
      1 + // Load implicit zero flag + strides from stack
      (half8Conv ? 1 : 0); // Implicit zero loop decision
  // worker partitions is indexed by [worker][partitions].
  std::uint64_t cumulativeFieldElems = 0;
  for (const auto &worker : workerPartitions) {
    std::uint64_t workerCycles = workerProcessGroupPreambleCycles;

    for (const auto &numFieldElems : worker) {
      workerCycles += (half8Conv ? 9 : 10); // Pre-amble, brnzdec
      if (implicitZeroing) {
        workerCycles += 1; // Extra branch to exit
      }
      std::uint64_t rowCycles = 0;

      if (outputStride == 1) {
        if (numFieldElems < loopDecisionThreshold) {
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
          if (implicitZeroing) {
            rowCycles += 15 + (numFieldElems - 5);
          } else {
            // Account for decisions on numFieldElements in half8Conv loop
            rowCycles += 15 + (numFieldElems - 5) + (half8Conv ? 3 : 0);
          }
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

      // For float partials, dummy dual load is used to incrememnt pointers
      if (floatPartials) {
        rowCycles -= 1;
      }

      // Account for the passes over input data
      workerCycles += (floatPartials ? 3 : 0) + rowCycles * inputDataPasses;
      // Count field elems total so we can account for the merging copy
      cumulativeFieldElems += numFieldElems;
    }
    // Account for the copy to merge the 2 outputs. Decision only
    workerCycles += (half8Conv ? 2 : 0);
    maxWorkerCycles = std::max(maxWorkerCycles, workerCycles);
  }
  // So far we have the total max cycles for any worker for all the work which
  // can be spread over many sub kernels.  Only on one pass (of 8 conv, half
  // vertex) will workers merge the 2 outputs together (When the last sub kernel
  // is used). Here we add the cycles to account for this on one pass - the
  // pass where implicit zeroing is used
  const std::uint64_t copyCycles =
      (half8Conv && implicitZeroing) ? (2 + 2 * cumulativeFieldElems) : 0;
  return maxWorkerCycles * numWorkerContexts + copyCycles;
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
                                       bool isPartialsFloat, bool singleInput,
                                       bool constrainPartials,
                                       unsigned numWorkers) {
  const unsigned accVectorWidth = isPartialsFloat ? 4 : 8;
  const unsigned partialsTypeSize = isPartialsFloat ? 4 : 2;
  const unsigned partialsInDataPathWidth =
      dataPathWidth / (partialsTypeSize * 8);

  if (singleInput) {
    unsigned supervisorCycles = 33;
    // If we use interleaved mem for partials, we can achieve double the
    // load bandwidth.
    const unsigned loadElemsPerCycle =
        partialsInDataPathWidth * (1 + constrainPartials);
    assert(loadElemsPerCycle % accVectorWidth == 0 ||
           accVectorWidth % loadElemsPerCycle == 0);
    // Simpler optimised vertex, 1 or 2 cycle inner loop
    const auto cyclesPerLoop =
        accVectorWidth / std::min(accVectorWidth, loadElemsPerCycle);
    const auto loops = outSize / accVectorWidth;
    auto loopsDividedBetweenWorkers = loops / numWorkers;
    if (loops % numWorkers) {
      loopsDividedBetweenWorkers++;
    }
    unsigned cycles = 20;
    unsigned outerLoopOverHead;
    if (isPartialsFloat) {
      outerLoopOverHead = isOutTypeFloat ? 8 : 7;
    } else {
      outerLoopOverHead = isOutTypeFloat ? 10 : 9;
    }
    cycles += (cyclesPerLoop * partialsSize + outerLoopOverHead) *
              loopsDividedBetweenWorkers;
    return cycles * numWorkers + supervisorCycles;
  }

  // Supervisor vertex, and new implementation
  unsigned cycles = 32;
  assert(partialsInDataPathWidth % accVectorWidth == 0 ||
         accVectorWidth % partialsInDataPathWidth == 0);
  // Cycles per loop is calculated based on processing accVectorWidth
  // per-loop, so assume 1 cycle for accumulator op, then load bandwidth
  // is limiting factor. We take the data path width for load bandwidth
  // because we do not use interleaved memory here and + 1 for cycle to
  // load next pointer because it is not a single input.
  const auto cyclesPerLoop =
      (accVectorWidth / std::min(accVectorWidth, partialsInDataPathWidth)) + 1;
  auto loops = outSize / accVectorWidth;
  for (unsigned i = 0; i < ceilLog2(accVectorWidth); ++i) {
    if (outSize & (1u << i)) {
      loops++;
    }
  }
  // Account for time at full load - all workers busy
  auto loopsDividedBetweenWorkers = loops / numWorkers;
  // and a remainder where only some are busy which can be a shorter loop
  if (loops % numWorkers) {
    if (outSize & (accVectorWidth - 1))
      cycles += (2 * partialsSize + (isPartialsFloat ? 13 : 11));
    else
      loopsDividedBetweenWorkers++;
  }

  // TODO: Should this account for store bandwidth/accumulator output vector
  // width?
  if (isOutTypeFloat) {
    cycles += (cyclesPerLoop * partialsSize + (isPartialsFloat ? 7 : 9)) *
              loopsDividedBetweenWorkers;
  } else {
    cycles += (cyclesPerLoop * partialsSize + (isPartialsFloat ? 6 : 8)) *
              loopsDividedBetweenWorkers;
  }

  return cycles * numWorkers;
}

namespace poplin {

inline std::uint64_t getNumberOfMACs(const poplin::ConvParams &params) {
  std::uint64_t numMACs = params.getNumConvGroups() * params.getBatchSize() *
                          params.getNumOutputChansPerConvGroup() *
                          params.getNumInputChansPerConvGroup();
  for (unsigned dim = 0; dim != params.getNumFieldDims(); ++dim) {
    unsigned fieldMACs = 0;
    auto kernelSize = params.kernelShape[dim];
    auto kernelTruncationLower = params.kernelTransform.truncationLower[dim];
    auto kernelTruncationUpper = params.kernelTransform.truncationUpper[dim];
    auto outputSize = params.getOutputSize(dim);
    auto outputStride = params.outputTransform.stride[dim];
    auto inputDilation = params.inputTransform.dilation[dim];
    // For a fixed kernel index the distance between elements in the output
    // whose calculation involves that kernel index.
    auto MACStride = lcm(outputStride, inputDilation) / outputStride;
    for (unsigned k = kernelTruncationLower;
         k != kernelSize - kernelTruncationUpper; ++k) {
      auto outRange =
          getOutputRangeForKernelIndex(dim, {0, outputSize}, k, params);
      auto outRangeSize = outRange.second - outRange.first;
      fieldMACs += (outRangeSize + MACStride - 1) / MACStride;
    }
    numMACs *= fieldMACs;
  }
  return numMACs;
}

inline unsigned getNumConvUnits(bool floatActivations, bool floatPartial,
                                const poplar::Target &target) {
  if (floatActivations) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  } else {
    return floatPartial ? target.getFp16InFp32OutConvUnitsPerTile()
                        : target.getFp16InFp16OutConvUnitsPerTile();
  }
}

inline std::uint64_t getConvPartialnx1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    const std::vector<unsigned> &kernelShape, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatWeights, bool floatPartials,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride) {
  const auto kernelElements = product(kernelShape);
  const auto partition = partitionConvPartialByWorker(
      batchElements, vectorConvert<unsigned>(outShape), numWorkerContexts,
      inputDilation, stride);

  // use conv nx1 vertex
  // workList is indexed by [context][numKernelPositions][numPartitions]
  std::vector<std::vector<std::vector<WorklistDataType>>> workList;
  const unsigned positionsOuter = ceildiv(kernelShape[0], filterHeight);
  const unsigned numKernelPositions =
      (positionsOuter * kernelElements / kernelShape[0]);
  const auto outStrideX =
      inputDilation.back() / gcd(inputDilation.back(), stride.back());

  workList.reserve(numWorkerContexts);
  for (unsigned context = 0; context < numWorkerContexts; ++context) {
    workList.emplace_back();
    for (auto k = 0U; k != numKernelPositions; ++k) {
      workList.back().emplace_back();
      for (const auto &partialRow : partition[context]) {
        const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
        const auto numFieldPos = ceildiv(workerOutWidth, outStrideX);
        if (numFieldPos) {
          workList.back().back().push_back(numFieldPos);
        }
      }
    }
  }
  const auto kernelInnerElems = numKernelPositions / positionsOuter;
  const auto kernelOuterElems = positionsOuter;

  return getConvPartialnx1SupervisorInnerLoopCycleEstimate(
      workList, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights,
      floatPartials);
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials, bool zeroPartials) {
  assert(inputDilation == stride);
  std::vector<std::vector<PartialRow>> partition = partitionConvPartialByWorker(
      batchElements, vectorConvert<unsigned>(outShape), numWorkerContexts,
      inputDilation, stride);
  // use conv 1x1 vertex
  std::vector<std::vector<WorklistDataType>> worklist(numWorkerContexts);
  for (unsigned context = 0; context != numWorkerContexts; ++context) {
    for (const auto &partialRow : partition[context]) {
      const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
      if (workerOutWidth == 0)
        continue;
      worklist[context].push_back(workerOutWidth);
    }
  }
  return getConvPartial1x1SupervisorInnerLoopCycleEstimate(
      worklist, numWorkerContexts, numConvUnits, zeroPartials, floatActivations,
      floatPartials);
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, floatActivations, floatPartials, true);
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, bool floatActivations,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, floatActivations, floatPartials, false);
}

inline std::uint64_t getConvPartialSlicInnerLoopCycles(
    unsigned outStride, bool implicitZeroing, unsigned batchElements,
    const std::vector<unsigned> &outShape, unsigned numWorkerContexts,
    unsigned numConvUnits, unsigned slicWindowWidth, bool floatActivations,
    bool floatPartials) {
  // SLIC doesn't support input dilation
  std::vector<unsigned> inputDilation(outShape.size(), 1);
  // SLIC only supports output striding (of 1 or 2) in the innermost dimension.
  std::vector<unsigned> outputStride(outShape.size(), 1);
  outputStride.back() = outStride;

  const auto partition = partitionConvPartialByWorker(
      batchElements, outShape, numWorkerContexts, inputDilation, outputStride);
  std::vector<std::vector<WorklistDataType>> worklist(numWorkerContexts);
  for (unsigned context = 0; context != numWorkerContexts; ++context) {
    for (const auto &partialRow : partition[context]) {
      const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
      if (workerOutWidth == 0) {
        continue;
      }

      worklist[context].push_back(workerOutWidth);
    }
  }
  return getConvPartialSlicSupervisorInnerLoopCycleEstimate(
      worklist, numWorkerContexts, numConvUnits, slicWindowWidth,
      floatActivations, floatPartials, outputStride.back(), implicitZeroing);
}

inline std::uint64_t estimateCastCycles(unsigned outputSize,
                                        unsigned partialsVectorWidth,
                                        unsigned outputVectorWidth,
                                        unsigned numWorkers) {
  const auto outputPerWorker = (outputSize + numWorkers - 1) / numWorkers;
  std::uint64_t loadPartialsCycles =
      (outputPerWorker + partialsVectorWidth - 1) / partialsVectorWidth;
  std::uint64_t writeOutputCycles =
      (outputPerWorker + outputVectorWidth - 1) / outputVectorWidth;
  std::uint64_t cycles = std::max(loadPartialsCycles, writeOutputCycles);
  return (cycles + 26) * numWorkers;
}

inline std::uint64_t
estimateConvReduceCycles(unsigned outputSize, unsigned reductionDepth,
                         unsigned inChanSerialSplit, bool floatOutput,
                         bool floatPartials, unsigned numWorkers,
                         unsigned dataPathWidth, unsigned partialsVectorWidth,
                         unsigned outputVectorWidth,
                         const std::vector<unsigned> &memoryElementOffsets,
                         unsigned bytesPerPartialsElement,
                         bool enableMultiStageReduce, bool enableFastReduce) {
  if (reductionDepth == 0)
    return 0;
  if (reductionDepth == 1) {
    // if input-channel serial splitting is involved, casting is deferred until
    // all the serial splits have been processed.
    if ((floatOutput == floatPartials) || (inChanSerialSplit > 1)) {
      return 0;
    } else {
      return estimateCastCycles(outputSize, partialsVectorWidth,
                                outputVectorWidth, numWorkers);
    }
  }

  // Determine number of stages used in the reduction
  auto reductionPlan =
      getMultiStageReducePlan(reductionDepth, enableMultiStageReduce);
  std::uint64_t cycles = 0;

  unsigned remainingDepth = reductionDepth;
  // Output size depends on the depth used in the reduction
  unsigned outputSizeThisStage = outputSize * reductionDepth;
  const unsigned widthForFastReduce = floatPartials ? 4 : 8;

  for (auto d : reductionPlan) {
    const auto depthThisStage = ceildiv(remainingDepth, d);
    remainingDepth = ceildiv(remainingDepth, depthThisStage);
    const auto stageOutputIsFloat =
        remainingDepth == 1 ? floatOutput : floatPartials;
    outputSizeThisStage = ceildiv(outputSizeThisStage, depthThisStage);

    const auto exchangedPartialsBytes =
        (depthThisStage - 1) * outputSizeThisStage * bytesPerPartialsElement;
    bool singleInputReduceIsPossible =
        (outputSizeThisStage % widthForFastReduce) == 0;
    bool singleInputReducePartialsSize = checkPartialsSizeForSingleInputReduce(
        exchangedPartialsBytes, memoryElementOffsets);
    bool useSingleInputReduce =
        singleInputReduceIsPossible &&
        (enableFastReduce || singleInputReducePartialsSize);
    const auto depthForEstimate = depthThisStage - useSingleInputReduce;

    cycles += getReduceCycleEstimate(outputSizeThisStage, depthForEstimate,
                                     dataPathWidth, stageOutputIsFloat,
                                     floatPartials, useSingleInputReduce,
                                     enableFastReduce, numWorkers);
  }

  if (remainingDepth > 1) {
    outputSizeThisStage =
        (outputSizeThisStage + remainingDepth - 1) / remainingDepth;
    const auto exchangedPartialsBytes =
        (remainingDepth - 1) * outputSizeThisStage * bytesPerPartialsElement;
    bool singleInputReduceIsPossible =
        (outputSizeThisStage % widthForFastReduce) == 0;
    bool singleInputReducePartialsSize = checkPartialsSizeForSingleInputReduce(
        exchangedPartialsBytes, memoryElementOffsets);
    bool useSingleInputReduce =
        singleInputReduceIsPossible &&
        (enableFastReduce || singleInputReducePartialsSize);
    const auto depthForEstimate = remainingDepth - useSingleInputReduce;

    cycles += getReduceCycleEstimate(
        outputSizeThisStage, depthForEstimate, dataPathWidth, floatOutput,
        floatPartials, useSingleInputReduce, enableFastReduce, numWorkers);
  }
  return cycles;
}

inline std::uint64_t estimateZeroSupervisorCycles(unsigned fieldSize,
                                                  unsigned numOutGroups,
                                                  unsigned numConvGroups,
                                                  unsigned outChansPerGroup,
                                                  unsigned dataPathWidth,
                                                  unsigned numWorkerContexts) {
  std::vector<WorklistDataType> zeroWorkList;
  zeroWorkList.reserve(numWorkerContexts);

  for (unsigned i = 0; i != numWorkerContexts; ++i) {
    zeroWorkList.push_back(
        (fieldSize * outChansPerGroup + numWorkerContexts - 1) /
        numWorkerContexts);
  }
  return getZeroSupervisorVertexCycleEstimate(
      zeroWorkList, numOutGroups * numConvGroups, dataPathWidth,
      numWorkerContexts, true);
}

inline std::uint64_t estimateConvPartialHorizontalMacInnerLoopCycles(
    unsigned numOutRows, unsigned tileOutWidth, unsigned outputStrideX,
    unsigned tileKernelHeight, unsigned tileKernelWidth, unsigned numWorkers,
    bool floatActivations, bool floatPartials, unsigned inChansPerGroup,
    unsigned outChansPerGroup, unsigned dataPathWidth) {
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numOutRows);
  unsigned numPartRows = numOutRows * rowSplitFactor;
  const auto maxPartRows = (numPartRows + numWorkers - 1) / numWorkers;
  const auto workerWholeRows = maxPartRows / rowSplitFactor;
  const auto workerPartRows = maxPartRows % rowSplitFactor;
  const auto wholeRowConvSize =
      (tileOutWidth + outputStrideX - 1) / outputStrideX;
  std::vector<std::vector<std::vector<unsigned>>> workerPartitions;
  workerPartitions.emplace_back();
  const auto kernelSize = tileKernelWidth * tileKernelHeight;
  for (auto k = 0U; k != kernelSize; ++k) {
    workerPartitions.back().emplace_back();
    if (wholeRowConvSize) {
      for (unsigned r = 0; r != workerWholeRows; ++r) {
        workerPartitions.back().back().push_back(wholeRowConvSize);
      }
      if (workerPartRows) {
        auto convSize = workerPartRows *
                        (wholeRowConvSize + rowSplitFactor - 1) /
                        rowSplitFactor;
        workerPartitions.back().back().push_back(convSize);
      }
    }
  }
  return getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
      workerPartitions, kernelSize, inChansPerGroup, outChansPerGroup,
      numWorkers, floatActivations, floatPartials);
}

inline std::uint64_t estimateConvPartialVerticalMacInnerLoopCycles(
    unsigned tileOutHeight, unsigned tileOutWidth, unsigned batchSize,
    unsigned tileKernelHeight, unsigned tileKernelWidth, unsigned numWorkers,
    bool floatActivations, bool floatPartials, unsigned inChansPerGroup,
    unsigned outChansPerGroup, unsigned convGroupsPerGroup) {
  unsigned numRows = tileOutHeight * tileOutWidth * batchSize;
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numRows);
  unsigned numPartRows = numRows * rowSplitFactor;
  const auto maxPartRows = (numPartRows + numWorkers - 1) / numWorkers;
  const auto workerWholeRows = maxPartRows / rowSplitFactor;
  const auto workerPartRows = maxPartRows % rowSplitFactor;
  std::vector<std::vector<unsigned>> workerPartitions;
  workerPartitions.emplace_back();
  unsigned wholeRowConvSize = tileKernelWidth;
  for (auto k = 0U; k != tileKernelHeight; ++k) {
    for (unsigned r = 0; r != workerWholeRows; ++r) {
      workerPartitions.back().push_back(wholeRowConvSize);
    }
    if (workerPartRows) {
      for (unsigned r = 0; r != workerPartRows; ++r) {
        auto partRowConvSize =
            (wholeRowConvSize + rowSplitFactor - 1) / rowSplitFactor;
        workerPartitions.back().push_back(partRowConvSize);
      }
    }
  }
  assert(inChansPerGroup == 1);
  assert(outChansPerGroup == 1);
  auto cycles = getConvPartialVerticalMacSupervisorInnerLoopCycleEstimate(
      workerPartitions, tileKernelHeight, convGroupsPerGroup, numWorkers,
      floatActivations, floatPartials);
  return cycles;
}

} // namespace poplin

#endif // _performance_estimation_h_
