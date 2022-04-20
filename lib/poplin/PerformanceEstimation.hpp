// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "ConvReducePlan.hpp"
#include "ConvUtilInternal.hpp"
#include <poplar/Target.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplin/ConvParams.hpp>
#include <poplin/ConvUtil.hpp>

#include <gccs/Algorithm.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

using namespace poplibs_support;

inline static std::uint64_t convHorizontalMacOverhead(bool floatActivations) {
  return floatActivations ? 58 : 63;
}

inline static std::uint64_t convNx1Overhead() { return 109; }

// Number of worker cycle savings if state retention is used.
// The first entry is the total savings and the second is
// because of retention of state related to input channel processing.
inline static std::pair<std::uint64_t, std::uint64_t>
conv1x1WorkerRetentionSavings(bool floatActivations, bool floatPartials,
                              unsigned numConvUnits) {
  if (!floatActivations &&
      (floatPartials || (!floatPartials && numConvUnits == 16))) {
    return std::make_pair(7, 3);
  } else {
    return std::make_pair(0, 0);
  }
}

inline static std::pair<std::uint64_t, std::uint64_t>
conv1x1WorkerRetentionSavings(unsigned numConvUnits) {
  return std::make_pair(0, 0);
}

inline static std::uint64_t
convnx1WorkerRetentionSavings(bool /*floatActivations */,
                              bool /*floatPartials */) {
  return 9;
}

inline static std::uint64_t convnx1WorkerRetentionSavings() { return 0; }

inline static std::uint64_t zeroPartialsRetentionSavings(bool floatPartials) {
  return floatPartials ? 9 : 10;
}

inline static std::uint64_t zeroPartialsRetentionSavings() { return 0; }

inline std::uint64_t getDenseDotProductCycles(unsigned activationsVectorWidth,
                                              bool floatActivations,
                                              bool floatPartials,
                                              unsigned size) {

  const unsigned actsPer64Bits = floatActivations ? 2 : 4;
  const bool isHalfHalf = !floatActivations && !floatPartials;

  const auto numOutChans = isHalfHalf ? 2 : 1;
  const auto innerCycles = (1 + 2) * numOutChans + // rpt + loop wind down
                           isHalfHalf + // format conversion for half/half
                           3 + // sum with previous partials (load, acc, store)
                           1;  // branch

  // float -> float
  // half -> float
  // half -> half
  assert(!floatActivations || floatPartials);
  for (unsigned i = gccs::ceilLog2(activationsVectorWidth); i >= 0; --i) {
    const unsigned currWidth = (1u << i);
    if ((size % currWidth) == 0) {
      // Limitation currently due to instruction set is that if we have
      // less than 64-bits per loop we cannot simultaneously load activations
      // and weights. Assume for higher vector widths we can always
      // simulataneously load 64-bit values.
      const auto cyclesPerWidth = currWidth >= actsPer64Bits ? 1 : 2;
      // Due to the same above limitation, we cannot simultaneously load
      // weights with activations during pipeline warmup so add a cycle
      // for this for each output channel.
      const unsigned additionalCycles =
          currWidth < actsPer64Bits ? numOutChans : 0;
      return innerCycles + additionalCycles +
             (size / currWidth) * cyclesPerWidth;
    }
  }
  // This should be unreachable.
  assert(false);
  POPLIB_UNREACHABLE();
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
    bool floatActivations, bool floatPartials, unsigned activationsVectorWidth,
    unsigned numInChans, unsigned numOutChans,
    const std::vector<unsigned> &convSizes) {
  uint64_t cycles = 16;
  for (auto convSize : convSizes) {
    if (convSize == 0) {
      cycles += 7;
    } else {
      if (!floatPartials) {
        numOutChans /= 2; // Processing two channels inside inner loop
      }
      cycles += 19;
      cycles += convSize *
                (7 + numOutChans * getDenseDotProductCycles(
                                       activationsVectorWidth, floatActivations,
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
    unsigned activationsVectorWidth, bool floatActivations,
    bool floatPartials) {
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
          floatActivations, floatPartials, activationsVectorWidth,
          numInChansPerGroup, numOutChansPerGroup,
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
    unsigned activationsVectorWidth, bool floatActivations,
    bool floatPartials) {
  auto innerLoopCycles =
      getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
          workerPartitions, kernelSize, numInChansPerGroup, numOutChansPerGroup,
          numWorkerContexts, activationsVectorWidth, floatActivations,
          floatPartials);
  auto cycles = getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
      innerLoopCycles, numConvGroups, numInGroups, numOutGroups,
      numWorkerContexts, floatActivations, floatPartials);
  return cycles;
}

inline std::uint64_t getVerticalMacDotProductCycles(bool floatActivations,
                                                    bool floatPartials,
                                                    unsigned size,
                                                    unsigned numChannels) {
  assert(!floatActivations);
  uint64_t cycles = 3; // inner cycles overhead
  if (numChannels >= 8) {
    cycles += (3 * (size - 1)) * (numChannels / 8);
  } else {
    cycles += (2 * (size - 1)) * (numChannels / 4);
  }
  return cycles;
}

inline std::uint64_t getConvPartialVerticalMacCycleEstimate(
    bool floatActivations, bool floatPartials, unsigned convGroupsPerGroup,
    const std::vector<unsigned> &convSizes) {

  uint64_t cycles = 0;

  if (convGroupsPerGroup >= 8 && !floatPartials) {
    // convVerticalMacFlattenedReentry to the end except PartitionLoop cycles
    cycles = (convGroupsPerGroup == 8) ? 13 : 25;

    uint64_t cyclesPer8Chans = 0;
    for (auto convSize : convSizes) {
      cyclesPer8Chans +=
          7 + // PARTITION_LOOP_8CHANS cycles - Partition_loop_start loop cycles
          15 + // Store and reload acc
          getVerticalMacDotProductCycles(floatActivations, floatPartials,
                                         convSize, 8);
    }

    // 16 groups processed as 2 x 8/
    cyclesPer8Chans *= (convGroupsPerGroup == 8) ? 1 : 2;
    cycles += cyclesPer8Chans;

  } else { // if (convGroupsPerGroup == 4)
    // Reload state
    // Cycles convVerticalMacFlattenedReentry to the end except PartitionLoop
    // cycles
    cycles = floatPartials ? 19 : 18;

    for (auto convSize : convSizes) {
      cycles += 7;

      // Cycles to store accumulators and then reload. Note that
      // this is an overestimate since these cycles should only be
      // incurred  when the output differs from that of the
      // previous worklist.
      cycles += floatPartials ? 5 : 3;
      auto dotProdCycles = getVerticalMacDotProductCycles(
          floatActivations, floatPartials, convSize, convGroupsPerGroup);
      cycles += dotProdCycles;
    }
  }

  return cycles;
}

inline std::uint64_t getConvPartialVerticalReductionCycleEstimate(
    unsigned numElems, unsigned numWorkers, bool floatPartials,
    unsigned numGroupsPerGroup) {
  const auto cyclesPerRpt = floatPartials ? 2 : 1;
  return 10 - floatPartials +
         ((7 + 2 * floatPartials + (cyclesPerRpt * (numWorkers - 1))) *
          numElems / numGroupsPerGroup);
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
    unsigned numOutElems, bool floatPartials) {
  unsigned elemsPerCycle = floatPartials ? 2 : 4;
  return 4 + gccs::ceildiv(numOutElems, elemsPerCycle);
}

inline std::uint64_t
getConvPartialVerticalMacSupervisorReductionInnerLoopCycleEstimate(
    unsigned numOutElems, unsigned numWorkerContexts, bool floatPartials,
    unsigned convGroupsPerGroup) {
  auto numElemsPerWorker =
      (numOutElems + numWorkerContexts - 1) / numWorkerContexts;
  uint64_t cycles = getConvPartialVerticalReductionCycleEstimate(
      numElemsPerWorker, numWorkerContexts, floatPartials, convGroupsPerGroup);
  return cycles;
}

inline std::uint64_t getConvPartialVerticalMacSupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCycles, std::uint64_t zeroInitInnerCycles,
    std::uint64_t reductionInnerCycles, unsigned numConvGroups,
    unsigned numInGroups, bool floatPartials) {
  const auto supOverheadCycles = 61;
  const auto wkrCoreVMACInit = 14;
  const auto wkrStateRetentionInit = 25 + floatPartials ? 1 : 0;
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
          numOutElems, floatPartials);
  auto reductionCycles =
      getConvPartialVerticalMacSupervisorReductionInnerLoopCycleEstimate(
          numOutElems, numWorkerContexts, floatPartials, convGroupsPerGroup);
  cycles += getConvPartialVerticalMacSupervisorOuterLoopCycleEstimate(
      innerLoopCycles, zeroInitCycles, reductionCycles, numConvGroups,
      numInGroups, floatPartials);
  return cycles;
}

inline std::uint64_t getConvPartial1x1SupervisorInnerLoopCycleEstimateHalfFloat(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvUnits, bool outputZeroing,
    bool floatActivations, bool floatPartials) {
  // Core loop cycles for 16x8 AMP vertex
  auto coreCycles = floatActivations ? 8 : 4;

  auto retentionSavings = conv1x1WorkerRetentionSavings(
      floatActivations, floatPartials, numConvUnits);
  unsigned usedContexts = workerPartitions.size();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  // This should be altered once we support different number of conv units.
  // This only takes care of 8 and 16 conv units
  auto zeroCyclesPerGroup = floatActivations && numConvUnits == 16 ? 8
                            : floatPartials                        ? 4
                                                                   : 2;
  // The code paths for half activations, half partials and 16 conv units is
  // = half activations, float partials and 8 conv units.
  bool commonHalfAndFloatPartialsCodePath =
      (!floatActivations && numConvUnits == 16) || floatPartials;
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
          if (commonHalfAndFloatPartialsCodePath) {
            thisWorkerCycles += 20 + (outputZeroing ? 1 : 2);
          } else {
            thisWorkerCycles += 24;
          }
        }
        break;
      case 1:
        if (floatActivations)
          thisWorkerCycles += 47 + (2 + zeroCyclesPerGroup) * outputZeroing;
        else {
          if (commonHalfAndFloatPartialsCodePath) {
            thisWorkerCycles += 31 + (outputZeroing ? 1 : 2);
          } else {
            thisWorkerCycles += 39 + (2 + zeroCyclesPerGroup) * outputZeroing;
          }
        }
        break;
      case 2:
        if (floatActivations)
          thisWorkerCycles += 46 + (2 + zeroCyclesPerGroup * 2) * outputZeroing;
        else {
          if (commonHalfAndFloatPartialsCodePath) {
            thisWorkerCycles += 40 + (outputZeroing ? 1 : 2);
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
          if (commonHalfAndFloatPartialsCodePath) {
            thisWorkerCycles +=
                (outputZeroing ? 37 : 38) + (numElems - 3) * coreCycles;
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

inline std::uint64_t getConvPartial1x1SupervisorInnerLoopCycleEstimateQuarter(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvUnits, bool outputZeroing) {

  assert(numConvUnits == 16);
  // Core loop cycles for AMP vertex - 32 QUARTER inputs
  auto coreCycles = 4;

  unsigned usedContexts = workerPartitions.size();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();

  const auto overhead = 20;
  for (const auto &worker : workerPartitions) {
    // 1x1 vertex doesn't support more than one worklist item per worker.
    assert(worker.size() <= 1);

    uint64_t thisWorkerCycles = overhead;
    if (!worker.empty()) {
      const auto numElems = worker.front();
      const unsigned outputZeroSaving = outputZeroing ? 5 : 0;
      switch (numElems) {
      case 0:
        thisWorkerCycles += 0;
        break;
      case 1:
        thisWorkerCycles += 17 - outputZeroSaving;
        break;
      case 2:
        thisWorkerCycles += 22 - outputZeroSaving;
        break;
      default:
        thisWorkerCycles += 29 - outputZeroSaving + (numElems - 3) * coreCycles;
      }
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

inline std::uint64_t getConvPartial1x1SupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvUnits, bool outputZeroing,
    const poplar::Type &actsType, bool floatPartials) {
  if (actsType == poplar::QUARTER) {
    return getConvPartial1x1SupervisorInnerLoopCycleEstimateQuarter(
        workerPartitions, numWorkerContexts, numConvUnits, outputZeroing);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartial1x1SupervisorInnerLoopCycleEstimateHalfFloat(
        workerPartitions, numWorkerContexts, numConvUnits, outputZeroing,
        floatActivations, floatPartials);
  }
}

inline unsigned getConvPartialAmpSupervisorWeightLoadCycleEstimate(
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, unsigned filterHeight) {

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

inline std::uint64_t getConvPartial1x1SupervisorOuterLoopCycleEstimateHalfFloat(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing, unsigned numConvGroups,
    unsigned numInGroups, unsigned numOutGroups, unsigned outChansPerGroup,
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, bool floatActivations,
    bool floatPartials, unsigned numWorkerContexts) {
  const auto outputPassesPerGroup =
      (outChansPerGroup + numConvUnits - 1) / numConvUnits;

  const auto retentionSavings = conv1x1WorkerRetentionSavings(
      floatActivations, floatPartials, numConvUnits);

  // Filter height is not applicable to 1x1 vertex so set it to 1
  const auto numLoads = getConvPartialAmpSupervisorWeightLoadCycleEstimate(
      weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle, 1);
  // The code paths for half activations, half partials and 16 conv units is
  // = half activations, float partials and 8 conv units.
  bool commonHalfAndFloatPartialsCodePath =
      (!floatActivations && numConvUnits == 16) || floatPartials;

  const uint64_t supervisorNonloopOverhead = 50;
  const unsigned outPassesOverhead = 7;
  const unsigned excessInChanOverhead = 1;
  return supervisorNonloopOverhead +
         numWorkerContexts *
             (retentionSavings.first +
              retentionSavings.second * (numInGroups * numConvGroups - 1)) +
         numConvGroups *
             (12 + commonHalfAndFloatPartialsCodePath +
              (numInGroups - 1) *
                  (15 - commonHalfAndFloatPartialsCodePath +
                   excessInChanOverhead +
                   numOutGroups * (19 + outputPassesPerGroup *
                                            (6 + numLoads +
                                             innerLoopCyclesWithoutZeroing))) +
              (10 + excessInChanOverhead +
               numOutGroups *
                   (19 - commonHalfAndFloatPartialsCodePath +
                    outputPassesPerGroup * (outPassesOverhead + numLoads +
                                            innerLoopCyclesWithZeroing))));
}

inline std::uint64_t getConvPartial1x1SupervisorOuterLoopCycleEstimateQuarter(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing, unsigned numConvGroups,
    unsigned numInGroups, unsigned numOutGroups, unsigned outChansPerGroup,
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, unsigned numWorkerContexts) {

  // Filter height is not applicable to 1x1 vertex so set it to 1
  const auto numLoads = getConvPartialAmpSupervisorWeightLoadCycleEstimate(
      weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle, 1);

  const uint64_t workerCallOverhead = numWorkerContexts * 6;
  const uint64_t supervisorNonloopOverhead = 46;
  const uint64_t cgLoopOverhead = numWorkerContexts * 2 + 9;
  const uint64_t ogLoopOverhead = numWorkerContexts * 6 + 10;
  const uint64_t igLoopOverhead =
      numWorkerContexts * 14 + 11 + workerCallOverhead;
  const uint64_t igLoopOverheadAfterSync =
      numWorkerContexts * 11 + 6 + workerCallOverhead;

  return supervisorNonloopOverhead +
         numConvGroups *
             (cgLoopOverhead +
              numOutGroups *
                  (ogLoopOverhead +
                   // Zeroing pass
                   igLoopOverhead + numLoads + innerLoopCyclesWithZeroing +
                   // Later passes - assuming loop to sync is zero cost
                   (numInGroups - 1) * (igLoopOverheadAfterSync + numLoads +
                                        innerLoopCyclesWithoutZeroing)));
}

inline std::uint64_t getConvPartial1x1SupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing, unsigned numConvGroups,
    unsigned numInGroups, unsigned numOutGroups, unsigned outChansPerGroup,
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, const poplar::Type &actsType,
    bool floatPartials, unsigned numWorkerContexts) {
  if (actsType == poplar::QUARTER) {
    return getConvPartial1x1SupervisorOuterLoopCycleEstimateQuarter(
        innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
        numConvGroups, numInGroups, numOutGroups, outChansPerGroup,
        weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
        numWorkerContexts);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartial1x1SupervisorOuterLoopCycleEstimateHalfFloat(
        innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
        numConvGroups, numInGroups, numOutGroups, outChansPerGroup,
        weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
        floatActivations, floatPartials, numWorkerContexts);
  }
}

inline std::uint64_t getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numConvGroups, unsigned numInGroups, unsigned numOutGroups,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, poplar::Type actsType, bool floatPartials) {
  auto innerLoopCyclesWithZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvUnits, true, actsType,
          floatPartials);
  auto innerLoopCyclesWithoutZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(
          workerPartitions, numWorkerContexts, numConvUnits, false, actsType,
          floatPartials);

  return getConvPartial1x1SupervisorOuterLoopCycleEstimate(
      innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing, numConvGroups,
      numInGroups, numOutGroups, outChansPerGroup, weightBytesPerConvUnit,
      numConvUnits, convUnitCoeffLoadBytesPerCycle, actsType, floatPartials,
      numWorkerContexts);
}

inline std::uint64_t getConvPartialnx1SupervisorOuterLoopCycleEstimateHalfFloat(
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

inline std::uint64_t getConvPartialnx1SupervisorOuterLoopCycleEstimateQuarter(
    std::uint64_t innerLoopCycles, unsigned numConvGroups,
    unsigned numOutGroups, unsigned numInGroups, unsigned outChansPerGroup,
    unsigned numConvUnits, unsigned numWorkerContexts) {

  const uint64_t supervisorNonloopOverhead = 83;
  const uint64_t cgLoopOverhead = numWorkerContexts * 5 + 7;
  const uint64_t ogLoopOverhead = numWorkerContexts * 8 + 10;
  const uint64_t igLoopOverhead = numWorkerContexts * 8 + 13;

  auto result = supervisorNonloopOverhead +
                numConvGroups *
                    // cg loop
                    (cgLoopOverhead +
                     numOutGroups *
                         // og loop
                         (ogLoopOverhead +
                          numInGroups *
                              // ig loop + cycles for inner loops and workers
                              (igLoopOverhead + innerLoopCycles)));
  return result;
}

inline std::uint64_t getConvPartialnx1SupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCycles, unsigned numConvGroups,
    unsigned numOutGroups, unsigned numInGroups, unsigned outChansPerGroup,
    unsigned numConvUnits, unsigned numWorkerContexts,
    const poplar::Type &actsType, bool floatPartials) {

  if (actsType == poplar::QUARTER) {
    return getConvPartialnx1SupervisorOuterLoopCycleEstimateQuarter(
        innerLoopCycles, numConvGroups, numOutGroups, numInGroups,
        outChansPerGroup, numConvUnits, numWorkerContexts);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartialnx1SupervisorOuterLoopCycleEstimateHalfFloat(
        innerLoopCycles, numConvGroups, numOutGroups, numInGroups,
        outChansPerGroup, numConvUnits, numWorkerContexts, floatActivations,
        floatPartials);
  }
}

// Cycles for a single run of the ConvPartialnx1 vertex worker code
// with the given worklist and parameters
static std::uint64_t inline getConvPartialnx1WorkerCycles(
    const std::vector<unsigned> &worklist, bool floatActivations,
    bool floatPartials, unsigned numConvUnits, unsigned retentionSavings) {

  // Core loop cycles for vertex will all engines in use
  auto coreCycles = floatActivations ? 8 : 4;

  unsigned extraStrideCycles = 0;
  bool hh16 = !floatActivations && !floatPartials && (numConvUnits == 16);
  bool hf8 = !floatActivations && floatPartials && (numConvUnits == 8);
  if (hh16 || hf8) {
    extraStrideCycles = 1; // Require one more special stride
  }

  // Cycles between worker start and <PartitionLoop> label
  std::uint64_t cycles = 17 + (floatPartials ? 0 : 1);
  for (auto &numElems : worklist) {
    // cost of construct second tripack pointer and special stride
    cycles += 3;
    switch (numElems) {
    case 0:
      cycles += 17;
      break;
    case 1:
      cycles += (floatActivations ? 33 : 29);
      break;
    case 2:
      cycles += (floatActivations ? 44 : 33 + extraStrideCycles);
      break;
    default:
      if (floatActivations)
        cycles += 45 + (numElems - 3) * coreCycles;
      else
        cycles += 34 + (numElems - 3) * coreCycles;
    }
    cycles -= retentionSavings;
  }

  return cycles;
}

// Cycles for a single run of the ConvPartialnx1 vertex worker code
// with the given worklist and parameters
static std::uint64_t inline getConvPartialnx1WorkerCyclesQuarter(
    const std::vector<unsigned> &worklist, unsigned numConvUnits,
    unsigned retentionSavings) {

  assert(numConvUnits == 16);
  // Core loop cycles for vertex with all engines in use
  auto coreCycles = 4;

  const auto overhead = 18;
  const auto worklistLoopOverhead = 22;
  std::uint64_t cycles = overhead;
  for (auto &numElems : worklist) {
    cycles += worklistLoopOverhead;
    switch (numElems) {
    case 0:
      cycles += 0;
      break;
    case 1:
      cycles += 17;
      break;
    case 2:
      cycles += 22;
      break;
    default:
      cycles += 29 + (numElems - 3) * coreCycles;
    }
    cycles -= retentionSavings;
  }

  return cycles;
}

// Overload with per kernel position worklists
static inline std::uint64_t
getConvPartialnx1SupervisorKernelLoopCycleEstimateHalfFloat(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems,
    unsigned numOutChanPasses, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials, unsigned numConvUnits) {
  unsigned usedContexts = workerPartitions.size();
  const auto retentionSavings =
      convnx1WorkerRetentionSavings(floatActivations, floatPartials);
  std::uint64_t cycles = 0;
  for (auto ky = 0U; ky != kernelOuterElems; ++ky) {
    for (auto kx = 0U; kx != kernelInnerElems; ++kx) {
      uint64_t maxWorkerCycles = 0;
      uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                     ? 0
                                     : std::numeric_limits<uint64_t>::max();
      for (auto context = 0U; context != usedContexts; ++context) {
        const auto k = ky * kernelInnerElems + kx;
        const auto thisWorkerCycles = getConvPartialnx1WorkerCycles(
            workerPartitions[context][k], floatActivations, floatPartials,
            numConvUnits, retentionSavings);
        maxWorkerCycles =
            std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
        minWorkerCycles =
            std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
      }
      cycles +=
          std::max(maxWorkerCycles, minWorkerCycles + 9) * numOutChanPasses;
    }
  }
  return cycles;
}

// Overload with assumed same worklist for every kernel position.
static inline std::uint64_t
getConvPartialnx1SupervisorKernelLoopCycleEstimateHalfFloat(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems,
    unsigned numOutChanPasses, unsigned numWorkerContexts,
    bool floatActivations, bool floatPartials, unsigned numConvUnits) {
  unsigned usedContexts = workerPartitions.size();
  const auto retentionSavings =
      convnx1WorkerRetentionSavings(floatActivations, floatPartials);
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  for (auto context = 0U; context != usedContexts; ++context) {
    const auto thisWorkerCycles = getConvPartialnx1WorkerCycles(
        workerPartitions[context], floatActivations, floatPartials,
        numConvUnits, retentionSavings);
    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }
  const auto cycles =
      std::max(maxWorkerCycles, minWorkerCycles + 9) * numOutChanPasses;
  return cycles * kernelOuterElems * kernelInnerElems;
}

// Overload with per kernel position worklists
static inline std::uint64_t
getConvPartialnx1SupervisorKernelLoopCycleEstimateQuarter(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems,
    unsigned numOutChanPasses, unsigned numWorkerContexts,
    unsigned numConvUnits) {
  unsigned usedContexts = workerPartitions.size();
  const auto retentionSavings = convnx1WorkerRetentionSavings();
  std::uint64_t cycles = 0;
  for (auto ky = 0U; ky != kernelOuterElems; ++ky) {
    for (auto kx = 0U; kx != kernelInnerElems; ++kx) {
      uint64_t maxWorkerCycles = 0;
      uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                     ? 0
                                     : std::numeric_limits<uint64_t>::max();
      for (auto context = 0U; context != usedContexts; ++context) {
        const auto k = ky * kernelInnerElems + kx;
        const auto thisWorkerCycles = getConvPartialnx1WorkerCyclesQuarter(
            workerPartitions[context][k], numConvUnits, retentionSavings);
        maxWorkerCycles =
            std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
        minWorkerCycles =
            std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
      }
      cycles +=
          std::max(maxWorkerCycles, minWorkerCycles + 9) * numOutChanPasses;
    }
  }
  return cycles;
}

// Overload with assumed same worklist for every kernel position.
static inline std::uint64_t
getConvPartialnx1SupervisorKernelLoopCycleEstimateQuarter(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems,
    unsigned numOutChanPasses, unsigned numWorkerContexts,
    unsigned numConvUnits) {
  unsigned usedContexts = workerPartitions.size();
  const auto retentionSavings = convnx1WorkerRetentionSavings();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts
                                 ? 0
                                 : std::numeric_limits<uint64_t>::max();
  for (auto context = 0U; context != usedContexts; ++context) {
    const auto thisWorkerCycles = getConvPartialnx1WorkerCyclesQuarter(
        workerPartitions[context], numConvUnits, retentionSavings);
    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }
  const auto cycles =
      std::max(maxWorkerCycles, minWorkerCycles + 9) * numOutChanPasses;
  return cycles * kernelOuterElems * kernelInnerElems;
}

template <typename WorkerWorklist>
inline std::uint64_t getConvPartialnx1SupervisorInnerLoopCycleEstimateHalfFloat(
    const std::vector<WorkerWorklist> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, bool floatActivations, bool floatPartials) {

  unsigned numOutChanPasses = outChansPerGroup / numConvUnits;

  // innermostLoopCycles is the cycles in the innermost supervisor loop
  uint64_t innermostLoopCycles =
      getConvPartialAmpSupervisorWeightLoadCycleEstimate(
          weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
          filterHeight);

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

  // Supervisor cycles for supervisor overhead on each outer loop as
  // well as loading of weights etc. in the innermost loop.
  uint64_t innerLoopCycles =
      kernelOuterElems *
      (14 +
       kernelInnerElems * (17 - 5 + numOutChanPasses * innermostLoopCycles));

  innerLoopCycles +=
      getConvPartialnx1SupervisorKernelLoopCycleEstimateHalfFloat(
          workerPartitions, kernelInnerElems, kernelOuterElems,
          numOutChanPasses, numWorkerContexts, floatActivations, floatPartials,
          numConvUnits);

  return innerLoopCycles;
}

template <typename WorkerWorklist>
inline std::uint64_t getConvPartialnx1SupervisorInnerLoopCycleEstimateQuarter(
    const std::vector<WorkerWorklist> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts) {

  const uint64_t weightLoadCycles =
      getConvPartialAmpSupervisorWeightLoadCycleEstimate(
          weightBytesPerConvUnit, numConvUnits, convUnitCoeffLoadBytesPerCycle,
          filterHeight);

  const uint64_t workerCallOverhead = numWorkerContexts * 6;
  const uint64_t kyLoopOverhead = numWorkerContexts * 5 + 3;
  const uint64_t kxLoopOverhead =
      numWorkerContexts * 9 + 13 + workerCallOverhead;
  const uint64_t kxLoopOverheadAfterSync =
      numWorkerContexts * 7 + 8 + workerCallOverhead;

  // Supervisor cycles for supervisor overhead on each outer loop as
  // well as loading of weights etc. in the innermost loop.
  const uint64_t innerLoopCycles =
      kernelOuterElems *
      (kyLoopOverhead + (
                            // One pass with this cost
                            kxLoopOverhead + weightLoadCycles +
                            // Later passes with this cost
                            (kernelInnerElems - 1) * kxLoopOverheadAfterSync +
                            weightLoadCycles));

  unsigned numOutChanPasses = outChansPerGroup / numConvUnits;
  return innerLoopCycles +
         getConvPartialnx1SupervisorKernelLoopCycleEstimateQuarter(
             workerPartitions, kernelInnerElems, kernelOuterElems,
             numOutChanPasses, numWorkerContexts, numConvUnits);
}

template <typename WorkerWorklist>
inline std::uint64_t getConvPartialnx1SupervisorInnerLoopCycleEstimate(
    const std::vector<WorkerWorklist> &workerPartitions,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, const poplar::Type &actsType,
    bool floatPartials) {
  if (actsType == poplar::QUARTER) {
    return getConvPartialnx1SupervisorInnerLoopCycleEstimateQuarter(
        workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
        outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
        convUnitCoeffLoadBytesPerCycle, numWorkerContexts);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartialnx1SupervisorInnerLoopCycleEstimateHalfFloat(
        workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
        outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
        convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatActivations,
        floatPartials);
  }
}

inline std::uint64_t getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups, unsigned numOutGroups, unsigned numInGroups,
    unsigned kernelInnerElems, unsigned kernelOuterElems, unsigned filterHeight,
    unsigned inChansPerGroup, unsigned outChansPerGroup,
    unsigned weightBytesPerConvUnit, unsigned numConvUnits,
    unsigned convUnitCoeffLoadBytesPerCycle, unsigned numWorkerContexts,
    const poplar::Type &actsType, bool floatPartials) {
  auto innerLoopCycles = getConvPartialnx1SupervisorInnerLoopCycleEstimate(
      workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, actsType,
      floatPartials);
  return getConvPartialnx1SupervisorOuterLoopCycleEstimate(
      innerLoopCycles, numConvGroups, numOutGroups, numInGroups,
      outChansPerGroup, numConvUnits, numWorkerContexts, actsType,
      floatPartials);
}

inline std::uint64_t getConvPartialSlicSupervisorWeightLoadCycleEstimate(
    unsigned convGroupsPerGroup, unsigned chansPerGroup,
    unsigned numWorkerContexts, unsigned slicWindowWidth, poplar::Type inType) {
  assert(slicWindowWidth == 4u);

  if (inType == poplar::QUARTER) {
    // Quarter code has a different structure
    // 32 weights load cycles, csr write with stall
    return 32 + 6;
  }
  std::uint64_t cycles = 0;
  if (chansPerGroup == 4) {
    assert(convGroupsPerGroup == 1u);
    cycles += (6 + // brnzdec
               6 + // put CCCSLOAD
               6); // bri
  } else {
    if (chansPerGroup == 2) {
      assert(convGroupsPerGroup == 2);
    }
    if (chansPerGroup == 1) {
      assert((convGroupsPerGroup % 4) == 0);
    }

    std::uint64_t workerLoadWeightsCycles = (convGroupsPerGroup == 2) ? 12 : 10;

    if (convGroupsPerGroup > 4) {
      // 8 and 16 requires one more cycle to calculate input offset
      workerLoadWeightsCycles++;
    }

    cycles += (9 + // brnzdec, put CCCSLOAD pointer (stall), store weights
                   // pointer for rearrangement.
               6 + // runall
               // Rearrange weights in workers
               (workerLoadWeightsCycles * numWorkerContexts) + 6); // sync
  }
  cycles += 16; // 16 * ld64putcs
  return cycles;
}

inline std::uint64_t
getConvPartialSlicSupervisorOuterLoopCycleEstimateHalfFloat(
    std::uint64_t implicitZeroingInnerLoopCycles, std::uint64_t innerLoopCycles,
    std::uint64_t weightLoadCycles, unsigned numConvGroupGroups,
    unsigned numSubKernels, unsigned numConvChains, unsigned slicWindowWidth,
    unsigned convGroupsPerGroup, bool floatActivations, bool floatPartials) {

  // TODO: we currently only target a kernel width of 4.
  assert(!floatActivations);
  assert(slicWindowWidth == 4);
  assert(numConvGroupGroups >= 1);
  assert(numSubKernels >= 1);

  const auto brnzdec_cycles = [](const unsigned n) {
    // 6 cycles brnzdec stall for all but the last one
    return 6 * (n - 1) + 1;
  };

  // Similar, but different function for the 2 chains of conv units,
  // half partials case.
  const bool half8Conv = (numConvChains == 2 && floatPartials == false);

  const std::uint64_t supervisorPreambleCycles = half8Conv ? 25 : 28;
  std::uint64_t supervisorConvGroupGroupsBodyCycles = half8Conv ? 12 : 15;
  std::uint64_t supervisorSubKernelBodyCycles = 0;

  if (convGroupsPerGroup <= 4) {
    supervisorSubKernelBodyCycles =
        weightLoadCycles +
        (half8Conv ? 0
                   : 3) + // deal with whether to swap output pointers or not
        2 +               // store new worklist pointer and increment
        (half8Conv ? 0 : 1) + // or, store implicit zero/stride
        6 +                   // runall
        1 +
        6 + // sync
        1;  // load new weights pointer

  } else {
    const unsigned numConvGroupStrides = convGroupsPerGroup / 4;
    const std::uint64_t convGroupStrideBodyCycles =
        weightLoadCycles + 6 + // runall
        1 + 6 +                // sync
        2 + 6; // update weights pointer (ld + add + register stall)
    const std::uint64_t convGroupStrideLoopCycles =
        convGroupStrideBodyCycles * numConvGroupStrides +
        brnzdec_cycles(numConvGroupStrides);

    supervisorSubKernelBodyCycles =
        convGroupStrideLoopCycles +
        2 +     // deal with whether to swap output pointers or not
        2 + 6 + // store implicit zero/stride + stall
        3 +     // deal with the weights pointer
        2;      // store new worklist pointer and increment

    // For 8 and 16 groups codelets store numGroupCounter on the stack
    // so that brings extra 2 commands (st + ld) and a penalty for
    // accessing loaded value straignt away
    supervisorConvGroupGroupsBodyCycles += 2 + 6;

    // Update worker cycles by number of group stides
    innerLoopCycles *= numConvGroupStrides;
  }

  const std::uint64_t supervisorSubKernelLoopCycles =
      supervisorSubKernelBodyCycles * numSubKernels +
      brnzdec_cycles(numSubKernels);

  const std::uint64_t supervisorConvGroupGroupsLoopCycles =
      supervisorConvGroupGroupsBodyCycles * numConvGroupGroups +
      brnzdec_cycles(numConvGroupGroups);

  const std::uint64_t cycles =
      supervisorPreambleCycles + supervisorConvGroupGroupsLoopCycles +
      supervisorSubKernelLoopCycles +
      // Workers make one pass for the first sub-kernel implicitly zeroing
      // partials, and the remainder of the sub-kernels not implicitly zeroing.
      (numConvGroupGroups * implicitZeroingInnerLoopCycles +
       numConvGroupGroups * (numSubKernels - 1) * innerLoopCycles);

  return cycles;
}

inline std::uint64_t getConvPartialSlicSupervisorOuterLoopCycleEstimateQuarter(
    std::uint64_t implicitZeroingInnerLoopCycles, std::uint64_t innerLoopCycles,
    std::uint64_t weightLoadCycles, unsigned numConvGroupGroups,
    unsigned numSubKernels, unsigned numConvChains, unsigned slicWindowWidth,
    unsigned convGroupsPerGroup, unsigned numWorkerContexts) {

  assert(slicWindowWidth == 4);
  assert(numConvGroupGroups >= 1);
  assert(numSubKernels >= 1);

  const uint64_t workerCallOverhead = numWorkerContexts * 6;
  const uint64_t supervisorNonloopOverhead = 43;
  const uint64_t cgLoopOverhead = numWorkerContexts * 13 + 15;
  const uint64_t kgLoopOverhead = numWorkerContexts * 6 + 12;
  const uint64_t kgLoopOverheadToSync = numWorkerContexts * 5 + 8;

  return supervisorNonloopOverhead +
         numConvGroupGroups *
             (cgLoopOverhead +
              (
                  // One pass with this cost
                  workerCallOverhead + weightLoadCycles + kgLoopOverhead +
                  implicitZeroingInnerLoopCycles +
                  // Later passes - assuming loop to sync is zero cost
                  (numSubKernels - 1) *
                      (workerCallOverhead + weightLoadCycles +
                       kgLoopOverheadToSync + innerLoopCycles)));
}

inline std::uint64_t getConvPartialSlicSupervisorOuterLoopCycleEstimate(
    std::uint64_t implicitZeroingInnerLoopCycles, std::uint64_t innerLoopCycles,
    std::uint64_t weightLoadCycles, unsigned numConvGroupGroups,
    unsigned numSubKernels, unsigned numConvChains, unsigned slicWindowWidth,
    unsigned convGroupsPerGroup, const poplar::Type &actsType,
    bool floatPartials, unsigned numWorkerContexts) {
  if (actsType == poplar::QUARTER) {
    return getConvPartialSlicSupervisorOuterLoopCycleEstimateQuarter(
        implicitZeroingInnerLoopCycles, innerLoopCycles, weightLoadCycles,
        numConvGroupGroups, numSubKernels, numConvChains, slicWindowWidth,
        convGroupsPerGroup, numWorkerContexts);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartialSlicSupervisorOuterLoopCycleEstimateHalfFloat(
        implicitZeroingInnerLoopCycles, innerLoopCycles, weightLoadCycles,
        numConvGroupGroups, numSubKernels, numConvChains, slicWindowWidth,
        convGroupsPerGroup, floatActivations, floatPartials);
  }
}
// This gives us the number of cycles in terms of supervisor cycles
// for all workers to process a single conv group/sub-kernel. There is
// a strong assumption that the amount of work is always the same between
// sub-kernels.
inline std::uint64_t
getConvPartialSlicSupervisorInnerLoopCycleEstimateHalfFloat(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvChains,
    unsigned slicWindowWidth, unsigned convGroupsPerGroup,
    bool floatActivations, bool floatPartials, unsigned outputStride,
    bool implicitZeroing) {
  // TODO: we currently only target kernel width of 4.
  assert(!floatActivations);
  assert(slicWindowWidth == 4);

  const unsigned inputDataPasses = numConvChains == 4 ? 1 : 2;
  // Similar, but different function for the 8 convUnits, half partials case
  const bool half8Conv = (numConvChains == 2 && floatPartials == false);
  const unsigned loopDecisionThreshold = (half8Conv ? 6 : 5);

  std::uint64_t maxWorkerCycles = 0;

  std::uint64_t workerProcessGroupPreambleCycles =
      2 +                   // Get worker ID
      (half8Conv ? 2 : 3) + // Load and maybe switch output pointers
      1 +                   // Load input pointer
      2 +                   // Load worklist DeltaN for worker
      4 +                   // Unpack DeltaN
      2 + // Load base pointer for DeltaN and add to form final worklist pointer
      2 + // Divide number of work items in the list by 3
      1 + // Load implicit zero flag + strides from stack
      (half8Conv ? 1 : 0); // Implicit zero loop decision

  if (convGroupsPerGroup > 4) {
    // 8 and 16 groupings workers uses extra 4 cycles
    // 1 - load group index
    // 3 - update 3 pointer by the group index
    workerProcessGroupPreambleCycles += 4;
  }
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

inline std::uint64_t getConvPartialSlicSupervisorInnerLoopCycleEstimateQuarter(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvChains,
    unsigned slicWindowWidth, unsigned convGroupsPerGroup,
    unsigned outputStride, bool implicitZeroing) {
  // TODO: we currently only target kernel width of 4.
  assert(slicWindowWidth == 4);

  const unsigned inputDataPasses = 2;
  // A loop or individual treatment for less than this number of outputs
  const unsigned loopDecisionThresholdStride1 = 5;
  const unsigned loopDecisionThresholdStride2 = 3;

  std::uint64_t maxWorkerCycles = 0;

  const std::uint64_t workerProcessGroupPreambleCycles = 20;
  const std::uint64_t whileLoopOverhead = 24;
  const std::uint64_t additionalOverheadWithIf = 18;

  // worker partitions is indexed by [worker][partitions].
  for (const auto &worker : workerPartitions) {
    std::uint64_t workerCycles = workerProcessGroupPreambleCycles;

    for (const auto &numFieldElems : worker) {
      workerCycles += whileLoopOverhead;

      std::uint64_t rowCycles = 0;

      if (outputStride == 1 && implicitZeroing) {
        if (numFieldElems < loopDecisionThresholdStride1) {
          const std::array<unsigned, loopDecisionThresholdStride1 - 1> cycles =
              {5, 9, 12, 15};
          rowCycles += 6 + cycles[numFieldElems];
        } else {
          rowCycles += 14 + (numFieldElems - loopDecisionThresholdStride1);
        }
      } else if (outputStride == 1 && !implicitZeroing) {
        // C++ if statement and pointer arithmetic
        rowCycles += additionalOverheadWithIf;

        if (numFieldElems < loopDecisionThresholdStride1) {
          const std::array<unsigned, loopDecisionThresholdStride1 - 1> cycles =
              {7, 10, 14, 14};
          rowCycles += 2 + cycles[numFieldElems];
        } else {
          rowCycles += 12 + (numFieldElems - loopDecisionThresholdStride1);
        }
      } else {
        // outputStride == 2
        if (numFieldElems < loopDecisionThresholdStride2) {
          rowCycles += 4 + (numFieldElems == 1 ? 6 : 11);
        } else {
          rowCycles += 13 + 2 * (numFieldElems - loopDecisionThresholdStride2);
        }
      }
      // Account for multiple passes per worker call
      workerCycles += rowCycles * inputDataPasses;
    }
    maxWorkerCycles = std::max(maxWorkerCycles, workerCycles);
  }
  return maxWorkerCycles * numWorkerContexts;
}

inline std::uint64_t getConvPartialSlicSupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, unsigned numConvChains,
    unsigned slicWindowWidth, unsigned convGroupsPerGroup,
    const poplar::Type &actsType, bool floatPartials, unsigned outputStride,
    bool implicitZeroing) {
  if (actsType == poplar::QUARTER) {
    return getConvPartialSlicSupervisorInnerLoopCycleEstimateQuarter(
        workerPartitions, numWorkerContexts, numConvChains, slicWindowWidth,
        convGroupsPerGroup, outputStride, implicitZeroing);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartialSlicSupervisorInnerLoopCycleEstimateHalfFloat(
        workerPartitions, numWorkerContexts, numConvChains, slicWindowWidth,
        convGroupsPerGroup, floatActivations, floatPartials, outputStride,
        implicitZeroing);
  }
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
  for (unsigned i = 0; i < gccs::ceilLog2(accVectorWidth); ++i) {
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
    auto MACStride = std::lcm(outputStride, inputDilation) / outputStride;
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

inline unsigned getConvLoadPerCycle(bool floatActivations,
                                    const poplar::Target &target) {
  if (floatActivations) {
    return target.getFp32ConvUnitInputLoadElemsPerCycle();
  } else {
    return target.getFp16ConvUnitInputLoadElemsPerCycle();
  }
}

inline std::uint64_t getConvPartialnx1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    const std::vector<unsigned> &kernelShape, unsigned filterHeight,
    unsigned outChansPerGroup, unsigned weightBytesPerConvUnit,
    unsigned numConvUnits, unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts, const poplar::Type &actsType,
    bool floatPartials, const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride) {
  const auto kernelElements = product(kernelShape);
  const auto partition = partitionConvPartialByWorker(
      batchElements, vectorConvert<unsigned>(outShape), numWorkerContexts,
      inputDilation, stride);

  // use conv nx1 vertex
  const unsigned positionsOuter = gccs::ceildiv(kernelShape[0], filterHeight);
  const unsigned numKernelPositions =
      (positionsOuter * kernelElements / kernelShape[0]);
  const auto outStrideX =
      inputDilation.back() / std::gcd(inputDilation.back(), stride.back());

  // workList is indexed by [context][numPartitions]
  // worklist for each kernel position is assumed to be the same.
  std::vector<std::vector<WorklistDataType>> workList;
  workList.reserve(partition.size());
  for (std::size_t context = 0; context < partition.size(); ++context) {
    workList.emplace_back();
    workList.back().reserve(partition[context].size());
    for (const auto &partialRow : partition[context]) {
      const auto workerOutWidth = partialRow.xEnd - partialRow.xBegin;
      const auto numFieldPos = gccs::ceildiv(workerOutWidth, outStrideX);
      if (numFieldPos) {
        workList.back().emplace_back(numFieldPos);
      }
    }
  }
  const auto kernelInnerElems = numKernelPositions / positionsOuter;
  const auto kernelOuterElems = positionsOuter;

  return getConvPartialnx1SupervisorInnerLoopCycleEstimate(
      workList, kernelInnerElems, kernelOuterElems, filterHeight,
      outChansPerGroup, weightBytesPerConvUnit, numConvUnits,
      convUnitCoeffLoadBytesPerCycle, numWorkerContexts, actsType,
      floatPartials);
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimate(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, const poplar::Type &actsType,
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
  if (actsType == poplar::QUARTER) {
    return getConvPartial1x1SupervisorInnerLoopCycleEstimateQuarter(
        worklist, numWorkerContexts, numConvUnits, zeroPartials);
  } else {
    return getConvPartial1x1SupervisorInnerLoopCycleEstimateHalfFloat(
        worklist, numWorkerContexts, numConvUnits, zeroPartials,
        actsType == poplar::FLOAT, floatPartials);
  }
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, const poplar::Type &actsType,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, actsType, floatPartials, true);
}

inline std::uint64_t getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
    unsigned batchElements, const std::vector<unsigned> &outShape,
    unsigned numWorkerContexts, unsigned numConvUnits,
    const std::vector<unsigned> &inputDilation,
    const std::vector<unsigned> &stride, const poplar::Type &actsType,
    bool floatPartials) {
  return getConvPartial1x1InnerLoopCycleEstimate(
      batchElements, outShape, numWorkerContexts, numConvUnits, inputDilation,
      stride, actsType, floatPartials, false);
}

inline std::uint64_t getConvPartialSlicInnerLoopCycles(
    unsigned outStride, bool implicitZeroing, unsigned batchElements,
    const std::vector<unsigned> &outShape, unsigned numWorkerContexts,
    unsigned numConvChains, unsigned slicWindowWidth,
    unsigned convGroupsPerGroup, const poplar::Type &actsType,
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
  if (actsType == poplar::QUARTER) {
    return getConvPartialSlicSupervisorInnerLoopCycleEstimateQuarter(
        worklist, numWorkerContexts, numConvChains, slicWindowWidth,
        convGroupsPerGroup, outputStride.back(), implicitZeroing);
  } else {
    auto floatActivations = actsType == poplar::FLOAT;
    return getConvPartialSlicSupervisorInnerLoopCycleEstimateHalfFloat(
        worklist, numWorkerContexts, numConvChains, slicWindowWidth,
        convGroupsPerGroup, floatActivations, floatPartials,
        outputStride.back(), implicitZeroing);
  }
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
    const auto depthThisStage = gccs::ceildiv(remainingDepth, d);
    remainingDepth = gccs::ceildiv(remainingDepth, depthThisStage);
    const auto stageOutputIsFloat =
        remainingDepth == 1 ? floatOutput : floatPartials;
    outputSizeThisStage = gccs::ceildiv(outputSizeThisStage, depthThisStage);

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
    unsigned activationsVectorWidth, bool floatActivations, bool floatPartials,
    unsigned inChansPerGroup, unsigned outChansPerGroup,
    unsigned dataPathWidth) {
  unsigned rowSplitFactor = numWorkers / std::gcd(numWorkers, numOutRows);
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
      numWorkers, activationsVectorWidth, floatActivations, floatPartials);
}

inline std::uint64_t estimateConvPartialVerticalMacInnerLoopCycles(
    unsigned tileOutHeight, unsigned tileOutWidth, unsigned batchSize,
    unsigned tileKernelHeight, unsigned tileKernelWidth, unsigned numWorkers,
    bool floatActivations, bool floatPartials, unsigned inChansPerGroup,
    unsigned outChansPerGroup, unsigned convGroupsPerGroup) {
  unsigned numRows = tileOutHeight * tileOutWidth * batchSize;
  unsigned rowSplitFactor = numWorkers / std::gcd(numWorkers, numRows);
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
