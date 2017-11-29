#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
#include <limits>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

inline std::uint64_t
getDenseDotProductCycles(bool isFloat, unsigned size, unsigned dataPathWidth) {
  if (isFloat) {
    const auto floatVectorWidth = dataPathWidth / 32;
    return (size + floatVectorWidth - 1) / floatVectorWidth + 2;
  }
  const auto halfVectorWidth = dataPathWidth / 16;
  return (size + halfVectorWidth - 1) / halfVectorWidth + 2;
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

inline std::uint64_t
getConvPartialHorizontalMacCycleEstimate(
    bool isFloat,
    unsigned numChans,
    const std::vector<unsigned> &convSizes,
    unsigned dataPathWidth) {
  uint64_t cycles = 0;
  for (auto convSize : convSizes) {
    // 3 to load in offset, out offset and width,
    // 3 to setup pointers + tapack
    // 1 additional cycles for pipeline
    cycles += 7 + convSize * getDenseDotProductCycles(isFloat, numChans,
                                                      dataPathWidth);
  }
  return cycles;
}

inline std::uint64_t
getZeroSupervisorVertexCycleEstimate(const std::vector<unsigned> &worklist,
                                     unsigned numGroups,
                                     unsigned dataPathWidth,
                                     unsigned numWorkerContexts,
                                     bool isFloat,
                                     bool useDeltasForEdges) {
  const unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  std::uint64_t maxWorkerCyclesZero = 0;
  for (unsigned context = 0; context != worklist.size(); ++context) {
    uint64_t numVectors = (worklist[context] + vectorWidth - 1) / vectorWidth;
    maxWorkerCyclesZero = std::max(maxWorkerCyclesZero, numVectors + 4);
  }
  uint64_t zeroCycles = ((maxWorkerCyclesZero * numGroups + useDeltasForEdges) *
                         numWorkerContexts + 12);
  return zeroCycles;
}

inline std::uint64_t
getConvPartialHorizontalMacSupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups,
    unsigned kernelSize,
    unsigned numInChansPerGroup,
    unsigned dataPathWidth,
    unsigned numWorkerContexts,
    bool isFloat) {
  unsigned usedContexts = workerPartitions.size();
  uint64_t cycles = 0;
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts ?
                               0 : std::numeric_limits<uint64_t>::max();
  for (auto context = 0U; context != usedContexts; ++context) {
    uint64_t thisWorkerCycles = 0;
    for (auto k = 0U; k != kernelSize; ++k) {
      thisWorkerCycles +=
        getConvPartialHorizontalMacCycleEstimate(isFloat, numInChansPerGroup,
                                                 workerPartitions[context][k],
                                                 dataPathWidth);
      // to load partition with post increment and branch
      thisWorkerCycles += 2;
    }
    const unsigned workerNonLoopOverhead = 6;
    thisWorkerCycles += workerNonLoopOverhead;
    maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }
  cycles += std::max(maxWorkerCycles, minWorkerCycles + 11);

  return 6  + numConvGroups
            * (16 + numOutGroups
                * (8 + numInGroups
                   * (2 + cycles)));
}

inline std::uint64_t
getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts,
    bool floatWeights,
    bool useDeltasForEdges) {
  unsigned usedContexts = workerPartitions.size();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts ?
                             0 : std::numeric_limits<uint64_t>::max();
  for (const auto &worker : workerPartitions) {
    uint64_t thisWorkerCycles = 9;
    for (auto wi : worker) {
      const auto numElems =  wi;
      if (numElems) {
        thisWorkerCycles += 20 + numElems * 4;
      } else {
        thisWorkerCycles += 3;
      }
    }
    maxWorkerCycles =
      std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
    minWorkerCycles =
      std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
  }

  // tag cost to worker with min cycles
  maxWorkerCycles = std::max(maxWorkerCycles, minWorkerCycles + 14);

  const auto numInputLoadsInnerLoop = 4;
  const auto numLoads = convUnitInputLoadElemsPerCycle * numInputLoadsInnerLoop
                          * numConvUnitsPerTile
                          * (floatWeights ? 4 : 2)
                          / convUnitCoeffLoadBytesPerCycle;
  if (useDeltasForEdges) {
    const uint64_t supervisorNonloopOverhead = 48;
    return supervisorNonloopOverhead + numConvGroups
           * (numInGroups
              * (9 + numOutGroups * (18 + numLoads + maxWorkerCycles)));
  } else {
    const uint64_t supervisorNonloopOverhead = 39;
    return supervisorNonloopOverhead + numConvGroups
           * (numInGroups
              * (8 + numOutGroups * (16 + numLoads + maxWorkerCycles)));
  }
}

inline std::uint64_t
getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numOutGroups,
    unsigned numInGroups,
    unsigned kernelSize,
    unsigned filterHeight,
    unsigned inChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts,
    bool floatWeights,
    unsigned useDeltaForEdges) {
  unsigned usedContexts = workerPartitions.size();
  const auto numInputLoadsInnerLoop = 4;
  const auto numLoads = convUnitInputLoadElemsPerCycle * numInputLoadsInnerLoop
                        * numConvUnitsPerTile
                        * (floatWeights ? 4 : 2)
                         / convUnitCoeffLoadBytesPerCycle;
  uint64_t cycles = 0;
  for (auto k = 0U; k != kernelSize; ++k) {
    // load coefficients
    cycles += numLoads + 9 + 8 * (filterHeight - 1);
    uint64_t maxWorkerCycles = 0;
    uint64_t minWorkerCycles = usedContexts < numWorkerContexts ?
                               0 : std::numeric_limits<uint64_t>::max();
    for (auto context = 0U; context != usedContexts; ++context) {
      uint64_t thisWorkerCycles = 15;
      for (auto &numElems :  workerPartitions[context][k]) {
        if (numElems) {
          thisWorkerCycles += 22 + numElems * 4;
        } else {
          thisWorkerCycles += 3;
        }
      }
      maxWorkerCycles =
        std::max(maxWorkerCycles, numWorkerContexts * thisWorkerCycles);
      minWorkerCycles =
        std::min(minWorkerCycles, numWorkerContexts * thisWorkerCycles);
    }
    cycles += std::max(maxWorkerCycles, minWorkerCycles + 9);
  }
  if (useDeltaForEdges) {
    return 6 + numConvGroups
               * (46 + numOutGroups
                 * (9 + numInGroups
                   * (21 + cycles)));
  } else {
    return 6 + numConvGroups
               * (42 + numOutGroups
                 * (9 + numInGroups
                   * (19 + cycles)));
  }
}

inline std::uint64_t
getMatMul1PartialCycleEstimate(bool isFloat, unsigned size,
                               unsigned dataPathWidth) {
  return 5 + getDenseDotProductCycles(isFloat, size, dataPathWidth);
}

inline std::uint64_t
getMatMul2CycleEstimate(unsigned size) {
  // Inner loop is dominated by loads (load pointer, load 64bits, load 16
  // bits). This could be improved if we uses strided loads instead of
  // pointers.
  return 5 + size * 3;
}

inline uint64_t getWgdDataTransformCycles(
                              unsigned numChannels,
                              bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 13 + 56 * ((numChannels + chansPerOp - 1)/chansPerOp);
}


inline uint64_t getWgdKernelTransformCycles(
                              unsigned numChannels,
                              bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 2 + 35 * ((numChannels + chansPerOp - 1)/chansPerOp);
}

inline uint64_t getWgdInvTransformCycles(
                              unsigned numChannels,
                              bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 15 + 30 * ((numChannels + chansPerOp - 1)/chansPerOp);
}

/**
 * The accumulator operates on pencils which are of depth "pencilDepth".
 * An inner product of a coefficient vector and data vector is computed.
 * "comPencils" gives the number of pencils which share a common coefficient
 * vector. "numPencils" gives a set of pencils which share common coefficients
 */
inline uint64_t getWgdAccumCycles(
                             unsigned numPencils,
                             unsigned comPencils,
                             unsigned pencilDepth,
                             unsigned outDepth,
                             unsigned numWorkers,
                             unsigned numConvUnits,
                             unsigned weightsPerConvUnit,
                             unsigned convUnitCoeffLoadBytesPerCycle,
                             bool isFloat) {

  unsigned numCoeffSets = (outDepth + numConvUnits - 1)/numConvUnits;
  numCoeffSets *= (pencilDepth + weightsPerConvUnit - 1)/weightsPerConvUnit;
  numCoeffSets *= numPencils;
  const auto coeffLoadCycles = numConvUnits * weightsPerConvUnit
                          * (isFloat ? 2 : 4) / convUnitCoeffLoadBytesPerCycle;
  const auto overhead = 4;

  const auto numPencilsPerWorker = (comPencils + numWorkers - 1) / numWorkers;
  return (overhead + coeffLoadCycles + numPencilsPerWorker
          * numWorkers * 4) * numCoeffSets;
}

inline uint64_t getWgdReduceCycles(unsigned numPencils, unsigned depth,
                          bool isFloat) {
  unsigned chansPerOp = isFloat ? 2 : 4;
  return 5 + ((numPencils * depth + chansPerOp - 1)/chansPerOp);
}


inline uint64_t getWgdCompleteCycles(
                            unsigned numChannels,
                            bool isFloat) {
  unsigned divFactor = isFloat ? 2 : 4;

  return 5 + numChannels/divFactor;
}

inline std::uint64_t
getOuterProductCycleEstimate(bool isFloat,
                             unsigned width, unsigned numChannels,
                             unsigned chansPerGroup,
                             unsigned dataPathWidth) {
  assert(numChannels % chansPerGroup == 0);
  const auto numChanGroups = numChannels / chansPerGroup;
  const auto elementWidth = isFloat ? 32 : 16;
  auto vectorsPerGroup =
      (chansPerGroup * elementWidth + dataPathWidth - 1) / dataPathWidth;
  // Taken from conv_outer_product_f16 microbenchmark.
  std::uint64_t cycles =
      9 + numChanGroups * (8 + width * (1 + vectorsPerGroup));
  return cycles;
}

#endif // _performance_estimation_h_
