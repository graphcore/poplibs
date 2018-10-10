#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <limits>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

inline std::uint64_t
getDenseDotProductCycles(bool isFloat, unsigned size, unsigned dataPathWidth) {
  if (isFloat) {
    unsigned vecLen = (size % 2) == 0 ? size / 2 : size;
    return vecLen + 4;
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

inline std::uint64_t
getConvPartialHorizontalMacCycleEstimate(
    bool isFloat,
    unsigned numInChans,
    unsigned numOutChans,
    const std::vector<unsigned> &convSizes,
    unsigned dataPathWidth) {
  uint64_t cycles = 16;
  for (auto convSize : convSizes) {
    if (convSize == 0) {
      cycles += 7;
    } else {
      cycles += 19 + convSize * (7 + numOutChans *
                                 getDenseDotProductCycles(isFloat, numInChans,
                                                          dataPathWidth));
    }
  }
  return cycles;
}

inline std::uint64_t
getZeroSupervisorVertexCycleEstimate(const std::vector<unsigned> &worklist,
                                     unsigned numGroups,
                                     unsigned dataPathWidth,
                                     unsigned numWorkerContexts,
                                     bool isFloat) {
  const unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  std::uint64_t maxWorkerCyclesZero = 0;
  for (unsigned context = 0; context != worklist.size(); ++context) {
    uint64_t numVectors = (worklist[context] + vectorWidth - 1) / vectorWidth;
    maxWorkerCyclesZero = std::max(maxWorkerCyclesZero, numVectors + 4);
  }
  uint64_t zeroCycles = ((maxWorkerCyclesZero * numGroups) *
                         numWorkerContexts + 12);
  return zeroCycles;
}

inline std::uint64_t
getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelSize,
    unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup,
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
                                                 numOutChansPerGroup,
                                                 workerPartitions[context][k],
                                                 dataPathWidth);
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
    std::uint64_t innerLoopCycles,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups) {
  uint64_t cycles = innerLoopCycles;
  return 62 + numConvGroups
            * (21 + numInGroups
                * (15 + numOutGroups
                   * (10 + cycles)));
}

inline std::uint64_t
getConvPartialHorizontalMacSupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups,
    unsigned kernelSize,
    unsigned numInChansPerGroup,
    unsigned numOutChansPerGroup,
    unsigned dataPathWidth,
    unsigned numWorkerContexts,
    bool isFloat) {
  auto cycles =
      getConvPartialHorizontalMacSupervisorInnerLoopCycleEstimate(
        workerPartitions, kernelSize, numInChansPerGroup, numOutChansPerGroup,
        dataPathWidth, numWorkerContexts, isFloat);
  return getConvPartialHorizontalMacSupervisorOuterLoopCycleEstimate(
        cycles, numConvGroups, numInGroups, numOutGroups);
}

inline std::uint64_t
getConvPartial1x1SupervisorInnerLoopCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numWorkerContexts, bool outputZeroing) {
  unsigned usedContexts = workerPartitions.size();
  uint64_t maxWorkerCycles = 0;
  uint64_t minWorkerCycles = usedContexts < numWorkerContexts ?
                             0 : std::numeric_limits<uint64_t>::max();
  // TODO: These estimates are incorrect for float inputs
  for (const auto &worker : workerPartitions) {
    // fixed overhead for loading pointers worklist pointers and dividing
    // partitions by 3
    uint64_t thisWorkerCycles = 16;
    for (auto wi : worker) {
      const auto numElems =  wi;
      switch (numElems) {
        case 0:
          thisWorkerCycles += 15;
          break;
        case 1:
          thisWorkerCycles += 40 + (2 + 4) * outputZeroing;
          break;
        case 2:
          thisWorkerCycles += 40 + (2 + 4 * 2) * outputZeroing;
          break;
        default:
          thisWorkerCycles += 40 + (2 + 4 * numElems) * outputZeroing +
                                   (numElems - 3) * 4;
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

inline std::uint64_t
getConvPartial1x1SupervisorOuterLoopCycleEstimate(
    std::uint64_t innerLoopCyclesWithZeroing,
    std::uint64_t innerLoopCyclesWithoutZeroing,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups,
    unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    bool floatWeights) {
  const auto outputPassesPerGroup =
      (outChansPerGroup + numConvUnitsPerTile - 1) / numConvUnitsPerTile;

  const auto numInputLoadsInnerLoop = 4;
  const auto numLoads = convUnitInputLoadElemsPerCycle * numInputLoadsInnerLoop
                          * numConvUnitsPerTile
                          * (floatWeights ? 4 : 2)
                          / convUnitCoeffLoadBytesPerCycle;
  const uint64_t supervisorNonloopOverhead = 55;
  return supervisorNonloopOverhead + numConvGroups
           * (13 + (numInGroups - 1)
              * (13 + numOutGroups
                 * (11 + outputPassesPerGroup
                   * (6 + numLoads + innerLoopCyclesWithoutZeroing))) +
                (13 + numOutGroups
                 * (11 + outputPassesPerGroup
                   * (6 + numLoads + innerLoopCyclesWithZeroing))));
}

inline std::uint64_t
getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<unsigned>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numInGroups,
    unsigned numOutGroups,
    unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts,
    bool floatWeights) {
  auto innerLoopCyclesWithZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(workerPartitions,
                                                        numWorkerContexts,
                                                        true);
  auto innerLoopCyclesWithoutZeroing =
      getConvPartial1x1SupervisorInnerLoopCycleEstimate(workerPartitions,
                                                        numWorkerContexts,
                                                        false);
  return getConvPartial1x1SupervisorOuterLoopCycleEstimate(
            innerLoopCyclesWithZeroing, innerLoopCyclesWithoutZeroing,
            numConvGroups, numInGroups, numOutGroups,
            outChansPerGroup, convUnitInputLoadElemsPerCycle,
            numConvUnitsPerTile, convUnitCoeffLoadBytesPerCycle, floatWeights);
}

inline std::uint64_t
getConvPartialnx1SupervisorCycleOuterLoopEstimate(
    std::uint64_t innerLoopCycles,
    unsigned numConvGroups,
    unsigned numOutGroups,
    unsigned numInGroups,
    unsigned outChansPerGroup,
    unsigned numConvUnitsPerTile) {
  uint64_t cycles = innerLoopCycles;
  return 93 + numConvGroups
             * (15 + numOutGroups
              * (16 + numInGroups
                * (16 + cycles)));
}

inline std::uint64_t
getConvPartialnx1SupervisorCycleInnerLoopEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned kernelInnerElems,
    unsigned kernelOuterElems,
    unsigned filterHeight,
    unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts,
    bool floatWeights) {
  unsigned usedContexts = workerPartitions.size();
  unsigned numOutChanPasses = outChansPerGroup / numConvUnitsPerTile;
  // TODO: Update for float input when assembler code is written
  if (filterHeight == 4 &&  convUnitCoeffLoadBytesPerCycle >= 8)
    convUnitCoeffLoadBytesPerCycle = 8;
  const auto numInputLoadsInnerLoop = 4;
  const auto numLoads = convUnitInputLoadElemsPerCycle * numInputLoadsInnerLoop
                        * numConvUnitsPerTile
                        * (floatWeights ? 4 : 2)
                        / convUnitCoeffLoadBytesPerCycle;
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
  uint64_t innerLoopCycles = 0;
  for (auto ky = 0U; ky != kernelOuterElems; ++ky) {
    innerLoopCycles += 15;
    for (auto kx = 0U; kx != kernelInnerElems; ++kx) {
      // remove cycles for branch in outChanPasses loop for last iteration
      innerLoopCycles += 18 - 5;
      for (auto ocp = 0U; ocp != numOutChanPasses; ++ocp) {
        uint64_t maxWorkerCycles = 0;
        uint64_t minWorkerCycles = usedContexts < numWorkerContexts ?
                                   0 : std::numeric_limits<uint64_t>::max();
        for (auto context = 0U; context != usedContexts; ++context) {
          uint64_t thisWorkerCycles = 19;
          const auto k = ky * kernelInnerElems + kx;
          for (auto &numElems :  workerPartitions[context][k]) {
            switch (numElems) {
            case 0:
              thisWorkerCycles += 18;
              break;
            case 1:
              thisWorkerCycles += 30;
              break;
            case 2:
              thisWorkerCycles += 34;
              break;
            default:
              thisWorkerCycles += 35 + (numElems - 3) * 4;
            }
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

inline std::uint64_t
getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &workerPartitions,
    unsigned numConvGroups,
    unsigned numOutGroups,
    unsigned numInGroups,
    unsigned kernelInnerElems,
    unsigned kernelOuterElems,
    unsigned filterHeight,
    unsigned inChansPerGroup,
    unsigned outChansPerGroup,
    unsigned convUnitInputLoadElemsPerCycle,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numWorkerContexts,
    bool floatWeights) {
  auto innerLoopCycles =
      getConvPartialnx1SupervisorCycleInnerLoopEstimate(
        workerPartitions, kernelInnerElems, kernelOuterElems, filterHeight,
        outChansPerGroup, convUnitInputLoadElemsPerCycle, numConvUnitsPerTile,
        convUnitCoeffLoadBytesPerCycle, numWorkerContexts, floatWeights);
  return getConvPartialnx1SupervisorCycleOuterLoopEstimate(
           innerLoopCycles,
           numConvGroups,
           numOutGroups,
           numInGroups,
           outChansPerGroup,
           numConvUnitsPerTile);
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

  int cycles;
  // Conditions for executing a fast or slow path, replicated from the assembly
  // implementation
  if(isFloat) {
      if( (chansPerGroup >= 6) &&             // Min size of unrolled loop
          ((chansPerGroup & 1) == 0)  &&      // Loop processes 2 at once
          ((chansPerGroup/2 -3) < 0x1000) &&  // hardware RPT count constraint
          ((chansPerGroup/2 +1)  < 512)) {    // Stride size contstraint

          // Float, Fast path cycle estimates
          cycles = 25 + numChanGroups *
                  (11 + width * (6 + (chansPerGroup-6)/2));
      }
      else {
          // Float, Slow path cycle estimates
          cycles = 25 + numChanGroups *
                  (11 + width * (10 + chansPerGroup*2));
      }
  }
  else {
       if( (chansPerGroup >= 12) &&           // Min size of unrolled loop
          ((chansPerGroup & 3) == 0)  &&      // Loop processes 2 at once
          ((chansPerGroup/4 -3) < 0x1000) &&  // hardware RPT count constraint
          ((chansPerGroup/4 +1) < 512)) {     // Stride size contstraint

          // Half, Fast path cycle estimates
          cycles = 25 + numChanGroups *
                  (10 + width * (6 + (chansPerGroup-12)/4));
      }
      else {
          // Half, Slow path cycle estimates
          cycles = 25 + numChanGroups *
                  (10 + width * (10 + (chansPerGroup*5)/2));
      }
  }
  return cycles;
}

inline uint64_t
getReduceCycleEstimate(unsigned outSize,
                       unsigned partialsSize,
                       unsigned dataPathWidth,
                       bool isOutTypeFloat,
                       bool isPartialsFloat,
                       unsigned numWorkers) {
  unsigned cycles = 0;

  // Supervisor vertex, and new implementation
  if (isPartialsFloat) {
    cycles = 32;
    // Float - workers process 4 at once, and account for remainder loops
    auto loops = outSize/4;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers =  loops/numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 3)
        cycles += (2 * partialsSize + 13);
      else
        loopsDividedBetweenWorkers++;
    }

    if(isOutTypeFloat)
      cycles += (3 * partialsSize + 7) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 6) * loopsDividedBetweenWorkers;
  }
  else{
    cycles = 32;
    // Half - workers process 8 at once, and account for remainder loops
    auto loops = outSize/8;
    if (outSize & 1)
      loops++;
    if (outSize & 2)
      loops++;
    if (outSize & 4)
      loops++;
    // Account for time at full load - all workers busy
    auto loopsDividedBetweenWorkers =  loops/numWorkers;
    // and a remainder where only some are busy which can be a shorter loop
    if (loops % numWorkers) {
      if (outSize & 7)
        cycles += (2 * partialsSize + 11);
      else
        loopsDividedBetweenWorkers++;
    }

    if(isOutTypeFloat)
      cycles += (3 * partialsSize + 9) * loopsDividedBetweenWorkers;
    else
      cycles += (3 * partialsSize + 8) * loopsDividedBetweenWorkers;
  }
  cycles = cycles * numWorkers;

  return cycles;
}

#endif // _performance_estimation_h_
