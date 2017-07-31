#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearity.hpp"
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
    unsigned numChans, unsigned convCount, unsigned totalConvSize,
    unsigned dataPathWidth) {
  std::uint64_t cycles = 5;
  cycles +=
    4 * convCount +
    totalConvSize * getDenseDotProductCycles(isFloat, numChans, dataPathWidth);
  return cycles;
}

inline std::uint64_t
getConvPartialHorizontalMacCycleEstimate(
    bool isFloat,
    unsigned numChans,
    const std::vector<unsigned> &convSizes,
    unsigned dataPathWidth) {
  const auto convCount = convSizes.size();
  const auto totalConvSize = std::accumulate(convSizes.begin(), convSizes.end(),
                                             0U);
  return getConvPartialHorizontalMacCycleEstimate(isFloat, numChans, convCount,
                                                  totalConvSize, dataPathWidth);
}

inline std::uint64_t
getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &
    convSizesByWeightAndWorker,
    unsigned convUnitPipelineDepth,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numInputPointers,
    bool useDeltasForEdges) {
  const unsigned numOutputPointers = 1;
  const auto numWorkerContexts = 6;
  const auto coeffBytesPerPipelineStage = 8;

  const unsigned supervisorNonLoopOverhead = 14U;

  unsigned cycles = supervisorNonLoopOverhead;

  for (const auto &convSizesByWorker : convSizesByWeightAndWorker) {
    assert(convSizesByWorker.size() <= numWorkerContexts);

    // Load weights in the supervisor.
    const auto numLoads = convUnitPipelineDepth
                          * numConvUnitsPerTile
                          * coeffBytesPerPipelineStage
                          / convUnitCoeffLoadBytesPerCycle;

    cycles += numLoads;

    const unsigned supervisorLoopOverhead = 2;

    cycles += supervisorLoopOverhead; // overhead to modify supervisor struct

    unsigned maxWorkerCycles = 0;
    // Start workers.
    for (const auto &convSizes : convSizesByWorker) {
      unsigned workerCycles = 0;
      const auto vertexOverhead = 21U;
      workerCycles += vertexOverhead;

      for (const auto convSize : convSizes) {
        /* inner loop overhead includes cycles to warm-up and cool down AMP loop
         */
        const unsigned innerLoopOverhead = 10;

        /* Cycles to form packed addresses */
        const unsigned packedAddrCompCyles = std::max(numInputPointers,
                                                      numOutputPointers) *
                                             (useDeltasForEdges ? 5 : 3);
        workerCycles += innerLoopOverhead +
                        packedAddrCompCyles +
                        convSize * convUnitPipelineDepth;
      }
      maxWorkerCycles = std::max(maxWorkerCycles, workerCycles);
    }
    cycles += maxWorkerCycles * numWorkerContexts;
    cycles += 2 * numWorkerContexts; // run instruction
    cycles += 1; // Sync.
    cycles += numWorkerContexts - 1; // Pipeline bubble.
  }
  return cycles;
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
