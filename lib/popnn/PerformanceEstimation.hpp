#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include "popnn/NonLinearityDef.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>

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
getConvPartialnx1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &
    convSizesByWeightAndWorker,
    unsigned convUnitPipelineDepth,
    unsigned numConvUnitsPerTile,
    unsigned convUnitCoeffLoadBytesPerCycle,
    unsigned numInputPointers) {
  const unsigned numOutputPointers = 1;
  const auto numWorkerContexts = 6;
  const auto coeffBytesPerPipelineStage = 8;

  const unsigned supervisorNonLoopOverhead = 21U;

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
      const auto vertexOverhead = 20U;
      workerCycles += vertexOverhead;

      for (const auto convSize : convSizes) {
        /* inner loop overhead includes cycles to warm-up and cool down AMP loop
         */
        const unsigned innerLoopOverhead = 8;

        /* Cycles to form packed addresses */
        const unsigned packedAddrCompCyles = std::max(numInputPointers,
                                                      numOutputPointers) *
                                             9;
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
getConvPartialByDotProductCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                                        unsigned kernelWidth,
                                        unsigned inputGroupsPerOutput,
                                        unsigned outputWidth,
                                        unsigned dataPathWidth,
                                        unsigned outputStride)
{
  unsigned vertexOverhead = 15;
  unsigned innerLoopCycles =
      getDenseDotProductCycles(isFloat, kernelWidth * inChansPerGroup,
                               dataPathWidth) / outputStride;
  unsigned middleLoopCycles = inputGroupsPerOutput * (5 + innerLoopCycles);
  unsigned outerLoopCycles = outputWidth * (10 + middleLoopCycles);

  return vertexOverhead + outerLoopCycles;
}

inline std::uint64_t
getFullyConnectedPartialCycleEstimate(bool isFloat, unsigned size,
                                      unsigned dataPathWidth) {
  return 5 + getDenseDotProductCycles(isFloat, size, dataPathWidth);
}

inline std::uint64_t
getFullyConnectedBwdCycleEstimate(unsigned size) {
  // Inner loop is dominated by loads (load pointer, load 64bits, load 16
  // bits). This could be improved if we uses strided loads instead of
  // pointers.
  return 5 + size * 3;
}

inline std::uint64_t
getWeightGradAopCycles(bool floatInput, bool floatPartials,
                       unsigned dataPathWidth, unsigned inChansPerGroup,
                       unsigned outChansPerGroup,
                       std::vector<std::vector<unsigned>> &shape) {
  // Outer loop overhead for stack variable initialisations
  const auto vertexOverhead = 22;
  std::uint64_t cycles = vertexOverhead;
  unsigned vectorWidth = dataPathWidth / (floatInput ? 32 : 16);
  unsigned partialsVectorWidth = dataPathWidth / (floatPartials ? 32 : 16);
  const auto ocgSubgroups = (outChansPerGroup + vectorWidth - 1) / vectorWidth;
  const auto icgSubgroups = (inChansPerGroup + vectorWidth - 1) / vectorWidth;

  for (const auto &w : shape) {
    const auto shapeOverhead = 19;

    auto innerLoopCycles = 0;
    for (auto deltasWidth : w) {
      // Inner loop.
      // Inner loop overhead required to set-up pointers, rpt loop and branching
      const auto innerLoopOverhead = 15;
      innerLoopCycles += deltasWidth + innerLoopOverhead;
    }

    // Accumulator restore and save
    const auto accSave = 1 + vectorWidth * vectorWidth / partialsVectorWidth;
    const auto ocgSubgroupOverhead = 13;
    const auto icgSubgroupOverhead = 6;

    cycles += shapeOverhead
              + ocgSubgroups
               * (ocgSubgroupOverhead + icgSubgroups * (icgSubgroupOverhead
                                                        + accSave
                                                        + innerLoopCycles));
  }
  return cycles;
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

inline uint64_t getNonLinearityCycles(unsigned numActivations,
                                      NonLinearityType nonLinearityType,
                                      bool isFloat,
                                      unsigned dataPathWidth)
{
  const auto floatVectorWidth = dataPathWidth / 32;
  const auto halfVectorWidth =  dataPathWidth / 16;
  const auto overhead = 15;
  switch (nonLinearityType) {
  case NON_LINEARITY_NONE:
  case NON_LINEARITY_RELU:
    {
      const unsigned vertexOverhead = 2      // run instruction
                                      + 12;   // overhead (assuming pointers)
      unsigned cycles = vertexOverhead;

      const unsigned numBlocks = isFloat ?
                (numActivations + floatVectorWidth - 1) / floatVectorWidth :
                (numActivations + halfVectorWidth - 1) / halfVectorWidth;
      cycles += (numBlocks / 2) * 3 + (numBlocks & 1);
      return cycles;
    }
  case NON_LINEARITY_SIGMOID:
    // scalar operation for floats, vector operation for halves
    // transcendtal operations are ~10cyles for float, ~1cycles for half
    if (isFloat) {
      return overhead + numActivations * 10;
    } else {
      return overhead + (numActivations + halfVectorWidth - 1)
                        / halfVectorWidth;
    }
  }
  throw std::runtime_error("Invalid nonlinearity type");
}


inline uint64_t getBwdNonlinearityDerivativeCycles(
                  unsigned numDeltas,
                  NonLinearityType nonLinearityType,
                  bool isFloat,
                  unsigned dataPathWidth) {

  unsigned vectorWidth = dataPathWidth / (isFloat ? 32 : 16);
  unsigned numVectors = (numDeltas + vectorWidth - 1) / vectorWidth;

  switch (nonLinearityType) {
  case NON_LINEARITY_SIGMOID:
    return 5 + numVectors * 3;
  case NON_LINEARITY_RELU:
    {
      const unsigned vertexOverhead = 2    // run instruction
                                      + 7; // remaining vertex overhead
      return vertexOverhead + numVectors * 3;
    }
    case NON_LINEARITY_NONE:
      return 5 + numVectors;
  }
  throw std::runtime_error("Invalid nonlinearity type");
}

#endif // _performance_estimation_h_
