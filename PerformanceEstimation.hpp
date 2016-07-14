#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <algorithm>
#include <cassert>
#include <cstdint>

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
getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &
    convSizesByWeightAndWorker,
    unsigned convUnitPipelineDepth,
    unsigned numConvUnitsPerTile) {
  const auto numWorkerContexts = 6;

  unsigned cycles = 0;
  for (const auto &convSizesByWorker : convSizesByWeightAndWorker) {
    assert(convSizesByWorker.size() <= numWorkerContexts);
    // Load weights in the supervisor.
    const auto numLoads = convUnitPipelineDepth * numConvUnitsPerTile;
    cycles += numLoads;
    unsigned maxWorkerCycles = 0;
    // Start workers.
    for (const auto &convSizes : convSizesByWorker) {
      unsigned workerCycles = 0;
      const auto vertexOverhead = 5U;
      const auto coolDownCycles = 5U;
      workerCycles += vertexOverhead;
      for (const auto convSize : convSizes) {
        workerCycles += 2 + convSize * convUnitPipelineDepth;
      }
      workerCycles += coolDownCycles;
      maxWorkerCycles = std::max(maxWorkerCycles, workerCycles);
    }
    cycles += maxWorkerCycles * numWorkerContexts;
    cycles += 1; // Sync.
    cycles += numWorkerContexts - 1; // Pipeline bubble.
  }
  return cycles;
}

inline std::uint64_t
getConvPartial1x1CycleWorkerEstimate(
    const std::vector<std::vector<unsigned>> &convSizesByWeight,
    unsigned convUnitPipelineDepth,
    unsigned numConvUnitsPerTile) {
  const unsigned vertexOverhead = 5;
  unsigned cycleCount = vertexOverhead;
  for (const auto &convSizes : convSizesByWeight) {
    const auto numElements = std::accumulate(convSizes.begin(), convSizes.end(),
                                             0);
    const auto pointerLoadCycles = 2 * convSizes.size();
    unsigned warmUpCycles = numConvUnitsPerTile * convUnitPipelineDepth + 3;


    unsigned innerLoopCycles =
        numElements * convUnitPipelineDepth;
    unsigned coolDownCycles = 5;
    cycleCount += warmUpCycles + innerLoopCycles + coolDownCycles +
                  pointerLoadCycles;
  }
  return cycleCount;
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

#endif // _performance_estimation_h_
