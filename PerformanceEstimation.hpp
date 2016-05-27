#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <algorithm>
#include <cassert>
#include <cstdint>

inline std::uint64_t getDenseDotProductCycles(bool isFloat, unsigned size) {
  if (isFloat) {
    return (size + 1) / 2 + 2;
  }
  return (size + 3) / 4 + 2;
}

inline bool
canUseConvolutionInstruction(bool isFloat, unsigned stride,
                             unsigned inChansPerGroup,
                             unsigned partialChansPerGroup) {
  return !isFloat && stride < (1 << 4) && inChansPerGroup == 16 &&
         partialChansPerGroup == 4;
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
getConvPartial1x1CycleEstimate(
    const std::vector<std::vector<unsigned>> &convSizesByWeight,
    bool supervisorVertex) {
  const auto numWorkerContexts = 6;
  const auto inChansPerGroup = 16;
  const auto partialChansPerGroup = 4;
  if (supervisorVertex) {
    unsigned cycleCount = 0;
    for (const auto &convSizes : convSizesByWeight) {
      if (convSizes.empty())
        continue;
      assert(allEqual(convSizes.begin(), convSizes.end()));
      const auto convSize = convSizes.front();
      // Load weights in the supervisor.
      const auto numBytes = inChansPerGroup * partialChansPerGroup * 2;
      cycleCount += (numBytes + 7) / 8;
      // Start 6 workers.
      const auto numElements =
          std::accumulate(convSizes.begin(), convSizes.end(), 0);
      unsigned maxWorkerCycleCount = 0;
      for (unsigned i = 0; i != numWorkerContexts; ++i) {
        const auto beginElement = (i * numElements) / numWorkerContexts;
        const auto endElement = ((i + 1) * numElements) / numWorkerContexts;
        if (beginElement == endElement)
          continue;
        const auto beginRow = beginElement / convSize;
        const auto endRow = 1 + (endElement - 1) / convSize;
        const auto workerPointerLoads = endRow - beginRow;
        const auto workerNumElements = endElement - beginElement;
        const auto vertexOverhead = 5U;
        const auto coolDownCycles = 5U;
        const auto workerCycleCount = vertexOverhead +
                                      workerNumElements * partialChansPerGroup +
                                      coolDownCycles +
                                      workerPointerLoads;
        maxWorkerCycleCount = std::max(maxWorkerCycleCount, workerCycleCount);
      }
      cycleCount += maxWorkerCycleCount * numWorkerContexts;
      cycleCount += 1; // Sync.
      cycleCount += numWorkerContexts - 1; // Pipeline bubble.
    }
    return cycleCount;
  }
  const unsigned vertexOverhead = 5;
  unsigned cycleCount = vertexOverhead;
  for (const auto &convSizes : convSizesByWeight) {
    const auto numElements = std::accumulate(convSizes.begin(), convSizes.end(),
                                             0);
    const auto pointerLoadCycles = convSizes.size();
    const auto partialChansPerGroup = 4;
    unsigned warmUpCycles = 19;
    unsigned innerLoopCycles =
        numElements * partialChansPerGroup;
    unsigned coolDownCycles = 5;
    cycleCount += warmUpCycles + innerLoopCycles + coolDownCycles +
                  pointerLoadCycles;
  }
  return cycleCount;
}

inline std::uint64_t
getConvPartialCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                            unsigned stride, unsigned kernelWidth,
                            unsigned inputGroupsPerOutput,
                            unsigned outputHeight,
                            unsigned outputWidth,
                            unsigned outChansPerGroup,
                            bool useSupervisorVertices)
{
  if (canUseConvolutionInstruction(isFloat, stride, inChansPerGroup,
                                   outChansPerGroup)) {
    std::vector<std::vector<unsigned>> convSizesByWeight;
    for (unsigned i = 0; i != inputGroupsPerOutput * kernelWidth; ++i) {
      convSizesByWeight.emplace_back();
      for (unsigned j = 0; j != outputHeight; ++j) {
        convSizesByWeight.back().push_back(outputWidth);
      }
    }
    return getConvPartial1x1CycleEstimate(convSizesByWeight,
                                          useSupervisorVertices);
  }
  assert(!useSupervisorVertices);
  unsigned vertexOverhead = 5;
  return vertexOverhead +
         outChansPerGroup * outputWidth * outputHeight * inputGroupsPerOutput *
         (1 + getDenseDotProductCycles(isFloat, kernelWidth * inChansPerGroup));
}

inline std::uint64_t
getFullyConnectedPartialCycleEstimate(bool isFloat, unsigned size) {
  return 5 + getDenseDotProductCycles(isFloat, size);
}

#endif // _performance_estimation_h_
