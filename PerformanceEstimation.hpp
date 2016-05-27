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
getConvPartial1x1SupervisorCycleEstimate(
    const std::vector<std::vector<std::vector<unsigned>>> &
    convSizesByWeightAndWorker) {
  const auto numWorkerContexts = 6;
  const auto inChansPerGroup = 16;
  const auto partialChansPerGroup = 4;

  unsigned cycles = 0;
  for (const auto &convSizesByWorker : convSizesByWeightAndWorker) {
    assert(convSizesByWorker.size() <= numWorkerContexts);
    // Load weights in the supervisor.
    const auto numBytes = inChansPerGroup * partialChansPerGroup * 2;
    cycles += (numBytes + 7) / 8;
    unsigned maxWorkerCycles = 0;
    // Start workers.
    for (const auto &convSizes : convSizesByWorker) {
      unsigned workerCycles = 0;
      const auto vertexOverhead = 5U;
      const auto coolDownCycles = 5U;
      workerCycles += vertexOverhead;
      for (const auto convSize : convSizes) {
        workerCycles += 1 + convSize * partialChansPerGroup;
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
getConvPartial1x1CycleEstimate(
    const std::vector<std::vector<unsigned>> &convSizesByWeight,
    bool supervisorVertex) {
  const auto numWorkerContexts = 6;
  if (supervisorVertex) {
    std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
    convSizesByWeightAndWorker.reserve(convSizesByWeight.size());
    for (const auto &convSizes : convSizesByWeight) {
      if (convSizes.empty())
        continue;
      convSizesByWeightAndWorker.emplace_back();
      assert(allEqual(convSizes.begin(), convSizes.end()));
      const auto convSize = convSizes.front();
      const auto numElements =
          std::accumulate(convSizes.begin(), convSizes.end(), 0);
      auto &convSizesByWorker = convSizesByWeightAndWorker.back();
      convSizesByWorker.reserve(numWorkerContexts);
      for (unsigned i = 0; i != numWorkerContexts; ++i) {
        convSizesByWorker.emplace_back();
        const auto beginElement = (i * numElements) / numWorkerContexts;
        const auto endElement = ((i + 1) * numElements) / numWorkerContexts;
        if (beginElement == endElement)
          continue;
        const auto beginRow = beginElement / convSize;
        const auto endRow = 1 + (endElement - 1) / convSize;
        for (unsigned j = beginRow; j != endRow; ++j) {
          unsigned beginIndex = j == beginRow ? beginElement % convSize :
                                                0;
          unsigned endIndex = j + 1 == endRow ? 1 + (endElement - 1) % convSize :
                                                convSize;
          convSizesByWorker.back().push_back(endIndex - beginIndex);
        }
      }
    }
    return getConvPartial1x1SupervisorCycleEstimate(convSizesByWeightAndWorker);
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
