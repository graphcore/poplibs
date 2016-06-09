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
    unsigned numConvUnitsPerTile) {
  const auto numWorkerContexts = 6;
  const auto weightsPerConvUnit = 16;

  unsigned cycles = 0;
  for (const auto &convSizesByWorker : convSizesByWeightAndWorker) {
    assert(convSizesByWorker.size() <= numWorkerContexts);
    // Load weights in the supervisor.
    const auto numBytes = weightsPerConvUnit * numConvUnitsPerTile * 2;
    cycles += (numBytes + 7) / 8;
    unsigned maxWorkerCycles = 0;
    // Start workers.
    for (const auto &convSizes : convSizesByWorker) {
      unsigned workerCycles = 0;
      const auto vertexOverhead = 5U;
      const auto coolDownCycles = 5U;
      workerCycles += vertexOverhead;
      for (const auto convSize : convSizes) {
        workerCycles += 1 + convSize * weightsPerConvUnit / 4;
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

struct PartialRow {
  unsigned rowNumber;
  unsigned begin;
  unsigned end;
  PartialRow(unsigned rowNumber, unsigned begin, unsigned end) :
    rowNumber(rowNumber),
    begin(begin),
    end(end) {}
};

inline std::vector<std::vector<PartialRow>>
partitionConvPartialByWorker(
    unsigned numConvolutions,
    unsigned convSize,
    unsigned numContexts) {
  std::vector<std::vector<PartialRow>> partitionByWorker;
  partitionByWorker.reserve(numContexts);
  const auto numElements = numConvolutions * convSize;
  for (unsigned i = 0; i != numContexts; ++i) {
    partitionByWorker.emplace_back();
    const auto beginElement = (i * numElements) / numContexts;
    const auto endElement = ((i + 1) * numElements) / numContexts;
    if (beginElement == endElement)
      continue;
    const auto beginRow = beginElement / convSize;
    const auto endRow = 1 + (endElement - 1) / convSize;
    for (unsigned j = beginRow; j != endRow; ++j) {
      unsigned beginIndex = j == beginRow ? beginElement % convSize :
                                            0;
      unsigned endIndex = j + 1 == endRow ? 1 + (endElement - 1) % convSize :
                                            convSize;
      partitionByWorker.back().emplace_back(j, beginIndex, endIndex);
    }
  }
  return partitionByWorker;
}

inline std::uint64_t
getConvPartial1x1CycleEstimate(
    const std::vector<std::vector<unsigned>> &convSizesByWeight,
    unsigned numConvUnitsPerTile) {
  const unsigned vertexOverhead = 5;
  unsigned cycleCount = vertexOverhead;
  for (const auto &convSizes : convSizesByWeight) {
    const auto numElements = std::accumulate(convSizes.begin(), convSizes.end(),
                                             0);
    const auto pointerLoadCycles = convSizes.size();
    unsigned warmUpCycles = numConvUnitsPerTile * 4 + 3;


    unsigned innerLoopCycles =
        numElements * 4;
    unsigned coolDownCycles = 5;
    cycleCount += warmUpCycles + innerLoopCycles + coolDownCycles +
                  pointerLoadCycles;
  }
  return cycleCount;
}

inline std::uint64_t
getConvPartial1x1CycleEstimate(unsigned kernelWidth,
                               unsigned inputGroupsPerOutput,
                               unsigned outputHeight,
                               unsigned outputWidth,
                               unsigned numConvUnitsPerTile,
                               bool useSupervisorVertices)
{
  if (useSupervisorVertices) {
    std::vector<std::vector<std::vector<unsigned>>> convSizesByWeightAndWorker;
    for (unsigned i = 0; i != inputGroupsPerOutput * kernelWidth; ++i) {
      const auto numWorkerContexts = 6;
      std::vector<std::vector<PartialRow>> partition =
          partitionConvPartialByWorker(outputHeight, outputWidth,
                                       numWorkerContexts);
      convSizesByWeightAndWorker.emplace_back();
      convSizesByWeightAndWorker.back().reserve(partition.size());
      for (const auto &entry : partition) {
        convSizesByWeightAndWorker.back().emplace_back();
        convSizesByWeightAndWorker.back().back().reserve(entry.size());
        for (const auto &partialRow : entry) {
          convSizesByWeightAndWorker.back().back().push_back(partialRow.end -
                                                             partialRow.begin);
        }
      }
    }
    return getConvPartial1x1SupervisorCycleEstimate(convSizesByWeightAndWorker,
                                                    numConvUnitsPerTile);
  }
  std::vector<std::vector<unsigned>> convSizesByWeight;
  for (unsigned i = 0; i != inputGroupsPerOutput * kernelWidth; ++i) {
    convSizesByWeight.emplace_back();
    for (unsigned j = 0; j != outputHeight; ++j) {
      convSizesByWeight.back().push_back(outputWidth);
    }
  }
  return getConvPartial1x1CycleEstimate(convSizesByWeight, numConvUnitsPerTile);
}

inline std::uint64_t
getConvPartialByDotProductCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                                        unsigned kernelWidth,
                                        unsigned inputGroupsPerOutput,
                                        unsigned outputHeight,
                                        unsigned outputWidth,
                                        unsigned outChansPerGroup)
{
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
