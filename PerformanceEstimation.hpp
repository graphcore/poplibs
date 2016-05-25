#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <algorithm>
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

inline std::uint64_t
get1x1ConvCycles(unsigned outputWidth) {
  const auto outChansPerGroup = 4;
  unsigned warmUpCycles = 19;
  unsigned innerLoopCycles =
      outputWidth * outChansPerGroup;
  unsigned coolDownCycles = 5;
  return warmUpCycles + innerLoopCycles + coolDownCycles;
}

inline std::uint64_t
getConvPartial1x1CycleEstimate(
    const std::vector<std::vector<unsigned>> &convSizesByWeight) {
  const unsigned vertexOverhead = 5;
  unsigned cycleCount = vertexOverhead;
  for (const auto &convSizes : convSizesByWeight) {
    const auto numElements = std::accumulate(convSizes.begin(), convSizes.end(),
                                             0);
    const auto pointerLoadCycles = convSizes.size();
    cycleCount += get1x1ConvCycles(numElements) + pointerLoadCycles;
  }
  return cycleCount;
}

inline std::uint64_t
getConvPartialCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                            unsigned stride, unsigned kernelWidth,
                            unsigned inputGroupsPerOutput,
                            unsigned outputHeight,
                            unsigned outputWidth,
                            unsigned outChansPerGroup)
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
    return getConvPartial1x1CycleEstimate(convSizesByWeight);
  }
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
