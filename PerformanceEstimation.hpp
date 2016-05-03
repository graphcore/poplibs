#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <cstdint>

inline std::uint64_t getDenseDotProductCycles(bool isFloat, unsigned size) {
  if (isFloat) {
    return (size + 1) / 2 + 2;
  }
  return (size + 3) / 4 + 2;
}

inline std::uint64_t
getConvPartialCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                            unsigned stride, unsigned kernelSize,
                            unsigned numInRows,
                            unsigned outputWidth)
{
  unsigned vertexOverhead = 5;
  if (inChansPerGroup == 4 && !isFloat && stride == 1) {
    // The 6 cycle overhead is estimated as 2 cycles to flush to convolution
    // instruction pipeline and 4 cycles to load the weights.
    unsigned innerLoopCycles = 6 + outputWidth;
    // A nx1 convolution must be computed as the sum of 3x1 convolutions.
    // The number of passes per row is the number of 3x1 convolutions
    // required.
    unsigned passesPerRow = (kernelSize + 2) / 3;
    unsigned outerLoopCycles;
    if (passesPerRow > 1) {
      unsigned middleLoopCycles = 1 + passesPerRow * innerLoopCycles;
      outerLoopCycles = 1 + numInRows * middleLoopCycles;
    } else {
      outerLoopCycles = 1 + numInRows * innerLoopCycles;
    }
    return vertexOverhead + outerLoopCycles;
  }
  return vertexOverhead +
         outputWidth *
         numInRows * (1 +
                      getDenseDotProductCycles(isFloat,
                                               kernelSize * inChansPerGroup));
}

inline std::uint64_t
getFullyConnectedPartialCycleEstimate(bool isFloat, unsigned size) {
  return 5 + getDenseDotProductCycles(isFloat, size);
}

#endif // _performance_estimation_h_
