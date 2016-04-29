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
    unsigned innerLoopCycles = outputWidth;
    // A nx1 convolution must be computed as the sum of 3x1 convolutions.
    // The number of passes per row is the number of 3x1 convolutions
    // required.
    unsigned passesPerRow = (kernelSize + 2) / 3;
    // The 1 here represents loop overheads. TODO take into account
    // the time to load weights at the beginning of the row and flush the
    // accumulator values out at the end of the row.
    return vertexOverhead + numInRows * passesPerRow * (1 + innerLoopCycles);
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
