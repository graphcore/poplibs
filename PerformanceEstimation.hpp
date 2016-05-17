#ifndef _performance_estimation_h_
#define _performance_estimation_h_

#include <cstdint>

inline std::uint64_t getDenseDotProductCycles(bool isFloat, unsigned size) {
  if (isFloat) {
    return (size + 1) / 2 + 2;
  }
  return (size + 3) / 4 + 2;
}

/// Estimate the number of cycles required for a partial convolution vertex.
/// Each partial convolution vertex computes a partial sum for one output row.
/// The number of in rows is the total number of inChansPerGroup deep input
/// rows that are required to compute the output row.
inline std::uint64_t
getConvPartialCycleEstimate(bool isFloat, unsigned inChansPerGroup,
                            unsigned stride, unsigned kernelSize,
                            unsigned numInRows,
                            unsigned outputWidth,
                            unsigned outChansPerGroup)
{
  unsigned vertexOverhead = 5;
  if (!isFloat && inChansPerGroup == 16 && outChansPerGroup == 4 &&
      stride == 1 && kernelSize == 1) {
    unsigned warmUpCycles = 19;
    unsigned innerLoopCycles =
        outputWidth * outChansPerGroup;
    unsigned coolDownCycles = 5;
    unsigned cycleCount = numInRows *
                          (warmUpCycles + innerLoopCycles + coolDownCycles);
    return cycleCount;
  }
  return vertexOverhead +
         outChansPerGroup * outputWidth *
         numInRows * (1 +
                      getDenseDotProductCycles(isFloat,
                                               kernelSize * inChansPerGroup));
}

inline std::uint64_t
getFullyConnectedPartialCycleEstimate(bool isFloat, unsigned size) {
  return 5 + getDenseDotProductCycles(isFloat, size);
}

#endif // _performance_estimation_h_
