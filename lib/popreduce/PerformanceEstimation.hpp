#ifndef _performance_estimation_h_
#define _performance_estimation_h_
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

template <typename OutType, typename PartialsType>
static uint64_t
reduceCycleEstimate(const std::vector<unsigned> &outSizes,
                    unsigned partialsSize,
                    unsigned dataPathWidth,
                    bool isUpdate, bool isScale) {
  bool isPartialsFloat = std::is_same<PartialsType, float>::value;
  bool isOutTypeFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth = dataPathWidth / (isPartialsFloat ? 32 : 16);
  bool conversionCyles = isPartialsFloat != isOutTypeFloat;
  unsigned cycles;
  const unsigned numReductions = outSizes.size();
  const unsigned numPartials = partialsSize / numReductions;
  const unsigned version=1;
  unsigned addCycles = 0;
  if (isUpdate) {
    addCycles = 2;
  }
  if (isScale) {
    addCycles = 1;
  }
  switch (version) {
  case 0: // Original optimistic estimate
  default:
    cycles = 4;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = outSizes[r];
      auto numVectors = (numElem + vectorWidth - 1) / vectorWidth;
      cycles += 1 + numPartials * (1 + numVectors)
                + conversionCyles * numVectors;
    }
    break;
  case 1:
    // Innermost loop accumulates vector across all input tiles
    // This estimate based on float->float code
    // Inner loop processes 128bits/2cycles
    // Inner loop cycles would halve for strided data given f32v4add IS addtion
    cycles = 2+5+1;
    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 6;
      const unsigned numElem = outSizes[r];
      auto numVectorWidths = (numElem + 2 * vectorWidth - 1)
                             / (2 * vectorWidth);
      cycles += (2 * numPartials + 1 + 3) * numVectorWidths;
      cycles += numVectorWidths * addCycles;
    }
    break;
  case 2:
    // Innermost loop adds one tile's input accross a region
    // This estimate based on float->float code Reductions
    // in loop overhead are expected given IS changes.
    // Note this isn't suitable for half->float reduction
    assert(isOutTypeFloat);
    cycles = 2+7+1;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = outSizes[r];
      auto numVectorWidths = (numElem + vectorWidth - 1) / vectorWidth;
      cycles += 9 + numVectorWidths + 1;
      cycles += (7 + numVectorWidths + 1) * (numPartials - 1);
      cycles += numVectorWidths * addCycles;
    }
    break;
  }
  return cycles;
}

#endif // _performance_estimation_h_
