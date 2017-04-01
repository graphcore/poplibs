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
                    unsigned dataPathWidth) {
  bool isPartialsFloat = std::is_same<PartialsType, float>::value;
  bool isOutTypeFloat = std::is_same<OutType, float>::value;
  unsigned vectorWidth = dataPathWidth / (isPartialsFloat ? 32 : 16);
  bool conversionCyles = isPartialsFloat != isOutTypeFloat;
  unsigned cycles;
  const unsigned numReductions = outSizes.size();
  const unsigned numPartials = partialsSize / numReductions;
  const unsigned version=1;
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
    // Inner loop cycles likely to halve given minor IS change,
    // may then be best practical choice
    cycles = 2+5+1;
    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 6;
      const unsigned numElem = outSizes[r];
      auto numVectors = (numElem + 2 * vectorWidth - 1) / (2 * vectorWidth);
      cycles += (2 * numPartials + 4) * numVectors;
    }
    break;
  case 2:
    // Innermost loop adds one tile's input accross a region
    // This estimate based on float->float code Reductions
    // in loop overhead are expected given IS changes.
    cycles = 2+3;
    for (unsigned r = 0; r < numReductions; ++r) {
      unsigned numElem = outSizes[r];
      auto numVectors = (numElem + vectorWidth - 1) / vectorWidth;
      cycles += 24 + numVectors;
      // inner loop processes 3 vectors
      unsigned numInners = (numVectors + 2) / 3;
      cycles += (7 + 3 * numInners + 4 + 3 + 1) * (numPartials - 1);
    }
    break;
  }
  return cycles;
}

#endif // _performance_estimation_h_
