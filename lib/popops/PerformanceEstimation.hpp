#ifndef _performance_estimation_h_
#define _performance_estimation_h_
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>
#include <poplar/Type.hpp>
static uint64_t
reduceCycleEstimate(const std::vector<unsigned> &outSizes,
                    unsigned partialsSize,
                    unsigned dataPathWidth,
                    bool isUpdate, bool isScale,
                    const poplar::Type &outType,
                    const poplar::Type &partialsType) {
  bool isPartialsFloat = partialsType == poplar::FLOAT;
  bool isOutTypeFloat = outType == poplar::FLOAT;
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
    // Inner loop processes 128bits/3cycles (1 for masking the deltaN)
    // Inner loop cycles would halve for strided data given f32v4add IS addition
    cycles = 2+5+1;
    // VectorList costs 6 or 8 cycles to load n+base+descriptorPtr
    // These vertices have two VectorList::DELTAN so we're likely to have one
    // of each
    cycles += 6 + 8;
    for (unsigned r = 0; r < numReductions; ++r) {
      cycles += 6;
      const unsigned numElem = outSizes[r];
      auto numVectorWidths = (numElem + 2 * vectorWidth - 1)
                             / (2 * vectorWidth);
      cycles += (3 * numPartials + 1 + 3) * numVectorWidths;
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

static uint64_t
reduceOpsCycleEstimate(const std::vector<unsigned> &outSizes,
                       unsigned partialsSize,
                       unsigned dataPathWidth,
                       const poplar::Type &outType,
                       const poplar::Type &partialsType) {
  bool isPartialsFloat = partialsType == poplar::FLOAT;
  bool isOutTypeFloat = outType == poplar::FLOAT;
  // assumed that bool is 16 bits. If it is 8, vector operations are possible
  // on the AUX side but the cycle count will be different
  unsigned vectorWidth = dataPathWidth / (isPartialsFloat ? 32 : 16);
  // if partials is bool, output is always bool
  bool conversionCyles = isPartialsFloat != isOutTypeFloat;
  uint64_t cycles = 10;
  const unsigned numReductions = outSizes.size();
  const unsigned numPartials = partialsSize / numReductions;

  for (unsigned r = 0; r < numReductions; ++r) {
    // overhead for each reduction
    cycles += 5;
    unsigned numElem = outSizes[r];
    const unsigned numVectors = (numElem + vectorWidth - 1) / vectorWidth;

    // process vectorWidth at a time with ld2xstpace. This may not be the best
    // option if numVectors is small
    // overhead of 5: ld ptrs, rpt, brnzdec
    cycles += (numVectors + 5) * numPartials + conversionCyles * numVectors;
  }
  return cycles;
}

#endif // _performance_estimation_h_
