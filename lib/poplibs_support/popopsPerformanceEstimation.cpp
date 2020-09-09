// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplibs_support/popopsPerformanceEstimation.hpp"

namespace popops {

std::uint64_t basicOpSupervisorOverhead(const bool isScaledPtr64Type) {

  // common supervisor overhead
  std::uint64_t cycles = 198;

  // extra 2 cycles needed to unpack A and B pointers if they are scaled.
  if (isScaledPtr64Type) {
    cycles += 2;
  }

  return cycles;
}

/* Cycle cost computation for basic operations */
std::uint64_t basicOpLoopCycles(const unsigned numElems,
                                const unsigned vectorSize,
                                const unsigned cyclesPerVector) {
  return cyclesPerVector * (numElems + vectorSize - 1) / vectorSize;
}

} // namespace popops
