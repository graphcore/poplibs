// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef ConvReducePlan_hpp
#define ConvReducePlan_hpp

#include <cassert>
#include <vector>

namespace poplin {

// Given partials depth return a plan.
// The plan is a vector of reduction factors of the partial depth.
// The number of stages in the  reduction is equal to 1 + sizeof(return value)
std::vector<unsigned> getMultiStageReducePlan(unsigned partialsDepth,
                                              bool enableMultiStageReduce);

bool inline checkPartialsSizeForSingleInputReduce(
    unsigned partialsBytes, const std::vector<unsigned> &memoryElementOffsets) {
  // We don't want to allocate all the partials in one huge chunk if this
  // is going to cause problems due to its size.
  // Use a heuristic of partialsBytes rounded up to the nearest memory element
  // < 1/16 of the total tile memory
  assert(memoryElementOffsets.size() >= 2);
  const auto memoryElementSize =
      memoryElementOffsets[1] - memoryElementOffsets[0];
  const auto memorySize = memoryElementOffsets.back() - memoryElementOffsets[0];

  const auto occupiedElements =
      (partialsBytes + memoryElementSize - 1) / memoryElementSize;

  return occupiedElements * memoryElementSize < memorySize / 16;
}

} // namespace poplin

#endif // ConvReducePlan_hpp
