// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef ConvReducePlan_hpp
#define ConvReducePlan_hpp

#include <vector>

namespace poplin {

// Given partials depth return a plan.
// The plan is a vector of reduction factors of the partial depth.
// The number of stages in the  reduction is equal to 1 + sizeof(return value)
std::vector<unsigned> getMultiStageReducePlan(unsigned partialsDepth,
                                              bool enableMultiStageReduce);

bool inline checkPartialsSizeForSingleInputReduce(unsigned partialsBytes,
                                                  unsigned bytesPerTile) {
  // We don't want to allocate all the partials in one huge chunk if this
  // is going to cause problems due to its size.
  // Use a heuristic of 1/32 of the total tile memory as a reasonable proportion
  return partialsBytes <= bytesPerTile / 32;
}

} // namespace poplin

#endif // ConvReducePlan_hpp
