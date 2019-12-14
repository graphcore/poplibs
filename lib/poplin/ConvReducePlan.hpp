// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef ConvReducePlan_hpp
#define ConvReducePlan_hpp

#include <vector>

namespace poplin {

// Given partials depth return a plan.
// The plan is a vector of reduction factors of the partial depth.
// The number of stages in the  reduction is equal to 1 + sizeof(return value)
std::vector<unsigned> getMultiStageReducePlan(unsigned partialsDepth);

} // namespace poplin

#endif // ConvReducePlan_hpp
