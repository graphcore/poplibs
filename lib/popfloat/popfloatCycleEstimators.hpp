// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef __popfloatCycleEstimators_hpp__
#define __popfloatCycleEstimators_hpp__

#include <poplibs_support/cyclesTables.hpp>

namespace experimental {
namespace popfloat {

poplibs::CycleEstimatorTable makeCyclesFunctionTable();

} // end namespace popfloat
} // end namespace experimental
#endif
