// Copyright (c) Graphcore Ltd, All rights reserved.
#ifndef __popfloatCycleEstimators_hpp__
#define __popfloatCycleEstimators_hpp__

#include <poplibs_support/cyclesTables.hpp>

namespace popfloat {
namespace experimental {

poplibs::CycleEstimatorTable makeCyclesFunctionTable();

} // end namespace experimental
} // end namespace popfloat
#endif
