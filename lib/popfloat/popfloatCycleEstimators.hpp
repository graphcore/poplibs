// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef __popfloatCycleEstimators_hpp__
#define __popfloatCycleEstimators_hpp__

#include <poplibs_support/cyclesTables.hpp>

namespace popfloat {
namespace experimental {

poplibs::PerfEstimatorTable makePerfFunctionTable();

} // end namespace experimental
} // end namespace popfloat
#endif
