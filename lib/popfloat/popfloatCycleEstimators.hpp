// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef __popfloatCycleEstimators_hpp__
#define __popfloatCycleEstimators_hpp__

#include <poputil/cyclesTables.hpp>

namespace popfloat {
namespace experimental {

poputil::internal::PerfEstimatorTable makePerfFunctionTable();

} // end namespace experimental
} // end namespace popfloat
#endif
