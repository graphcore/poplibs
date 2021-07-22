// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#ifndef __popnnCycleEstimators_hpp__
#define __popnnCycleEstimators_hpp__

#include "poputil/exceptions.hpp"

#include <poputil/cyclesTables.hpp>

namespace popnn {

poputil::internal::PerfEstimatorTable makePerfFunctionTable();
}

#endif
