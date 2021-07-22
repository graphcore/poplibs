// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef popsparse_popsparseCycleEstimators_hpp
#define popsparse_popsparseCycleEstimators_hpp

#include <poputil/cyclesTables.hpp>

namespace popsparse {

poputil::internal::PerfEstimatorTable makePerfFunctionTable();

} // end namespace popsparse

#endif // popsparse_popsparseCycleEstimators_hpp
