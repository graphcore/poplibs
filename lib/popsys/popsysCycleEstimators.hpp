#ifndef __popsysCycleEstimators_hpp__
#define __popsysCycleEstimators_hpp__

#include <poplibs_support/cyclesTables.hpp>
#include <poplar/Target.hpp>

namespace popsys {

poplibs::CycleEstimatorTable
makeCyclesFunctionTable(const poplar::Target &target);

}

#endif
