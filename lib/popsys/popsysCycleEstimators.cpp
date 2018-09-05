#include "popsysCycleEstimators.hpp"

using namespace poplar;

namespace popsys {
std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(TimeItStart)(const VertexIntrospector &vertex,
                                       const Target &target) {
  return 39;
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(TimeItEnd)(const VertexIntrospector &vertex,
                                     const Target &target) {
  return 75;
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
    CYCLE_ESTIMATOR_ENTRY(popsys, TimeItStart),
    CYCLE_ESTIMATOR_ENTRY(popsys, TimeItEnd)
  };
};

} // end namespace popsys
