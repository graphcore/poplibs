#include "popreduceCycleEstimators.hpp"
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popreduce {

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAdd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             false, false,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAddUpdate)(const VertexIntrospector &vertex,
                                           const Target &target,
                                           const Type &outType,
                                           const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             true, false,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAddScale)(const VertexIntrospector &vertex,
                                          const Target &target,
                                          const Type &outType,
                                          const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate(outSizes,
                             partials.size(),
                             dataPathWidth,
                             false, true,
                             outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceOpsCycleEstimate(outSizes,
                                partials.size(),
                                dataPathWidth,
                                outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMax)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceMin)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceAnd)(const VertexIntrospector &vertex,
                                     const Target &target,
                                     const Type &outType,
                                     const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)(vertex, target, outType, partialsType);
}

std::uint64_t
MAKE_CYCLE_ESTIMATOR_NAME(ReduceOr)(const VertexIntrospector &vertex,
                                    const Target &target,
                                    const Type &outType,
                                    const Type &partialsType) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceAnd)(vertex, target, outType, partialsType);
}

poplibs::CycleEstimatorTable makeCyclesFunctionTable() {
  return {
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceOr, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAnd, BOOL, BOOL),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, INT, INT),

    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, FLOAT, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, HALF, FLOAT),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, HALF, HALF),
    CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, INT, INT)
  };
};

} // end namespace popreduce
