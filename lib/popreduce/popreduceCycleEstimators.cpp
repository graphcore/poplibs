#include "popreduceCycleEstimators.hpp"
#include <poplar/HalfFloat.hpp>
#include "PerformanceEstimation.hpp"

using namespace poplar;

namespace popreduce {

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceAdd, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                    partials.size(),
                                                    dataPathWidth,
                                                    false, false);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceAddUpdate, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                    partials.size(),
                                                    dataPathWidth,
                                                    true, false);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceAddScale, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceCycleEstimate<OutType, PartialsType>(outSizes,
                                                    partials.size(),
                                                    dataPathWidth,
                                                    false, true);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceMul, vertex, target) {
  CODELET_FIELD(out);
  CODELET_FIELD(partials);
  const auto dataPathWidth = target.getDataPathWidth();

  std::vector<unsigned> outSizes;
  for (auto i = 0u; i < out.size(); ++i) outSizes.push_back(out[i].size());

  return reduceOpsCycleEstimate<OutType, PartialsType>(outSizes,
                                                       partials.size(),
                                                       dataPathWidth);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceMax, vertex, target) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)<OutType, PartialsType>(vertex, target);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceMin, vertex, target) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)<OutType, PartialsType>(vertex, target);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceAnd, vertex, target) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceMul)<OutType, PartialsType>(vertex, target);
}

template <typename OutType, typename PartialsType>
MAKE_CYCLE_ESTIMATOR(ReduceOr, vertex, target) {
  return
    MAKE_CYCLE_ESTIMATOR_NAME(ReduceAnd)<OutType, PartialsType>(vertex, target);
}

poplibs::CycleEstimatorTable cyclesFunctionTable = {
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceOr, bool, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAnd, bool, bool),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMin, int, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMax, int, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceMul, int, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddScale, int, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAddUpdate, int, int),

  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, float, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, half, float),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, half, half),
  TEMPLATE_CYCLE_ESTIMATOR_ENTRY(popreduce, ReduceAdd, int, int)
};

} // end namespace popreduce
