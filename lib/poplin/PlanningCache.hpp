// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplin_PlanningCache_hpp
#define poplin_PlanningCache_hpp

#include "ConvPlanTypes.hpp"
#include "PerformanceEstimation.hpp"
#include <map>
#include <poplibs_support/Memoize.hpp>

namespace poplin {

class PlanningCacheImpl {
public:
  using Key = ConvDescription;

  class CycleEstimationImpl {
  public:
    decltype(poplibs_support::memoize(
        getConvPartial1x1InnerLoopCycleEstimateWithZeroing))
        mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing;
    decltype(poplibs_support::memoize(
        getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing))
        mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing;
    decltype(poplibs_support::memoize(getConvPartialnx1InnerLoopCycleEstimate))
        mGetConvPartialnx1InnerLoopCycleEstimate;
    decltype(poplibs_support::memoize(
        estimateConvPartialHorizontalMacInnerLoopCycles))
        mEstimateConvPartialHorizontalMacInnerLoopCycles;
    decltype(poplibs_support::memoize(
        estimateConvPartialVerticalMacInnerLoopCycles))
        mEstimateConvPartialVerticalMacInnerLoopCycles;
    decltype(poplibs_support::memoize(
        estimateConvReduceCycles)) mEstimateConvReduceCycles;
    decltype(poplibs_support::memoize(getNumberOfMACs)) mGetNumberOfMACs;
    decltype(poplibs_support::memoize(
        estimateZeroSupervisorCycles)) mEstimateZeroSupervisorCycles;
    decltype(poplibs_support::memoize(
        getConvPartialSlicSupervisorOuterLoopCycleEstimate))
        mGetConvPartialSlicSupervisorOuterLoopCycleEstimate;
    decltype(poplibs_support::memoize(
        getConvPartialSlicInnerLoopCycles)) mGetConvPartialSlicInnerLoopCycles;

    CycleEstimationImpl()
        : mGetConvPartial1x1InnerLoopCycleEstimateWithZeroing(
              getConvPartial1x1InnerLoopCycleEstimateWithZeroing),
          mGetConvPartial1x1InnerLoopCycleEstimateWithoutZeroing(
              getConvPartial1x1InnerLoopCycleEstimateWithoutZeroing),
          mGetConvPartialnx1InnerLoopCycleEstimate(
              getConvPartialnx1InnerLoopCycleEstimate),
          mEstimateConvPartialHorizontalMacInnerLoopCycles(
              estimateConvPartialHorizontalMacInnerLoopCycles),
          mEstimateConvPartialVerticalMacInnerLoopCycles(
              estimateConvPartialVerticalMacInnerLoopCycles),
          mEstimateConvReduceCycles(estimateConvReduceCycles),
          mGetNumberOfMACs(getNumberOfMACs),
          mEstimateZeroSupervisorCycles(estimateZeroSupervisorCycles),
          mGetConvPartialSlicSupervisorOuterLoopCycleEstimate(
              getConvPartialSlicSupervisorOuterLoopCycleEstimate),
          mGetConvPartialSlicInnerLoopCycles(
              getConvPartialSlicInnerLoopCycles) {}
  };

  // The plan's cycleEstimation can be used and updated in parallel.
  CycleEstimationImpl cycleEstimation;

private:
  // Updates to plans must be single-threaded.
  std::map<Key, std::pair<Plan, Cost>> planCache;

public:
  boost::optional<std::pair<Plan, Cost>> getPlan(const Key &key) {
    const auto plan = planCache.find(key);
    if (plan == planCache.end()) {
      return boost::none;
    } else {
      return (*plan).second;
    }
  }

  void addPlanToCache(Key key, std::pair<Plan, Cost> value) {
    planCache.emplace(std::move(key), std::move(value));
  }

  std::size_t size() const { return planCache.size(); }
};

} // namespace poplin

#endif // poplin_PlanningCache_hpp
