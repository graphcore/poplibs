// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE MultiConvolutionPlan
#include <boost/test/unit_test.hpp>

#include "poplin/CanonicalConvParams.hpp"
#include "poplin/ConvOptions.hpp"
#include "poplin/ConvPlan.hpp"
#include "poplin/ConvUtilInternal.hpp"
#include <poplibs_support/TestDevice.hpp>

#include <algorithm>
#include <functional>
using namespace poplibs_support;

const poplar::OptionFlags multiConvOptions{{"planType", "parallel"}};

std::vector<poplin::CanonicalConvParams> getGenericOctConvParams() {
  const auto dataType = poplar::FLOAT;
  const auto batchSize = 1;
  const std::vector<std::size_t> kernelShape{3, 3};
  const auto numConvGroups = 1;

  const std::vector<std::size_t> highFreqInputField{8, 8};
  const std::vector<std::size_t> lowFreqInputField{4, 4};

  const auto highFreqInputChans = 8;
  const auto lowFreqInputChans = 3;

  const auto highFreqOutputChans = 10;
  const auto lowFreqOutputChans = 15;

  const auto HH = poplin::ConvParams{
      dataType,           batchSize,           highFreqInputField, kernelShape,
      highFreqInputChans, highFreqOutputChans, numConvGroups};
  const auto HL = poplin::ConvParams{
      dataType,           batchSize,          lowFreqInputField, kernelShape,
      highFreqInputChans, lowFreqOutputChans, numConvGroups};
  const auto LH = poplin::ConvParams{
      dataType,          batchSize,           lowFreqInputField, kernelShape,
      lowFreqInputChans, highFreqOutputChans, numConvGroups};
  const auto LL = poplin::ConvParams{
      dataType,          batchSize,          lowFreqInputField, kernelShape,
      lowFreqInputChans, lowFreqOutputChans, numConvGroups};

  return {HH, HL, LH, LL};
}
std::vector<poplin::ConvOptions> getGenericOctConvOptions() {
  poplar::OptionFlags po;
  po.set("pass", "INFERENCE_FWD");
  const poplin::ConvOptions options(po);
  return {options, options, options, options};
}

BOOST_AUTO_TEST_CASE(DividesTilesUnevenlyOnFLOPs) {
  // Given
  const auto tilesOnIPU = 100;
  const auto device = createTestDevice(TEST_TARGET, 1, tilesOnIPU);
  const auto params = getGenericOctConvParams();
  const auto options = getGenericOctConvOptions();
  poplin::PlanningCache cache;

  // When
  const auto concurrentPlans =
      boost::get<poplin::ParallelPlan>(getMultiPlan(device.getTarget(), params,
                                                    options, &cache,
                                                    multiConvOptions))
          .plans;

  // Then
  const auto tiles = [&concurrentPlans]() {
    std::vector<unsigned> tiles;
    std::transform(concurrentPlans.begin(), concurrentPlans.end(),
                   std::back_inserter(tiles),
                   [](const auto &plan) { return plan.totalTiles(); });
    return tiles;
  }();

  bool sameNumberOfTilesPerPlan = true;
  for (size_t i = 1; i < tiles.size(); i++) {
    if (tiles[i - 1] != tiles[i]) {
      sameNumberOfTilesPerPlan = false;
    }
  }

  BOOST_CHECK_EQUAL(sameNumberOfTilesPerPlan, false);
  // Sometimes we don't use all the tiles, and it's acceptable for this test
  BOOST_CHECK_LE(std::accumulate(tiles.begin(), tiles.end(), 0), tilesOnIPU);
}

std::vector<poplin::CanonicalConvParams> getLargeOctConvParams() {
  const auto genericParams = getGenericOctConvParams();
  std::vector<poplin::CanonicalConvParams> largeParams;
  std::transform(genericParams.cbegin(), genericParams.cend(),
                 std::back_inserter(largeParams),
                 [](const poplin::CanonicalConvParams &p) {
                   auto largerParam = p.getParams();
                   largerParam.numConvGroups = 1500;
                   return largerParam;
                 });
  return largeParams;
}

// This requires serial split support, currently for large convs (artifically
// increasing numConvGroups), a solution is failed to be found. The error is
// reported to be an OOM, but after increasing tile memory on the remaining
// plans (e.g. planned HH and increasing memory on HL, LH, LL) it still cannot
// find a solution. I suspect this is because of a factorisation problem
// occuring such that it's impossible to create the same number of serial
// splits regardless of the memory provided.
#if 0
BOOST_AUTO_TEST_CASE(ConsistentNumberOfSerialSplitsAcrossPlans) {
  // Given
  const auto device = createTestDevice(TEST_TARGET, 1, 100);
  const auto params = getLargeOctConvParams();
  auto options = getGenericOctConvOptions();

  poplin::PlanningCache cache;

  // When
  const auto concurrentPlans =
      boost::get<poplin::ParallelPlan>(
          getMultiPlan(device.getTarget(), params, options, &cache, multiConvOptions))
          .plans;

  // Then
  bool atLeastOneSerialSplit = false;
  for (size_t i = 1; i < concurrentPlans.size(); i++) {
    const auto prevPlan = concurrentPlans[i - 1];
    const auto currPlan = concurrentPlans[i];
    BOOST_CHECK_EQUAL(prevPlan.partitions.size(), currPlan.partitions.size());
    for (size_t j = 0; j < prevPlan.partitions.size(); j++) {
      BOOST_CHECK_EQUAL(prevPlan.partitions[j].inChanSplit.serial,
                        currPlan.partitions[j].inChanSplit.serial);
      BOOST_CHECK_EQUAL(prevPlan.partitions[j].outChanSplit.serial,
                        currPlan.partitions[j].outChanSplit.serial);
      if (prevPlan.partitions[j].inChanSplit.serial != 1 ||
          currPlan.partitions[j].inChanSplit.serial != 1 ||
          prevPlan.partitions[j].outChanSplit.serial != 1 ||
          currPlan.partitions[j].outChanSplit.serial != 1) {
        atLeastOneSerialSplit = true;
      }
    }
  }

  // Given
  BOOST_CHECK(atLeastOneSerialSplit);
}
#endif

BOOST_AUTO_TEST_CASE(FallsBackToSerialPlanningIfCannotFit) {
  // Given
  const auto totalTilesOnIPU = 10;
  const auto device = createTestDevice(TEST_TARGET, 1, totalTilesOnIPU);
  // Warning: Brittle choice of plan to fit serial and not parallel.
  const auto params = getLargeOctConvParams();
  auto options = getGenericOctConvOptions();

  poplin::PlanningCache cache;

  // When
  const auto plans = getMultiPlan(device.getTarget(), params, options, &cache,
                                  multiConvOptions);

  // Then
  BOOST_CHECK(plans.type() == typeid(poplin::SerialPlan));
  auto totalTilesUsedAcrossSteps = 0;
  const auto serialPlans = boost::get<poplin::SerialPlan>(plans).plans;
  for (const auto &plan : serialPlans) {
    totalTilesUsedAcrossSteps += plan.totalTiles();
    BOOST_CHECK_LE(plan.totalTiles(), totalTilesOnIPU);
  }
  // We should be using more than the total tiles agregating across the steps or
  // somethings likely gone wrong.
  BOOST_CHECK_GT(totalTilesUsedAcrossSteps, totalTilesOnIPU);
}

// Manual test
BOOST_AUTO_TEST_CASE(ChoosesBetterPlanWhenGivenReference) {
  // Given
  const auto tilesOnIPU = 100;
  const auto device = createTestDevice(TEST_TARGET, 1, tilesOnIPU);
  const auto params = getGenericOctConvParams();
  const auto options = getGenericOctConvOptions();
  poplin::PlanningCache cache;

// TODO: automate this test
#ifndef automated
  // suppress macos unused (still want to see logging output for manual
  // inspection)

  // Plan serially
  getMultiPlan(device.getTarget(), {params[0]}, {options[0]}, &cache,
               multiConvOptions);
  getMultiPlan(device.getTarget(), {params[1]}, {options[1]}, &cache,
               multiConvOptions);
  getMultiPlan(device.getTarget(), {params[2]}, {options[2]}, &cache,
               multiConvOptions);
  getMultiPlan(device.getTarget(), {params[3]}, {options[3]}, &cache,
               multiConvOptions);

  // Plan concurrently
  getMultiPlan(device.getTarget(), params, options, &cache, multiConvOptions);
#else
  // Given the reference is planned serially with no reference
  const auto individuallyPlanned = {
      getMultiPlan(device.getTarget(), {params[0]}, {options[0]}, &cache,
                   multiConvOptions)[0],
      getMultiPlan(device.getTarget(), {params[1]}, {options[1]}, &cache,
                   multiConvOptions)[0],
      getMultiPlan(device.getTarget(), {params[2]}, {options[2]}, &cache,
                   multiConvOptions)[0],
      getMultiPlan(device.getTarget(), {params[3]}, {options[3]}, &cache,
                   multiConvOptions)[0],
  };

  // When we plan together
  const auto concurrentlyPlanned = getMultiPlan(
      device.getTarget(), params, options, &cache, multiConvOptions);
  // Then
  const auto getCost = [](const auto &multiplan) {
    std::vector<unsigned> cost;
    for (const auto &plan : multiplan) {
      // TODO: Method to get step breakdown of plan
      const std::vector<unsigned> planCosts = getPlanStepCost(plan);
      for (size_t i = 0; i < planCosts.size(); i++) {
        if (cost.size() < i) {
          cost.push_back(0);
        }
        cost[i] = std::max(planCosts[i], cost[i]);
      }
    }
    return std::accumulate(cost.begin(), cost.end(), 0);
  };
  // Currently it appears to choose the same plans, and it's not erroneous to be
  // the same BUT we want to validate that it improves the plan, there's value
  // that it doesn't regress however we are specifically ensuring that we can
  // improve.
  BOOST_CHECK_LT(getCost(concurrentlyPlanned), getCost(individuallyPlanned));
#endif
}

// Manual test
BOOST_AUTO_TEST_CASE(FindsMultiPlanInCache) {
  // Given
  const auto device = createTestDevice(TEST_TARGET, 1, 8);
  const auto params = getGenericOctConvParams();
  const auto options = getGenericOctConvOptions();
  poplin::PlanningCache cache;

  // When
  getMultiPlan(device.getTarget(), params, options, &cache, multiConvOptions);
  getMultiPlan(device.getTarget(), params, options, &cache, multiConvOptions);

  // Then planner doesn't run twice in the logs
}

// TODO: test for multistage reduction constraint
