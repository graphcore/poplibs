// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PlanMultipleCorrectnessTest
#include <boost/test/unit_test.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/DynamicSlice.hpp>
#include <vector>

using namespace poplar;
using namespace popops;
using namespace popops::embedding;
using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(PlanMultipleCorrectness) {
  constexpr static unsigned tilesPerIPU = 4;
  auto device = createTestDevice(TEST_TARGET, 1, tilesPerIPU);
  const auto &target = device.getTarget();
  Graph graph(target);

  std::vector<SlicePlanningParameters> descriptions = {
      SlicePlanningParameters(graph, HALF, 2, 1, 1, {1, 1}, {}),
      SlicePlanningParameters(graph, FLOAT, 1, 4, 16, {2}, {}),
  };
  std::vector<SlicePlan> sequential_result;
  for (auto d : descriptions) {
    sequential_result.push_back(plan(d.graph, d.dataType, d.groupSize,
                                     d.numEntries, d.outputSize, d.numLookups,
                                     d.optionFlags));
  }
  std::vector<SlicePlan> parallel_result = planMultiple(descriptions);
  // Would be pointless to compare elements if they're all the same.
  BOOST_CHECK((sequential_result[0] != sequential_result[1]));
  BOOST_REQUIRE(sequential_result == parallel_result);
}
