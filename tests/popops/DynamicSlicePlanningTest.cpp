// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DynamicSliceTest
#include <boost/test/framework.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <numeric>
#include <poplar/Type.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/DynamicSliceInternal.hpp>
#include <set>
#include <sstream>
#include <vector>

using namespace poplar;
using namespace popops;
using namespace poplibs_support;

void checkPlanner(
    Type dType, unsigned numEntries, unsigned embeddingSize,
    const std::vector<std::size_t> numLookups,
    const sliceInternal::Partition<std::size_t> &expectedPartition) {
  BOOST_TEST_MESSAGE(
      "Test " << boost::unit_test::framework::current_test_case().p_name);

  // Always check with the number of tiles on ipu1
  auto device = createTestDevice(TEST_TARGET, 1, 1216);
  Graph graph(device.getTarget());
  auto plan = popops::embedding::plan(graph, dType, numEntries, embeddingSize,
                                      numLookups, {});
  const auto &e = expectedPartition;
  const auto &p = plan.getImpl();
  BOOST_CHECK_EQUAL(e.lookupSplit, p.partition.lookupSplit);
  BOOST_CHECK_EQUAL(e.slicedDimSplit, p.partition.slicedDimSplit);
  BOOST_CHECK_EQUAL(e.unslicedDimSplit, p.partition.unslicedDimSplit);
  BOOST_CHECK_EQUAL(e.unslicedGrainSize, p.partition.unslicedGrainSize);
}

BOOST_AUTO_TEST_CASE(BigEmbedding) {
  checkPlanner(HALF, 100000, 200, {1440}, {1, 12, 100, 2});
}
BOOST_AUTO_TEST_CASE(ManyLookups) {
  checkPlanner(HALF, 1000, 200, {18000}, {12, 1, 100, 2});
}
BOOST_AUTO_TEST_CASE(VeryManyLookups) {
  checkPlanner(HALF, 1000, 200, {40000}, {12, 1, 100, 2});
}
BOOST_AUTO_TEST_CASE(LargeAndManyLookups) {
  checkPlanner(HALF, 100000, 200, {11520}, {1, 12, 100, 2});
}
BOOST_AUTO_TEST_CASE(Square) {
  checkPlanner(HALF, 1000, 1000, {1000}, {1, 2, 500, 2});
}
BOOST_AUTO_TEST_CASE(Small) {
  checkPlanner(FLOAT, 50, 50, {20}, {2, 5, 50, 1});
}

BOOST_AUTO_TEST_CASE(PlanComparison) {
  auto device = createTestDevice(TEST_TARGET, 1, 1216);
  Graph graph(device.getTarget());
  auto plan1a =
      popops::embedding::plan(graph, poplar::HALF, 10000, 200, {400}, {});
  auto plan1b =
      popops::embedding::plan(graph, poplar::HALF, 10000, 200, {400}, {});
  auto plan2 =
      popops::embedding::plan(graph, poplar::HALF, 10000, 100, {300}, {});

  BOOST_CHECK(plan1a == plan1b);
  BOOST_CHECK(plan1a != plan2);

  BOOST_CHECK(plan1b == plan1a);
  BOOST_CHECK(plan1b != plan2);

  BOOST_CHECK(plan2 != plan1a);
  BOOST_CHECK(plan2 != plan1b);

  std::set<popops::SlicePlan> plans;
  plans.insert(plan1a);
  BOOST_CHECK(plans.size() == 1);
  plans.insert(plan1b);
  BOOST_CHECK(plans.size() == 1);
  plans.insert(plan2);
  BOOST_CHECK(plans.size() == 2);
  BOOST_CHECK(plans.count(plan1a) == 1);
  BOOST_CHECK(plans.count(plan1b) == 1);
  BOOST_CHECK(plans.count(plan2) == 1);
  plans.erase(plan1a);
  BOOST_CHECK(plans.count(plan1a) == 0);
  BOOST_CHECK(plans.count(plan1b) == 0);
  plans.erase(plan2);
  BOOST_CHECK(plans.empty());
}
