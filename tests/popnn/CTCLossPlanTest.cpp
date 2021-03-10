// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCLossPlanTest
#include <boost/test/unit_test.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <popnn/CTCInference.hpp>
#include <popnn/CTCLoss.hpp>

// Common simple plan parameters
const poplar::Type inType = poplar::FLOAT;
const poplar::Type outType = poplar::FLOAT;
constexpr unsigned batchSize = 10;
constexpr unsigned maxTime = 40;
constexpr unsigned maxLabelLength = 10;
constexpr unsigned numClasses = 4;

BOOST_AUTO_TEST_CASE(SimplePlan) {
  auto device = createTestDeviceFullSize(TEST_TARGET, 1);
  auto &target = device.getTarget();
  poplar::Graph graph(target);

  // Finds reasonably sized plan
  popnn::ctc::plan(graph, inType, outType, batchSize, maxTime, maxLabelLength,
                   numClasses, {{"availableMemoryProportion", "0.9"}});
}

BOOST_AUTO_TEST_CASE(SimplePlanWithMemoryBound) {
  auto device = createTestDeviceFullSize(TEST_TARGET, 1);
  auto &target = device.getTarget();
  poplar::Graph graph(target);

  // Can't find plan when no available memory
  BOOST_CHECK_THROW(popnn::ctc::plan(graph, inType, outType, batchSize, maxTime,
                                     maxLabelLength, numClasses,
                                     {{"availableMemoryProportion", "0.0"}}),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(SimplePlanWithConstrainedVar) {
  auto device = createTestDeviceFullSize(TEST_TARGET, 1);
  auto &target = device.getTarget();
  poplar::Graph graph(target);

  // Finds plan with a resonable constraint
  popnn::ctc::plan(
      graph, inType, outType, batchSize, maxTime, maxLabelLength, numClasses,
      {{"planConstraints", R"delim({"parallel": {"time": 1}})delim"}});

  // Can't find plan with un-satisfiable constraint
  BOOST_CHECK_THROW(
      popnn::ctc::plan(
          graph, inType, outType, batchSize, maxTime, maxLabelLength,
          numClasses,
          {{"planConstraints", R"delim({"parallel": {"batch": 0}})delim"}}),
      poputil::poplibs_error);

  // Throws on invalid name
  BOOST_CHECK_THROW(
      popnn::ctc::plan(
          graph, inType, outType, batchSize, maxTime, maxLabelLength,
          numClasses,
          {{"planConstraints", R"delim({"parallel": {"xyz": 0}})delim"}}),
      poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(WrongPlanType) {
  auto device = createTestDeviceFullSize(TEST_TARGET, 1);
  auto &target = device.getTarget();
  poplar::Graph graph(target);

  // Given an inference plan
  auto invalidPlan = popnn::ctc_infer::plan(graph, poplar::FLOAT, 1, 10, 5, 5);

  // When we call the CTC Loss API, Then we expect exceptions thrown
  BOOST_CHECK_THROW(
      popnn::ctc::createDataInput(graph, poplar::FLOAT, 1, 10, 5, invalidPlan),
      poputil::poplibs_error);
  BOOST_CHECK_THROW(
      popnn::ctc::createLabelsInput(graph, poplar::FLOAT, 1, 10, invalidPlan),
      poputil::poplibs_error);

  poplar::program::Sequence prog{};
  BOOST_CHECK_THROW(
      popnn::ctc::calcLossAndGradientLogProbabilities(
          graph, poplar::FLOAT, {}, {}, {}, {}, prog, 0, invalidPlan),
      poputil::poplibs_error);
  BOOST_CHECK_THROW(popnn::ctc::calcLossAndGradientLogits(graph, poplar::FLOAT,
                                                          {}, {}, {}, {}, prog,
                                                          0, invalidPlan),
                    poputil::poplibs_error);
}
