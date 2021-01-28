// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE CTCLossPlanTest
#include <boost/test/unit_test.hpp>

#include <poplibs_support/TestDevice.hpp>
#include <popnn/CTCLoss.hpp>

BOOST_AUTO_TEST_CASE(SimplePlanWithMemoryBound) {
  poplar::Type inType = poplar::FLOAT;
  poplar::Type outType = poplar::FLOAT;
  unsigned batchSize = 10;
  unsigned maxTime = 40;
  unsigned maxLabels = 10;
  unsigned numClasses = 4;

  auto device = createTestDeviceFullSize(TEST_TARGET, 1);
  auto &target = device.getTarget();
  poplar::Graph graph(target);

  // Finds reasonably sized plan
  popnn::ctc::plan(graph, inType, outType, batchSize, maxTime, maxLabels,
                   numClasses, {{"availableMemoryProportion", "0.9"}});

  // Can't find plan when no available memory
  BOOST_CHECK_THROW(popnn::ctc::plan(graph, inType, outType, batchSize, maxTime,
                                     maxLabels, numClasses,
                                     {{"availableMemoryProportion", "0.0"}}),
                    poputil::poplibs_error);
}
