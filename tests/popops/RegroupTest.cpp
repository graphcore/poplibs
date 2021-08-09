// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE RegroupTest
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/logging.hpp>
#include <popops/Rearrange.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_support;

const OptionFlags options;

constexpr unsigned dim0 = 32;
constexpr unsigned dim1 = 16;

static std::pair<std::vector<GroupingInfo>, std::vector<GroupingInfo>>
runTest(const GroupingInfo &requestedGrouping) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  auto in = graph.addVariable(FLOAT, {dim0, dim1});
  float hIn[dim0][dim1];
  float hOut[dim0][dim1];

  for (unsigned i = 0; i != dim0; ++i) {
    for (unsigned j = 0; j != dim1; ++j) {
      hIn[i][j] = i * dim1 + j;
      hOut[i][j] = dim0 * dim1 - hIn[i][j];
    }
  }

  mapTensorLinearly(graph, in, 8, 8);
  auto prog = Sequence();
  auto out =
      popops::rearrange::regroupIfPossible(graph, in, prog, requestedGrouping);

  graph.createHostWrite("in", in);
  graph.createHostRead("out", out);

  Engine eng(graph, prog, options);
  device.bind([&](const Device &d) {
    eng.load(d);
    eng.writeTensor("in", hIn, &hIn[0][0] + dim0 * dim1);
    eng.run();
    eng.readTensor("out", hOut, &hOut[0][0] + dim0 * dim1);
  });

  for (unsigned i = 0; i != dim0; ++i) {
    for (unsigned j = 0; j != dim1; ++j) {
      BOOST_CHECK_EQUAL(hIn[i][j], hOut[i][j]);
    }
  }

  return std::make_pair(detectDimGroupings(graph, in),
                        detectDimGroupings(graph, out));
}

// Requested regrouping is performed
BOOST_AUTO_TEST_CASE(RegroupGrouping_0_16) {
  auto [inG, outG] = runTest({0, 16});
  BOOST_CHECK_EQUAL(std::get<0>(inG[0]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(inG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(inG[1]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(inG[1]), 8);

  BOOST_CHECK_EQUAL(std::get<0>(outG[0]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(outG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(outG[1]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(outG[1]), 8);
}

// No change in grouping as grouping is requested exists in the innermost
// dimension
BOOST_AUTO_TEST_CASE(RegroupGrouping_1_16) {
  auto [inG, outG] = runTest({1, 16});
  BOOST_CHECK_EQUAL(std::get<0>(inG[0]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(inG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(inG[1]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(inG[1]), 8);

  BOOST_CHECK_EQUAL(std::get<0>(outG[0]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(outG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(outG[1]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(outG[1]), 8);
}

// There be no change in the grouping as requested grouping is not a multiple
// of the dimension
BOOST_AUTO_TEST_CASE(RegroupGrouping_0_7) {
  auto [inG, outG] = runTest({0, 7});
  BOOST_CHECK_EQUAL(std::get<0>(inG[0]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(inG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(inG[1]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(inG[1]), 8);

  BOOST_CHECK_EQUAL(std::get<0>(outG[0]), 1);
  BOOST_CHECK_EQUAL(std::get<1>(outG[0]), 16);
  BOOST_CHECK_EQUAL(std::get<0>(outG[1]), 0);
  BOOST_CHECK_EQUAL(std::get<1>(outG[1]), 8);
}
