// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"
#include <poplar/Graph.hpp>
#include <popops/HostSliceTensor.hpp>
#define BOOST_TEST_MODULE HostSliceTensor
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace poplar;
using namespace popops;

static unsigned getNumTiles(const std::vector<std::vector<Interval>> &ans,
                            const bool print = false /*for debugging*/) {
  unsigned count = 0;
  if (print) {
    std::cout << "Printin used tiles\n";
  }
  for (unsigned i = 0; i < ans.size(); ++i) {
    count += (!ans[i].empty());
    if (print && !ans[i].empty()) {
      std::cout << i << ",";
    }
  }
  if (print) {
    std::cout << std::endl;
  }
  return count;
}

BOOST_AUTO_TEST_CASE(Basic) {
  auto device = createTestDevice(TEST_TARGET, 1, 1216);
  Graph graph(device.getTarget());

  auto t0 = createHostSliceableTensor(graph, FLOAT, {1, 63}, false);
  auto m0 = graph.getTileMapping(t0.tensor);
  auto i0 = graph.getTileMapping(t0.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m0), 1);
  BOOST_CHECK_EQUAL(getNumTiles(i0), 1);
  BOOST_CHECK_EQUAL(m0[0].size(), 1);
  BOOST_CHECK_EQUAL(i0[0].size(), 1);
  BOOST_CHECK_EQUAL(m0[0][0].size(), 63);
  BOOST_CHECK_EQUAL(i0[0][0].size(), 1);

  auto t1 = createHostSliceableTensor(graph, FLOAT, {2, 10}, false);
  auto m1 = graph.getTileMapping(t1.tensor);
  auto i1 = graph.getTileMapping(t1.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m1), 2);
  BOOST_CHECK_EQUAL(getNumTiles(i1), 2);

  // one packet per tile
  auto t2 = createHostSliceableTensor(graph, FLOAT, {1, 64 * 1216}, false);
  auto m2 = graph.getTileMapping(t2.tensor);
  auto i2 = graph.getTileMapping(t2.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m2), 1216);
  for (const auto &m : m2) {
    BOOST_CHECK_EQUAL(m.size(), 1);
  }
  BOOST_CHECK_EQUAL(i2[0].size(), 1);

  auto t3 = createHostSliceableTensor(graph, FLOAT, {1, 64 * 1217}, false);
  auto m3 = graph.getTileMapping(t3.tensor);
  auto i3 = graph.getTileMapping(t3.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m3), 1216);
  for (unsigned i = 0; i < m3.size(); ++i) {
    BOOST_CHECK_EQUAL(m3[i].size(), 1);
    if (i == 0) {
      BOOST_CHECK_EQUAL(m3[0][0].size(), 128);
      BOOST_CHECK_EQUAL(i3[0].size(), 1);
      BOOST_CHECK_EQUAL(i3[0][0].size(), 1);
    } else {
      BOOST_CHECK_EQUAL(m3[i][0].size(), 64);
      BOOST_CHECK(i3[i].empty());
    }
  }

  auto t4 = createHostSliceableTensor(graph, FLOAT, {6, 70}, false);
  auto m4 = graph.getTileMapping(t4.tensor);
  auto i4 = graph.getTileMapping(t4.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m4), 12);
  BOOST_CHECK_EQUAL(getNumTiles(i4), 6);

  auto t5 =
      createHostSliceableTensor(graph, FLOAT, {30, (128 * 1216) + 70}, false);
  auto m5 = graph.getTileMapping(t5.tensor);
  auto i5 = graph.getTileMapping(t5.indices);
  BOOST_CHECK_EQUAL(getNumTiles(m5), 1216);
  BOOST_CHECK_EQUAL(getNumTiles(i5), 30);

  for (unsigned i = 0; i < m5.size(); ++i) {
    BOOST_CHECK_EQUAL(m5[i].size(), 1);
    if (!i5[i].empty()) {
      BOOST_CHECK_EQUAL(i5[i].size(), 1);
      BOOST_CHECK_EQUAL(i5[i][0].size(), 1);
    }
  }

  auto t6 = createHostTransferableTensor(graph, FLOAT, {4, 8, 64}, false);
  auto m6 = graph.getTileMapping(t6);
  std::vector<size_t> expected = {4, 8, 64};
  BOOST_CHECK(t6.shape() == expected);
  BOOST_CHECK_EQUAL(getNumTiles(m6), 32);
}
