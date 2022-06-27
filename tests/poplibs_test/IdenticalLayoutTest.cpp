// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE IdenticalLayoutTest

#include "poplibs_test/Util.hpp"
#include <poplar/Graph.hpp>
#include <poplibs_support/TestDevice.hpp>

using namespace poplar;
using namespace poplibs_test::util;

BOOST_AUTO_TEST_CASE(Basic) {
  const auto numIPUs = 1;
  const auto tilesPerIPU = 1;
  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());

  auto a = graph.addVariable(INT, {4});
  auto b = graph.addVariable(INT, {4});
  graph.setTileMapping(a, 0);
  graph.setTileMapping(b, 0);
  BOOST_CHECK(identicalLayout(graph, a, b));
}

BOOST_AUTO_TEST_CASE(RobustToOrdering) {
  const auto numIPUs = 1;
  const auto tilesPerIPU = 1;
  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());

  auto a = graph.addVariable(INT, {4});
  auto b = graph.addVariable(INT, {4});
  auto c = graph.addVariable(INT, {4});
  auto d = graph.addVariable(INT, {4});
  graph.setTileMapping(a, 0);
  graph.setTileMapping(b, 0);
  graph.setTileMapping(c, 0);
  graph.setTileMapping(d, 0);
  BOOST_CHECK(identicalLayout(graph, concat(a, b), concat(d, c)));
}

BOOST_AUTO_TEST_CASE(RobustToOrderingWithMapping) {
  const auto numIPUs = 1;
  const auto tilesPerIPU = 4;
  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());

  auto a = graph.addVariable(INT, {4});
  auto b = graph.addVariable(INT, {4});
  auto c = graph.addVariable(INT, {4});
  auto d = graph.addVariable(INT, {4});
  graph.setTileMapping(a, 0);
  graph.setTileMapping(b, 1);
  graph.setTileMapping(c, 1);
  graph.setTileMapping(d, 0);
  BOOST_CHECK(identicalLayout(graph, concat(a, b), concat(d, c)));
}

BOOST_AUTO_TEST_CASE(DifferentMapping) {
  const auto numIPUs = 1;
  const auto tilesPerIPU = 4;
  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());

  auto a = graph.addVariable(INT, {4});
  auto b = graph.addVariable(INT, {4});
  auto c = graph.addVariable(INT, {4});
  auto d = graph.addVariable(INT, {4});
  graph.setTileMapping(a, 0);
  graph.setTileMapping(b, 1);
  graph.setTileMapping(c, 1);
  graph.setTileMapping(d, 0);
  BOOST_CHECK(!identicalLayout(graph, concat(a, b), concat(c, d)));
}

BOOST_AUTO_TEST_CASE(Transposed) {
  const auto numIPUs = 1;
  const auto tilesPerIPU = 1;
  auto device = createTestDevice(TEST_TARGET, numIPUs, tilesPerIPU);
  Graph graph(device.getTarget());

  auto a = graph.addVariable(INT, {4, 4}).transpose();
  auto b = graph.addVariable(INT, {4, 4}).transpose();
  graph.setTileMapping(a, 0);
  graph.setTileMapping(b, 0);
  BOOST_CHECK(identicalLayout(graph, a, b));
}
