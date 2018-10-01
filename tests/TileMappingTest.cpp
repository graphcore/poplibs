#define BOOST_TEST_MODULE TileMappingTest
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

BOOST_AUTO_TEST_CASE(Imbalance) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device);
  popops::addCodelets(graph);
  auto t = graph.addVariable(FLOAT, {1000});
  graph.setTileMapping(t.slice(0, 100), 0);
  graph.setTileMapping(t.slice(100, 800), 1);
  graph.setTileMapping(t.slice(800, 900), 2);
  graph.setTileMapping(t.slice(900, 1000), 3);

  // A perfectly balanced tensor would have 250 elems per tile, but
  // we have 700 on tile 1, so the imbalance is 450.
  BOOST_CHECK_EQUAL(poputil::getTileImbalance(graph, t), 450);

  poputil::rebalanceTensor(graph, t, 0, 1, 0);

  BOOST_CHECK_EQUAL(poputil::getTileImbalance(graph, t), 0);

  graph.setTileMapping(t.slice(0, 100), 0);
  graph.setTileMapping(t.slice(100, 150), 1);
  graph.setTileMapping(t.slice(250, 900), 1);
  graph.setTileMapping(t.slice(150, 250), 2);
  graph.setTileMapping(t.slice(900, 1000), 3);

  // A perfectly balanced tensor would have 250 elems per tile, but
  // we have 700 on tile 1, so the imbalance is 450.
  BOOST_CHECK_EQUAL(poputil::getTileImbalance(graph, t), 450);

  poputil::rebalanceTensor(graph, t, 0, 1, 0);

  BOOST_CHECK_EQUAL(poputil::getTileImbalance(graph, t), 0);
}
