#define BOOST_TEST_MODULE TileMappingTest
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include "TestDevice.hpp"

#include <boost/integer/common_factor.hpp>
#include <algorithm>
#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace {

bool hasAtLeastGrainSize(const Graph &graph, const Tensor &tensor,
                         std::size_t grainSize) {
  const auto mapping = graph.getTileMapping(tensor);
  std::size_t commonGrainSize = grainSize;
  for (unsigned t = 0; t < mapping.size(); ++t) {
    const auto &tileMapping = mapping[t];
    if (tileMapping.empty())
      continue;
    for (const auto &i : tileMapping) {
      commonGrainSize =
        boost::integer::gcd(commonGrainSize, i.size());
    }
  }
  return commonGrainSize == grainSize;
}

std::size_t getMinElementsPerTile(const Graph &graph, const Tensor &tensor) {
  const auto mapping = graph.getTileMapping(tensor);
  std::size_t minimum = std::numeric_limits<std::size_t>::max();
  for (unsigned t = 0; t < mapping.size(); ++t) {
    const auto &tileMapping = mapping[t];
    if (tileMapping.empty())
      continue;
    auto thisTile = std::accumulate(tileMapping.begin(), tileMapping.end(),
                                    std::size_t(0),
        [&](std::size_t total , const Interval &i) {
          return total + i.size();
        });
    minimum = std::min(minimum, thisTile);
  }
  return minimum;
}

}

BOOST_AUTO_TEST_CASE(Imbalance) {
  auto device = createTestDevice(TEST_TARGET, 1, 4);
  Graph graph(device.getTarget());
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

BOOST_AUTO_TEST_CASE(ElementWiseBasic) {
  constexpr static std::size_t numTiles = 13;
  constexpr static std::size_t nElems = 1000;
  constexpr static std::size_t grainSize = 4;

  static_assert((nElems % grainSize) == 0,
                "nElems must be multiple of grainSize for test to pass");
  static_assert((((nElems + numTiles - 1) / numTiles) % grainSize) != 0,
                "nElems split among numTiles must result in a greater maximum "
                "no. of elements per-tile with grainSize vs. no grainSize for "
                "this test to pass");

  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  auto in1 = graph.addVariable(FLOAT, {nElems});
  auto in2 = graph.addVariable(FLOAT, {nElems});
  auto out = graph.addVariable(FLOAT, {nElems});
  {
    // in1 mapped with grain size 1.
    unsigned tile = 0;
    for (std::size_t i = 0; i < nElems; ++i) {
      graph.setTileMapping(in1[i], tile);
      tile = (tile + 1) % numTiles;
    }

    // in2 mapped with chosen grain size.
    tile = 0;
    for (std::size_t i = 0; i < nElems; i += grainSize) {
      graph.setTileMapping(in2.slice(i, i + grainSize), tile);
      tile = (tile + 1) % numTiles;
    }
  }

  // Default case, no limitations on grain size or grains per-tile.
  // We expect output mapping to match in1 as this has the smallest
  // maximum no. of elements per-tile.
  poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out);
  BOOST_CHECK(graph.getTileMapping(in1) == graph.getTileMapping(out));

  // Constrain grainSize, we expect in2 mapping to be chosen as the mapping
  // for output as in1 does not meet the grainSize requirement.
  poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out, grainSize);
  BOOST_CHECK(graph.getTileMapping(in2) == graph.getTileMapping(out));
  BOOST_CHECK(hasAtLeastGrainSize(graph, out, grainSize));

  // Constrain minGrainsPerTile, we expect out to not match the mapping of
  // either in1 or in2 as neither has a high enough minGrainsPerTile.
  poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out, 1, 100);
  BOOST_CHECK_GT(getMinElementsPerTile(graph, out), 100 - 1);

  // Constrain both grainSize and minGrainsPerTile.
  poputil::mapOutputForElementWiseOp(graph, {in1, in2}, out, grainSize, 100);
  BOOST_CHECK_GT(getMinElementsPerTile(graph, out), (grainSize * 100) - 1);
  BOOST_CHECK(hasAtLeastGrainSize(graph, out, grainSize));
}

BOOST_AUTO_TEST_CASE(ElementWiseEdgeCase) {
  constexpr static std::size_t numTiles = 4;
  constexpr static std::size_t nElems = 1000;
  constexpr static std::size_t broadcastFactor = 4;
  static_assert((nElems % broadcastFactor) == 0, "");
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  auto in1 = graph.addVariable(FLOAT, {nElems / broadcastFactor});
  auto out1 = graph.addVariable(FLOAT, {nElems});
  unsigned tile = 0;
  for (std::size_t i = 0; i < nElems / broadcastFactor; ++i) {
    graph.setTileMapping(in1[i], tile);
    tile = (tile + 1) % numTiles;
  }

  // We expect output mapping be valid and to differ from this input as
  // the input has aliasing elements.
  auto in1b = in1.broadcast(broadcastFactor, 0);
  BOOST_CHECK_NO_THROW(
    poputil::mapOutputForElementWiseOp(graph, {in1b}, out1));
  BOOST_CHECK_NO_THROW(graph.getTileMapping(out1));
  BOOST_CHECK(graph.getTileMapping(in1b) != graph.getTileMapping(out1));

  // We expect mapping to succeed and produce a valid mapping when
  // a tensor with constants is given as an input.
  auto in2 = graph.addConstant(FLOAT, {nElems}, 0);
  graph.setTileMapping(in2, 0);
  auto out2 = graph.addVariable(FLOAT, {nElems});
  BOOST_CHECK_NO_THROW(poputil::mapOutputForElementWiseOp(graph, {in2}, out2));
  BOOST_CHECK_NO_THROW(graph.getTileMapping(out2));
}
