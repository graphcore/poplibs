// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TileMappingTest
#include "TestDevice.hpp"
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

#include <algorithm>
#include <boost/integer/common_factor.hpp>
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
      commonGrainSize = boost::integer::gcd(commonGrainSize, i.size());
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
    auto thisTile = std::accumulate(
        tileMapping.begin(), tileMapping.end(), std::size_t(0),
        [&](std::size_t total, const Interval &i) { return total + i.size(); });
    minimum = std::min(minimum, thisTile);
  }
  return minimum;
}

} // namespace

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
  BOOST_CHECK_NO_THROW(poputil::mapOutputForElementWiseOp(graph, {in1b}, out1));
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

BOOST_AUTO_TEST_CASE(DimIsSplitOverTiles) {
  constexpr static std::size_t nElems = 4;
  constexpr static std::size_t numTiles = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  auto t = graph.addVariable(FLOAT, {nElems, numTiles});
  for (std::size_t i = 0; i < numTiles; ++i) {
    graph.setTileMapping(t.slice(i, i + 1, 1), i);
  }

  BOOST_CHECK(!dimIsSplitOverTiles(graph, t, 0));
  BOOST_CHECK(dimIsSplitOverTiles(graph, t, 1));
}

BOOST_AUTO_TEST_CASE(DimIsSplitOverIPUs) {
  constexpr static std::size_t nElems = 4;
  constexpr static std::size_t numIPUs = 4;
  auto device = createTestDevice(TEST_TARGET, numIPUs, 1);
  Graph graph(device.getTarget());

  auto t = graph.addVariable(FLOAT, {nElems, numIPUs});
  for (std::size_t i = 0; i < numIPUs; ++i) {
    graph.setTileMapping(t.slice(i, i + 1, 1), i);
  }

  BOOST_CHECK(!dimIsSplitOverIPUs(graph, t, 0));
  BOOST_CHECK(dimIsSplitOverIPUs(graph, t, 1));
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerBasic) {
  constexpr std::size_t numTiles = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t grainSize = 4;
  constexpr std::size_t numElems = numTiles * grainSize;
  auto t = graph.addVariable(FLOAT, {numElems});

  // Map each grainSize * grainSize elements to each grainSize tiles.
  for (unsigned tile = 0; tile < target.getNumTiles(); ++tile) {
    tracker.add(graph, tile,
                t.slice((tile / grainSize) * grainSize * grainSize,
                        (tile / grainSize + 1) * grainSize * grainSize));
  }

  tracker.mapTensorsByUse(graph, grainSize /*grainSize*/,
                          grainSize /*minElementsPerTile*/,
                          false /*optimizeHaloRegions*/);

  // Check the tensor is fully mapped
  BOOST_CHECK_NO_THROW(graph.getTileMapping(t));
  // Check grainSize is respected
  BOOST_CHECK_EQUAL(getTileImbalance(graph, t, grainSize, grainSize), 0);
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerGrainSize) {
  constexpr std::size_t numTiles = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t grainSize = 4;
  constexpr std::size_t numElems = numTiles;
  auto t = graph.addVariable(FLOAT, {numElems});

  // Map each grainSize elements to each grainSize tiles.
  for (unsigned tile = 0; tile < target.getNumTiles(); ++tile) {
    tracker.add(graph, tile,
                t.slice((tile / grainSize) * grainSize,
                        (tile / grainSize + 1) * grainSize));
  }

  tracker.mapTensorsByUse(graph, grainSize /*grainSize*/,
                          grainSize /*minElementsPerTile*/,
                          false /*optimizeHaloRegions*/);

  // Check the tensor is fully mapped
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(t));
  auto totalTilesUsed =
      std::accumulate(mapping.begin(), mapping.end(), std::size_t(0),
                      [](std::size_t total, const std::vector<Interval> &is) {
                        return total + !is.empty();
                      });
  // Sanity check used tiles against grainSize/tile balance.
  BOOST_CHECK_EQUAL(totalTilesUsed, numTiles / grainSize);
  // Check grainSize and tile min is respected
  BOOST_CHECK_EQUAL(getTileImbalance(graph, t, grainSize, grainSize), 0);
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerHaloRegions) {
  constexpr std::size_t numTiles = 4;
  static_assert((numTiles & (numTiles - 1)) == 0,
                "numTiles must be a power of 2");
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  // Expressing this in a sort of convolutional way because this optimisation
  // is targeted at data usage patterns sometimes introduced by partitioning
  // a convolution a certain way.
  constexpr std::size_t kernelSize = 8;
  constexpr std::size_t stride = 6;
  constexpr std::size_t padding = kernelSize - stride;
  constexpr std::size_t kernelPositions = numTiles / 2;
  constexpr std::size_t inputFieldSize = kernelPositions * stride + padding;
  auto t = graph.addVariable(FLOAT, {inputFieldSize, inputFieldSize});

  for (unsigned tile = 0; tile < target.getNumTiles(); ++tile) {
    tracker.add(graph, tile,
                t.slice((tile % kernelPositions) * stride,
                        (tile % kernelPositions) * stride + kernelSize, 1)
                    .slice((tile / kernelPositions) * stride,
                           (tile / kernelPositions) * stride + kernelSize, 0));
  }

  tracker.mapTensorsByUse(graph, 1 /*grainSize*/, 1 /*minElementsPerTile*/,
                          true /*optimizeHaloRegions*/);

  // Check the tensor is fully mapped
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(t));

  BOOST_TEST_MESSAGE("Mapping of t:");
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    BOOST_TEST_MESSAGE("Tile #" << tile << ":");
    for (const auto &i : mapping[tile]) {
      BOOST_TEST_MESSAGE("[" << i.begin() << "," << i.end() << ")");
    }
  }

  // Optimising halo regions means we make consistent the tile mapping of the
  // overlapping regions of the input covered by the kernel on different
  // tiles. Without this optimisation the 'halo' regions of the input used
  // by multiple tiles would be split evenly without consideration for the
  // surrounding, otherwise contiguous regions increasing exchange code size
  // needed to broadcast these 'halo' regions to the tiles that use them
  // (because these messages can be incorporated into those used to send the
  // regions that precede or follow the halo region).
  //
  // Expected result:
  //
  //   +-----------+-------+
  //   |           |       |
  //   |           |       |
  //   |           |       | inputFieldSize / 2
  //   |           |       |
  //   |           |       |
  //   +-------------------+
  //   |           |       |
  //   |           |       |
  //   |           |       | inputFieldSize / 2
  //   |           |       |
  //   |           |       |
  //   +-----------+-------+
  //    kernelSize   stride
  //
  for (unsigned tile = 0; tile < mapping.size(); ++tile) {
    auto width = (tile % kernelPositions == 0) ? kernelSize : stride;
    auto height = inputFieldSize / 2;
    auto xOffset = (tile % kernelPositions != 0) ? kernelSize : 0;
    auto yOffset = (tile / kernelPositions != 0) ? height : 0;
    BOOST_CHECK(std::all_of(
        mapping[tile].begin(), mapping[tile].end(), [&](const Interval &i) {
          return i.begin() % inputFieldSize == xOffset &&
                 i.begin() / inputFieldSize >= yOffset && i.size() == width;
        }));
    BOOST_CHECK_EQUAL(mapping[tile].size(), height);
  }
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerExtendPartialUsage) {
  constexpr std::size_t numTiles = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t splits = 2;
  constexpr std::size_t sharing = 4;
  constexpr std::size_t numElems = numTiles;
  auto t = graph.addVariable(FLOAT, {numElems, splits});

  // Map the first slice.
  for (unsigned tile = 0; tile < target.getNumTiles(); ++tile) {
    tracker.add(
        graph, tile,
        t.slice(0, 1, 1).squeeze({1}).slice((tile / sharing) * sharing,
                                            (tile / sharing + 1) * sharing));
  }

  tracker.mapTensorsByUse(graph, 1 /*grainSize*/, 1 /*minElementsPerTile*/,
                          false /*optimizeHaloRegions*/,
                          true /*extendPartialUsage*/);

  // Check the tensor is fully mapped.
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(t));

  // We expect that partial uses of the variable are extended based on their
  // neighbouring elements' uses. Hence we expect in this example for tile
  // balance to be perfect despite only marking data uses for half the tensor.
  BOOST_CHECK_EQUAL(getTileImbalance(graph, t, 1, 1), 0);
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerResolve) {
  constexpr std::size_t numTiles = 4;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t splits = 2;
  constexpr std::size_t grainSize = 4;
  constexpr std::size_t numElems = numTiles * grainSize;
  auto t = graph.addVariable(FLOAT, {splits, numElems});

  for (std::size_t s = 0; s < splits; ++s) {
    TensorUseTracker subTracker(target.getNumTiles());
    for (unsigned tile = 0; tile < target.getNumTiles(); ++tile) {
      subTracker.add(
          graph, tile,
          t[s].slice((tile / grainSize) * grainSize * grainSize,
                     (tile / grainSize + 1) * grainSize * grainSize));
    }
    subTracker.resolve(graph, grainSize, grainSize, false, false);
    tracker.add(std::move(subTracker));
  }

  tracker.mapTensorsByUse(graph, grainSize /*grainSize*/,
                          grainSize /*minElementsPerTile*/,
                          false /*optimizeHaloRegions*/);

  // Check the tensor is fully mapped
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(t));

  // We expect that usages that were resolved independently are
  // independent of one another. What this means in this context
  // is that the condition that the used elements are spread evenly
  // over the tiles they were used on, and with a minimum count
  // and grain size, holds on any set of independent usages in
  // isolation.
  for (std::size_t s = 0; s < splits; ++s) {
    BOOST_CHECK_EQUAL(getTileImbalance(graph, t[s], grainSize, grainSize), 0);
  }
}
