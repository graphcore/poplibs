// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE TileMappingTest
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <popops/ElementWiseUtil.hpp>
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
using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(ElementWiseEdgeCase) {
  constexpr static std::size_t numTiles = 4;
  constexpr static std::size_t nElems = 1000;
  constexpr static std::size_t broadcastFactor = 4;
  static_assert((nElems % broadcastFactor) == 0, "");
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  auto in1 = graph.addVariable(FLOAT, {nElems / broadcastFactor});
  unsigned tile = 0;
  for (std::size_t i = 0; i < nElems / broadcastFactor; ++i) {
    graph.setTileMapping(in1[i], tile);
    tile = (tile + 1) % numTiles;
  }

  // We expect output mapping be valid and to differ from this input as
  // the input has aliasing elements.
  auto in1b = in1.broadcast(broadcastFactor, 0);
  auto out1 = popops::createOutputForElementWiseOp(graph, {in1b}, FLOAT);
  BOOST_CHECK_NO_THROW(graph.getTileMapping(out1));
  BOOST_CHECK(graph.getTileMapping(in1b) != graph.getTileMapping(out1));

  // We expect mapping to succeed and produce a valid mapping when
  // a tensor with constants is given as an input.
  auto in2 = graph.addConstant(FLOAT, {nElems}, 0);
  graph.setTileMapping(in2, 0);
  auto out2 = popops::createOutputForElementWiseOp(graph, {in2}, FLOAT);
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
                          grainSize /*minElementsPerTile*/);

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
                          grainSize /*minElementsPerTile*/);

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
                          false /*extendPartialUsage*/,
                          TensorUseTracker::MappingMethod::OptimizeHaloRegions);

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
                          true /*extendPartialUsage*/,
                          TensorUseTracker::MappingMethod::None);

  // Check the tensor is fully mapped.
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(t));

  // We expect that partial uses of the variable are extended based on their
  // neighbouring elements' uses. Hence we expect in this example for tile
  // balance to be perfect despite only marking data uses for half the tensor.
  BOOST_CHECK_EQUAL(getTileImbalance(graph, t, 1, 1), 0);
}

BOOST_AUTO_TEST_CASE(TensorUseTrackerConstrainMappingToUsedTiles) {
  constexpr std::size_t numTiles = 4;
  constexpr std::size_t numRows = 3;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t grainSize = 4;
  constexpr std::size_t numElems = numTiles * grainSize;
  auto t = graph.addVariable(HALF, {numRows, numElems});

  // Map the first slice.
  std::size_t firstTileElems = 1;
  std::size_t lastIndex = firstTileElems;
  graph.setTileMapping(t.slice(0, firstTileElems, 1), 0);

  for (unsigned tile = 1; tile != numTiles; ++tile) {
    graph.setTileMapping(t.slice(lastIndex, (tile + 1) * grainSize, 1), tile);
    lastIndex = (tile + 1) * grainSize;
  }

  auto rhsOpTensor =
      createBroadcastOperand(graph, t, t.elementType(), 1, false);

  // Check the tensor is fully mapped.
  std::vector<std::vector<Interval>> mapping;
  BOOST_CHECK_NO_THROW(mapping = graph.getTileMapping(rhsOpTensor));

  BOOST_CHECK_EQUAL(mapping[0].size(), 1);
  // There should only be firstTileElems on the first tile even though grain
  // size is set to a different value
  BOOST_CHECK_EQUAL(mapping[0][0].size(), firstTileElems);
  BOOST_CHECK_EQUAL(mapping[1][0].size(), 2 * grainSize - firstTileElems);
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
    subTracker.resolve(graph, grainSize, grainSize, false,
                       TensorUseTracker::MappingMethod::None);
    tracker.add(std::move(subTracker));
  }

  tracker.mapTensorsByUse(graph, grainSize /*grainSize*/, grainSize);

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

BOOST_AUTO_TEST_CASE(CloneToGraph) {
  constexpr std::size_t numTiles = 8;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  auto vgraph1 = graph.createVirtualGraph(0, 4);
  auto vgraph2 = graph.createVirtualGraph(4, 8);

  auto t1 = vgraph1.addVariable(poplar::FLOAT, {1024});
  poputil::mapTensorLinearly(vgraph1, t1);

  auto t2 = poputil::cloneToGraph(vgraph1, vgraph2, t1);

  // The tile mappings within the virtual graphs should be the same.
  BOOST_CHECK(vgraph1.getTileMapping(t1) == vgraph2.getTileMapping(t2));

  auto t1_mapping = graph.getTileMapping(t1);
  auto t2_mapping = graph.getTileMapping(t2);

  // Rotating the tile mapping of t2 by 4 should be the same as the tile mapping
  // of t1 on the top-level graph.
  std::rotate(t2_mapping.begin(), t2_mapping.begin() + 4, t2_mapping.end());
  BOOST_CHECK(t1_mapping == t2_mapping);
}

BOOST_AUTO_TEST_CASE(CloneToGraphBadRange) {
  constexpr std::size_t numTiles = 8;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  auto vgraph1 = graph.createVirtualGraph(0, 4);
  auto vgraph2 = graph.createVirtualGraph(4, 5);

  auto t1 = vgraph1.addVariable(poplar::FLOAT, {1024});
  poputil::mapTensorLinearly(vgraph1, t1);

  // vgraph2 doesn't have enough tiles. Expect an exception here.
  BOOST_CHECK_THROW(poputil::cloneToGraph(vgraph1, vgraph2, t1),
                    poputil::poplibs_error);
}

BOOST_AUTO_TEST_CASE(ChooseMappingOffset) {
  constexpr std::size_t numTiles = 1024;

  // Verifying specific values as observed - the test is that they are in
  // an acceptable range and are different to each other
  BOOST_CHECK_EQUAL(chooseMappingOffset(numTiles, {4, 64}), 239);
  // Check the max tile is respected, result is < 4 and would otherwise be 239
  BOOST_CHECK_EQUAL(chooseMappingOffset(4, {4, 64}), 3);
  // Check the seed has an effect, result would be 239 otherwise
  BOOST_CHECK_EQUAL(chooseMappingOffset(numTiles, {4, 64}, 0x1234), 2);
  // Check the shape has an effect, result would be 239 otherwise
  BOOST_CHECK_EQUAL(chooseMappingOffset(numTiles, {256}), 206);
}

BOOST_AUTO_TEST_CASE(OffsetLinearMapping) {
  constexpr std::size_t numTiles = 64;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  const auto &target = device.getTarget();
  Graph graph(target);

  TensorUseTracker tracker(target.getNumTiles());

  constexpr std::size_t numElems = numTiles * 10;

  struct Test {
    unsigned offset;
    bool ascending;
  };

  struct MappingSummary {
    std::size_t first;
    std::size_t last;
    std::size_t used;
    std::size_t firstUnused;
    std::size_t lastUnused;
  };
  auto summariseMap = [](const std::vector<std::vector<Interval>> &map) {
    MappingSummary result = {map.size(), 0, 0, map.size(), 0};
    for (std::size_t j = 0; j < map.size(); j++) {
      if (map[j].size()) {
        result.first = std::min(j, result.first);
        result.last = std::max(j, result.last);
        result.used++;
      } else {
        result.firstUnused = std::min(j, result.firstUnused);
        result.lastUnused = std::max(j, result.lastUnused);
      }
    }
    return result;
  };
  std::vector<Test> tests = {{16, true},  {32, false},         {60, true},
                             {60, false}, {1, true},           {0, false},
                             {1, false},  {numTiles - 1, true}};
  std::vector<Tensor> tensors(tests.size());
  std::vector<MappingSummary> results(tests.size());

  for (unsigned i = 0; i < tests.size(); i++) {
    tensors[i] = graph.addVariable(FLOAT, {numElems});
    mapTensorLinearlyWithOffset(graph, tensors[i], tests[i].offset,
                                tests[i].ascending);
    auto map = graph.getTileMapping(tensors[i]);
    results[i] = summariseMap(map);
  }
  // As a reference
  auto tFromZero = graph.addVariable(FLOAT, {numElems});
  mapTensorLinearly(graph, tFromZero);
  auto mapFromZero = graph.getTileMapping(tFromZero);
  auto mapFromZeroSummary = summariseMap(mapFromZero);

  for (unsigned i = 0; i < tests.size(); i++) {
    BOOST_CHECK_EQUAL(results[i].used, mapFromZeroSummary.used);
    if (tests[i].ascending) {
      bool wrapAround = tests[i].offset + mapFromZeroSummary.used > numTiles;
      if (wrapAround) {
        BOOST_CHECK_EQUAL(results[i].first, 0);
        BOOST_CHECK_EQUAL(results[i].last, numTiles - 1);
        BOOST_CHECK_EQUAL(results[i].lastUnused, tests[i].offset - 1);
        BOOST_CHECK_EQUAL(results[i].firstUnused,
                          (tests[i].offset + mapFromZeroSummary.used) %
                              numTiles);
      } else {
        BOOST_CHECK_EQUAL(results[i].first, tests[i].offset);
        BOOST_CHECK_EQUAL(results[i].last + 1,
                          tests[i].offset + mapFromZeroSummary.used);
      }
    } else {
      const auto firstMappedTile = (numTiles - tests[i].offset - 1);
      bool wrapAround = firstMappedTile < mapFromZeroSummary.used;
      if (wrapAround) {
        BOOST_CHECK_EQUAL(results[i].first, 0);
        BOOST_CHECK_EQUAL(results[i].last, numTiles - 1);
        BOOST_CHECK_EQUAL(results[i].firstUnused, firstMappedTile + 1);
        BOOST_CHECK_EQUAL(results[i].lastUnused,
                          numTiles + firstMappedTile - mapFromZeroSummary.used);
      } else {
        BOOST_CHECK_EQUAL(results[i].first,
                          firstMappedTile - mapFromZeroSummary.used + 1);
        BOOST_CHECK_EQUAL(results[i].last, firstMappedTile);
      }
    }
  }
}
