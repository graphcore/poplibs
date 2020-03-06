// Copyright (c) 2019 Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE ElementWiseUtilTest
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/ContiguousRegionsByTile.hpp"
#include "popops/ElementWiseUtilInternal.hpp"
#include <boost/test/unit_test.hpp>
#include <popops/ElementWiseUtil.hpp>

#include <poplar/Engine.hpp>

#include "TestDevice.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poplibs_support;
using poplibs::getSortedContiguousRegionsByTile;

BOOST_AUTO_TEST_CASE(CreateOutputBasic) {
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

  const auto in1 = graph.addVariable(FLOAT, {nElems});
  const auto in2 = graph.addVariable(FLOAT, {nElems});
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

  // Default case, we expect output to match layout of in1 as this has
  // the smallest maximum no. of elements per-tile.
  const auto out = createOutputForElementWiseOp(graph, {in1, in2}, FLOAT);
  const auto in1Layout =
      getSortedContiguousRegionsByTile(graph, in1, graph.getTileMapping(in1));
  const auto outLayout =
      getSortedContiguousRegionsByTile(graph, out, graph.getTileMapping(out));
  BOOST_CHECK(in1Layout == outLayout);
}

BOOST_AUTO_TEST_CASE(CreateOutputTensorSpreadOverMoreTiles) {
  constexpr std::size_t numTiles = 16;
  constexpr std::size_t numElems = 24;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  // in1 will have 2 elems on all tiles for a total of 12
  // tiles
  //
  // in2 will have 2 elems on 8 tiles and 1 elem on 8 other
  // tiles for a total of 16 tiles.
  //
  // Both will have the same maximum elements per tile but
  // occupy different numbers of tiles.
  //
  const auto in1 = graph.addVariable(FLOAT, {numElems});
  const auto in2 = graph.addVariable(FLOAT, {numElems});
  {
    const auto ceil = ceildiv(numElems, numTiles);

    // A necessary condition for this test to be valid is
    // that the balanced partition spreads over more tiles
    // than the other.
    BOOST_CHECK_LT(ceildiv(numElems, ceil), numTiles);

    const auto floor = floordiv(numElems, numTiles);
    const auto p = balancedPartition(numElems, numTiles);
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      const auto begin1 = std::min(numElems, tile * ceil);
      const auto end1 = std::min(numElems, (tile + 1) * ceil);
      graph.setTileMapping(in1.slice(begin1, end1), tile);

      std::size_t begin2 = 0;
      std::size_t end2 = 0;
      if (tile < p.first) {
        begin2 = tile * ceil;
        end2 = (tile + 1) * ceil;
      } else {
        begin2 = p.first * ceil + (tile - p.first) * floor;
        end2 = p.first * ceil + (tile + 1 - p.first) * floor;
      }
      graph.setTileMapping(in2.slice(begin2, end2), tile);
    }
  }

  const auto out = createOutputForElementWiseOp(graph, {in1, in2}, FLOAT);
  // We expect out to match the layout of in2 because this has
  // the greatest spread of elements over tiles even though the max
  // elements per-tile was the same.
  const auto in2Layout =
      getSortedContiguousRegionsByTile(graph, in2, graph.getTileMapping(in2));
  const auto outLayout =
      getSortedContiguousRegionsByTile(graph, out, graph.getTileMapping(out));
  BOOST_CHECK(in2Layout == outLayout);
}

BOOST_AUTO_TEST_CASE(CreateOutputTensorWithFewerContiguousRegions) {
  constexpr std::size_t numTiles = 16;
  constexpr std::size_t A = numTiles;
  constexpr std::size_t B = numTiles;
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  // in1 will have B contiguous regions per-tile
  // in2 will have 1 contiguous region per-tile
  //
  // Both will have the same maximum elements per tile and
  // occupy the same number of tiles.
  //
  const auto in1 = graph.addVariable(FLOAT, {B, A}).transpose();
  const auto in2 = graph.addVariable(FLOAT, {A, B});
  {
    unsigned tile = 0;
    for (std::size_t i = 0; i < A; ++i) {
      graph.setTileMapping(in1[i], tile);
      graph.setTileMapping(in2[i], tile);
      tile = (tile + 1) % numTiles;
    }
  }

  const auto out = createOutputForElementWiseOp(graph, {in1, in2}, FLOAT);
  // We expect out to match the layout of in2 because this has
  // the fewest contiguous regions per-tile and every other measure
  // used was the same.
  const auto in2Layout =
      getSortedContiguousRegionsByTile(graph, in2, graph.getTileMapping(in2));
  const auto outLayout =
      getSortedContiguousRegionsByTile(graph, out, graph.getTileMapping(out));
  BOOST_CHECK(in2Layout == outLayout);
}

BOOST_AUTO_TEST_CASE(CreateOutputNonParallelWriteable) {
  constexpr static std::size_t numTiles = 4;
  constexpr static std::size_t nElems = 1000;
  constexpr static std::size_t broadcastFactor = 4;
  static_assert((nElems % broadcastFactor) == 0, "");
  auto device = createTestDevice(TEST_TARGET, 1, numTiles);
  Graph graph(device.getTarget());

  const auto in1 = graph.addVariable(FLOAT, {nElems / broadcastFactor});
  unsigned tile = 0;
  for (std::size_t i = 0; i < nElems / broadcastFactor; ++i) {
    graph.setTileMapping(in1[i], tile);
    tile = (tile + 1) % numTiles;
  }

  // We expect output mapping be valid and to differ from this input as
  // the input has aliasing elements.
  const auto in1b = in1.broadcast(broadcastFactor, 0);
  Tensor out1;
  BOOST_CHECK_NO_THROW(out1 =
                           createOutputForElementWiseOp(graph, {in1b}, FLOAT));
  BOOST_CHECK_NO_THROW(graph.getTileMapping(out1));
  BOOST_CHECK(graph.getTileMapping(in1b) != graph.getTileMapping(out1));

  // We expect mapping to succeed and produce a valid mapping when
  // a tensor with constants is given as an input.
  const auto in2 = graph.addConstant(FLOAT, {nElems}, 0);
  graph.setTileMapping(in2, 0);
  Tensor out2;
  BOOST_CHECK_NO_THROW(out2 =
                           createOutputForElementWiseOp(graph, {in2}, FLOAT));
  BOOST_CHECK_NO_THROW(graph.getTileMapping(out2));
}

void CheckCutRegionSectionResults(
    const std::vector<poplar::Interval> &sectionActual,
    const std::vector<poplar::Interval> &sectionExpected,
    const unsigned indexActual, const unsigned indexExpected,
    const unsigned offsetActual, const unsigned offsetExpected,
    const unsigned regIndexActual, const unsigned regIndexExpected) {
  BOOST_CHECK(sectionActual == sectionExpected);
  BOOST_CHECK(indexActual == indexExpected);
  BOOST_CHECK(offsetActual == offsetExpected);
  BOOST_CHECK(regIndexActual == regIndexExpected);
}

BOOST_AUTO_TEST_CASE(CutRegionSectionSingleInterval) {
  std::vector<poplar::Interval> regions = {{0, 43}};
  unsigned index = 0;
  unsigned offset = 0;
  unsigned regIndex = 0;
  auto section = cutRegionSection(regions, 11, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{0, 11}}, index, 0, offset, 11,
                               regIndex, 0);

  section = cutRegionSection(regions, 31, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{11, 42}}, index, 0, offset, 42,
                               regIndex, 0);

  section = cutRegionSection(regions, 1, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{42, 43}}, index, 0, offset, 0,
                               regIndex, 1);

  // Cut out the entire single interval
  index = 0;
  offset = 0;
  regIndex = 0;
  section = cutRegionSection(regions, 43, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{0, 43}}, index, 0, offset, 0,
                               regIndex, 1);
}

BOOST_AUTO_TEST_CASE(CutRegionSectionIntervalBoundary) {
  std::vector<poplar::Interval> regions = {
      {0, 43}, {56, 57}, {58, 68}, {75, 85}, {101, 117}};

  // Section starts after a few intervals
  // Section ends at the beginning of an interval
  unsigned index = 0;
  unsigned offset = 0;
  unsigned regIndex = 0;
  auto section = cutRegionSection(regions, 45, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{0, 43}, {56, 57}, {58, 59}}, index, 2,
                               offset, 1, regIndex, 0);

  // Section ends at the end of an interval
  index = 0;
  offset = 42;
  section = cutRegionSection(regions, 11, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{42, 43}, {56, 57}, {58, 67}}, index,
                               2, offset, 9, regIndex, 0);

  // Section ends at the end of the region
  index = 0;
  offset = 42;
  section = cutRegionSection(regions, 38, index, offset, regIndex);
  CheckCutRegionSectionResults(
      section, {{42, 43}, {56, 57}, {58, 68}, {75, 85}, {101, 117}}, index, 0,
      offset, 0, regIndex, 1);
}

BOOST_AUTO_TEST_CASE(CutRegionSectionMultipleExecutions) {
  std::vector<poplar::Interval> regions = {
      {0, 43}, {56, 57}, {58, 68}, {75, 85}, {101, 117}, {90, 95}, {96, 101}};

  unsigned index = 0;
  unsigned offset = 42;
  unsigned regIndex = 0;
  auto section = cutRegionSection(regions, 5, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{42, 43}, {56, 57}, {58, 61}}, index,
                               2, offset, 3, regIndex, 0);

  section = cutRegionSection(regions, 11, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{61, 68}, {75, 79}}, index, 3, offset,
                               4, regIndex, 0);

  section = cutRegionSection(regions, 3, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{79, 82}}, index, 3, offset, 7,
                               regIndex, 0);

  section = cutRegionSection(regions, 22, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{82, 85}, {101, 117}, {90, 93}}, index,
                               5, offset, 3, regIndex, 0);

  section = cutRegionSection(regions, 7, index, offset, regIndex);
  CheckCutRegionSectionResults(section, {{93, 95}, {96, 101}}, index, 0, offset,
                               0, regIndex, 1);
}
