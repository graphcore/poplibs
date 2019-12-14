// Copyright (c) Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE LargeSplitRegionsTest
#include <boost/test/unit_test.hpp>
#include <poplar/Interval.hpp>
#include <poputil/Util.hpp>

using namespace poputil;
using namespace poplar;

BOOST_AUTO_TEST_CASE(largeSplitRegionsTest) {
  auto N = 100000000UL;
  auto split = splitRegions({{0, N}}, 1, 6);
  for (const auto &intervals : split)
    for (const auto &interval : intervals)
      BOOST_CHECK_LE(interval.end(), N);
}

unsigned
maxRegionElements(std::vector<std::vector<std::vector<Interval>>> &regions) {
  unsigned maxElements = 0;
  for (unsigned i = 0; i < regions.size(); i++) {
    for (unsigned j = 0; j < regions[i].size(); j++) {
      const unsigned regionElements = std::accumulate(
          regions[i][j].begin(), regions[i][j].end(), 0,
          [](std::size_t a, Interval b) { return a + b.size(); });
      maxElements = std::max(maxElements, regionElements);
    }
  }
  return maxElements;
}

BOOST_AUTO_TEST_CASE(splitRegionsForRepeatTest1) {
  std::vector<std::vector<Interval>> regions(1);
  unsigned x = 0;
  const unsigned expectedRegions[] = {3, 2, 2};
  for (unsigned maxElements = 49999; maxElements < 50002; maxElements++, x++) {
    regions.resize(1);
    regions[0].resize(1);
    regions[0] = {{0, 100000}};
    auto result = splitRegions(regions, 4, 1, 0, UINT_MAX, maxElements);
    BOOST_CHECK_LE(maxRegionElements(result), maxElements);
    BOOST_CHECK_EQUAL(result[0].size(), expectedRegions[x]);
  }
}

BOOST_AUTO_TEST_CASE(splitRegionsForRepeatTest2) {
  std::vector<std::vector<Interval>> regions(1);
  regions[0].resize(2);
  regions[0] = {{0, 30000}, {40000, 60001}};
  unsigned maxElements = 50000;
  auto result = splitRegions(regions, 4, 1, 0, UINT_MAX, maxElements);

  BOOST_CHECK_LE(maxRegionElements(result), maxElements);
  BOOST_CHECK_EQUAL(result[0].size(), 2);
}

BOOST_AUTO_TEST_CASE(splitRegionsForRepeatTest3) {
  std::vector<std::vector<Interval>> regions(2);

  regions[0].resize(2);
  regions[1].resize(2);
  regions[0] = {{0, 30000}, {40000, 70000}};
  regions[1] = {{0, 20000}, {40000, 60000}};
  unsigned maxElements = 50000;
  auto result = splitRegions(regions, 4, 1, 0, UINT_MAX, maxElements);

  BOOST_CHECK_LE(maxRegionElements(result), maxElements);
  BOOST_CHECK_EQUAL(result[0].size(), 3);
}

BOOST_AUTO_TEST_CASE(splitRegionsForRepeatTest4) {
  std::vector<std::vector<Interval>> regions(1);
  unsigned x = 0;
  const unsigned expectedRegions[] = {1, 1, 2};
  unsigned maxElements = 50000;
  for (unsigned regionSize = 49999; regionSize < 50002; regionSize++, x++) {
    regions.resize(1);
    regions[0].resize(1);
    regions[0] = {{0, regionSize}};
    auto result = splitRegions(regions, 4, 1, 0, UINT_MAX, maxElements);

    BOOST_CHECK_LE(maxRegionElements(result), maxElements);
    BOOST_CHECK_EQUAL(result[0].size(), expectedRegions[x]);
  }
}

BOOST_AUTO_TEST_CASE(splitRegionsForRepeatTest5) {
  std::vector<std::vector<Interval>> regions(2);

  regions[0].resize(2);
  regions[1].resize(2);
  regions[0] = {{0, 300}, {400, 700}};
  regions[1] = {{0, 200}, {400, 600}};
  unsigned maxElements = 32;
  auto result = splitRegions(regions, 4, 6, 0, UINT_MAX, maxElements);

  BOOST_CHECK_LE(maxRegionElements(result), maxElements);
  for (unsigned i = 0; i < result.size(); i++) {
    BOOST_CHECK_GE(result[i].size(), 2);
  }
}
