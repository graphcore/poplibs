// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ReductionPatternsTest
#include "TestDevice.hpp"
#include "popops/reduction/ReductionStages.hpp"
#include <boost/test/unit_test.hpp>
#include <poplar/Engine.hpp>

#include <iostream>

using namespace poplar;
using namespace popops;

void printResult(const std::vector<PartialsDescription> &partialsDescription) {
  for (auto &par : partialsDescription) {
    std::cout << "Reduction patterns for column(s):";
    for (auto &column : par.columns) {
      std::cout << " " << column;
    }
    std::cout << "\n";
    for (auto &pattern : par.patterns) {
      std::cout << "Pattern innerFactor: " << pattern.innerFactor
                << " Start:" << pattern.regionOffset
                << " Stride:" << pattern.stride
                << " outerFactor:" << pattern.outerFactor
                << " Region:" << pattern.regionIdx << "\n";
    }
  }
}

bool operator!=(const PartialsPattern &lhs, const PartialsPattern &rhs) {
  return (lhs.innerFactor != rhs.innerFactor) ||
         (lhs.regionOffset != rhs.regionOffset) || (lhs.stride != rhs.stride) ||
         (lhs.outerFactor != rhs.outerFactor) ||
         (lhs.regionIdx != rhs.regionIdx);
}

bool checkResult(const std::vector<PartialsDescription> &generatedPatterns,
                 const std::vector<std::vector<PartialsPattern>> &patterns,
                 const std::vector<std::vector<unsigned>> &columns) {
  if (generatedPatterns[0].patterns.size() == 0 && patterns.size() == 0) {
    return true;
  }
  if (generatedPatterns.size() != patterns.size()) {
    return false;
  }
  for (unsigned i = 0; i < generatedPatterns.size(); i++) {
    if (generatedPatterns[i].patterns.size() != patterns[i].size()) {
      return false;
    }
    for (unsigned j = 0; j < generatedPatterns[i].patterns.size(); j++) {
      if (generatedPatterns[i].patterns[j] != patterns[i][j]) {
        return false;
      }
    }
  }

  for (unsigned i = 0; i < columns.size(); i++) {
    if (columns[i] != generatedPatterns[i].columns) {
      return false;
    }
  }
  return true;
}

BOOST_AUTO_TEST_CASE(ReducePatternsSimple) {
  // Define one reduction which, being empty causes the function to
  // identify columns
  std::vector<PartialsDescription> reductions;
  // The reductions operate on a matrix with 2 columns
  unsigned columns = 2;

  // Define a single region with 10 elements in it, starting at the
  // beginning of the Tensor
  std::vector<std::vector<Interval>> regions = {{{0, 10}}};

  // Given 2 columns, 10 elements in the region the elements expected in
  // column 0 are given by 1's:
  // 1 0 1 0 1 0 1 0 1 0
  // And for column 1:
  // 0 1 0 1 0 1 0 1 0 1
  //
  //   start (0 for column 0, 1 for column 1)
  //   Pattern size 1 element
  //   Pattern stride (repeat length) = 2
  //   Pattern repetitions (of the pattern 1 0 = 5). Lack of the  last
  //   tailing 0 doesn't matter.
  std::vector<std::vector<PartialsPattern>> expected = {{{1, 0, 2, 5, 0}},
                                                        {{1, 1, 2, 5, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}, {1}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsTwoReductions) {
  // Define two reductions which describe the reduction of columns 0, 2 each
  // spanning 1 column
  std::vector<PartialsDescription> reductions(2);
  reductions[0].columns.push_back(0);
  reductions[1].columns.push_back(2);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a single region with 20 elements in it, starting at the
  // beginning of the Tensor
  std::vector<std::vector<Interval>> regions = {{{0, 20}}};

  // Given 4 columns, 20 elements in the region the elements expected in
  // column 0 are given by 1's:
  // 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0
  // And column 2:
  // 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0
  std::vector<std::vector<PartialsPattern>> expected = {{{1, 0, 4, 5, 0}},
                                                        {{1, 2, 4, 5, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}, {2}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsMultiPattern) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(1);
  // The reductions operate on a matrix with 10 columns
  unsigned columns = 10;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{1, 2},
                                                 {11, 13},
                                                 {21, 22},
                                                 {31, 33},
                                                 {41, 42},
                                                 {51, 54},
                                                 {61, 62},
                                                 {71, 74},
                                                 {81, 82},
                                                 {91, 95}}};

  // Given 10 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 1  11 12 21 31 32 41 51 52 53 61 71 72 73 81 91 92 93
  // 1  1  0  1  1  0  1  1  0  0  1  1  0  0  1  1  0  0
  // So, 2 patterns: 1 1 0 and 1 1 0 0
  std::vector<std::vector<PartialsPattern>> expected = {
      {{2, 0, 3, 3, 0}, {2, 10, 4, 2, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{1}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsTruncatedPattern) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{1, 2},
                                                 {4, 5},
                                                 {8, 9},
                                                 {12, 14},
                                                 {16, 17},
                                                 {20, 21},
                                                 {24, 26},
                                                 {28, 29},
                                                 {32, 33}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 0 1 1 1 0 1 1 1 0 1 1
  // So, 2 patterns: 1 1 1 0 and 1 1 are expected:
  std::vector<std::vector<PartialsPattern>> expected = {
      {{3, 1, 4, 2, 0}, {2, 9, 1, 1, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsStop) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {
      {{1, 2}, {4, 5}, {8, 10}, {12, 13}, {16, 18}, {20, 21}, {24, 28}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 0 1 1 0 1 1 0 1 1 0 0 0
  // So, the pattern: 1 1 0 is expected:
  std::vector<std::vector<PartialsPattern>> expected = {{{2, 1, 3, 3, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsAllOnePattern) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{4, 5}, {8, 9}, {12, 13}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 1 1 1
  std::vector<std::vector<PartialsPattern>> expected = {{{3, 0, 1, 1, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsNoPattern) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(1);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{4, 5}, {8, 9}, {12, 13}}};

  // Given 4 columns, and concatenating the region described, there is nothing
  // in column 1.
  std::vector<std::vector<PartialsPattern>> expected;

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{1}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsMultiRegion) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;
  // Define a series of regions - illustrated below
  std::vector<std::vector<Interval>> regions = {
      {{4, 5}}, {{0, 3}, {8, 9}}, {{12, 13}}};

  // Given 4 columns, and concatenating the regions described, we have
  // Region 0: 1
  // Region 1: 1 0 0 1
  // Region 2: 1
  std::vector<std::vector<PartialsPattern>> expected = {
      {{1, 0, 1, 1, 0}, {1, 0, 3, 2, 1}, {1, 0, 1, 1, 2}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsLongerOne) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{1, 3},
                                                 {4, 5},
                                                 {8, 10},
                                                 {12, 13},
                                                 {16, 18},
                                                 {20, 21},
                                                 {24, 25},
                                                 {28, 29},
                                                 {32, 35}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 0 0 1 1 0 1 1 0 1 1 1 1 0 0
  // So, the patterns: 1 1 0, 1 1 are expected:
  std::vector<std::vector<PartialsPattern>> expected = {
      {{2, 2, 3, 2, 0}, {4, 8, 1, 1, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsShorterOne) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{1, 3},
                                                 {4, 5},
                                                 {8, 9},
                                                 {12, 14},
                                                 {16, 17},
                                                 {20, 21},
                                                 {24, 26},
                                                 {28, 33}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 0 0 1 1 1 0 1 1 1 0 1 0 0 0 1
  // So, the patterns: 1 1 1 0, 1 0 0 0 are expected:
  std::vector<std::vector<PartialsPattern>> expected = {
      {{3, 2, 4, 2, 0}, {1, 10, 4, 2, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsEndAtEnd) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  // The reductions operate on a matrix with 4 columns
  unsigned columns = 4;

  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {
      {{1, 3}, {4, 5}, {8, 10}, {12, 13}, {16, 18}, {20, 21}, {24, 25}}};

  // Given 4 columns, and concatenating the region described, those in
  // column 0 are given by 1's:
  // 0 0 1 1 0 1 1 0 1 1
  // So, the patterns: 1 1 0 is expected:
  std::vector<std::vector<PartialsPattern>> expected = {{{2, 2, 3, 3, 0}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsGroupedSimple) {
  std::vector<PartialsDescription> reductions;
  unsigned columns = 4;
  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{0, 24}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  auto groupedReductions = groupPartials(reductions, columns);
  std::cout << "Grouped:\n";
  printResult(groupedReductions);
  // As reductions is empty at the start, this will gather information on
  // all 4 columns. The intervals span 6 rows so we should see a sequence of
  // columns in our groupedReductions = 0, 1, 2, 3
  // Elements repeat once, start at the beginning of the region, have
  // a stride of 4 and repeat 6 times:
  std::vector<std::vector<PartialsPattern>> expected = {{{1, 0, 4, 6, 0}}};
  BOOST_TEST(checkResult(groupedReductions, expected, {{0, 1, 2, 3}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsGrouped2Groups) {
  std::vector<PartialsDescription> reductions;
  unsigned columns = 4;
  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {
      {{0, 2}, {4, 6}, {8, 10}, {12, 14}, {16, 18}, {20, 22}, {24, 27}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  auto groupedReductions = groupPartials(reductions, columns);
  std::cout << "Grouped:\n";
  printResult(groupedReductions);
  // Here we have a groupable pattern with columns 0, 1 in it and then an
  // individual patten with column 2 in it.
  std::vector<std::vector<PartialsPattern>> expected = {{{1, 0, 2, 7, 0}},
                                                        {{1, 14, 1, 1, 0}}};
  BOOST_TEST(checkResult(groupedReductions, expected, {{0, 1}, {2}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsGroupedTruncatedRegion) {
  std::vector<PartialsDescription> reductions;
  unsigned columns = 6;
  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{0, 23}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  auto groupedReductions = groupPartials(reductions, columns);
  std::cout << "Grouped:\n";
  printResult(groupedReductions);
  // Here the region almost conatains a whole 4 x 6 matrix but the last
  // element is missing.  We should get 2 grouped patterns:
  std::vector<std::vector<PartialsPattern>> expected = {{{1, 0, 6, 4, 0}},
                                                        {{1, 5, 6, 3, 0}}};
  BOOST_TEST(checkResult(groupedReductions, expected, {{0, 1, 2, 3, 4}, {5}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsGroupedMultiRegion) {
  std::vector<PartialsDescription> reductions;
  unsigned columns = 2;
  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {{{0, 24}}, {{24, 48}}};

  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  auto groupedReductions = groupPartials(reductions, columns);
  std::cout << "Grouped:\n";
  printResult(groupedReductions);
  // Here there are 2 identical sets of patterns for column 0, 1 but split over
  // 2 regions.  They can be grouped - the one group contains 2 patterns.
  std::vector<std::vector<PartialsPattern>> expected = {
      {{1, 0, 2, 12, 0}, {1, 0, 2, 12, 1}}};
  BOOST_TEST(checkResult(groupedReductions, expected, {{0, 1}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsMultiRegion3Patterns) {
  std::vector<PartialsDescription> reductions(1);
  reductions[0].columns.push_back(0);
  unsigned columns = 10;
  // Define a series of intervals in one region - illustrated below
  std::vector<std::vector<Interval>> regions = {
      {{0, 1}, {10, 11}, {11, 13}, {40, 41}, {50, 51}, {60, 61}}, {{0, 1}}};
  // Data in memory: column 0 or don't care : x
  //           01234567890123
  // Region 0: 00xx000
  // Region 1: 0
  gatherReductionPatterns(reductions, regions, columns);
  printResult(reductions);
  std::vector<std::vector<PartialsPattern>> expected = {
      {{2, 0, 4, 1, 0}, {3, 4, 1, 1, 0}, {1, 0, 1, 1, 1}}};
  BOOST_TEST(checkResult(reductions, expected, {{0}}));
}

BOOST_AUTO_TEST_CASE(ReducePatternsDivideDifferentLengths) {
  std::vector<PartialsDescription> reductions;
  std::vector<unsigned> columns = {1, 2};
  reductions.push_back({columns, {}});
  // 2 patterns where we have >1 column, and patterns with a large and different
  // innerFactor parameter.  The other parameters are arbitrary. These should be
  // split up.
  reductions[0].patterns.push_back({8, 0, (8 * 2), 3, 0});
  reductions[0].patterns.push_back({12, (8 * 2 * 3), (12 * 2), 6, 0});
  printResult(reductions);

  auto device = createTestDevice(DeviceType::IpuModel);
  Graph graph(device.getTarget());
  auto dividedReductions =
      dividePartials(reductions, graph, HALF, popops::Operation::ADD);
  std::cout << "Divided:\n";
  printResult(dividedReductions);

  std::vector<std::vector<PartialsPattern>> expected = {
      {{8, 0, 16, 3, 0}, {12, 48, 24, 6, 0}},
      {{8, 8, 16, 3, 0}, {12, 60, 24, 6, 0}}};
  std::vector<std::vector<unsigned>> expectedColumns(columns.size());
  for (unsigned i = 0; i < columns.size(); i++) {
    expectedColumns[i].push_back(columns[i]);
  }
  BOOST_TEST(checkResult(dividedReductions, expected, expectedColumns));
}
