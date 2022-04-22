// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE UtilTest

#include <boost/test/unit_test.hpp>
#include <poplar/Interval.hpp>
#include <poputil/Util.hpp>

#include <vector>

BOOST_AUTO_TEST_CASE(CalculateUnshufflingIntervals) {
  auto test = [](const std::vector<poplar::Interval> &intervals,
                 const std::vector<poplar::Interval> &expected) {
    const auto actual = poputil::calculateUnshufflingIntervals(intervals);
    BOOST_CHECK_EQUAL(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); i++) {
      BOOST_CHECK_EQUAL(actual[i].lower(), expected[i].lower());
      BOOST_CHECK_EQUAL(actual[i].upper(), expected[i].upper());
    }
  };

  // original: 0, 1, 2, 3, 4
  // shuffled: 0, 1, 2, 3, 4
  test({{0, 1}, {1, 2}, {2, 5}}, {{0, 1}, {1, 2}, {2, 5}});

  // original: 0, 1, 2, 3, 4
  // shuffled: 3, 4, 0, 1, 2
  test({{3, 5}, {0, 1}, {1, 3}}, {{2, 3}, {3, 5}, {0, 2}});

  // original: 0, 1, 2, 3, 4
  // shuffled: 4, 3, 2, 1, 0
  test({{4, 5}, {3, 4}, {2, 3}, {1, 2}, {0, 1}},
       {{4, 5}, {3, 4}, {2, 3}, {1, 2}, {0, 1}});

  // original: 0, 1, 2, 3, 4
  // shuffled: 4, 3, 0, 1, 2
  test({{4, 5}, {3, 4}, {0, 3}}, {{2, 5}, {1, 2}, {0, 1}});
}
