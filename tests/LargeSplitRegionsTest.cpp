#define BOOST_TEST_MODULE LargeSplitRegionsTest
#include <boost/test/unit_test.hpp>
#include <poputil/Util.hpp>

using namespace poputil;

BOOST_AUTO_TEST_CASE(largeSplitRegionsTest){
  auto N = 100000000UL;
  auto split = splitRegions({{0, N}}, 1, 6);
  for (const auto &intervals : split)
    for (const auto &interval : intervals)
      BOOST_CHECK_LE(interval.end(), N);
}
