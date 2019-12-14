// Copyright (c) Graphcore Ltd, All rights reserved.
#define BOOST_TEST_MODULE AlgorithmTest
#include <boost/test/unit_test.hpp>
#include <poplibs_support/Algorithm.hpp>

using namespace poplibs_support;

BOOST_AUTO_TEST_CASE(BalancedPartitionZeroPartitions) {
  // The correct answer is 0 of either size of partition as
  // ceil(1/0) and floor(1/0) are both infinity.
  auto p = balancedPartition(1u, 0u);
  BOOST_CHECK_EQUAL(p.first, 0u);
  BOOST_CHECK_EQUAL(p.second, 0u);
}

BOOST_AUTO_TEST_CASE(BalancedPartitionZeroSize) {
  // Any answer is correct here so just check this works.
  BOOST_CHECK_NO_THROW(balancedPartition(0u, 1u));
}

BOOST_AUTO_TEST_CASE(BalancedPartitionNLessThanD) {
  auto p = balancedPartition(1u, 2u);
  // We expect n partitions of size 1 and d - n partitions of size 0.
  BOOST_CHECK_EQUAL(p.first, 1u);
  BOOST_CHECK_EQUAL(p.second, 1u);

  p = balancedPartition(3u, 5u);
  BOOST_CHECK_EQUAL(p.first, 3u);
  BOOST_CHECK_EQUAL(p.second, 2u);
}

BOOST_AUTO_TEST_CASE(BalancedPartitionDividesEqually) {
  auto p = balancedPartition(20u, 5u);
  // We expect the non-zero answer in the first return even
  // though mathematically the answer being in the second return would
  // be correct.
  BOOST_CHECK_EQUAL(p.first, 5u);
  BOOST_CHECK_EQUAL(p.second, 0u);
}

BOOST_AUTO_TEST_CASE(BalancedPartitionDividesUnequally) {
  constexpr unsigned n = 20u;

  // For all divisors which give ceil(n/d)=2 and floor(n/d)=1
  for (unsigned d = n / 2 + 1; d < n; ++d) {
    auto p = balancedPartition(n, d);
    BOOST_CHECK_EQUAL(p.first * ceildiv(n, d) + p.second * floordiv(n, d), n);
    BOOST_CHECK_EQUAL(p.first + p.second, d);
  }
}
