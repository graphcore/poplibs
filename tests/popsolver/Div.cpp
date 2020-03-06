// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
// Simple tests for popsolver.
//
#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Div
#include "poplibs_support/Algorithm.hpp"
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(CeilDiv) {
  Model m;
  auto a = m.addConstant(10);
  auto b = m.addConstant(4);
  auto c = m.ceildiv(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 3);
}

BOOST_AUTO_TEST_CASE(CeilDiv2) {
  Model m;
  auto a = m.addVariable();
  auto b = m.addConstant(4);
  auto c = m.ceildiv(a, b);
  m.lessOrEqual(3, c);
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 9);
  BOOST_CHECK_EQUAL(s[c], 3);
}

BOOST_AUTO_TEST_CASE(FloorDiv) {
  Model m;
  auto a = m.addVariable();
  auto b = m.addConstant(4);
  auto c = m.floordiv(a, b);
  m.lessOrEqual(3, c);
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 12);
  BOOST_CHECK_EQUAL(s[c], 3);
}

BOOST_AUTO_TEST_CASE(CeilDivZero) {
  Model m;
  auto a = m.addVariable();
  auto b = m.addConstant(0);
  auto c = m.ceildiv(a, b);
  BOOST_CHECK_EQUAL(m.minimize(c).validSolution(), false);
}

BOOST_AUTO_TEST_CASE(CeilDivConstrainDivisor) {
  using poplibs_support::ceildiv;
  const unsigned maxDivisor = 20;
  for (unsigned divisor = 1; divisor != maxDivisor; ++divisor) {
    Model m;
    const unsigned dividend = 49;
    auto a = m.addConstant(dividend, "a");
    auto b = m.addConstant(divisor, "b");
    auto c = m.ceildivConstrainDivisor(a, b, "c");

    auto s = m.minimize(c);
    // The constrained div result is only valid when a smaller divisor would
    // give a different result
    auto expectSolution = divisor < 2 || ceildiv(dividend, divisor) !=
                                             ceildiv(dividend, divisor - 1);
    BOOST_CHECK_EQUAL(s.validSolution(), expectSolution);
    // division correct?
    if (expectSolution && s.validSolution()) {
      BOOST_CHECK_EQUAL(s[c], poplibs_support::ceildiv(dividend, s[b]));
    }
  }
}
