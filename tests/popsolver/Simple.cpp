// Copyright (c) Graphcore Ltd, All rights reserved.
// Simple tests for popsolver.
//
#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Simple
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(NoConstraints) {
  Model m;
  auto a = m.addVariable(5, 10);
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 5);
}

BOOST_AUTO_TEST_CASE(Unsatisfiable) {
  Model m;
  auto a = m.addVariable(2, 5);
  m.lessOrEqual(a, 1);
  BOOST_CHECK_EQUAL(m.minimize(a).validSolution(), false);
}

BOOST_AUTO_TEST_CASE(MultiObjective) {
  Model m;
  auto a = m.addVariable(1, 10);
  auto b = m.addVariable(1, 10);
  m.lessOrEqual(5, m.sum({a, b}));
  auto s1 = m.minimize({a, b});
  BOOST_CHECK_EQUAL(s1[a], 1);
  BOOST_CHECK_EQUAL(s1[b], 4);
  auto s2 = m.minimize({b, a});
  BOOST_CHECK_EQUAL(s2[a], 4);
  BOOST_CHECK_EQUAL(s2[b], 1);
}
