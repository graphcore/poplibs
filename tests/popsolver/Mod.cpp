// Copyright (c) 2020 Graphcore Ltd, All rights reserved.
//
#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Mod
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(Mod) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(1);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 0);
}

BOOST_AUTO_TEST_CASE(Mod0) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(0);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  // Can't divide by zero
  BOOST_CHECK(!s.validSolution());
}

BOOST_AUTO_TEST_CASE(ModAEqualB) {
  Model m;
  auto a = m.addConstant(4);
  auto b = m.addConstant(4);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 0);
}

BOOST_AUTO_TEST_CASE(ModBEqual1) {
  Model m;
  auto a = m.addConstant(4);
  auto b = m.addConstant(1);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 0);
}

BOOST_AUTO_TEST_CASE(ModAGreaterThanB) {
  Model m;
  auto a = m.addConstant(8);
  auto b = m.addConstant(4);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 0);
}

BOOST_AUTO_TEST_CASE(ModAGreaterThanBNonZero) {
  Model m;
  auto a = m.addConstant(7);
  auto b = m.addConstant(4);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 3);
}

BOOST_AUTO_TEST_CASE(ModALessThanB) {
  Model m;
  auto a = m.addConstant(4);
  auto b = m.addConstant(8);
  auto c = m.mod(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK_EQUAL(s[c], 4);
}
