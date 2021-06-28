// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE ReifiedEqual
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(ReifiedEqualCase1) {
  Model m;

  auto a = m.addConstant(0);
  auto b = m.addConstant(0);
  auto c = m.reifiedEqual(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(ReifiedEqualCase2) {
  Model m;

  auto a = m.addConstant(0);
  auto b = m.addConstant(1);
  auto c = m.reifiedEqual(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(ReifiedEqualCase3) {
  Model m;

  auto a = m.addConstant(50);
  auto b = m.addConstant(150);
  auto c = m.reifiedEqual(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(ReifiedEqualCase4) {
  Model m;

  auto a = m.addConstant(DataType::max());
  auto b = m.addConstant(DataType::max());
  auto c = m.reifiedEqual(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(ReifiedEqualCase5) {
  Model m;

  auto a = m.addConstant(DataType::min());
  auto b = m.addConstant(DataType::max());
  auto c = m.reifiedEqual(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}
