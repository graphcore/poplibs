// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE ReifiedLess
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(ReifiedLessCase1) {
  Model m;

  auto a = m.addConstant(0);
  auto b = m.addConstant(1);
  auto c = m.reifiedLess(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(ReifiedLessCase2) {
  Model m;

  auto a = m.addConstant(0);
  auto b = m.addConstant(0);
  auto c = m.reifiedLess(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(ReifiedLessCase3) {
  Model m;

  auto a = m.addConstant(20);
  auto b = m.addConstant(19);
  auto c = m.reifiedLess(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(ReifiedLessCase4) {
  Model m;

  auto a = m.addConstant(DataType::max());
  auto b = m.addConstant(DataType::max());
  auto c = m.reifiedLess(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(ReifiedLessCase5) {
  Model m;

  auto a = m.addConstant(DataType::min());
  auto b = m.addConstant(DataType::max());
  auto c = m.reifiedLess(a, b);
  auto s = m.minimize(c);

  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}
