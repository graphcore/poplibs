// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE BooleanOps
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(OrCase1) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(1);
  auto c = m.booleanOr(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(OrCase2) {
  Model m;
  auto a = m.addConstant(0);
  auto b = m.addConstant(0);
  auto c = m.booleanOr(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(OrCase3) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(0);
  auto c = m.booleanOr(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(OrCase4) {
  Model m;
  auto a = m.addConstant(0);
  auto b = m.addConstant(1);
  auto c = m.booleanOr(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(OrCase5) {
  Model m;
  auto a = m.addConstant(DataType::max());
  auto b = m.addConstant(DataType::max());
  auto c = m.booleanOr(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(AndCase1) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(1);
  auto c = m.booleanAnd(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(AndCase2) {
  Model m;
  auto a = m.addConstant(0);
  auto b = m.addConstant(0);
  auto c = m.booleanAnd(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(AndCase3) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.addConstant(0);
  auto c = m.booleanAnd(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(AndCase4) {
  Model m;
  auto a = m.addConstant(0);
  auto b = m.addConstant(1);
  auto c = m.booleanAnd(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{0});
}

BOOST_AUTO_TEST_CASE(AndCase5) {
  Model m;
  auto a = m.addConstant(DataType::max());
  auto b = m.addConstant(DataType::max());
  auto c = m.booleanAnd(a, b);
  auto s = m.minimize(c);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[c], DataType{1});
}

BOOST_AUTO_TEST_CASE(NotCase1) {
  Model m;
  auto a = m.addConstant(0);
  auto b = m.booleanNot(a);
  auto s = m.minimize(b);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[b], DataType{1});
}

BOOST_AUTO_TEST_CASE(NotCase2) {
  Model m;
  auto a = m.addConstant(1);
  auto b = m.booleanNot(a);
  auto s = m.minimize(b);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[b], DataType{0});
}

BOOST_AUTO_TEST_CASE(NotCase3) {
  Model m;
  auto a = m.addConstant(20000);
  auto b = m.booleanNot(a);
  auto s = m.minimize(b);
  BOOST_CHECK(s.validSolution());
  BOOST_CHECK_EQUAL(s[b], DataType{0});
}
