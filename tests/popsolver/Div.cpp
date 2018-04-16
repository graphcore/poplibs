// Simple tests for popsolver.
//
#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Div
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
