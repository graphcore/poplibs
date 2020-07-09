// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "Constraint.hpp"
#include "Scheduler.hpp"

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Min
#include <boost/test/unit_test.hpp>

using namespace popsolver;

const Variable a(0);
const Variable b(1);
const Variable c(2);

BOOST_AUTO_TEST_CASE(PropagateConstrainResult) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({DataType{15}, DataType{40}}); // a
  domains.push_back({DataType{20}, DataType{30}}); // b
  domains.push_back({DataType{25}, DataType{35}}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(min.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{20});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{30});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{20});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{30});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{25});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{35});
}

BOOST_AUTO_TEST_CASE(PropagateConstrainValues) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({DataType{15}, DataType{20}}); // a
  domains.push_back({DataType{0}, DataType{30}});  // b
  domains.push_back({DataType{10}, DataType{35}}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(min.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{15});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{20});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{15});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{30});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{15});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{35});
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultUpperBound) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({DataType{35}, DataType{40}}); // a
  domains.push_back({DataType{25}, DataType{30}}); // b
  domains.push_back({DataType{20}, DataType{35}}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(!min.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultLowerBound) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({DataType{0}, DataType{10}});  // a
  domains.push_back({DataType{15}, DataType{20}}); // b
  domains.push_back({DataType{25}, DataType{35}}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(!min.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(MinimizeBelowVariable) {
  Model m;

  const auto b = m.addConstant(5);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], DataType{5});
}

BOOST_AUTO_TEST_CASE(MinimizeInsideVariable) {
  Model m;

  const auto b = m.addConstant(20);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], DataType{15});
}

BOOST_AUTO_TEST_CASE(MinimizeAboveVariable) {
  Model m;

  const auto b = m.addConstant(35);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], DataType{15});
}
