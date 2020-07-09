// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "Constraint.hpp"
#include "Scheduler.hpp"

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE Max
#include <boost/test/unit_test.hpp>

using namespace popsolver;

const Variable a(0);
const Variable b(1);
const Variable c(2);

BOOST_AUTO_TEST_CASE(PropagateConstrainResult) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({DataType{15}, DataType{40}}); // a
  domains.push_back({DataType{20}, DataType{30}}); // b
  domains.push_back({DataType{25}, DataType{35}}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(max.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{25});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{35});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{20});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{30});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{25});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{35});
}

BOOST_AUTO_TEST_CASE(PropagateConstrainValues) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({DataType{15}, DataType{20}}); // a
  domains.push_back({DataType{0}, DataType{30}});  // b
  domains.push_back({DataType{10}, DataType{35}}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(max.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{15});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{20});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{0});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{20});

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{10});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{20});
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultUpperBound) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({DataType{15}, DataType{20}}); // a
  domains.push_back({DataType{25}, DataType{30}}); // b
  domains.push_back({DataType{30}, DataType{35}}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(!max.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultLowerBound) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({DataType{30}, DataType{40}}); // a
  domains.push_back({DataType{15}, DataType{20}}); // b
  domains.push_back({DataType{10}, DataType{25}}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(!max.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(Minimize) {
  Model m;

  const auto b = m.addConstant(25);
  const auto c = m.addVariable(5, 20);

  const auto a = m.max({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], DataType{25});
}
