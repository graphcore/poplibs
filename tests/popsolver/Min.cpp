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
  domains.push_back({15, 40}); // a
  domains.push_back({20, 30}); // b
  domains.push_back({25, 35}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(min.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 20);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 30);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 20);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 30);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 25);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 35);
}

BOOST_AUTO_TEST_CASE(PropagateConstrainValues) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({15, 20}); // a
  domains.push_back({0, 30});  // b
  domains.push_back({10, 35}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(min.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 15);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 20);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 15);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 30);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 15);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 35);
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultUpperBound) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({35, 40}); // a
  domains.push_back({25, 30}); // b
  domains.push_back({20, 35}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(!min.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultLowerBound) {
  Min min(a, {b, c});

  Domains domains;
  domains.push_back({0, 10});  // a
  domains.push_back({15, 20}); // b
  domains.push_back({25, 35}); // c

  Scheduler scheduler(domains, {&min});
  BOOST_CHECK(!min.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(MinimizeBelowVariable) {
  Model m;

  const auto b = m.addConstant(5);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 5);
}

BOOST_AUTO_TEST_CASE(MinimizeInsideVariable) {
  Model m;

  const auto b = m.addConstant(20);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 15);
}

BOOST_AUTO_TEST_CASE(MinimizeAboveVariable) {
  Model m;

  const auto b = m.addConstant(35);
  const auto c = m.addVariable(15, 30);

  const auto a = m.min({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 15);
}
