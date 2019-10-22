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
  domains.push_back({15, 40}); // a
  domains.push_back({20, 30}); // b
  domains.push_back({25, 35}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(max.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 25);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 35);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 20);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 30);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 25);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 35);
}

BOOST_AUTO_TEST_CASE(PropagateConstrainValues) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({15, 20}); // a
  domains.push_back({0, 30});  // b
  domains.push_back({10, 35}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(max.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 15);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 20);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 0);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 20);

  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 10);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 20);
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultUpperBound) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({15, 20}); // a
  domains.push_back({25, 30}); // b
  domains.push_back({30, 35}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(!max.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(PropagateFailsResultLowerBound) {
  Max max(a, {b, c});

  Domains domains;
  domains.push_back({30, 40}); // a
  domains.push_back({15, 20}); // b
  domains.push_back({10, 25}); // c

  Scheduler scheduler(domains, {&max});
  BOOST_CHECK(!max.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(Minimize) {
  Model m;

  const auto b = m.addConstant(25);
  const auto c = m.addVariable(5, 20);

  const auto a = m.max({b, c});
  auto s = m.minimize(a);
  BOOST_CHECK_EQUAL(s[a], 25);
}
