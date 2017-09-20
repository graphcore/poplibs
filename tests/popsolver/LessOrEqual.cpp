#include "Constraint.hpp"
#include "Scheduler.hpp"
#include <memory>
#define BOOST_TEST_MODULE LessOrEqual
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(PropagateNoChange) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({1, 5}); // a
  domains.push_back({7, 8}); // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 8);
}

BOOST_AUTO_TEST_CASE(PropagateChangeA) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({5, 10}); // a
  domains.push_back({7, 8});  // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 8);
}

BOOST_AUTO_TEST_CASE(PropagateChangeB) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({5, 10}); // a
  domains.push_back({1, 11});  // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 10);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 11);
}

BOOST_AUTO_TEST_CASE(PropagateChangeBoth) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({5, 10}); // a
  domains.push_back({1, 8});  // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 8);
}

BOOST_AUTO_TEST_CASE(PropagateFail) {
  Variable a(0), b(1);
  auto c = new LessOrEqual(a, b);
  Domains domains;
  domains.push_back({5, 6});
  domains.push_back({1, 4});
  Scheduler scheduler(domains, {c});
  bool success = c->propagate(scheduler);
  BOOST_CHECK(!success);
}
