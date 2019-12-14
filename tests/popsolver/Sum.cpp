// Copyright (c) Graphcore Ltd, All rights reserved.
#include "Constraint.hpp"
#include "Scheduler.hpp"
#include <memory>
#define BOOST_TEST_MODULE Sum
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(PropagateNoChange) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({7, 8});  // a
  domains.push_back({2, 5});  // b
  domains.push_back({9, 13}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 9);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 13);
}

BOOST_AUTO_TEST_CASE(PropagateResult) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({7, 8});  // a
  domains.push_back({2, 5});  // b
  domains.push_back({1, 20}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 9);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 13);
}

BOOST_AUTO_TEST_CASE(PropagateOperands) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({7, 1000}); // a
  domains.push_back({0, 9});    // b
  domains.push_back({9, 13});   // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 13);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 0);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 6);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 9);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 13);
}

BOOST_AUTO_TEST_CASE(PropagateBoth) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({7, 8});  // a
  domains.push_back({0, 5});  // b
  domains.push_back({9, 20}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 9);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 13);
}

BOOST_AUTO_TEST_CASE(AvoidOverflow) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({1, std::numeric_limits<unsigned>::max() - 1}); // a
  domains.push_back({2, std::numeric_limits<unsigned>::max() - 1}); // b
  domains.push_back({20, 21});                                      // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 19);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 20);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 20);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 21);
}
