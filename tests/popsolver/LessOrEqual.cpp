// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
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
  domains.push_back({DataType{1}, DataType{5}}); // a
  domains.push_back({DataType{7}, DataType{8}}); // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{8});
}

BOOST_AUTO_TEST_CASE(PropagateChangeA) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({DataType{5}, DataType{10}}); // a
  domains.push_back({DataType{7}, DataType{8}});  // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{8});
}

BOOST_AUTO_TEST_CASE(PropagateChangeB) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({DataType{5}, DataType{10}}); // a
  domains.push_back({DataType{1}, DataType{11}}); // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{10});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{11});
}

BOOST_AUTO_TEST_CASE(PropagateChangeBoth) {
  Variable a(0), b(1);
  auto lessOrEqual = std::unique_ptr<LessOrEqual>(new LessOrEqual(a, b));
  Domains domains;
  domains.push_back({DataType{5}, DataType{10}}); // a
  domains.push_back({DataType{1}, DataType{8}});  // b
  Scheduler scheduler(domains, {lessOrEqual.get()});
  bool success = lessOrEqual->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{8});
}

BOOST_AUTO_TEST_CASE(PropagateFail) {
  Variable a(0), b(1);
  LessOrEqual c(a, b);
  Domains domains;
  domains.push_back({DataType{5}, DataType{6}});
  domains.push_back({DataType{1}, DataType{4}});
  Scheduler scheduler(domains, {&c});
  bool success = c.propagate(scheduler);
  BOOST_CHECK(!success);
}
