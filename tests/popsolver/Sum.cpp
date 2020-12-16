// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
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
  domains.push_back({DataType{7}, DataType{8}});  // a
  domains.push_back({DataType{2}, DataType{5}});  // b
  domains.push_back({DataType{9}, DataType{13}}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{9});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{13});
}

BOOST_AUTO_TEST_CASE(PropagateResult) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({DataType{7}, DataType{8}});  // a
  domains.push_back({DataType{2}, DataType{5}});  // b
  domains.push_back({DataType{1}, DataType{20}}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{9});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{13});
}

BOOST_AUTO_TEST_CASE(PropagateOperands) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({DataType{7}, DataType{1000}}); // a
  domains.push_back({DataType{0}, DataType{9}});    // b
  domains.push_back({DataType{9}, DataType{13}});   // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{13});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{0});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{6});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{9});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{13});
}

BOOST_AUTO_TEST_CASE(PropagateOperands2) {
  // This is essentially the case we have when we add this to the model:
  // m.sub(m.addConstant(1u), m.addConstant(0u));
  Variable zero(0), one(1), unknownOperand(2);
  auto sum = std::unique_ptr<Sum>(new Sum(one, {zero, unknownOperand}));
  Domains domains;
  domains.push_back({DataType{0}, DataType{0}});         // zero
  domains.push_back({DataType{1}, DataType{1}});         // one
  domains.push_back({DataType::min(), DataType::max()}); // unknown operand
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[zero].min(), DataType{0});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[zero].max(), DataType{0});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[one].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[one].max(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[unknownOperand].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[unknownOperand].max(), DataType{1});
}

BOOST_AUTO_TEST_CASE(PropagateBoth) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({DataType{7}, DataType{8}});  // a
  domains.push_back({DataType{0}, DataType{5}});  // b
  domains.push_back({DataType{9}, DataType{20}}); // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{9});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{13});
}

BOOST_AUTO_TEST_CASE(AvoidOverflow) {
  Variable a(0), b(1), c(2);
  auto sum = std::unique_ptr<Sum>(new Sum(c, {a, b}));
  Domains domains;
  domains.push_back({DataType{1}, DataType::max() - DataType{1}}); // a
  domains.push_back({DataType{2}, DataType::max() - DataType{1}}); // b
  domains.push_back({DataType{20}, DataType{21}});                 // c
  Scheduler scheduler(domains, {sum.get()});
  bool success = sum->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{19});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{20});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{20});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{21});
}
