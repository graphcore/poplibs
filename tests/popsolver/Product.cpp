// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "Constraint.hpp"
#include "Scheduler.hpp"
#include <memory>
#define BOOST_TEST_MODULE Product
#include <boost/test/unit_test.hpp>

using namespace popsolver;

BOOST_AUTO_TEST_CASE(PropagateNoChange) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{7}, DataType{8}});   // a
  domains.push_back({DataType{2}, DataType{5}});   // b
  domains.push_back({DataType{14}, DataType{40}}); // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{7});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{14});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{40});
}

BOOST_AUTO_TEST_CASE(PropagateResult) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{3}, DataType{4}});   // a
  domains.push_back({DataType{2}, DataType{5}});   // b
  domains.push_back({DataType{0}, DataType{100}}); // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{3});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{4});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{5});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{6});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{20});
}

BOOST_AUTO_TEST_CASE(PropagateOperands) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{2}, DataType{10}});   // a
  domains.push_back({DataType{4}, DataType{1000}}); // b
  domains.push_back({DataType{10}, DataType{12}});  // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{3});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{4});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{6});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{10});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{12});
}

BOOST_AUTO_TEST_CASE(PropagateBoth) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{2}, DataType{10}});   // a
  domains.push_back({DataType{4}, DataType{1000}}); // b
  domains.push_back({DataType{0}, DataType{12}});   // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{3});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{4});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{6});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{8});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{12});
}

BOOST_AUTO_TEST_CASE(PropagateBoth2) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{1}, DataType{40}}); // a
  domains.push_back({DataType{2}, DataType{3}});  // b
  domains.push_back({DataType{4}, DataType{5}});  // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{2});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), DataType{4});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), DataType{4});
}

BOOST_AUTO_TEST_CASE(AvoidOverflow) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({DataType{0}, DataType::max() - DataType{1}}); // a
  domains.push_back({DataType{0}, DataType::max() - DataType{1}}); // b
  domains.push_back({DataType{27}, DataType{64}});                 // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{64});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), DataType{64});
}
