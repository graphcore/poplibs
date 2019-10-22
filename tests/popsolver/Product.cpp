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
  domains.push_back({7, 8});   // a
  domains.push_back({2, 5});   // b
  domains.push_back({14, 40}); // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 7);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 14);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 40);
}

BOOST_AUTO_TEST_CASE(PropagateResult) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({3, 4});   // a
  domains.push_back({2, 5});   // b
  domains.push_back({0, 100}); // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 3);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 4);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 5);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 6);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 20);
}

BOOST_AUTO_TEST_CASE(PropagateOperands) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({2, 10});   // a
  domains.push_back({4, 1000}); // b
  domains.push_back({10, 12});  // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 3);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 4);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 6);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 10);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 12);
}

BOOST_AUTO_TEST_CASE(PropagateBoth) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({2, 10});   // a
  domains.push_back({4, 1000}); // b
  domains.push_back({0, 12});   // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 3);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 4);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 6);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 8);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 12);
}

BOOST_AUTO_TEST_CASE(PropagateBoth2) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({1, 40}); // a
  domains.push_back({2, 3});  // b
  domains.push_back({4, 5});  // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 2);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].min(), 4);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[c].max(), 4);
}

BOOST_AUTO_TEST_CASE(AvoidOverflow) {
  Variable a(0), b(1), c(2);
  auto product = std::unique_ptr<Product>(new Product(c, a, b));
  Domains domains;
  domains.push_back({0, std::numeric_limits<unsigned>::max() - 1}); // a
  domains.push_back({0, std::numeric_limits<unsigned>::max() - 1}); // b
  domains.push_back({27, 64});                                      // c
  Scheduler scheduler(domains, {product.get()});
  bool success = product->propagate(scheduler);
  BOOST_CHECK(success);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 64);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[b].max(), 64);
}
