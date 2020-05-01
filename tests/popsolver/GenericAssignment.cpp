// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "Constraint.hpp"
#include "Scheduler.hpp"

#include <popsolver/Model.hpp>
#define BOOST_TEST_MODULE GenericAssignment
#include <boost/test/unit_test.hpp>

using namespace popsolver;

const Variable a(0);
const Variable b(1);

BOOST_AUTO_TEST_CASE(GenericAssignmentSimple) {
  GenericAssignment assign(
      a, {},
      [](const std::vector<unsigned> &values) -> boost::optional<unsigned> {
        return 1;
      });

  Domains domains;
  domains.emplace_back(0, 10); // a

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(assign.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 1);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 1);
}

BOOST_AUTO_TEST_CASE(GenericAssignmentTakingValues) {
  GenericAssignment assign(
      a, {b},
      [](const std::vector<unsigned> &values) -> boost::optional<unsigned> {
        return values[0] * 2u;
      });

  Domains domains;
  domains.emplace_back(0, 10); // a
  domains.emplace_back(2, 2);  // b

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(assign.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), 4);
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), 4);
}

BOOST_AUTO_TEST_CASE(GenericAssignmentOutOfRange) {
  GenericAssignment assign(
      a, {},
      [](const std::vector<unsigned> &values) -> boost::optional<unsigned> {
        return 1;
      });

  Domains domains;
  domains.emplace_back(5, 5);

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(!assign.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(GenericAssignmentReturnsInvalid) {
  GenericAssignment assign(
      a, {},
      [](const std::vector<unsigned> &values) -> boost::optional<unsigned> {
        return boost::none;
      });

  Domains domains;
  domains.emplace_back(0, 10);

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(!assign.propagate(scheduler));
}
