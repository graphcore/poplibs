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
  GenericAssignment<DataType> assign(
      a, {},
      [](const std::vector<DataType> &values) -> boost::optional<DataType> {
        return DataType{1};
      });

  Domains domains;
  domains.emplace_back(DataType{0}, DataType{10}); // a

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(assign.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{1});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{1});
}

BOOST_AUTO_TEST_CASE(GenericAssignmentTakingValues) {
  GenericAssignment<DataType> assign(
      a, {b}, [](const std::vector<DataType> &values) -> DataType {
        return values[0] * DataType{2};
      });

  Domains domains;
  domains.emplace_back(DataType{0}, DataType{10}); // a
  domains.emplace_back(DataType{2}, DataType{2});  // b

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(assign.propagate(scheduler));

  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].min(), DataType{4});
  BOOST_CHECK_EQUAL(scheduler.getDomains()[a].max(), DataType{4});
}

BOOST_AUTO_TEST_CASE(GenericAssignmentOutOfRange) {
  GenericAssignment<DataType> assign(
      a, {}, [](const std::vector<DataType> &values) -> DataType {
        return DataType{1};
      });

  Domains domains;
  domains.emplace_back(DataType{5}, DataType{5});

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(!assign.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(GenericAssignmentReturnsInvalid) {
  GenericAssignment<DataType> assign(
      a, {},
      [](const std::vector<DataType> &values) -> boost::optional<DataType> {
        return boost::none;
      });

  Domains domains;
  domains.emplace_back(DataType{0}, DataType{10});

  Scheduler scheduler(domains, {&assign});

  BOOST_CHECK(!assign.propagate(scheduler));
}

BOOST_AUTO_TEST_CASE(GenericAssignmentConstrainsDomainToRange) {
  // Given
  const auto min = DataType{std::numeric_limits<unsigned>::max()};
  const auto max = DataType{std::numeric_limits<unsigned>::max()} + DataType{5};

  unsigned callCount = 0;
  GenericAssignment<unsigned> assign(
      a, {b}, [&](const auto &values) -> boost::optional<DataType> {
        // Then expect the bounds constrained
        BOOST_CHECK_EQUAL(values[0], std::numeric_limits<unsigned>::max());
        callCount++;
        return DataType{values[0]};
      });
  Domains domains;
  domains.emplace_back(min, max); // a
  domains.emplace_back(min, max); // b
  Scheduler scheduler(domains, {&assign});

  // When
  BOOST_CHECK(assign.propagate(scheduler));
  // Then only called once with only value in range unsigned
  BOOST_CHECK_EQUAL(callCount, 1);
}
