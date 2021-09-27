// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE OptionParserTest
#include <boost/test/unit_test.hpp>
#include <poputil/OptionParsing.hpp>

using namespace poplibs;

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsLowerBoundIsInclusiveEqTest) {
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(0.0, 0.0, true, 10.0, false));
}

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsLowerBoundIsNotInclusiveEqTest) {
  BOOST_CHECK_THROW(
      OptionHandler::ensureValueIsWithinBounds(0.0, 0.0, false, 10.0, false),
      poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsHigherBoundIsInclusiveEqTest) {
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(10.0, 0.0, false, 10.0, true));
}

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsHigherBoundIsNotInclusiveEqTest) {
  BOOST_CHECK_THROW(
      OptionHandler::ensureValueIsWithinBounds(10.0, 0.0, false, 10.0, false),
      poplar::invalid_option);
}

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsMiddleGroundTest) {
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5.0, 0.0, false, 10.0, true));
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5.0, 0.0, true, 10.0, false));
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5.0, 0.0, false, 10.0, false));
}

BOOST_AUTO_TEST_CASE(ensureValueIsWithinBoundsIntegerTest) {
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5, 0, false, 10, true));
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5, 0, true, 10, false));
  BOOST_CHECK_NO_THROW(
      OptionHandler::ensureValueIsWithinBounds(5, 0, false, 10, false));
}
