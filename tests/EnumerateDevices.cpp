// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "TestDevice.hpp"

#define BOOST_TEST_MODULE EnumerateDevices
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Enumerate) {
  // This test relies on createTestDevice() throwing in case of failure to
  // enumerate at least one device. Could be more sophisticed in future:
  auto device = createTestDeviceFullSize(DeviceType::Hw);
  const auto argc = boost::unit_test::framework::master_test_suite().argc;
  const auto argv = boost::unit_test::framework::master_test_suite().argv;
  if (argc > 1) {
    if (argc != 2) {
      throw std::logic_error(
          "Too many arguments: Takes one optional arguments: <arch>");
    }
    const auto archString = device.getTarget().getTargetArchString();
    const auto requiredArchString = argv[1];
    BOOST_CHECK(archString == requiredArchString);
  }
}
