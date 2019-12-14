// Copyright (c) Graphcore Ltd, All rights reserved.
#include "TestDevice.hpp"

#define BOOST_TEST_MODULE EnumerateDevices
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Enumerate) {
  // This test relies on createTestDevice() throwing in case of failure to
  // enumerate at least one device. Could be more sophisticed in future:
  auto device = createTestDevice(DeviceType::Hw);
}
