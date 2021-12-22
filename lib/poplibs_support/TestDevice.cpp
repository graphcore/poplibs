// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <poplibs_support/TestDevice.hpp>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>
#include <poplar/TargetType.hpp>

#include <chrono>
#include <iostream>
#include <thread>

poplibs_support::DeviceType TEST_TARGET;

namespace poplibs_support {

const char *deviceTypeToIPUName(DeviceType d) {
  switch (d) {
  case DeviceType::Sim2:
  case DeviceType::IpuModel2:
    return "ipu2";

  case DeviceType::Sim21:
  case DeviceType::IpuModel21:
    return "ipu21";

  case DeviceType::Cpu:
    throw poplar::poplar_error(
        "deviceTypeToIPUName(DeviceType::Cpu) not supported");
  case DeviceType::Hw:
    throw poplar::poplar_error(
        "deviceTypeToIPUName(DeviceType::Hw) not supported");

  default:
    throw poplar::poplar_error("Unknown device type");
  }
}

TestDevice::TestDevice(std::vector<poplar::Device> ds) : device(std::move(ds)) {
  using poplar::Device;

  const auto equalTarget = [](const Device &lhs, const Device &rhs) -> bool {
    return lhs.getTarget() == rhs.getTarget();
  };

  // all devices must have the same target for test determinism.
  const auto &devices = boost::get<std::vector<Device>>(device);
  const auto allEqual = std::equal(std::begin(devices) + 1, std::end(devices),
                                   std::begin(devices), equalTarget);
  if (!allEqual) {
    throw poplar::poplar_error("Acquired devices with different targets");
  }
}

const poplar::Target &TestDevice::getTarget() const {
  struct Visitor : public boost::static_visitor<const poplar::Target &> {
    result_type operator()(const poplar::Device &d) const {
      return d.getTarget();
    }

    result_type operator()(const std::vector<poplar::Device> &ds) const {
      assert(!ds.empty());
      return ds.front().getTarget();
    }
  };

  return boost::apply_visitor(Visitor(), device);
}

// Return a device of the requested type. If requestedTilesPerIPU is boost::none
// the default number of tiles (all) are used.
// Set requestedTilesPerIPU to DeviceTypeDefaultTiles to use all tile on the
// device.
TestDevice
createTestDevice(const DeviceType deviceType, const unsigned numIPUs,
                 const boost::optional<unsigned> requestedTilesPerIPU,
                 const bool compileIPUCode) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return poplar::Device::createCPUDevice();
  case DeviceType::Sim2:
  case DeviceType::Sim21: {
    auto targetName = deviceTypeToIPUName(deviceType);
    auto target = requestedTilesPerIPU.has_value()
                      ? poplar::Target::createIPUTarget(
                            numIPUs, *requestedTilesPerIPU, targetName)
                      : poplar::Target::createIPUTarget(numIPUs, targetName);

    return poplar::Device::createSimulatorDevice(std::move(target));
  }
  case DeviceType::Hw: {
    static auto manager = poplar::DeviceManager::createDeviceManager();
    auto devices = manager.getDevices(poplar::TargetType::IPU, numIPUs);
    if (devices.empty()) {
      throw poplar::poplar_error("No devices exist with the requested "
                                 "configuration.");
    }

    // all devices will be for the same target
    auto tilesPerIPU = requestedTilesPerIPU.get_value_or(
        devices.front().getTarget().getTilesPerIPU());
    // transform each device into a virtual device if needed.
    for (auto &device : devices) {
      if (tilesPerIPU != device.getTarget().getTilesPerIPU()) {
        device = device.createVirtualDevice(tilesPerIPU);
      }
    }

    return std::move(devices);
  }
  case DeviceType::IpuModel2:
  case DeviceType::IpuModel21: {
    auto targetName = deviceTypeToIPUName(deviceType);
    poplar::IPUModel model(targetName);
    model.numIPUs = numIPUs;
    if (requestedTilesPerIPU.has_value())
      model.tilesPerIPU = *requestedTilesPerIPU;
    model.compileIPUCode = compileIPUCode;
    return model.createDevice();
  }
  default:
    throw std::logic_error(
        R"XX(deviceType must be "Cpu", "IpuModel2", "Sim2" or "Hw")XX");
  }
}

std::istream &operator>>(std::istream &is, DeviceType &type) {
  std::string token;
  is >> token;
  if (token == "Cpu")
    type = DeviceType::Cpu;
  else if (token == "IpuModel2")
    type = DeviceType::IpuModel2;
  else if (token == "IpuModel21")
    type = DeviceType::IpuModel21;
  else if (token == "Sim2")
    type = DeviceType::Sim2;
  else if (token == "Sim21")
    type = DeviceType::Sim21;
  else if (token == "Hw")
    type = DeviceType::Hw;
  else
    throw std::logic_error(
        "Unsupported device type <" + token +
        ">; must be one of "
        R"XX("Cpu", "IpuModel2", "Sim2" or "Hw")XX");
  return is;
}

std::ostream &operator<<(std::ostream &os, const DeviceType &type) {
  os << asString(type);
  return os;
}

const char *asString(const DeviceType &deviceType) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return "Cpu";
  case DeviceType::IpuModel2:
    return "IpuModel2";
  case DeviceType::IpuModel21:
    return "IpuModel21";
  case DeviceType::Sim2:
    return "Sim2";
  case DeviceType::Sim21:
    return "Sim21";
  case DeviceType::Hw:
    return "Hw";
  default:
    break;
  }
  throw std::logic_error("Invalid device type");
}

DeviceType getDeviceType(const std::string &deviceString) {
  if (deviceString == "Cpu")
    return DeviceType::Cpu;
  if (deviceString == "IpuModel2")
    return DeviceType::IpuModel2;
  if (deviceString == "IpuModel21")
    return DeviceType::IpuModel21;
  if (deviceString == "Sim2")
    return DeviceType::Sim2;
  if (deviceString == "Sim21")
    return DeviceType::Sim21;
  if (deviceString == "Hw")
    return DeviceType::Hw;
  throw std::logic_error("Invalid device string");
}

} // namespace poplibs_support
