#ifndef TestDevice_hpp__
#define TestDevice_hpp__

#include <poplar/Engine.hpp>
#include <poplar/Target.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

#include <thread>
#include <chrono>
#include <iostream>

namespace {
// In CMakeLists.txt there is a regex on "Hw*" so be
// careful when adding new enums that begin with Hw:
enum class DeviceType {Cpu, Sim, Hw, IpuModel};

// In order to run multiple hardware tests concurrently loop indefinitely
// trying to acquire devices (relying on test timeout to terminate the
// test if none are available). To work sensibly this requires that at
// most N tests are run in parallel on any Hw build-bot
// e.g. via 'ctest -jN'.
inline
std::pair<bool, poplar::Device>
acquireHardwareDevice(std::vector<poplar::Device> &devices)
{
  bool success = false;
  poplar::Device device;

  unsigned waitIterCount = 0;
  while (!success) {
    for (auto &candidateDevice : devices) {
      success = candidateDevice.attach();
      if (success) {
        device = std::move(candidateDevice);
        break;
      }
    }
    if (!success) {
      std::cout << "Could not acquire hardware device, retrying ... "
                   "(retry count: " << waitIterCount << ")\n";
      waitIterCount += 1;
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }

  if (waitIterCount && success) {
    std::cout << "Successfully acquired device after retry.";
  }

  return std::make_pair(success, std::move(device));
}

// Create an device for testing:
inline poplar::Device createTestDevice(DeviceType deviceType,
                                       unsigned numIPUs = 1,
                                       unsigned tilesPerIPU = 1,
                                       const poplar::OptionFlags &opt = {}) {
  poplar::Target target;
  switch (deviceType) {
  case DeviceType::Cpu:
    target = poplar::Target::createCPUTarget();
    return poplar::Device::createCPUDevice();
  case DeviceType::Sim:
    target = poplar::Target::createIPUTarget(numIPUs, tilesPerIPU,
                                             "_TEST_SYSTEM");
    return poplar::Device::createSimulatorDevice(target);
  case DeviceType::Hw:
    {
      static auto manager = poplar::DeviceManager::getDeviceManager();
      auto devices = manager.getDevices(poplar::TargetType::IPU, numIPUs);
      if (devices.empty()) {
        // If the device list is empty we have to terminate here otherwise
        // acquireHardwareDevice() will retry forever.
        throw poplar::poplar_error("No devices exist with the requested "
                                   "configuration");
      }
      bool success = false;
      poplar::Device device;
      std::tie(success, device) = acquireHardwareDevice(devices);

      if (!success) {
        throw poplar::poplar_error("Could not acquire any Hw devices");
      }
      if (tilesPerIPU != device.getTarget().getTilesPerIPU()) {
        device = device.createVirtualDevice(tilesPerIPU);
      }

      return device;
    }
  case DeviceType::IpuModel:
    {
      poplar::IPUModel model;
      model.numIPUs = numIPUs;
      model.tilesPerIPU = tilesPerIPU;
      return model.createDevice();
    }
  default:
      throw std::logic_error(
        "deviceType must be \"Cpu\", \"IpuModel\", \"Sim\" or \"Hw\"\n");
  }
}


inline const char *asString(const DeviceType &deviceType) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return "Cpu";
  case DeviceType::IpuModel:
    return "IpuModel";
  case DeviceType::Sim:
    return "Sim";
  case DeviceType::Hw:
    return "Hw";
  default:
    break;
  }
  throw std::logic_error("Invalid device type");
}

inline std::istream &operator>>(std::istream &is, DeviceType &type) {
  std::string token;
  is >> token;
  if (token == "Cpu")
    type = DeviceType::Cpu;
  else if (token == "IpuModel")
    type = DeviceType::IpuModel;
  else if (token == "Sim")
    type = DeviceType::Sim;
  else if (token == "Hw")
    type = DeviceType::Hw;
  else
    throw std::logic_error(
      "Unsupported device type <" + token + ">" +
      "; must be one of \"Cpu\", \"IpuModel\", \"Sim\" or \"Hw\"");
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const DeviceType &type) {
  os << asString(type);
  return os;
}

}
#endif // __TestDevice_hpp
