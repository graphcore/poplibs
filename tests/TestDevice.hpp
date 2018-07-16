#ifndef TestDevice_hpp__
#define TestDevice_hpp__

#include <poplar/Engine.hpp>
#include <poplar/Target.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

namespace {
// In CMakeLists.txt there is a regex on "Hw*" so be
// careful when adding new enums that begin with Hw:
enum class DeviceType {Cpu, Sim, Hw, IpuModel};

// Create an engine for testing
inline poplar::Device createTestDevice(DeviceType deviceType,
                                       unsigned numIPUs = 1,
                                       unsigned tilesPerIPU = 1,
                                       const poplar::OptionFlags &opt = {}) {
  poplar::Target target;
  switch (deviceType) {
  case DeviceType::Cpu:
    target = poplar::Target::createCPUTarget();
    return poplar::Device::createCPUDevice();
    break;
  case DeviceType::IpuModel:
  {
    poplar::IPUModel ipuModel;
    ipuModel.numIPUs = numIPUs;
    ipuModel.tilesPerIPU = tilesPerIPU;
    return ipuModel.createDevice();
    break;
  }
  case DeviceType::Sim:
  {
    target = poplar::Target::createIPUTarget(numIPUs,
                                             tilesPerIPU, "_TEST_SYSTEM");
    return poplar::Device::createSimulatorDevice(target, opt);
  }
  case DeviceType::Hw:
  {
    static auto set = poplar::DeviceManager::getDeviceManager();
    try {
      auto device = set.getDevices(poplar::TargetType::IPU, 1).at(0);
      auto success = device.attach();
      if (!success) {
        throw std::logic_error("Could not acquire device " +
                                std::to_string(device.getId()));
      }
      if (tilesPerIPU != device.getTarget().getTilesPerIPU()) {
        device = device.createVirtualDevice(tilesPerIPU);
      }
      return device;
    } catch (const std::exception &e) {
      throw std::logic_error("Could not acquire physical IPU device. "
                             "Error: '" +
                              std::string(e.what()) + "'");
  }

    break;
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
