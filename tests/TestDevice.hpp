#ifndef __TestDevice_hpp
#define __TestDevice_hpp

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
namespace {
enum class DeviceType {
  Cpu,         //CPU
  IpuModel,    //IPU modelled on CPU, number of tiles specified by parameter
  Sim };       //IPU simulated by CISS, number of tiles specified by parameter

// Create an engine for testing
inline poplar::Device createTestDevice(DeviceType deviceType,
                                       unsigned numIPUs = 1,
                                       unsigned tilesPerIPU = 1)
{
  poplar::Target target;
  poplar::Device d;
  switch (deviceType) {
  case DeviceType::Cpu:
    target = poplar::Target::createCPUTarget();
    d = poplar::Device::createCPUDevice();
    break;
  case DeviceType::IpuModel:
  {
    poplar::IPUModel ipuModel;
    ipuModel.tilesPerIPU = tilesPerIPU;
    d = ipuModel.createDevice();
    break;
  }
  case DeviceType::Sim:
  {
    if ((tilesPerIPU!= 1 && (tilesPerIPU % 4) != 0) || tilesPerIPU > 1216)
      throw std::logic_error(
      "createDevice:: tilesPerIPU must be 1 or a multiple of 4 "
      "less than 1216\n");
    target = poplar::Target::createIPUTarget(numIPUs,
                                             tilesPerIPU, "_TEST_SYSTEM");
    poplar::OptionFlags opt;
    opt.set("debug.trace", tilesPerIPU <= 16 ? "true" : "false");
    d = poplar::Device::createSimulatorDevice(target, opt);
    break;
  }
  default:
      throw std::logic_error(
        "deviceType must be \"Cpu\", \"IpuModel\" or \"Sim\"\n");
  }
  return d;
}


inline const char *asString(const DeviceType &deviceType) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return "Cpu";
  case DeviceType::IpuModel:
    return "IpuModel";
  case DeviceType::Sim:
    return "Sim";
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
  else
    throw std::logic_error("Unsupported device type <" + token + ">");
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const DeviceType &type) {
  os << asString(type);
  return os;
}

}
#endif // __TestDevice_hpp
