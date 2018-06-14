#ifndef __TestDevice_hpp
#define __TestDevice_hpp

#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>
namespace {
enum class DeviceType {
  Cpu,         //CPU
  IpuModel,    //IPU modelled on CPU, number of tiles specified by parameter
  Sim,         //IPU simulated by CISS, quantised min number of tiles
  Sim4IPU4,    //4 IPUs simulated by CISS, 4 tiles each
  Sim1IPU};    //IPU simulated by CISS, all tiles

// Create an engine for testing
// minNumTiles = the minimum number of tiles required; 0 to accept the default
// (0 not allowed for IPUModel)
inline poplar::Device createTestDevice(DeviceType deviceType,
                                       unsigned minNumTiles = 1)
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
    if (minNumTiles)
      ipuModel.tilesPerIPU = minNumTiles;
    d = ipuModel.createDevice();
    break;
  }
  case DeviceType::Sim:
  {
    // support a 2tile sim using a 4tile sim
    if (minNumTiles == 2)
      minNumTiles = 4;
    // any multi-tile sim must have a multiple of 4 tiles
    if ((minNumTiles != 1 && (minNumTiles % 4) != 0) || minNumTiles > 1216)
      throw std::logic_error(
      "createDevice:: minNumTiles must be 1 or a multiple of 4 "
      "less than 1216\n");
    target = poplar::Target::createIPUTarget(1, minNumTiles, "_TEST_SYSTEM");
    poplar::OptionFlags opt;
    opt.set("debug.trace", minNumTiles <= 16 ? "true" : "false");
    d = poplar::Device::createSimulatorDevice(target, opt);
    break;
  }
  case DeviceType::Sim4IPU4:
    if (minNumTiles > 4)
      throw std::logic_error("Sim1IPU4 supports only 1 or 4 tiles\n");
    target = poplar::Target::createIPUTarget(4, 4, "_TEST_SYSTEM");
    d = poplar::Device::createSimulatorDevice(target);
    break;
  case DeviceType::Sim1IPU:
    // any multi-tile sim must have a multiple of 4 tiles
    if ((minNumTiles != 1 && (minNumTiles % 4) != 0) || minNumTiles > 1216)
      throw std::logic_error(
      "createDevice:: minNumTiles must be 1 or a multiple of 4 "
      "less than 1216\n");

    target = poplar::Target::createIPUTarget(1, "_TEST_SYSTEM");
    d = poplar::Device::createSimulatorDevice(target);
    break;

  default:
      throw std::logic_error(
        "deviceType must be \"Cpu\", \"IpuModel\", \"Sim1\" , \"Sim4IPU4\""
        " or \"Sim1IPU\"\n");
  }
  return d;
}
}
#endif // __TestDevice_hpp
