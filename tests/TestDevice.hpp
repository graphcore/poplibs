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
    assert(minNumTiles <= 1216);
    std::string system;
    if (minNumTiles == 1)
      system = "_TEST_SYSTEM_ONE_TILE";
    else if (minNumTiles <= 4)
      system = "_TEST_SYSTEM_FOUR_TILE";
    else if (minNumTiles <= 16)
      system = "_TEST_SYSTEM_SIXTEEN_TILE";
    else
      system = "_TEST_SYSTEM_ALL_TILES";
    target = poplar::Target::createIPUTarget(1, system);
    d = poplar::Device::createSimulatorDevice(target, minNumTiles <=16);
    break;
  }
  case DeviceType::Sim4IPU4:
    assert(minNumTiles <= 4);
    target = poplar::Target::createIPUTarget(4, "_TEST_SYSTEM_FOUR_TILE");
    d = poplar::Device::createSimulatorDevice(target);
    break;
  case DeviceType::Sim1IPU:
    assert(minNumTiles <= 1216);
    target = poplar::Target::createIPUTarget(1, "_TEST_SYSTEM_ALL_TILES");
    d = poplar::Device::createSimulatorDevice(target, false);
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
