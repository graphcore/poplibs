// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#ifndef TestDevice_hpp__
#define TestDevice_hpp__

#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/Target.hpp>

#include <chrono>
#include <iostream>
#include <thread>

namespace {
// In CMakeLists.txt there is a regex on "Hw*" so be
// careful when adding new enums that begin with Hw:
enum class DeviceType { Cpu, Sim, Sim2, Hw, IpuModel, IpuModel2 };
constexpr bool isSimulator(DeviceType d) {
  return d == DeviceType::Sim || d == DeviceType::Sim2;
}
constexpr bool isIpuModel(DeviceType d) {
  return d == DeviceType::IpuModel || d == DeviceType::IpuModel2;
}
constexpr bool isHw(DeviceType d) { return d == DeviceType::Hw; }

const char *deviceTypeToIPUName(DeviceType d) {
  switch (d) {
  case DeviceType::Sim:
  case DeviceType::IpuModel:
    return "ipu1";

  case DeviceType::Sim2:
  case DeviceType::IpuModel2:
    return "ipu2";

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

// an abstraction from one or more poplar::Devices that supports lazy attaching.
struct TestDevice {
  TestDevice(poplar::Device d) : device(std::move(d)) {}
  TestDevice(std::vector<poplar::Device> ds) : device(std::move(ds)) {
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

  const poplar::Target &getTarget() const {
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

  // lazily attach to a device and run the provided function.
  template <typename Fn> void bind(const Fn &fn) {
    struct Visitor : public boost::static_visitor<void> {
      Visitor(const Fn &f) : fn(f) {}

      bool tryCall(const poplar::Device &d) const {
        try {
          if (d.attach()) {
            fn(d);
            d.detach();
            return true;
          }

          return false;
        } catch (...) {
          d.detach();
          throw;
        }
      }

      result_type operator()(const poplar::Device &d) const {
        while (true) {
          if (tryCall(d)) {
            return;
          }

          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }

      result_type operator()(const std::vector<poplar::Device> &ds) const {
        while (true) {
          for (auto &d : ds) {
            if (tryCall(d)) {
              return;
            }
          }

          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }

    private:
      const Fn &fn;
    };

    return boost::apply_visitor(Visitor(fn), device);
  }

private:
  boost::variant<poplar::Device, std::vector<poplar::Device>> device;
};

// Return a device of the requested type. If requestedTilesPerIPU is boost::none
// the default number of tiles (all) are used.
// Set requestedTilesPerIPU to DeviceTypeDefaultTiles to use all tile on the
// device.
const auto DeviceTypeDefaultTiles = boost::none;
inline TestDevice
createTestDevice(const DeviceType deviceType, const unsigned numIPUs,
                 const boost::optional<unsigned> requestedTilesPerIPU,
                 const bool compileIPUCode = false) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return poplar::Device::createCPUDevice();
  case DeviceType::Sim:
  case DeviceType::Sim2: {
    auto targetName = deviceTypeToIPUName(deviceType);
    auto target = requestedTilesPerIPU
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
  case DeviceType::IpuModel:
  case DeviceType::IpuModel2: {
    auto targetName = deviceTypeToIPUName(deviceType);
    poplar::IPUModel model(targetName);
    model.numIPUs = numIPUs;
    if (requestedTilesPerIPU)
      model.tilesPerIPU = *requestedTilesPerIPU;
    model.compileIPUCode = compileIPUCode;
    return model.createDevice();
  }
  default:
    throw std::logic_error(
        R"XX(deviceType must be "Cpu", "IpuModel", "IpuModel2", "Sim", "Sim2" or "Hw")XX");
  }
}

// Helper to create a device with a single IPU with a single tile
inline TestDevice createTestDevice(const DeviceType deviceType) {
  return createTestDevice(deviceType, 1, 1);
}

// Helper to create a device with full-sized IPUs
inline TestDevice createTestDeviceFullSize(const DeviceType deviceType,
                                           unsigned numIPUs = 1,
                                           bool compileIPUCode = false) {
  return createTestDevice(deviceType, numIPUs, DeviceTypeDefaultTiles,
                          compileIPUCode);
}

inline const char *asString(const DeviceType &deviceType) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return "Cpu";
  case DeviceType::IpuModel:
    return "IpuModel";
  case DeviceType::IpuModel2:
    return "IpuModel2";
  case DeviceType::Sim:
    return "Sim";
  case DeviceType::Sim2:
    return "Sim2";
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
  else if (token == "IpuModel2")
    type = DeviceType::IpuModel2;
  else if (token == "Sim")
    type = DeviceType::Sim;
  else if (token == "Sim2")
    type = DeviceType::Sim2;
  else if (token == "Hw")
    type = DeviceType::Hw;
  else
    throw std::logic_error(
        "Unsupported device type <" + token +
        ">; must be one of "
        R"XX("Cpu", "IpuModel", "IpuModel2", "Sim", "Sim2" or "Hw")XX");
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const DeviceType &type) {
  os << asString(type);
  return os;
}

} // namespace

#endif // __TestDevice_hpp
