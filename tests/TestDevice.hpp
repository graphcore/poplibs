#ifndef TestDevice_hpp__
#define TestDevice_hpp__

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
enum class DeviceType { Cpu, Sim, Sim0, Hw, IpuModel, IpuModel0 };

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

inline TestDevice createTestDevice(const DeviceType deviceType,
                                   const unsigned numIPUs = 1,
                                   const unsigned tilesPerIPU = 1,
                                   const bool compileIPUCode = false) {
  switch (deviceType) {
  case DeviceType::Cpu:
    return poplar::Device::createCPUDevice();
  case DeviceType::Sim:
  case DeviceType::Sim0: {
    auto target = poplar::Target::createIPUTarget(
        numIPUs, tilesPerIPU, deviceType == DeviceType::Sim ? "ipu1" : "ipu0");
    return poplar::Device::createSimulatorDevice(std::move(target));
  }
  case DeviceType::Hw: {
    static auto manager = poplar::DeviceManager::createDeviceManager();
    auto devices = manager.getDevices(poplar::TargetType::IPU, numIPUs);
    if (devices.empty()) {
      throw poplar::poplar_error("No devices exist with the requested "
                                 "configuration.");
    }

    // transform each device into a virtual device if needed.
    for (auto &device : devices) {
      if (tilesPerIPU != device.getTarget().getTilesPerIPU()) {
        device = device.createVirtualDevice(tilesPerIPU);
      }
    }

    return std::move(devices);
  }
  case DeviceType::IpuModel:
  case DeviceType::IpuModel0: {
    poplar::IPUModel model(deviceType == DeviceType::IpuModel ? "ipu1"
                                                              : "ipu0");
    model.numIPUs = numIPUs;
    model.tilesPerIPU = tilesPerIPU;
    model.compileIPUCode = compileIPUCode;
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
  case DeviceType::IpuModel0:
    return "IpuModel0";
  case DeviceType::Sim:
    return "Sim";
  case DeviceType::Sim0:
    return "Sim0";
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

} // namespace

#endif // __TestDevice_hpp
