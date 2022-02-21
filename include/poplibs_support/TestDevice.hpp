// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef TestDevice_hpp__
#define TestDevice_hpp__

#ifdef TEST_WITH_TARGET
/* When TEST_WITH_TARGET is defined this file will cause a main() to be
 * generated that will set TEST_TARGET based on the commandline. The requires
 * that BOOST_TEST_MODULE is defined before including this file, and
 * <boost/test/unit_test.hpp> must NOT be included beforehand.
 */
#ifndef BOOST_TEST_MODULE
#error                                                                         \
    "When TEST_WITH_TARGET is defined BOOST_TEST_MODULE must be defined before including TestDevice.hpp or any boost test headers"
#endif
#endif // TEST_WITH_TARGET

#if defined(TEST_WITH_TARGET) || defined(TEST_WITHOUT_TARGET)
#include <boost/test/unit_test.hpp>
#endif

#include <boost/optional.hpp>
#include <boost/variant.hpp>
#include <poplar/Device.hpp>
#include <poplar/Target.hpp>

#include <chrono>
#include <iosfwd>
#include <thread>

namespace poplibs_support {

// In CMakeLists.txt there is a regex on "Hw*" so be
// careful when adding new enums that begin with Hw:
enum class DeviceType { Cpu, Sim2, Sim21, Hw, IpuModel2, IpuModel21 };

const auto deviceTypeHelp = "Device type: Cpu | IpuModel2 | Sim2 | Hw";

constexpr bool isSimulator(DeviceType d) {
  return d == DeviceType::Sim2 || d == DeviceType::Sim21;
}
constexpr bool isIpuModel(DeviceType d) {
  return d == DeviceType::IpuModel2 || d == DeviceType::IpuModel21;
}
constexpr bool isHw(DeviceType d) { return d == DeviceType::Hw; }

const char *deviceTypeToIPUName(DeviceType d);

// an abstraction from one or more poplar::Devices that supports lazy
// attaching.
struct TestDevice {
  TestDevice(poplar::Device d) : device(std::move(d)) {}
  TestDevice(std::vector<poplar::Device> ds);

  const poplar::Target &getTarget() const;

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
TestDevice
createTestDevice(const DeviceType deviceType, const unsigned numIPUs,
                 const boost::optional<unsigned> requestedTilesPerIPU,
                 const bool compileIPUCode = false);

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

const char *asString(const DeviceType &deviceType);

DeviceType getDeviceType(const std::string &deviceString);

std::istream &operator>>(std::istream &is, DeviceType &type);

std::ostream &operator<<(std::ostream &os, const DeviceType &type);

} // namespace poplibs_support

// TEST_TARGET is be defined by TestDevice.cpp. When TEST_WITH_TARGET is defined
// it is initialised by the boost test infrastructure. When TEST_WITH_TARGET is
// not defined explicit initialisation is required.
extern poplibs_support::DeviceType TEST_TARGET;

#ifdef TEST_WITH_TARGET
// Defines to allow the test device to be specified on the command line, and
// for test predication.

struct CommandLineDeviceInit {
  CommandLineDeviceInit() {
    // configure TEST_TARGET based on the test device argument
    if (boost::unit_test::framework::master_test_suite().argc != 3 ||
        boost::unit_test::framework::master_test_suite().argv[1] !=
            std::string("--device-type"))
      BOOST_FAIL("This test requires the device to be specified on the "
                 "command-line via <test-command> [ctest arguments] -- "
                 "--device-type <device-type>");
    auto deviceString =
        boost::unit_test::framework::master_test_suite().argv[2];
    TEST_TARGET = poplibs_support::getDeviceType(deviceString);
  }
  void setup() {}
  void teardown() {}
};

// Note this defines main(); BOOST_TEST_MODULE must be defined at this point
BOOST_TEST_GLOBAL_FIXTURE(CommandLineDeviceInit);

struct enableIfSimulator {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isSimulator(TEST_TARGET));
    ans.message() << "test only supported for simulator targets";
    return ans;
  }
};
struct enableIfNotSim {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(!isSimulator(TEST_TARGET));
    ans.message() << "test only supported for simulator targets";
    return ans;
  }
};

struct enableIfHw {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isHw(TEST_TARGET));
    ans.message() << "test only supported for hardware targets";
    return ans;
  }
};

struct enableIfIpuModel {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(isIpuModel(TEST_TARGET));
    ans.message() << "test only supported for IpuModel targets";
    return ans;
  }
};

struct enableIfSimOrHw {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    bool is_sim_target = isSimulator(TEST_TARGET);
    bool is_hw_target = isHw(TEST_TARGET);
    bool is_ipu_target = is_sim_target || is_hw_target;
    boost::test_tools::assertion_result ans(is_ipu_target);
    ans.message() << "test only supported for Sim and Hw targets";
    return ans;
  }
};

struct enableIfIpu21Sim {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(TEST_TARGET ==
                                            poplibs_support::DeviceType::Sim21);
    ans.message() << "test only supported for IPU21 simulator target";
    return ans;
  }
};

struct enableIfIpu21 {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(
        TEST_TARGET == poplibs_support::DeviceType::Sim21 ||
        TEST_TARGET == poplibs_support::DeviceType::IpuModel21);
    ans.message() << "test only supported for IPU21 targets";
    return ans;
  }
};

struct enableIfNotCpu {
  boost::test_tools::assertion_result
  operator()(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result ans(TEST_TARGET !=
                                            poplibs_support::DeviceType::Cpu);
    ans.message() << "test not supported for Cpu target";
    return ans;
  }
};

#endif // TEST_WITH_TARGET

#endif // __TestDevice_hpp
