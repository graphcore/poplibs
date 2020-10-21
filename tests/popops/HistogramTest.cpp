// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplar/Target.hpp"
#include "poplibs_test/Check.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/GatherStatistics.hpp"
#include "popops/codelets.hpp"
#include <boost/program_options.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poputil/TileMapping.hpp>
#include <stdexcept>
#include <string.h>
#include <utility>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poplibs_test::util;
using namespace poputil;

using namespace poplibs_support;
namespace po = boost::program_options;
// Host model of the histogram.  Intentionally coded in a straightforward way
template <typename HistType>
std::vector<HistType>
histogram(const std::vector<float> &data, const std::vector<float> &limits,
          const std::vector<HistType> &initialHistogram, bool isAbsolute) {

  std::vector<HistType> histogram = initialHistogram;
  for (unsigned i = 0; i < data.size(); i++) {
    auto dataElem = data[i];
    if (isAbsolute) {
      dataElem = std::fabs(dataElem);
    }
    for (unsigned j = 0; j < limits.size(); j++) {
      if (j == 0 && dataElem < limits[j]) {
        // Below the lower bound
        histogram[j]++;
      } else {
        if (j == limits.size() - 1 && dataElem >= limits[j]) {
          // Above the upper bound
          histogram[j + 1]++;
        }
        if (dataElem < limits[j] && dataElem >= limits[j - 1]) {
          // Between a bound and the one below it
          histogram[j]++;
        }
      }
    }
  }
  return histogram;
}

template <typename HistType>
bool doTest(TestDevice &device, DeviceType deviceType, bool profile,
            const std::vector<double> &data, const std::vector<float> &limits,
            const std::vector<HistType> &initialHistogram,
            bool useFloatArithmetic, bool withOutput, bool update,
            const poplar::Type &dataType, bool isAbsolute) {
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  const auto rawSize = target.getTypeSize(dataType);

  std::vector<char> rawData(data.size() * rawSize);
  std::vector<char> rawLimits(limits.size() * rawSize);

  copy(target, data.data(), data.size(), dataType, rawData.data());
  copy(target, limits.data(), limits.size(), dataType, rawLimits.data());

  auto ipuData = graph.addVariable(dataType, {data.size()});
  auto ipuLimits = graph.addVariable(dataType, {limits.size()});
  mapTensorLinearly(graph, ipuData);
  graph.setTileMapping(ipuLimits, 0);

  auto prog = Sequence();
  poplar::OptionFlags options = {
      {"useFloatArithmetic", useFloatArithmetic ? "true" : "false"}};
  Tensor ipuHistogram;
  if (withOutput) {
    ipuHistogram = graph.addVariable(useFloatArithmetic ? FLOAT : UNSIGNED_INT,
                                     {limits.size() + 1});
    mapTensorLinearly(graph, ipuHistogram);
    graph.createHostWrite("histogram", ipuHistogram);

    popops::histogram(graph, ipuData, ipuHistogram, update, ipuLimits,
                      isAbsolute, prog, "Test Histogram");
  } else {
    ipuHistogram = popops::histogram(graph, ipuData, ipuLimits, isAbsolute,
                                     prog, "Test Histogram", options);
  }
  graph.createHostRead("histogram", ipuHistogram);
  graph.createHostWrite("data", ipuData);
  graph.createHostWrite("limits", ipuLimits);

  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Engine e(graph, prog, engineOptions);
  std::vector<HistType> resultHistogram(limits.size() + 1);

  device.bind([&](const Device &d) {
    e.load(d);

    e.writeTensor("data", rawData.data(), rawData.data() + rawData.size());
    e.writeTensor("limits", rawLimits.data(),
                  rawLimits.data() + rawLimits.size());
    if (withOutput) {
      e.writeTensor("histogram", initialHistogram.data(),
                    initialHistogram.data() + initialHistogram.size());
    }
    e.run();
    e.readTensor("histogram", resultHistogram.data(),
                 resultHistogram.data() + resultHistogram.size());
  });

  // Cast half data back to get float data that represents half accuracy, or
  // just copy if float.  This matters where limits are equal to the histogram
  // entries in the numeric format of the codelet.
  std::vector<float> dataConvert(data.size());
  std::vector<float> limitsConvert(limits.size());
  copy(target, dataType, rawData.data(), dataConvert.data(),
       dataConvert.size());
  copy(target, dataType, rawLimits.data(), limitsConvert.data(),
       limitsConvert.size());

  bool success = true;
  // Generate host result, compare
  auto hostResult = histogram<HistType>(dataConvert, limitsConvert,
                                        initialHistogram, isAbsolute);

  std::cout << "Limits:" << limitsConvert << "\n";

  std::cout << "    Hist:" << resultHistogram << "\nExpected:" << hostResult;
  std::cout << "\n";

  for (unsigned i = 0; i < hostResult.size(); ++i) {
    if (hostResult[i] != resultHistogram[i]) {
      success = false;
      break;
    }
  }

  if (profile && deviceType != DeviceType::Cpu) {
    e.printProfileSummary(std::cout,
                          OptionFlags{{"showExecutionSteps", "true"}});
  }
  return success;
}

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel2;
  // Default input parameters.
  Type dataType = FLOAT;
  bool isAbsolute = false;
  bool useFloatArithmetic = false;
  bool withOutput = false;
  bool update = false;
  unsigned dataSize;
  unsigned limitSize;
  unsigned tiles = 4;
  double dataMin = -65504;
  double dataRange = 65504 * 2;
  float limitsMin = -60000;
  float limitsRange = 60000 * 2;
  float limitsStep = 0.0;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       deviceTypeHelp)
    ("tiles-per-ipu", po::value(&tiles)->default_value(tiles),
      "Number of tiles per IPU")
    ("profile", "Output profiling report")
    ("type", po::value(&dataType)->default_value(dataType),
      "Data type of data and limits")
    ("data-size", po::value(&dataSize)->required(),
      "Number of data elements")
    ("data-min", po::value(&dataMin)->default_value(dataMin),
      "Minimum data value")
    ("data-range", po::value(&dataRange)->default_value(dataRange),
      "Data range")
    ("limits-size", po::value(&limitSize)->required(),
      "Number of limits, histogram results = limits + 1")
    ("limits-min", po::value(&limitsMin)->default_value(limitsMin),
      "Minimum limit")
    ("limits-range", po::value(&limitsRange)->default_value(limitsRange),
      "Limits range")
    ("limits-step", po::value(&limitsStep)->default_value(limitsStep),
      "Limits step, found from range unless this parameter is used")
    ("absolute", po::value(&isAbsolute)->default_value(isAbsolute),
      "Use absolute values of the data")
    ("use-float-arithmetic", po::value(&useFloatArithmetic)->default_value(useFloatArithmetic),
      "Use float arithmetic, producing a float result for speed")
    ("with-output", po::value(&withOutput)->default_value(withOutput),
      "Provide an output external to the histogram function")
    ("update", po::value(&update)->default_value(update),
      "Update (continue to gather) histogram results, implies with-output=true")
    ;
  // clang-format on
  po::variables_map vm;
  bool profile = false;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    if (vm.count("profile")) {
      profile = true;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error parsing command line: " << e.what() << "\n";
    return 1;
  }

  if (update) {
    withOutput = true;
  }

  // Random data within the range specified
  std::mt19937 randomEngine;
  std::vector<double> data(dataSize);
  auto device = createTestDevice(deviceType, 1, tiles);
  writeRandomValues(device.getTarget(), dataType, data.data(),
                    data.data() + data.size(), dataMin, dataMin + dataRange,
                    randomEngine);

  // Evenly spaced limits
  std::vector<float> limits(limitSize);
  if (limitsStep == 0.0) {
    limitsStep = limits.size() == 1
                     ? 1
                     : limitsRange / static_cast<float>(limits.size() - 1);
  }
  for (unsigned i = 0; i < limits.size(); i++) {
    limits[i] = limitsMin + static_cast<float>(i) * limitsStep;
  }
  bool success;
  if (useFloatArithmetic) {
    std::vector<float> initialHistogram(limitSize + 1);
    for (unsigned i = 0; i < limitSize + 1; i++) {
      initialHistogram[i] = update ? i + 1 : 0;
    }
    success = doTest<float>(device, deviceType, profile, data, limits,
                            initialHistogram, useFloatArithmetic, withOutput,
                            update, dataType, isAbsolute);
  } else {
    std::vector<unsigned> initialHistogram(limitSize + 1);
    for (unsigned i = 0; i < limitSize + 1; i++) {
      initialHistogram[i] = update ? i + 1 : 0;
    }
    success = doTest<unsigned>(device, deviceType, profile, data, limits,
                               initialHistogram, useFloatArithmetic, withOutput,
                               update, dataType, isAbsolute);
  }
  if (!success) {
    std::cerr << "Failure\n";
  }
  return !success;
}
