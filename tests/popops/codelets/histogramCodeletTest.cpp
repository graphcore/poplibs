// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poplar/Engine.hpp>
#include <poplibs_support/TestDevice.hpp>
// codelets
#include "poplar/Target.hpp"
#include "poplibs_test/Check.hpp"
#include "poplibs_test/Util.hpp"
#include "popops/codelets.hpp"
#include "poputil/VertexTemplates.hpp"
#include <boost/program_options.hpp>
#include <stdexcept>
#include <string.h>
#include <utility>

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;
using namespace poplibs_test::util;

using namespace poplibs_support;
namespace po = boost::program_options;
// Host model of the histogram.  Intentionally coded in a straightforward way
// that is different to the codelet implementation
std::vector<float> histogram(const std::vector<float> &data,
                             const unsigned unpaddedDataSize,
                             const unsigned offset,
                             const std::vector<float> &limits,
                             bool isAbsolute) {

  std::vector<float> histogram(limits.size() + 1, 0);
  for (unsigned i = offset; i < unpaddedDataSize + offset; i++) {
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

bool doTest(TestDevice &device, DeviceType deviceType, bool profile,
            unsigned padding, const unsigned rows,
            const std::vector<double> &data, const std::vector<float> &limits,
            bool isSupervisor, const poplar::Type &dataType, bool isAbsolute) {
  auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);

  const auto rawSize = target.getTypeSize(dataType);

  std::vector<char> rawData(data.size() * rawSize);
  std::vector<char> rawLimits(limits.size() * rawSize);

  copy(target, data.data(), data.size(), dataType, rawData.data());
  copy(target, limits.data(), limits.size(), dataType, rawLimits.data());

  auto cs = graph.addComputeSet("cs");

  const auto vertexClass = templateVertex(
      isSupervisor ? "popops::HistogramSupervisor" : "popops::Histogram2D",
      dataType, isAbsolute);

  // We create one data tensor, with multiple vertices reading from it, each
  // with a different offset to ensure we check each input alignment.
  // This is kind of possible with the 2D vertex and multiple rows each of
  // an odd length, but not for the supervisor case.  Use this mechanism for
  // both vertex types
  auto vertexData = graph.addVariable(dataType, {data.size()});
  auto vertexLimits = graph.addVariable(dataType, {limits.size()});
  const auto unpaddedDataSize = data.size() - padding;
  graph.setTileMapping(vertexData, 0);
  graph.setTileMapping(vertexLimits, 0);
  graph.createHostWrite("data", vertexData);
  graph.createHostWrite("limits", vertexLimits);

  const auto numVertices = padding + 1;
  std::vector<std::string> readHandles(numVertices);

  for (unsigned offset = 0; offset < numVertices; offset++) {
    auto vertexHistogram = graph.addVariable(FLOAT, {limits.size() + 1});

    auto v = graph.addVertex(cs, vertexClass);
    graph.setTileMapping(v, 0);

    if (isSupervisor) {
      graph.connect(v["data"],
                    vertexData.slice({offset, offset + unpaddedDataSize}));
    } else {
      std::vector<Tensor> dataSlices(rows);
      auto remainder = unpaddedDataSize % rows;
      auto rowLength = unpaddedDataSize / rows;

      for (unsigned i = 0; i < rows - 1; i++) {
        dataSlices[i] = vertexData.slice(offset + rowLength * i,
                                         offset + rowLength * (i + 1));
      }
      // The last row will contain the remainder, if any
      dataSlices[rows - 1] =
          vertexData.slice(offset + rowLength * (rows - 1),
                           offset + rowLength * rows + remainder);
      graph.connect(v["data"], dataSlices);
    }
    graph.setInitialValue(v["histogramCount"], limits.size() + 1);
    graph.connect(v["limits"], vertexLimits);
    graph.connect(v["histogram"], vertexHistogram);
    graph.setTileMapping(vertexHistogram, 0);

    readHandles[offset] = "histogram_" + std::to_string(offset);
    graph.createHostRead(readHandles[offset], vertexHistogram);
  }
  OptionFlags engineOptions;
  if (profile) {
    engineOptions.set("debug.instrumentCompute", "true");
  }
  Sequence prog;
  prog.add(Execute(cs));
  Engine e(graph, prog, engineOptions);
  std::vector<std::vector<float>> resultHistogram(numVertices);

  device.bind([&](const Device &d) {
    e.load(d);

    e.writeTensor("data", rawData.data(), rawData.data() + rawData.size());
    e.writeTensor("limits", rawLimits.data(),
                  rawLimits.data() + rawLimits.size());

    e.run();
    for (unsigned i = 0; i < numVertices; i++) {
      resultHistogram[i].resize(limits.size() + 1);
      e.readTensor(readHandles[i], resultHistogram[i].data(),
                   resultHistogram[i].data() + resultHistogram[i].size());
    }
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
  // Generate host result, compare for each offset
  for (unsigned offset = 0; offset < numVertices; offset++) {
    std::vector<float> hostResult;
    hostResult = histogram(dataConvert, unpaddedDataSize, offset, limitsConvert,
                           isAbsolute);
    std::cout << "Padding:" << padding << " Data with pad:" << dataConvert
              << "\n";
    std::cout << "Limits:" << limitsConvert << "\n";

    std::cout << "    Hist:" << resultHistogram[offset]
              << "\nExpected:" << hostResult;
    std::cout << "\n";

    for (unsigned i = 0; i < hostResult.size(); ++i) {
      if (hostResult[i] != resultHistogram[offset][i]) {
        success = false;
        break;
      }
    }
  }
  if (profile && deviceType != DeviceType::Cpu) {
    e.printProfileSummary(std::cout,
                          OptionFlags{{"showExecutionSteps", "true"}});
  }
  return success;
}

int main(int argc, char **argv) {
  DeviceType deviceType = DeviceType::IpuModel;
  // Default input parameters.
  Type dataType = FLOAT;
  bool isAbsolute = false;
  bool isSupervisor = false;
  unsigned dataSize;
  unsigned limitSize;
  unsigned rows = 1;
  double dataMin = -65504;
  double dataRange = 65504 * 2;
  float limitsMin = -60000;
  float limitsRange = 60000 * 2;
  float limitsStep = 0.0;
  bool checkMisaligned = true;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
       po::value<DeviceType>(&deviceType)->default_value(deviceType),
       "Device type: Cpu | Sim | Sim2 | Hw | IpuModel | IpuModel2")
    ("profile", "Output profiling report")
    ("type", po::value(&dataType)->default_value(dataType),
      "Data type of data and limits")
    ("data-size", po::value(&dataSize)->required(),
      "Number of data elements")
    ("data-min", po::value(&dataMin)->default_value(dataMin),
      "Minimum data value")
    ("data-range", po::value(&dataRange)->default_value(dataRange),
      "Data range")
    ("data-rows", po::value(&rows)->default_value(rows),
      "Data rows, for use with the 2D vertex")
    ("limits-size", po::value(&limitSize)->required(),
      "Number of limits, histogram results = limits + 1")
    ("limits-min", po::value(&limitsMin)->default_value(limitsMin),
      "Minimum limit")
    ("limits-range", po::value(&limitsRange)->default_value(limitsRange),
      "Limits range")
    ("limits-step", po::value(&limitsStep)->default_value(limitsStep),
      "Limits step, found from range unless this parameter is used")
    ("supervisor", po::value(&isSupervisor)->default_value(isSupervisor),
      "Test supervisor vertex")
    ("absolute", po::value(&isAbsolute)->default_value(isAbsolute),
      "Use absolute values of the data")
    ("check-misaligned", po::value(&checkMisaligned)->default_value(checkMisaligned),
      "Run the codelet multiple times with input data with each misalignment"
      " that is relevant to the data type")
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
  // Create some padding so that we can run multiple times and check alignment
  const auto padding = checkMisaligned ? (dataType == HALF ? 4 : 2) - 1 : 0;

  // Random data within the range specified
  std::mt19937 randomEngine;
  std::vector<double> data(dataSize + padding);
  auto device = createTestDevice(deviceType);
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
  auto success = doTest(device, deviceType, profile, padding, rows, data,
                        limits, isSupervisor, dataType, isAbsolute);
  if (!success) {
    std::cerr << "Failure\n";
  }
  return !success;
}
