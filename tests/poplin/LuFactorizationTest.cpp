// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <poplibs_support/TestDevice.hpp>

#include <iostream>
#include <random>

#include <boost/program_options.hpp>

#include <poplar/CycleCount.hpp>
#include <poplar/Engine.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/ConvPreplan.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <poplin/experimental/LuFactorization.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplin;
using namespace poplibs_support;

namespace {

std::vector<float> generateInput(size_t height, size_t width) {
  std::vector<float> data(height * width);
  std::random_device rd;
  auto engine = std::mt19937(45678);
  std::uniform_real_distribution<float> dist(1.5, 9.9);
  std::generate(data.begin(), data.end(), std::bind(dist, std::ref(engine)));
  return data;
}

void writeTensor(const poplar::Target &target, poplar::Engine &engine,
                 poplar::Type type, StringRef handle,
                 const std::vector<float> &values) {
  if (type == HALF) {
    std::vector<char> buf(values.size() * target.getTypeSize(HALF));
    copyFloatToDeviceHalf(target, values.data(), buf.data(), values.size());
    engine.writeTensor(handle, buf.data(), buf.data() + buf.size());
  } else if (type == FLOAT) {
    engine.writeTensor(handle, values.data(), values.data() + values.size());
  } else {
    throw std::runtime_error("invalid type");
  }
}

void readTensor(const poplar::Target &target, poplar::Engine &engine,
                poplar::Type type, StringRef handle,
                std::vector<float> &values) {
  if (type == HALF) {
    std::vector<char> buf(values.size() * target.getTypeSize(HALF));
    engine.readTensor(handle, buf.data(), buf.data() + buf.size());
    copyDeviceHalfToFloat(target, buf.data(), values.data(), values.size());
  } else if (type == FLOAT) {
    engine.readTensor(handle, values.data(), values.data() + values.size());
  } else {
    throw std::runtime_error("invalid type");
  }
}

bool deviceLuDecomposition(const DeviceType &deviceType, poplar::Type type,
                           size_t height, size_t width, bool printTensors,
                           bool benchmark) {

  const unsigned ipuTiles = benchmark ? 1472 : 16;
  auto device = createTestDevice(deviceType, 1, ipuTiles);
  auto &target = device.getTarget();
  bool matchesModel = false;

  Tensor tStampBegin;
  Tensor tStampEnd;
  uint64_t stampBegin, stampEnd;

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  auto inputData = generateInput(height, width);

  Tensor tA = graph.addVariable(type, {height, width}, "inputTensor");

  assert(tA.numElements() == inputData.size());

  poputil::mapTensorLinearly(graph, tA);

  graph.createHostWrite("inputData", tA);

  Sequence seq;

  if (benchmark) {
    tStampBegin = cycleStamp(graph, seq, 0, SyncType::INTERNAL);
  }
  auto [tL, tU] =
      experimental::LUFactorization(graph, tA, seq, "LU - Decomposition");
  if (benchmark) {
    tStampEnd = cycleStamp(graph, seq, 0, SyncType::INTERNAL);
  }
  assert(tL.numElements() == height * height);
  assert(tU.numElements() == height * width);
  if (printTensors) {
    seq.add(PrintTensor("input", tA));
    seq.add(PrintTensor("lower", tL));
    seq.add(PrintTensor("upper", tU));
  }
  Tensor tR = matMul(graph, tL, tU, seq, FLOAT, "LU result check");
  if (printTensors) {
    seq.add(PrintTensor("MatMul result", tR));
  }
  graph.createHostRead("resultData", tR);
  if (benchmark) {
    graph.createHostRead("tStampBegin", tStampBegin);
    graph.createHostRead("tStampEnd", tStampEnd);
  }
  std::vector<float> rData(tR.numElements());

  Engine eng(graph, seq);
  device.bind([&](const Device &dev) {
    eng.load(dev);
    writeTensor(target, eng, type, "inputData", inputData);
    eng.run();
    readTensor(target, eng, FLOAT, "resultData", rData);
    if (benchmark) {
      eng.readTensor("tStampBegin", &stampBegin, &stampBegin + 1);
      eng.readTensor("tStampEnd", &stampEnd, &stampEnd + 1);
    }
  });

  if (benchmark) {
    unsigned opCount = (2.0f / 3.0f) * height * height * width;
    uint64_t ipuTimeClk = stampEnd - stampBegin;
    double gFlops = (double)opCount / ipuTimeClk *
                    (target.getTileClockFrequency() / 1000000000);
    std::cout << "IPU_Clock;" << target.getTileClockFrequency()
              << "; IPU_Tiles;" << ipuTiles << "; Time_in_clk;" << ipuTimeClk
              << "; GFLOPS;" << gFlops << std::endl;
  }

  const double absTolerance = 0.3;
  const double relTolerance = 0.001;
  matchesModel = poplibs_test::util::checkIsClose(
      "result", rData.data(), {height, width}, inputData.data(),
      inputData.size(), relTolerance, absTolerance);
  return matchesModel;
}

} // namespace

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = TEST_TARGET;
  poplar::Type dataType = poplar::FLOAT;
  unsigned m = 32;
  unsigned n = 32;
  bool printTensors = false;
  bool benchmark = false;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type", po::value<DeviceType>(&deviceType)->required(), "Device type")
    ("height", po::value<unsigned>(&m)->required(), "Input width")
    ("width", po::value<unsigned>(&n)->required(), "Input height")
    ("data-type", po::value<poplar::Type>(&dataType)->default_value(dataType), "Data type")
    ("print-tensors", po::value<bool>(&printTensors)->default_value(printTensors), "Print tensors")
    ("benchmark", po::value<bool>(&benchmark)->default_value(benchmark), "Benchmark using IPU cycleStamp")
    ;
  // clang-format on

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  // Ignore benchmark option if test is not run on Hw
  benchmark = benchmark && (deviceType == DeviceType::Hw);

  const bool success = deviceLuDecomposition(deviceType, dataType, m, n,
                                             printTensors, benchmark);
  if (!success) {
    std::cerr << "Test failed\n";
    return 1;
  }

  return 0;
}
