// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplin/MatMul.hpp"
#include "poplin/TriangularSolve.hpp"
#include <boost/assign/list_of.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/version.hpp>
#include <fstream>
#include <iostream>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <sstream>

using namespace poplibs_support;

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType;
  boost::optional<unsigned> tilesPerIPU;

  po::options_description desc("Options");

  unsigned numBatches = 1;
  unsigned aRank;
  unsigned bRank;
  bool leftSide = true;
  poplar::Type dataType;
  boost::optional<unsigned> blockSizeParam;
  bool lower = true;
  bool unitDiagonal = true;
  boost::optional<std::string> profileFormat;
  boost::optional<std::string> jsonProfileOut;

  // clang-format off
  desc.add_options()
    ("help,h", "produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(DeviceType::IpuModel2),
      deviceTypeHelp)
    ("profile", "Output profiling report to standard output")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("profile-format",
     po::value<decltype(profileFormat)>(&profileFormat)
      ->default_value(boost::none),
     "Profile formats: v1 | experimental | unstable")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
    ("data-type",
     po::value(&dataType)->required(),
     "Data Type")
    ("a-rank",
     po::value(&aRank)->required(),
     "Rank of the A matrix.")
    ("b-rank",
     po::value(&bRank)->required(),
     "Rank of the B matrix.")
    ("batches",
     po::value(&numBatches)->default_value(numBatches),
     "Number of batch dimensions.")
    ("left-side",
     po::value(&leftSide)->default_value(leftSide),
     "Left side - solve AX = B, XA = B overwise.")
    ("lower",
     po::value(&lower)->default_value(lower),
     "Generate lower A, upper A overwise.")
    ("unit-diagonal",
     po::value(&unitDiagonal)->default_value(unitDiagonal),
     "Assume A has unit diagonal.")
    ("block-size",
     po::value(&blockSizeParam),
     "Solver block size if specified, no block solver overwise.")
    ;
  // clang-format on

  po::variables_map vm;
  try {
    const po::positional_options_description p;
    po::store(
        po::command_line_parser(argc, argv).options(desc).positional(p).run(),
        vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const boost::program_options::error &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  poplar::OptionFlags engineOptions;
  if (vm.count("profile") || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (profileFormat) {
      engineOptions.set("profiler.format", *profileFormat);
    }
  }

  const bool ignoreData = vm.count("ignore-data");

  const unsigned numIPUs = 1;
  const bool compileIPUCode = true;
  auto device =
      tilesPerIPU
          ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, compileIPUCode)
          : createTestDeviceFullSize(deviceType, numIPUs, compileIPUCode);

  const auto &target = device.getTarget();
  poplar::Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  poplin::matmul::PlanningCache cache;
  poplar::program::Sequence uploadProg, prog, downloadProg;

  poplar::DebugContext debugContext;

  std::vector<std::size_t> inputAShape{numBatches, aRank, aRank};
  std::vector<std::size_t> inputBShape{numBatches, leftSide ? aRank : bRank,
                                       leftSide ? bRank : aRank};

  auto blockSize = blockSizeParam ? *blockSizeParam : aRank;
  auto inputA = poplin::createTriangularSolveInputLHS(
      graph, dataType, dataType, inputAShape, inputBShape, leftSide, blockSize,
      debugContext, {}, &cache);

  auto inputB = poplin::createTriangularSolveInputRHS(
      graph, dataType, dataType, inputAShape, inputBShape, leftSide, blockSize,
      debugContext, {}, &cache);

  auto out = poplin::triangularSolve(graph, inputA, inputB, leftSide, lower,
                                     unitDiagonal, blockSize, prog,
                                     debugContext, {}, &cache);

  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostInputA, rawHostInputB, rawHostOutput;
  if (!ignoreData) {
    rawHostInputA = poplibs_test::util::allocateHostMemoryForTensor(
        inputA, "A", graph, uploadProg, boost::none, tmap);

    rawHostInputB = poplibs_test::util::allocateHostMemoryForTensor(
        inputB, "B", graph, uploadProg, boost::none, tmap);

    rawHostOutput = poplibs_test::util::allocateHostMemoryForTensor(
        out, "X", graph, boost::none, downloadProg, tmap);
  }

  poplar::Engine engine(graph, {uploadProg, prog, downloadProg}, engineOptions);

  if (vm.count("compile-only"))
    return 0;

  boost::multi_array<double, 3> hostInputA;
  boost::multi_array<double, 3> hostInputB;
  if (!ignoreData) {
    poplibs_test::util::attachStreams(engine, tmap);

    std::mt19937 randomEngine;
    boost::random::uniform_real_distribution<> dist(0.01, 1.0);
    boost::random::uniform_real_distribution<> diagonalDist(0.95, 1.05);

    hostInputA.resize(boost::extents[numBatches][aRank][aRank]);
    for (std::size_t g = 0; g < numBatches; ++g) {
      auto matrix = hostInputA[g];
      for (std::size_t i = 0; i < aRank; ++i) {
        auto rows = matrix[i];
        for (std::size_t j = 0; j < aRank; ++j) {
          double value;
          if (i == j) {
            value = unitDiagonal ? 1.0 : diagonalDist(randomEngine);
          } else if ((i <= j && !lower) || (j <= i && lower)) {
            value = dist(randomEngine);
          } else {
            value = 0.0;
          }
          rows[j] = value;
        }
      }
    }
    poplibs_test::util::copy(target, hostInputA, dataType, rawHostInputA.get());

    hostInputB.resize(
        boost::extents[numBatches][inputBShape[1]][inputBShape[2]]);
    poplibs_test::util::writeRandomValues(target, dataType, hostInputB, -1.0,
                                          1.0, randomEngine);
    poplibs_test::util::copy(target, hostInputB, dataType, rawHostInputB.get());
  }

  device.bind([&](const poplar::Device &d) {
    engine.load(d);
    if (!ignoreData) {
      // upload
      engine.run(0);
    }

    // convolve
    engine.run(1);

    if (!ignoreData) {
      // download
      engine.run(2);
    }
  });

  bool matchesModel = true;
  if (!ignoreData) {
    boost::multi_array<double, 3> hostOutput(
        boost::extents[numBatches][inputBShape[1]][inputBShape[2]]);
    poplibs_test::util::copy(target, dataType, rawHostOutput.get(), hostOutput);

    boost::multi_array<double, 3> modelOutput(
        boost::extents[numBatches][inputBShape[1]][inputBShape[2]]);

    if (leftSide) {
      poplibs_test::gemm::generalGroupedMatrixMultiply(hostInputA, hostOutput,
                                                       modelOutput);
    } else {
      poplibs_test::gemm::generalGroupedMatrixMultiply(hostOutput, hostInputA,
                                                       modelOutput);
    }

    const auto tolerance = dataType == poplar::HALF ? 0.3 : 0.001;
    matchesModel = poplibs_test::util::checkIsClose(
        "AX vs B: ", modelOutput, hostInputB, tolerance, tolerance);
  }

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();

    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }
  if (deviceType != DeviceType::Cpu && vm.count("profile")) {
    engine.printProfileSummary(
        std::cout, poplar::OptionFlags{{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  std::cerr << e.what() << std::endl;

  // this exit code has been marked as a "skip" for ctest.
  return 77;
}
