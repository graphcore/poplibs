// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "poplin/Cholesky.hpp"
#include "poplin/MatMul.hpp"
#include "poplin/TriangularSolve.hpp"
#include <boost/assign/list_of.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/version.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/TempDir.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/ConvPreplan.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <sstream>

using namespace poplibs_support;

void printArray(std::string name, boost::multi_array<double, 3> a) {
  std::cout << name << ": " << std::endl;
  std::size_t ng = a.shape()[0];
  std::size_t nr = a.shape()[1];
  std::size_t nc = a.shape()[2];

  for (std::size_t g = 0; g < ng; g++) {
    std::cout << g << std::endl;
    for (std::size_t r = 0; r < nr; r++) {
      std::cout << " ";
      for (std::size_t c = 0; c < nc; c++) {
        std::cout << std::setw(10) << a[g][r][c];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

boost::multi_array<double, 3> createPositiveDefiniteMatrix(std::size_t batches,
                                                           std::size_t rank) {
  boost::multi_array<double, 3> l(boost::extents[batches][rank][rank]);
  boost::multi_array<double, 3> pd(boost::extents[batches][rank][rank]);

  std::mt19937 randomEngine;
  boost::random::uniform_real_distribution<> dist(0.01, 1.0);

  for (std::size_t b = 0; b < batches; b++) {
    for (std::size_t r = 0; r < rank; r++) {
      for (std::size_t c = 0; c < rank; c++) {
        double v = dist(randomEngine);
        l[b][r][c] = v;
      }
    }
  }

  poplibs_test::gemm::generalGroupedMatrixMultiply(l, l, pd, false, true);

  for (std::size_t b = 0; b < batches; b++) {
    for (std::size_t r = 0; r < rank; r++) {
      pd[b][r][r] += rank;
    }
  }

  return pd;
}

const boost::multi_array<double, 3>
maskTriangularMatrix(const boost::multi_array<double, 3> &m,
                     bool lower = true) {
  std::size_t batches = m.shape()[0];
  std::size_t rank = m.shape()[1];
  boost::multi_array<double, 3> tm(boost::extents[batches][rank][rank]);

  for (std::size_t b = 0; b < batches; b++) {
    for (std::size_t r = 0; r < rank; r++) {
      for (std::size_t c = 0; c < rank; c++) {
        if ((lower && c <= r) || (!lower && c >= r))
          tm[b][r][c] = m[b][r][c];
        else
          tm[b][r][c] = 0;
      }
    }
  }

  return tm;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType;
  boost::optional<unsigned> tilesPerIPU;

  po::options_description desc("Options");

  unsigned numBatches = 1;
  unsigned aRank;
  unsigned bRank = -1;
  bool leftSide = true;
  poplar::Type dataType;
  boost::optional<unsigned> blockSizeParam;
  bool lower = true;
  bool unitDiagonal = true;
  boost::optional<std::string> profileDir;

  // clang-format off
  desc.add_options()
    ("help,h", "produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(DeviceType::IpuModel2),
      deviceTypeHelp)
    ("profile", "Output profiling report to standard output")
    ("profile-dir",
     po::value<decltype(profileDir)>(&profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("cholesky", "Run cholesky solver")
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
     po::value(&bRank),
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

  const bool runCholesky = vm.count("cholesky");

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

  poplin::PlanningCache cache;
  poplar::program::Sequence uploadProg, prog, downloadProg;
  poplar::OptionFlags options;

  poplar::DebugContext debugContext;

  if (blockSizeParam) {
    options.set("blockSize", std::to_string(*blockSizeParam));
  }

  if (runCholesky) {
    bRank = 0;
    unitDiagonal = false;

    if (!leftSide)
      throw poplar::poplar_error(
          "left-side must be true when using the cholesky solver.");
  } else if (bRank < 0) {
    throw poplar::poplar_error(
        "--b-rank is mandatory option for triangular solver");
  }

  std::vector<std::size_t> inputAShape{numBatches, aRank, aRank};
  std::vector<std::size_t> inputBShape{numBatches, leftSide ? aRank : bRank,
                                       leftSide ? bRank : aRank};

  std::vector<std::pair<poplin::MatMulParams, poplar::OptionFlags>>
      matmulOptPairs;
  if (runCholesky) {
    matmulOptPairs = poplin::getCholeskyMatMulPrePlanParameters(
        dataType, inputAShape, lower, options);
  } else {
    matmulOptPairs = poplin::getTriangularSolveMatMulPrePlanParameters(
        dataType, dataType, inputAShape, inputBShape, leftSide, lower, options);
  }

  std::set<poplin::MatMulPlanParams> params;
  for (auto &pair : matmulOptPairs)
    params.emplace(&target, pair.first, &pair.second);
  preplan({}, params, cache);

  poplar::Tensor inputA;
  if (runCholesky) {
    inputA = poplin::createCholeskyInput(graph, dataType, inputAShape, lower,
                                         debugContext, options, &cache);
  } else {
    inputA = poplin::createTriangularSolveInputLHS(
        graph, dataType, dataType, inputAShape, inputBShape, leftSide,
        debugContext, options, &cache);
  }

  poplar::Tensor inputB;
  if (!runCholesky) {
    inputB = poplin::createTriangularSolveInputRHS(
        graph, dataType, dataType, inputAShape, inputBShape, leftSide,
        debugContext, options, &cache);
  }

  poplar::Tensor out;
  if (runCholesky) {
    poplin::choleskyInPlace(graph, inputA, lower, prog, debugContext, options,
                            &cache);
    out = inputA;
  } else {
    out = poplin::triangularSolve(graph, inputA, inputB, leftSide, lower,
                                  unitDiagonal, prog, debugContext, options,
                                  &cache);
  }

  std::vector<std::pair<std::string, char *>> tmap;
  std::unique_ptr<char[]> rawHostInputA, rawHostInputB, rawHostOutput,
      rawHostOutputT;
  if (!ignoreData) {
    rawHostInputA = poplar_test::allocateHostMemoryForTensor(
        inputA, "A", graph, uploadProg, boost::none, tmap);

    if (!runCholesky) {
      rawHostInputB = poplar_test::allocateHostMemoryForTensor(
          inputB, "B", graph, uploadProg, boost::none, tmap);
    }

    rawHostOutput = poplar_test::allocateHostMemoryForTensor(
        out, "X", graph, boost::none, downloadProg, tmap);

    rawHostOutputT = poplar_test::allocateHostMemoryForTensor(
        poplin::transposeGroupedMatrix(out), "XT", graph, boost::none,
        downloadProg, tmap);
  }

  std::optional<TempDir> tempDir;
  poplar::OptionFlags engineOptions;
  if (vm.count("profile") || profileDir) {
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    if (profileDir) {
      engineOptions.set("autoReport.directory", *profileDir);
    } else {
      tempDir.emplace(TempDir::create());
      engineOptions.set("autoReport.directory", tempDir->getPath());
    }
  }
  poplar::Engine engine(graph, {uploadProg, prog, downloadProg}, engineOptions);

  if (vm.count("compile-only"))
    return 0;

  boost::multi_array<double, 3> hostInputA;
  boost::multi_array<double, 3> hostInputAFilled;
  boost::multi_array<double, 3> hostInputB;
  if (!ignoreData) {
    std::mt19937 randomEngine;
    boost::random::uniform_real_distribution<> dist(0.01, 1.0);
    boost::random::uniform_real_distribution<> diagonalDist(0.95, 1.05);

    poplar_test::attachStreams(engine, tmap);

    if (runCholesky) {
      hostInputAFilled.resize(boost::extents[numBatches][aRank][aRank]);
      hostInputA.resize(boost::extents[numBatches][aRank][aRank]);

      hostInputAFilled = createPositiveDefiniteMatrix(numBatches, aRank);
      hostInputA = maskTriangularMatrix(hostInputAFilled, lower);
    } else {

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
    }
    poplar_test::copy(target, hostInputA, dataType, rawHostInputA.get());

    if (!runCholesky) {
      hostInputB.resize(
          boost::extents[numBatches][inputBShape[1]][inputBShape[2]]);
      poplibs_test::util::writeRandomValues(target, dataType, hostInputB, -1.0,
                                            1.0, randomEngine);
      poplar_test::copy(target, hostInputB, dataType, rawHostInputB.get());
    }
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
    uint64_t dim1, dim2;
    if (runCholesky) {
      dim1 = inputAShape[1];
      dim2 = inputAShape[2];
    } else {
      dim1 = inputBShape[1];
      dim2 = inputBShape[2];
    }

    boost::multi_array<double, 3> hostOutput(
        boost::extents[numBatches][dim1][dim2]);
    poplar_test::copy(target, dataType, rawHostOutput.get(), hostOutput);

    boost::multi_array<double, 3> hostOutputT(
        boost::extents[numBatches][dim1][dim2]);
    poplar_test::copy(target, dataType, rawHostOutputT.get(), hostOutputT);

    boost::multi_array<double, 3> modelOutput(
        boost::extents[numBatches][dim1][dim2]);

    boost::multi_array<double, 3> *testOutput;

    if (runCholesky) {
      testOutput = &hostInputAFilled;

      if (lower)
        poplibs_test::gemm::generalGroupedMatrixMultiply(
            hostOutput, hostOutputT, modelOutput);
      else
        poplibs_test::gemm::generalGroupedMatrixMultiply(
            hostOutputT, hostOutput, modelOutput);
    } else {
      testOutput = &hostInputB;

      if (leftSide) {
        poplibs_test::gemm::generalGroupedMatrixMultiply(hostInputA, hostOutput,
                                                         modelOutput);
      } else {
        poplibs_test::gemm::generalGroupedMatrixMultiply(hostOutput, hostInputA,
                                                         modelOutput);
      }
    }

    const auto tolerance = dataType == poplar::HALF ? 0.3 : 0.001;
    matchesModel = poplibs_test::util::checkIsClose(
        "AX vs B: ", *testOutput, modelOutput, tolerance, tolerance);
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
