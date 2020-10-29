// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/Compiler.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/exceptions.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popops;
using namespace poplibs_support;

// Default tolerances used in tests
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1
#define FLOAT_ABS_TOL 1e-6
#define HALF_ABS_TOL 1e-5

// Class to specify matrix operation
enum class MatrixOp { NORMAL, TRANSPOSE };

const char *asString(const MatrixOp &op) {
  switch (op) {
  case MatrixOp::NORMAL:
    return "normal";
  case MatrixOp::TRANSPOSE:
    return "transpose";
  }
  POPLIB_UNREACHABLE();
}

std::istream &operator>>(std::istream &is, MatrixOp &op) {
  std::string token;
  is >> token;
  if (token == "normal")
    op = MatrixOp::NORMAL;
  else if (token == "transpose")
    op = MatrixOp::TRANSPOSE;
  else
    throw poputil::poplibs_error("Invalid pass <" + token + ">");
  return is;
}

std::ostream &operator<<(std::ostream &os, const MatrixOp &op) {
  return os << asString(op);
}

const OptionFlags defaultEngineOptions;

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  // Operation performed is is alpha * op(matA) x op(matB) + beta * matC
  // where  op(matA)  is a m x k matrix
  //        op(matB)  is a k x n matrix
  unsigned m, k, n, g;
  float alpha, beta;
  Type inputType;
  Type outputType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  MatrixOp matAOp = MatrixOp::NORMAL;
  MatrixOp matBOp = MatrixOp::NORMAL;
  DeviceType deviceType = DeviceType::IpuModel2;
  double availableMemoryProportion;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  unsigned numExecutions;
  std::string planConstraints;
  std::string planConstraintsFile;
  bool remapOutputTensor;
  bool enableFastReduce;

  boost::optional<std::string> jsonProfileOut;
  boost::optional<std::string> profileFormat;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(deviceType),
      deviceTypeHelp)
    ("profile", "Output profiling report to standard output")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("use-unstable-format", "Deprecated: use \"--profile-format experimental\"")
    ("profile-format",
     po::value<decltype(profileFormat)>(&profileFormat)
      ->default_value(boost::none),
     "Profile formats: v1 | experimental | unstable")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("m", po::value<unsigned>(&m)->required(),
     "Number of rows of left matrix, left-matrix-op(A)")
    ("k", po::value<unsigned>(&k)->required(),
     "Number of columns of left matrix left-matrix-op(A) and number of rows of "
     "right matrix right-matrix-op(B)")
    ("n",  po::value<unsigned>(&n)->required(),
      "Number of columns of the right matrix right-matrix-op(B)")
    ("g",  po::value<unsigned>(&g)->default_value(1),
      "Number of groups)")
    ("data-type",
     po::value<Type>(&inputType)->default_value(HALF),
     "Type of the input and output data")
    ("input-type",
     po::value<Type>(&inputType),
     "Type of the input data")
    ("output-type",
      po::value<Type>(&outputType),
      "Output data type")
    ("partials-type",
     po::value<Type>(&partialsType),
     "Type of the partials")
    ("alpha",
      po::value<float>(&alpha)->default_value(1.0),
      "alpha in the operation "
      "alpha * left-matrix-op(A) * right-matrix-op(B) + beta * C")
    ("beta",
     po::value<float>(&beta)->default_value(1.0),
     "beta in the operation "
     "alpha * left-matrix-op(A) * right-matrix-op(B) + beta * C")
    ("left-matrix-op",
      po::value<MatrixOp>(&matAOp)->default_value(matAOp),
      "Operation on left matrix  normal | transpose")
    ("right-matrix-op",
      po::value<MatrixOp>(&matBOp)->default_value(matBOp),
      "Operation on right matrix  normal | transpose")
    ("tolerance", po::value<double>(&relativeTolerance),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("available-memory-proportion",
     po::value<double>(&availableMemoryProportion),
     "the estimated proportion of memory available to perform this operation")
    ("enable-fast-reduce",
     po::value<bool>(&enableFastReduce)->default_value(false),
     "enable a faster reduction vertex")
    ("report-plan", "Show plan info")
    ("show-execution-steps", "Show execution steps (requires profiling)")
    ("show-var-storage", "Show variable liveness (requires profiling)")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
    ("plan-constraints",
     po::value<std::string>(&planConstraints),
     "Constraints on the chosen convolution plan as a JSON string")
    ("num-executions",
      po::value<unsigned>(&numExecutions)->default_value(1u),
     "Number of times to repeat the multiply")
    ("remap-output-tensor",
     po::value<bool>(&remapOutputTensor)->default_value(false),
     "Remap output tensor if layout is detected to be poor")
    ("plan-constraints-file",
     po::value<std::string>(&planConstraintsFile)
       ->default_value(planConstraintsFile),
     "Constraints on the chosen convolution plan as a file "
     "path to a JSON file")
  ;
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  if ((vm["output-type"].empty() != vm["input-type"].empty()) ||
      (!vm["data-type"].defaulted() && !vm["output-type"].empty())) {
    throw poputil::poplibs_error("Please specify either --data-type OR "
                                 "(--input-type AND --output-type), not both.");
  }
  if (vm["output-type"].empty()) {
    outputType = inputType;
  }

  if (outputType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
    relativeTolerance = FLOAT_REL_TOL;
  } else {
    absoluteTolerance = HALF_ABS_TOL;
    relativeTolerance = HALF_REL_TOL;
  }

  const bool profile = deviceType != DeviceType::Cpu && vm.count("profile");
  const bool reportPlan = vm.count("report-plan");
  const bool showExecutionSteps = vm.count("show-execution-steps");
  const bool showVarStorage = vm.count("show-var-storage");
  const bool ignoreData = vm.count("ignore-data");
  if (vm.count("use-unstable-format")) {
    throw poputil::poplibs_error("\"--use-unstable-format\" is deprecated. Use "
                                 "\"--profile-format experimental\" instead");
  }

  const bool compileIPUCode = true;
  auto device =
      tilesPerIPU
          ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, compileIPUCode)
          : createTestDeviceFullSize(deviceType, numIPUs, compileIPUCode);

  const auto &target = device.getTarget();
  Graph graph(target);
  poplin::addCodelets(graph);
  popops::addCodelets(graph);

  const bool transposeA = matAOp == MatrixOp::TRANSPOSE;
  const bool transposeB = matBOp == MatrixOp::TRANSPOSE;

  /* set up row and column dimensions for the right and left matrix */
  const auto rowsMatA = transposeA ? k : m;
  const auto colsMatA = transposeA ? m : k;
  const auto rowsMatB = transposeB ? n : k;
  const auto colsMatB = transposeB ? k : n;

  if (!planConstraints.empty() && !planConstraintsFile.empty()) {
    throw poputil::poplibs_error("Both plan-constraints and "
                                 "plan-constraints-file were specified");
  }

  // If constraints were specified in a file, put them into the plan
  // constraints option.
  if (!planConstraintsFile.empty()) {
    std::ifstream is(planConstraintsFile, std::ios_base::in);
    if (!is.good()) {
      throw poputil::poplibs_error("Plan constraints file doesn't exist");
    }
    is.seekg(0, std::ios::end);
    const auto bytes = is.tellg();
    planConstraints = std::string(bytes, '\0');
    is.seekg(0);
    is.read(&planConstraints[0], bytes);
  }

  matmul::PlanningCache cache;
  poplar::OptionFlags mmOpt;

  // For single layer tests always disable output remapping
  mmOpt.set({{"remapOutputTensor", remapOutputTensor ? "true" : "false"}});

  if (!vm["partials-type"].empty()) {
    mmOpt.set("partialsType", partialsType.toString());
  }
  if (!planConstraints.empty()) {
    mmOpt.set("planConstraints", planConstraints);
  }
  if (transposeB) {
    mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  } else if (transposeA) {
    mmOpt.set("fullyConnectedPass", "TRAINING_WU");
  }

  if (!vm["available-memory-proportion"].empty()) {
    mmOpt.set("availableMemoryProportion",
              std::to_string(availableMemoryProportion));
  }
  mmOpt.set("enableFastReduce", enableFastReduce ? "true" : "false");

  if (reportPlan) {
    matMulGroupedReportPlan(std::cout, graph, inputType, outputType, {g, m, k},
                            {g, k, n}, mmOpt, &cache);
  }

  auto matA =
      createMatMulGroupedInputLHS(graph, inputType, outputType, {g, m, k},
                                  {g, k, n}, "matA", mmOpt, &cache);
  if (transposeA) {
    matA = matA.dimShufflePartial({1, 2}, {2, 1});
  }

  auto matB =
      createMatMulGroupedInputRHS(graph, inputType, outputType, {g, m, k},
                                  {g, k, n}, "matB", mmOpt, &cache);
  if (transposeB) {
    matB = matB.dimShufflePartial({1, 2}, {2, 1});
  }

  auto outerProg = Sequence();
  auto prog = Sequence();

  auto matLhs = transposeA ? matA.dimShufflePartial({1, 2}, {2, 1}) : matA;
  auto matRhs = transposeB ? matB.dimShufflePartial({1, 2}, {2, 1}) : matB;

  auto matAxB = matMulGrouped(graph, matLhs, matRhs, prog, outputType,
                              "op(A) x op(B)", mmOpt, &cache);

  auto matC = graph.clone(outputType, matAxB, "matC");
  // inner repeat loop copies the source and performs the multiply and
  // accumulate. Only the final result is returned to avoid repeated
  // accumulation saturating the result
  Tensor matD;
  if (numExecutions > 1) {
    matD = graph.clone(outputType, matC, "matD");
    prog.add(Copy(matC, matD));
  } else {
    matD = matC;
  }

  scaledAddTo(graph, matD, beta, matAxB, alpha, prog);
  outerProg.add(Repeat(numExecutions, prog));

  if (numExecutions > 1)
    outerProg.add(Copy(matD, matC));

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostMatA = allocateHostMemoryForTensor(
      matA, "matA", graph, uploadProg, downloadProg, tmap);
  auto rawHostMatB = allocateHostMemoryForTensor(
      matB, "matB", graph, uploadProg, downloadProg, tmap);
  auto rawHostMatC = allocateHostMemoryForTensor(
      matC, "matC", graph, uploadProg, downloadProg, tmap);

  auto engineOptions = defaultEngineOptions;
  if (profile || jsonProfileOut) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (profileFormat) {
      engineOptions.set("profiler.format", *profileFormat);
    }
  }

  Sequence ctrlProg;
  if (!ignoreData) {
    ctrlProg.add(uploadProg);
  }
  ctrlProg.add(outerProg);
  if (!ignoreData) {
    ctrlProg.add(downloadProg);
  }

  Engine engine(graph, ctrlProg, engineOptions);

  if (vm.count("compile-only"))
    return 0;

  boost::multi_array<double, 3> hostMatC(boost::extents[g][m][n]);
  boost::multi_array<double, 3> refMatC(boost::extents[g][m][n]);
  if (!ignoreData) {
    boost::multi_array<double, 3> hostMatA(
        boost::extents[g][rowsMatA][colsMatA]);
    boost::multi_array<double, 3> hostMatB(
        boost::extents[g][rowsMatB][colsMatB]);

    attachStreams(engine, tmap);

    std::mt19937 randomEngine;
    writeRandomValues(target, inputType, hostMatA, -4.0, 4.0, randomEngine);
    writeRandomValues(target, inputType, hostMatB, -3.0, 3.0, randomEngine);
    writeRandomValues(target, inputType, hostMatC, -2.0, 2.0, randomEngine);

    // validate against a reference model
    poplibs_test::gemm::generalGroupedMatrixMultiply(
        hostMatA, hostMatB, hostMatC, refMatC, alpha, beta, transposeA,
        transposeB);

    copy(target, hostMatA, inputType, rawHostMatA.get());
    copy(target, hostMatB, inputType, rawHostMatB.get());
    copy(target, hostMatC, outputType, rawHostMatC.get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0); // matrix operation
  });

  bool matchesModel = true;
  if (!ignoreData) {
    copy(target, outputType, rawHostMatC.get(), hostMatC);

    matchesModel = checkIsClose("gemm", hostMatC, refMatC, relativeTolerance,
                                absoluteTolerance);
  }

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();

    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }

  if (profile) {
    engine.printProfileSummary(
        std::cout,
        {{"showExecutionSteps", showExecutionSteps ? "true" : "false"},
         {"showVarStorage", showVarStorage ? "true" : "false"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
}
