#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <cassert>
#include <exception>
#include <istream>
#include <ostream>
#include <fstream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Reduce.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplibs_test/Util.hpp>
#include "TestDevice.hpp"
#include <poplibs_support/Compiler.hpp>
#include <poputil/exceptions.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poplin;
using namespace poputil;
using namespace popops;

// Default tolerances used in tests
#define FLOAT_REL_TOL  0.1
#define HALF_REL_TOL   0.3
#define FLOAT_ABS_TOL  1e-5
#define HALF_ABS_TOL   7e-2

// Class to specify matrix operation
enum class MatrixOp {
  NORMAL,
  TRANSPOSE
};


const char *asString(const MatrixOp &op) {
  switch (op) {
  case MatrixOp::NORMAL: return "normal";
  case MatrixOp::TRANSPOSE: return "transpose";
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

const OptionFlags defaultEngineOptions {
  {"target.workerStackSizeInBytes", "0x180"}
};

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  // Operation performed is is alpha * op(matA) x op(matB) + beta * matC
  // where  op(matA)  is a m x k matrix
  //        op(matB)  is a k x n matrix
  unsigned m, k, n;
  float alpha, beta;
  Type inputType;
  Type outputType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  MatrixOp matAOp = MatrixOp::NORMAL;
  MatrixOp matBOp = MatrixOp::NORMAL;
  DeviceType deviceType = DeviceType::IpuModel;
  unsigned tempMemoryBudget;
  unsigned cycleBackoffPercent;
  double maxOutputMemoryProportion;
  unsigned numIPUs;
  unsigned tilesPerIPU;
  // create an IPUModel to get the default values out. do it in a scope so that
  // it isn't mistaken for the IPUModel that is actually used by the tool.
  {
    IPUModel defaultModel;
    numIPUs = defaultModel.numIPUs;
    tilesPerIPU = defaultModel.tilesPerIPU;
  }

  boost::optional<std::string> jsonProfileOut;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(deviceType),
      "Device type: Cpu | Sim | Hw | IpuModel")
    ("profile", "Output profiling report to standard output")
    ("profile-json",
     po::value<decltype(jsonProfileOut)>(&jsonProfileOut)
      ->default_value(boost::none),
     "Write the profile report as JSON to the specified file.")
    ("ignore-data", "Don't upload and download the results from the device. "
     "Note that this means the result is not validated against the model.")
    ("m", po::value<unsigned>(&m)->required(),
     "Number of rows of left matrix, left-matrix-op(A)")
    ("k", po::value<unsigned>(&k)->required(),
     "Number of columns of left matrix left-matrix-op(A) and number of rows of "
     "right matrix right-matrix-op(B)")
    ("n",  po::value<unsigned>(&n)->required(),
      "Number of columns of the right matrix right-matrix-op(B)")
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
     po::value<Type>(&partialsType)->default_value(FLOAT),
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
     po::value<unsigned>(&tilesPerIPU)->default_value(tilesPerIPU),
     "Number of tiles per IPU")
    ("temp-memory-budget",
     po::value<unsigned>(&tempMemoryBudget)->default_value(tempMemoryBudget),
     "Temporary memory budget for matmul in bytes per-tile")
    ("cycle-backoff-percent",
     po::value<unsigned>(&cycleBackoffPercent)
      ->default_value(cycleBackoffPercent),
     "Percentage of best possible matmul cycles to trade for possible memory "
     "savings")
    ("max-output-memory-proportion",
     po::value<double>(&maxOutputMemoryProportion)
      ->default_value(maxOutputMemoryProportion),
     "Proportion of memory outputs from the matmul may take up before "
     "serializing matmul by output channels")
    ("report-plan", "Show plan info")
    ("show-execution-steps", "Show execution steps (requires profiling)")
    ("show-var-storage", "Show variable liveness (requires profiling)")
    ("ipus",
     po::value<unsigned>(&numIPUs)->default_value(numIPUs),
     "Number of IPUs")
  ;
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
    po::notify(vm);
  } catch (std::exception& e) {
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

  if (beta != 1.0) {
    throw poputil::poplibs_error("Only beta = 1.0 is supported");
  }

  const bool profile = deviceType != DeviceType::Cpu && vm.count("profile");
  const bool reportPlan = vm.count("report-plan");
  const bool showExecutionSteps = vm.count("show-execution-steps");
  const bool showVarStorage = vm.count("show-var-storage");
  const bool ignoreData = vm.count("ignore-data");

  const bool compileIPUCode = true;
  auto device = createTestDevice(deviceType,
                                 numIPUs,
                                 tilesPerIPU,
                                 compileIPUCode);

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

  matmul::PlanningCache cache;
  poplar::OptionFlags mmOpt{
    { "partialsType", partialsType.toString() }
  };
  if (transposeB) {
    mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  } else if (transposeA) {
    mmOpt.set("fullyConnectedPass", "TRAINING_WU");
  }

  if (!vm["temp-memory-budget"].empty()) {
    mmOpt.set("tempMemoryBudget", std::to_string(tempMemoryBudget));
  }
  if (!vm["cycle-backoff-percent"].empty()) {
    mmOpt.set("cycleBackoffPercent", std::to_string(cycleBackoffPercent));
  }
  if (!vm["max-output-memory-proportion"].empty()) {
    mmOpt.set("maxOutputMemoryProportion",
              std::to_string(maxOutputMemoryProportion));
  }

  auto matA = createMatMulInputLHS(
    graph, inputType, outputType, {m, k}, {k, n}, "matA", mmOpt, &cache);
  if (transposeA) {
    matA = matA.transpose();
  }

  auto matB = createMatMulInputRHS(
    graph, inputType, outputType, {m, k}, {k, n}, "matB", mmOpt, &cache);
  if (transposeB) {
    matB = matB.transpose();
  }

  auto prog = Sequence();

  auto matLhs = transposeA ? matA.transpose() : matA;
  auto matRhs = transposeB ? matB.transpose() : matB;

  if(reportPlan) {
    matMulReportPlan(std::cout,
                     graph,
                     inputType,
                     outputType,
                     matLhs.shape(),
                     matRhs.shape(),
                     mmOpt,
                     &cache);
  }
  auto matAxB = matMul(graph,
                       matLhs,
                       matRhs,
                       prog,
                       outputType,
                       "op(A) x op(B)",
                       mmOpt,
                       &cache);

  auto matC = graph.clone(outputType, matAxB, "matC");
  scaledAddTo(graph, matC, matAxB, alpha, prog);

  Sequence uploadProg, downloadProg;
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostMatA = allocateHostMemoryForTensor(matA, "matA", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);
  auto rawHostMatB = allocateHostMemoryForTensor(matB, "matB", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);
  auto rawHostMatC = allocateHostMemoryForTensor(matC, "matC", graph,
                                                 uploadProg, downloadProg,
                                                 tmap);

  auto engineOptions = defaultEngineOptions;
  if (profile || jsonProfileOut) {
    engineOptions.set("debug.executionProfile", "compute_sets");
  }

  Sequence ctrlProg;
  if (!ignoreData) {
    ctrlProg.add(uploadProg);
  }
  ctrlProg.add(prog);
  if (!ignoreData) {
    ctrlProg.add(downloadProg);
  }

  Engine engine(graph, ctrlProg, engineOptions);

  boost::multi_array<double, 2> hostMatC(boost::extents[m][n]);
  boost::multi_array<double, 2> refMatC(boost::extents[m][n]);
  if (!ignoreData) {
    boost::multi_array<double, 2> hostMatA(boost::extents[rowsMatA][colsMatA]);
    boost::multi_array<double, 2> hostMatB(boost::extents[rowsMatB][colsMatB]);

    attachStreams(engine, tmap);

    std::mt19937 randomEngine;
    writeRandomValues(target, inputType, hostMatA, -4.0, 4.0, randomEngine);
    writeRandomValues(target, inputType, hostMatB, -3.0, 3.0, randomEngine);
    writeRandomValues(target, inputType, hostMatC, -2.0, 2.0, randomEngine);

    // validate against a reference model
    poplibs_test::gemm::generalMatrixMultiply(hostMatA, hostMatB, hostMatC,
                                             refMatC, alpha, beta, transposeA,
                                             transposeB);

    copy(target, hostMatA, inputType, rawHostMatA.get());
    copy(target, hostMatB, inputType, rawHostMatB.get());
    copy(target, hostMatC, outputType, rawHostMatC.get());
  }

  device.bind([&](const Device &d) {
    engine.load(d);
    engine.run(0);    // matrix operation
  });

  bool matchesModel = true;
  if (!ignoreData) {
    copy(target, outputType, rawHostMatC.get(), hostMatC);

    matchesModel = checkIsClose("gemm", hostMatC, refMatC,
                                relativeTolerance, absoluteTolerance);
  }

  if (jsonProfileOut) {
    const auto pr = engine.getProfile();

    std::ofstream os(*jsonProfileOut);
    poplar::serializeToJSON(os, pr);
  }

  if (profile) {
    engine.printProfileSummary(std::cout, {
      {"showExecutionSteps", showExecutionSteps ? "true" : "false"} ,
      {"showVarStorage", showVarStorage ? "true" : "false"}
    });
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
}
