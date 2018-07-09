#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>
#include <cassert>
#include <exception>
#include <istream>
#include <ostream>
#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Reduce.hpp>
#include <popconv/codelets.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
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
    throw poputil::poplib_error("Invalid pass <" + token + ">");
  return is;
}

std::ostream &operator<<(std::ostream &os, const MatrixOp &op) {
  return os << asString(op);
}

const OptionFlags engineOptions {
  {"target.textSectionSizeInBytes", "0xe000"},
  {"target.workerStackSizeInBytes", "0x180"}
};

const OptionFlags simDebugOptions {
  {"debug.trace", "false"}
};

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  // Operation performed is is alpha * op(matA) x op(matB) + beta * matC
  // where  op(matA)  is a m x k matrix
  //        op(matB)  is a k x n matrix
  unsigned m, k, n;
  float alpha, beta;
  Type dataType;
  Type partialsType;
  double relativeTolerance, absoluteTolerance;
  MatrixOp matAOp = MatrixOp::NORMAL;
  MatrixOp matBOp = MatrixOp::NORMAL;
  DeviceType deviceType = DeviceType::IpuModel;
  IPUModel ipuModel;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
      po::value<DeviceType>(&deviceType)->default_value(deviceType),
      "Device type: Cpu | Sim | Hw | IpuModel")
    ("m", po::value<unsigned>(&m)->required(),
     "Number of rows of left matrix, left-matrix-op(A)")
    ("k", po::value<unsigned>(&k)->required(),
     "Number of columns of left matrix left-matrix-op(A) and number of rows of "
     "right matrix right-matrix-op(B)")
    ("n",  po::value<unsigned>(&n)->required(),
      "Number of columns of the right matrix right-matrix-op(B)")
    ("data-type",
      po::value<Type>(&dataType)->default_value(HALF),
      "Input and output data type")
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
     po::value<unsigned>(&ipuModel.tilesPerIPU)->
                           default_value(ipuModel.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&ipuModel.numIPUs)->
                           default_value(ipuModel.numIPUs),
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

  if (dataType == FLOAT) {
    absoluteTolerance = FLOAT_ABS_TOL;
    relativeTolerance = FLOAT_REL_TOL;
  } else {
    absoluteTolerance = HALF_ABS_TOL;
    relativeTolerance = HALF_REL_TOL;
  }
  if (beta != 1.0) {
    throw poputil::poplib_error("Only beta = 1.0 is supported");
  }
  auto device = createTestDevice(deviceType, ipuModel.numIPUs,
                                  ipuModel.tilesPerIPU, simDebugOptions);

  const auto &target = device.getTarget();
  Graph graph(device);
  popconv::addCodelets(graph);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);

  const bool transposeA = matAOp == MatrixOp::TRANSPOSE;
  const bool transposeB = matBOp == MatrixOp::TRANSPOSE;

  /* set up row and column dimensions for the right and left matrix */
  const auto rowsMatA = transposeA ? k : m;
  const auto colsMatA = transposeA ? m : k;
  const auto rowsMatB = transposeB ? n : k;
  const auto colsMatB = transposeB ? k : n;

  PlanningCache cache;
  poplar::OptionFlags mmOpt{
    { "partialsType", partialsType.toString() }
  };
  if (transposeB) {
    mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  } else if (transposeA) {
    mmOpt.set("fullyConnectedPass", "TRAINING_WU");
  }
  auto matA = createMatMulInputLHS(graph, dataType,
                                   {m, k}, {k, n}, "matA", mmOpt, &cache);
  if (transposeA)
    matA = matA.transpose();

  auto matB = createMatMulInputRHS(graph, dataType,
                                   {m, k},
                                   {k, n},
                                   "matB", mmOpt, &cache);
  if (transposeB)
    matB = matB.transpose();

  auto prog = Sequence();

  auto matAxB = matMul(graph,
                       transposeA ? matA.transpose() : matA,
                       transposeB ? matB.transpose() : matB,
                       prog, "op(A) x op(B)", mmOpt, &cache);

  auto matC = graph.addVariable(dataType, {m, n}, "matC");
  mapTensorLinearly(graph, matC);

  if (matC.shape() != matAxB.shape()) {
    std::cerr << "Output matrix shape doesn't match expected shape\n";
    return 1;
  }

  scaledAddTo(graph, matC, matAxB, alpha, prog);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawHostMatA = allocateHostMemoryForTensor(matA, "matA", graph, tmap);
  auto rawHostMatB = allocateHostMemoryForTensor(matB, "matB", graph, tmap);
  auto rawHostMatC = allocateHostMemoryForTensor(matC, "matC", graph, tmap);

  Engine engine(device, graph, prog, engineOptions);

  boost::multi_array<double, 2>
      hostMatA(boost::extents[rowsMatA][colsMatA]);
  boost::multi_array<double, 2>
      hostMatB(boost::extents[rowsMatB][colsMatB]);
  boost::multi_array<double, 2>
      hostMatC(boost::extents[m][n]);
  std::mt19937 randomEngine;
  writeRandomValues(target, dataType, hostMatA, -4.0, 4.0, randomEngine);
  writeRandomValues(target, dataType, hostMatB, -3.0, 3.0, randomEngine);
  writeRandomValues(target, dataType, hostMatC, -2.0, 2.0, randomEngine);

  // validate against a reference model
  boost::multi_array<double, 2> refMatC(boost::extents[m][n]);
  poplibs_test::gemm::generalMatrixMultiply(hostMatA, hostMatB, hostMatC,
                                           refMatC, alpha, beta, transposeA,
                                           transposeB);

  copy(target, hostMatA, dataType, rawHostMatA.get());
  copy(target, hostMatB, dataType, rawHostMatB.get());
  copy(target, hostMatC, dataType, rawHostMatC.get());

  upload(engine, tmap);
  engine.run(0);    // matrix operation
  download(engine, tmap);

  copy(target, dataType, rawHostMatC.get(), hostMatC);

  const bool matchesModel = checkIsClose("gemm", hostMatC, refMatC,
                                         relativeTolerance, absoluteTolerance);
  if (deviceType == DeviceType::IpuModel) {
    engine.printSummary(std::cout, OptionFlags{
      { "doLayerWiseBreakdown", "true" }
    });
  }
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
