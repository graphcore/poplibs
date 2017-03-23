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
#include <popnn/ActivationMapping.hpp>
#include <popnn/MatMul.hpp>
#include <popnn/Add.hpp>
#include <popnn/Reduce.hpp>
#include <poplar/HalfFloat.hpp>
#include <popnn/codelets.hpp>
#include <popnn_ref/Util.hpp>
#include <popnn/Compiler.hpp>
#include <popnn/exceptions.hpp>
#include <popnn_ref/GeneralMatrixMultiply.hpp>
#include <random>

using namespace poplar;
using namespace poplar::program;
using namespace ref::util;
using namespace poplin;
using namespace popstd;


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
  POPNN_UNREACHABLE();
}

std::istream &operator>>(std::istream &is, MatrixOp &op) {
  std::string token;
  is >> token;
  if (token == "normal")
    op = MatrixOp::NORMAL;
  else if (token == "transpose")
    op = MatrixOp::TRANSPOSE;
  else
    throw popnn::popnn_error("Invalid pass <" + token + ">");
  return is;
}

std::ostream &operator<<(std::ostream &os, const MatrixOp &op) {
  return os << asString(op);
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  // Operation performed is is alpha * op(matA) x op(matB) + beta * matC
  // where  op(matA)  is a m x k matrix
  //        op(matB)  is a k x n matrix
  unsigned m, k, n;
  float alpha, beta;
  FPDataType dataType;
  FPDataType partialsType;
  double relativeTolerance;
  MatrixOp matAOp = MatrixOp::NORMAL;
  MatrixOp matBOp = MatrixOp::NORMAL;


  DeviceInfo info;
  info.IPUExchangeType =
      DeviceInfo::ExchangeType::BARE_NAKED_WITH_AGGRESSIVE_MULTICAST;

  po::options_description desc("Options");
  desc.add_options()
    ("help", "Produce help message")
    ("m", po::value<unsigned>(&m)->required(),
     "Number of rows of left matrix, left-matrix-op(A)")
    ("k", po::value<unsigned>(&k)->required(),
     "Number of columns of left matrix left-matrix-op(A) and number of rows of "
     "right matrix right-matrix-op(B)")
    ("n",  po::value<unsigned>(&n)->required(),
      "Number of columns of the right matrix right-matrix-op(B)")
    ("data-type",
      po::value<FPDataType>(&dataType)->default_value(FPDataType::HALF),
      "Input and output data type")
    ("partials-type",
     po::value<FPDataType>(&partialsType)->default_value(FPDataType::FLOAT),
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
    ("tolerance", po::value<double>(&relativeTolerance)->default_value(0.01),
     "Relative tolerance to use when validating results against the reference "
     "model")
    ("tiles-per-ipu",
     po::value<unsigned>(&info.tilesPerIPU)->default_value(info.tilesPerIPU),
     "Number of tiles per IPU")
    ("ipus",
     po::value<unsigned>(&info.numIPUs)->default_value(info.numIPUs),
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

  if (beta != 1.0) {
    throw popnn::popnn_error("Only beta = 1.0 is supported");
  }

  Graph graph(createIPUModelDevice(info));
  popnn::addCodelets(graph);

  std::string dataTypeStr(asString(dataType));
  std::string partialsTypeStr(asString(partialsType));

  const bool transposeA = matAOp == MatrixOp::TRANSPOSE;
  const bool transposeB = matBOp == MatrixOp::TRANSPOSE;

  /* set up row and column dimensions for the right and left matrix */
  const auto rowsMatA = transposeA ? k : m;
  const auto colsMatA = transposeA ? m : k;
  const auto rowsMatB = transposeB ? n : k;
  const auto colsMatB = transposeB ? k : n;

  auto matB = graph.addTensor(dataTypeStr,  {rowsMatB, colsMatB}, "matB");
  mapActivations(graph, matB);

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = !transposeA;
  mmOpt.cache = &cache;

  auto matA = createMatMulInputA(graph, dataTypeStr,
                                 {rowsMatA, colsMatA},
                                 transposeB ? matB.transpose() : matB,
                                 "matA", mmOpt);

  auto prog = Sequence();

  auto matAxB = matMul(graph,
                       transposeA ? matA.transpose() : matA,
                       transposeB ? matB.transpose() : matB,
                       prog, "op(A) x op(B)", mmOpt);

  auto matC = graph.addTensor(dataTypeStr, {m, n}, "matC");
  mapActivations(graph, matC);

  addTo(graph, matC, matAxB, alpha, prog);

  auto upload = Sequence();
  auto download = Sequence();
  auto rawHostMatA = allocateHostMemoryForTensor(graph, matA, upload, download);
  auto rawHostMatB = allocateHostMemoryForTensor(graph, matB, upload, download);
  auto rawHostMatC = allocateHostMemoryForTensor(graph, matC, upload, download);

  Engine engine(graph, {std::move(upload),
                        std::move(download),
                        std::move(prog)});

  boost::multi_array<double, 2>
      hostMatA(boost::extents[rowsMatA][colsMatA]);
  boost::multi_array<double, 2>
      hostMatB(boost::extents[rowsMatB][colsMatB]);
  boost::multi_array<double, 2>
      hostMatC(boost::extents[m][n]);
  std::mt19937 randomEngine;
  writeRandomValues(hostMatA, -4.0, 4.0, randomEngine);
  writeRandomValues(hostMatB, -3.0, 3.0, randomEngine);
  writeRandomValues(hostMatC, -2.0, 2.0, randomEngine);

  // validate against a reference model
  boost::multi_array<double, 2> refMatC(boost::extents[m][n]);
  ref::gemm::generalMatrixMultiply(hostMatA, hostMatB, hostMatC, refMatC,
                                   alpha, beta, transposeA, transposeB);

  copy(hostMatA, dataTypeStr, rawHostMatA.get());
  copy(hostMatB, dataTypeStr, rawHostMatB.get());
  copy(hostMatC, dataTypeStr, rawHostMatC.get());

  engine.run(0);    // Upload
  engine.run(2);    // matrix operation
  engine.run(1);    // download

  copy(dataTypeStr, rawHostMatC.get(), hostMatC);

  const bool matchesModel = checkIsClose("gemm", hostMatC, refMatC,
                                         relativeTolerance);

  Engine::ReportOptions opt;
  opt.doLayerWiseProfile = true;
  engine.report(std::cout, opt);
  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }
  return 0;
}
