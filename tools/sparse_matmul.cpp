// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplibs_support/TestDevice.hpp"

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include "poplibs_test/TempDir.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/SparseMatrix.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/TileMapping.hpp>

#include "poplin/codelets.hpp"
#include "popops/codelets.hpp"

#include "popsparse/MatMul.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "popsparse/codelets.hpp"

#include <fstream>

// Tolerances used when data is not ignored
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1

using namespace poplar;
using namespace poplar::program;

using namespace poplibs_test::util;
using namespace poplibs_support;

using namespace popsparse;
using namespace popsparse::dynamic;
using namespace poputil;

// Tolerances used when data is not ignored
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  constexpr unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  boost::optional<std::string> profileDir;
  unsigned groups = 1, m, k, n;
  double sparsityFactor;
  Type dataType = HALF;
  Type partialsType = FLOAT;
  ShapeOption<std::size_t> weightedAreaBegin;
  ShapeOption<std::size_t> weightedAreaEnd;
  ShapeOption<std::size_t> blockSize;

  weightedAreaBegin.val = weightedAreaEnd.val = {0, 0};
  double weightedAreaWeighting = 1.0;
  bool transposeLHS = false;
  bool transposeRHS = false;
  std::string matmulOptionsString;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("compile-only", "Stop after compilation; don't run the program")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("profile", "Enable profiling and print profiling report")
    ("profile-dir",
     po::value<decltype(profileDir)>(&profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("ignore-data", "Don't validate results")
    ("data-type", po::value(&dataType)->default_value(dataType), "Data type of operands")
    ("groups", po::value(&groups)->default_value(groups), "Number of groups")
    ("m", po::value(&m)->required(), "Rows in left-hand operand")
    ("k", po::value(&k)->required(), "Columns in left-hand operand/Rows in right-hand operand")
    ("n", po::value(&n)->required(), "Columns in right-hand operand")
    ("sparsity-factor", po::value(&sparsityFactor)->required(),
     "Proportion of elements of left-hand operand which are non-zero")
    ("block-size",
     po::value<ShapeOption<std::size_t>>(&blockSize)->default_value(1),
     "Block size as rows and columns (only square blocks are supported)")
    ("transpose-lhs", po::value(&transposeLHS),
     "Transpose the left-hand operand of the matmul such that the matmul "
     "becomes {k, m} * {m, n} = {k, n}")
    ("transpose-rhs", po::value(&transposeRHS),
     "Transpose the right-hand operand of the matmul")
    ("weighted-area-begin",
     po::value<ShapeOption<std::size_t>>(&weightedAreaBegin)->default_value(weightedAreaBegin),
     "Starting indices of an area of the sparse operand with a different "
     "level of sparsity to the rest")
    ("weighted-area-end",
     po::value<ShapeOption<std::size_t>>(&weightedAreaEnd)->default_value(weightedAreaEnd),
     "Ending indices of an area of the sparse operand with a different "
     "level of sparsity to the rest")
    ("weighted-area-weighting",
     po::value<double>(&weightedAreaWeighting)->default_value(weightedAreaWeighting),
     "Weighting for probability that a sparse element resides within the "
     "specified area")
    ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
    ("matmul-options", po::value<std::string>(&matmulOptionsString),
     "Options to use for the matrix multiplication, specified as a JSON "
     "string, e.g. {\"key\":\"value\"}")
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

  bool profile = vm.count("profile");
  bool profilingEnabled = profile || profileDir;
  bool ignoreData = vm.count("ignore-data");

  const std::size_t blockRows = blockSize[0];
  const std::size_t blockCols =
      blockSize.val.size() == 1 ? blockRows : blockSize[1];
  const auto blockArea = blockRows * blockCols;

  if (m % blockRows) {
    throw poputil::poplibs_error("Input size must be an integer multiple of "
                                 "rows in a block");
  }

  if (k % blockCols) {
    throw poputil::poplibs_error("output size must be an integer multiple of "
                                 "columns in a block");
  }

  // align weighted area to a block size grid
  weightedAreaBegin.val[0] = roundDown(weightedAreaBegin.val[0], blockRows);
  weightedAreaBegin.val[1] = roundDown(weightedAreaBegin.val[1], blockCols);
  weightedAreaEnd.val[0] = roundDown(weightedAreaEnd.val[0], blockRows);
  weightedAreaEnd.val[1] = roundDown(weightedAreaEnd.val[1], blockCols);

  PlanningCache cache;

  OptionFlags matmulOptions;
  // User options specified via --matmul-options override defaults
  if (!matmulOptionsString.empty()) {
    poplar::readJSON(matmulOptionsString, matmulOptions);
  }

  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, true)
                    : createTestDeviceFullSize(deviceType, numIPUs, true);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  popsparse::addCodelets(graph);
  Sequence uploadProg, prog, downloadProg;

  const auto sparsityType =
      blockArea == 1 ? SparsityType::Element : SparsityType::Block;
  SparsityParams sparsityParams(sparsityType, SparsityStructure::Unstructured,
                                {blockRows, blockCols});
  const auto params = MatMulParams::createWithNzRatio(
      std::move(sparsityParams), sparsityFactor, groups, m, k, n);
  // No support for groups yet
  assert(groups == 1);
  const SparseTensor lhs = createSparseDenseMatMulLHS(
      graph, dataType, params, "lhs", matmulOptions, &cache);

  Tensor rhs;
  if (transposeLHS) {
    std::vector<std::size_t> rhsShape = {groups, m, n};
    if (transposeRHS) {
      std::swap(rhsShape.at(1), rhsShape.at(2));
    }
    // If the left-hand operand is to be transposed, the right-hand
    // operand becomes of shape {m, n}. We don't have a way to allocate this
    // correctly externally so just allocate linearly in this case.
    rhs = graph.addVariable(dataType, rhsShape, "rhs");
    poputil::mapTensorLinearly(graph, rhs);
  } else {
    rhs = createSparseDenseMatMulRHS(graph, dataType, params, "rhs",
                                     matmulOptions, &cache);
    if (transposeRHS) {
      rhs = rhs.dimRoll(1, 2);
    }
  }

  const Tensor out =
      sparseDenseMatMul(graph, lhs, rhs, prog, transposeLHS, transposeRHS,
                        "multiply", matmulOptions, &cache);

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawMetaInfo =
      allocateHostMemoryForTensor(lhs.getMetaInfoTensor(), "lhs.meta", graph,
                                  uploadProg, downloadProg, tmap);
  auto rawNzInfo =
      allocateHostMemoryForTensor(lhs.getNzValuesTensor(), "lhs.values", graph,
                                  uploadProg, downloadProg, tmap);
  auto rawRHS = allocateHostMemoryForTensor(rhs, "rhs", graph, uploadProg,
                                            downloadProg, tmap);
  auto rawOut = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  Sequence controlProg({std::move(uploadProg), std::move(prog)});
  if (!ignoreData) {
    controlProg.add(downloadProg);
  }

  using EType = float;
  Partitioner<EType> partitioner(params, dataType, target, matmulOptions,
                                 &cache);

  std::optional<TempDir> tempDir;
  OptionFlags engineOptions;
  if (profilingEnabled) {
    engineOptions.set("autoReport.outputExecutionProfile", "true");
    if (profileDir) {
      engineOptions.set("autoReport.directory", *profileDir);
    } else {
      tempDir.emplace(TempDir::create());
      engineOptions.set("autoReport.directory", tempDir->getPath());
    }
  }
  Engine engine(graph, std::move(controlProg), engineOptions);

  if (vm.count("compile-only"))
    return 0;

  attachStreams(engine, tmap);

  std::mt19937 randomEngine;

  const bool floatingPointCouldRepresentMaxAccum = [&] {
    const auto maxVal = maxContiguousInteger(dataType);

    double weightedThreshold, remainingThreshold;
    std::tie(weightedThreshold, remainingThreshold) =
        poplibs_test::sparse::calculateWeightedVsRemainingSparsityFactor(
            {params.getM() / blockRows, params.getK() / blockCols},
            sparsityFactor,
            {weightedAreaBegin[0] / blockRows,
             weightedAreaBegin[1] / blockCols},
            {weightedAreaEnd[0] / blockRows, weightedAreaEnd[1] / blockCols},
            weightedAreaWeighting);
    const auto numWeightedK = (weightedAreaEnd[1] - weightedAreaBegin[1]);
    const auto numWeightedM = (weightedAreaEnd[0] - weightedAreaBegin[0]);
    std::size_t maxK = numWeightedK * weightedThreshold +
                       (params.getK() - numWeightedK) * remainingThreshold;
    std::size_t maxM = numWeightedM * weightedThreshold +
                       (params.getM() - numWeightedM) * remainingThreshold;
    maxM = roundDown(maxM, blockRows);
    maxK = roundDown(maxK, blockCols);
    const auto getOpsPerOutputElementEstimate =
        [&](const bool lhsTransposed) -> int {
      const auto numAccumulations = lhsTransposed ? maxM : maxK;
      return numAccumulations;
    };
    // We use a modifier to account for the unlikeliness of picking all positive
    // or negative 1s which would actually get us to the max precisely
    // represented integer.
    constexpr int modifier = 10;
    // We use another modifier to account for the chance that sparsity is not
    // perfectly evenly spread in this instant.
    constexpr double wiggleRoom = 1.3;
    if (wiggleRoom * getOpsPerOutputElementEstimate(transposeLHS) >
        maxVal * modifier) {
      return false;
    }
    return true;
  }();
  bool useBipolarDistribution = floatingPointCouldRepresentMaxAccum;
  std::array<std::size_t, 2> blockDims = {blockRows, blockCols};
  CSRMatrix<EType> csrMatrix(blockDims);
  std::tie(csrMatrix.nzValues, csrMatrix.columnIndices, csrMatrix.rowIndices) =
      poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
          randomEngine, {m, k}, {blockRows, blockCols}, sparsityFactor,
          weightedAreaBegin, weightedAreaEnd, weightedAreaWeighting,
          useBipolarDistribution);
  boost::multi_array<double, 2> hostRHS(boost::extents[rhs.dim(1)][rhs.dim(2)]);
  if (useBipolarDistribution) {
    writeRandomBinaryValues(target, dataType, hostRHS, -1.0, 1.0, randomEngine);
  } else {
    writeRandomValues(target, dataType, hostRHS, -3.0, 3.0, randomEngine);
  }
  boost::multi_array<double, 2> hostOut(boost::extents[out.dim(1)][out.dim(2)]);

  const auto buckets = partitioner.createSparsityDataImpl(csrMatrix);
  copy(target, hostRHS, dataType, rawRHS.get());
  copy(target, buckets.metaInfo, UNSIGNED_SHORT, rawMetaInfo.get());
  copy(target, buckets.nzValues, dataType, rawNzInfo.get());

  device.bind([&](const Device &d) { engine.loadAndRun(d); });

  bool matchesModel = true;
  if (!ignoreData) {
    const double relTolerance = dataType == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
    copy(target, out.elementType(), rawOut.get(), hostOut);
    boost::multi_array<double, 2> hostDenseLHS(boost::extents[m][k]);
    boost::multi_array<double, 2> modelOut(
        boost::extents[out.dim(1)][out.dim(2)]);
    hostDenseLHS = poplibs_test::sparse::csrToDenseMatrix(
        csrMatrix.nzValues.data(), csrMatrix.columnIndices.data(),
        csrMatrix.rowIndices.data(), csrMatrix.nzValues.size(), m, k, blockRows,
        blockCols);
    poplibs_test::gemm::generalMatrixMultiply(hostDenseLHS, hostRHS, modelOut,
                                              transposeLHS, transposeRHS);
    matchesModel &= checkIsClose("out", hostOut, modelOut, relTolerance);
  }

  if (profile) {
    engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  if (!e.profilePath.empty()) {
    poplar::printGraphSummary(std::cerr, e.profilePath,
                              {{"showVarStorage", "true"}});
  }
  throw;
}
