// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Adds a sparse matmul on each shard and does a concatenated
// matrix multiply. Requires the matrix to be square for ease
// of specification.
#include "poplibs_support/TestDevice.hpp"

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

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

#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1

int main(int argc, char **argv) {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  unsigned numShards = 2;
  boost::optional<unsigned> tilesPerIPU;
  boost::optional<std::string> profileDir;
  // matrix multi is is mxm and mxn
  unsigned groups = 1, m = 16, n = 16;
  double sparsityFactor = 0.1;
  Type dataType = HALF;
  Type partialsType = FLOAT;

  std::string matmulOptionsString;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     deviceTypeHelp)
    ("profile", "Enable profiling and print profiling report")
    ("profile-dir",
     po::value<decltype(profileDir)>(&profileDir)
      ->default_value(boost::none),
     "Write profile files to the specified directory.")
    ("left-square-matrix-size", po::value<unsigned>(&m)->default_value(m),
     "Square matrix size for the left matrix")
    ("num-shards", po::value<unsigned>(&numShards)->default_value(numShards),
     "Number of shards (each shard on an IPU)")
    ("right-matrix-cols", po::value<unsigned>(&n)->default_value(n),
     "Right hand matrix input cols")
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
  const std::size_t blockRows = 1;
  const std::size_t blockCols = 1;
  const auto blockArea = blockRows * blockCols;

  if (m % blockRows) {
    throw poputil::poplibs_error("size of matrix must be an integer multiple "
                                 "of rows in a block");
  }

  PlanningCache cache;

  OptionFlags matmulOptions;
  // User options specified via --matmul-options override defaults
  if (!matmulOptionsString.empty()) {
    poplar::readJSON(matmulOptionsString, matmulOptions);
  }

  auto device =
      tilesPerIPU ? createTestDevice(deviceType, numShards, *tilesPerIPU, true)
                  : createTestDeviceFullSize(deviceType, numShards, true);
  const auto &target = device.getTarget();
  Graph mainGraph(target);
  popops::addCodelets(mainGraph);
  poplin::addCodelets(mainGraph);
  popsparse::addCodelets(mainGraph);
  Sequence uploadProg, prog, downloadProg;

  std::vector<poplar::Graph> shardedGraphs;

  for (unsigned i = 0; i != numShards; ++i) {
    auto shard = mainGraph.createVirtualGraph(
        i * target.getTilesPerIPU(), (i + 1) * target.getTilesPerIPU());
    shardedGraphs.push_back(std::move(shard));
  }

  const auto sparsityType =
      blockArea == 1 ? SparsityType::Element : SparsityType::Block;
  SparsityParams sparsityParams(sparsityType, SparsityStructure::Unstructured,
                                {blockRows, blockCols});
  const auto params = MatMulParams::createWithNzRatio(
      std::move(sparsityParams), sparsityFactor, groups, m, m, n);

  // No support for groups yet
  assert(groups == 1);

  std::vector<SparseTensor> lhsMM;
  // We need the first rhs and the last output but we keep all anyway
  // even if they are not used.
  std::vector<Tensor> rhsMM, outMM;
  std::vector<std::pair<std::string, char *>> tmap;
  std::vector<std::unique_ptr<char[]>> rawMetaInfoMM, rawNzInfoMM;

  if (numShards == 1) {
    std::cerr << "failure: number of shards must be greater than 1\n";
    return 1;
  }
  for (unsigned i = 0; i != numShards; ++i) {
    const auto shardNum = std::to_string(i);
    auto &shardGraph = shardedGraphs[i];
    auto lhs =
        createSparseDenseMatMulLHS(shardGraph, dataType, params,
                                   "lhsMM" + shardNum, matmulOptions, &cache);
    lhsMM.push_back(lhs);
    auto rhs =
        i == 0
            ? createSparseDenseMatMulRHS(shardGraph, dataType, params,
                                         "rhsMM" + shardNum, matmulOptions,
                                         &cache)
            : poputil::copyToIpu(
                  mainGraph, outMM.back(), prog, 1, "inter-ipu-copy" + shardNum,
                  poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    rhsMM.push_back(rhs);
    auto out = sparseDenseMatMul(shardGraph, lhs, rhs, prog, false, false,
                                 "MM" + shardNum, matmulOptions, &cache);
    outMM.push_back(out);
    std::unique_ptr<char[]> rawMetaInfo = allocateHostMemoryForTensor(
        lhs.getMetaInfoTensor(), "lhsMM" + shardNum + ".meta", mainGraph,
        uploadProg, downloadProg, tmap);
    rawMetaInfoMM.push_back(std::move(rawMetaInfo));
    std::unique_ptr<char[]> rawNzInfo = allocateHostMemoryForTensor(
        lhs.getNzValuesTensor(), "lhsMM" + shardNum + ".values", mainGraph,
        uploadProg, downloadProg, tmap);
    rawNzInfoMM.push_back(std::move(rawNzInfo));
  }

  auto rawRHS = allocateHostMemoryForTensor(rhsMM.front(), "rhsMM0", mainGraph,
                                            uploadProg, downloadProg, tmap);
  auto rawOut = allocateHostMemoryForTensor(outMM.back(), "out", mainGraph,
                                            uploadProg, downloadProg, tmap);

  Sequence controlProg({std::move(uploadProg), std::move(prog)});
  controlProg.add(downloadProg);

  using EType = float;
  Partitioner<EType> partitioner(params, dataType, target, matmulOptions,
                                 &cache);

  OptionFlags engineOptions;
  if (profilingEnabled) {
    engineOptions.set("debug.instrumentCompute", "true");
    if (profileDir) {
      engineOptions.set("autoReport.all", "true");
      engineOptions.set("autoReport.directory", *profileDir);
    }
  }
  Engine engine(mainGraph, std::move(controlProg), engineOptions);
  attachStreams(engine, tmap);

  std::mt19937 randomEngine;
  std::array<std::size_t, 2> blockDims = {blockRows, blockCols};
  std::vector<CSRMatrix<EType>> csrMatrices(numShards, blockDims);
  for (unsigned i = 0; i != numShards; ++i) {
    std::tie(csrMatrices[i].nzValues, csrMatrices[i].columnIndices,
             csrMatrices[i].rowIndices) =
        poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
            randomEngine, {m, m}, {blockRows, blockCols}, sparsityFactor,
            {0, 0}, {m, m}, 1.0, false);
  }

  boost::multi_array<double, 2> hostRHS(
      boost::extents[rhsMM.front().dim(1)][rhsMM.front().dim(2)]);
  writeRandomBinaryValues(target, dataType, hostRHS, -1.0, 1.0, randomEngine);
  boost::multi_array<double, 2> hostOut(
      boost::extents[outMM.back().dim(1)][outMM.back().dim(2)]);

  copy(target, hostRHS, dataType, rawRHS.get());
  for (unsigned i = 0; i != numShards; ++i) {
    const auto bucketsMM = partitioner.createSparsityDataImpl(csrMatrices[i]);
    copy(target, bucketsMM.metaInfo, UNSIGNED_SHORT, rawMetaInfoMM[i].get());
    copy(target, bucketsMM.nzValues, dataType, rawNzInfoMM[i].get());
  }

  device.bind([&](const Device &d) { engine.loadAndRun(d); });

  bool matchesModel = true;
  const double relTolerance = dataType == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
  copy(target, outMM[1].elementType(), rawOut.get(), hostOut);

  boost::multi_array<double, 2> modelOutPrev(
      boost::extents[outMM.front().dim(1)][outMM.front().dim(2)]);

  // Concatenated matrix multiply
  for (unsigned i = 0; i != numShards; ++i) {
    boost::multi_array<double, 2> hostDenseLHS(boost::extents[m][m]);
    hostDenseLHS = poplibs_test::sparse::csrToDenseMatrix(
        csrMatrices[i].nzValues.data(), csrMatrices[i].columnIndices.data(),
        csrMatrices[i].rowIndices.data(), csrMatrices[i].nzValues.size(), m, m,
        blockRows, blockCols);
    const auto rhsTensor = i == 0 ? rhsMM.front() : outMM[i - 1];
    boost::multi_array<double, 2> rhs(
        boost::extents[rhsTensor.dim(1)][rhsTensor.dim(2)]);
    rhs = i == 0 ? hostRHS : modelOutPrev;
    boost::multi_array<double, 2> modelOutput(
        boost::extents[outMM[i].dim(1)][outMM[i].dim(2)]);
    poplibs_test::gemm::generalMatrixMultiply(hostDenseLHS, rhs, modelOutput,
                                              false, false);
    modelOutPrev = modelOutput;
  }

  matchesModel &= checkIsClose("out", hostOut, modelOutPrev, relTolerance);

  if (profilingEnabled) {
    engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
}
