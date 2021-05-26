// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "poplibs_support/TestDevice.hpp"

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include "poputil/exceptions.hpp"
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_test/Embedding.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/SparseMatrix.hpp>
#include <poplibs_test/Util.hpp>
#include <poputil/TileMapping.hpp>

#include "poplin/codelets.hpp"
#include "popops/codelets.hpp"

#include "popsparse/Embedding.hpp"
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

enum class Pass : std::uint8_t { FWD, WU, BOTH };

std::ostream &operator<<(std::ostream &os, const Pass p) {
  switch (p) {
  case Pass::FWD:
    return os << "fwd";
  case Pass::WU:
    return os << "wu";
  case Pass::BOTH:
    return os << "both";
  }

  throw poputil::poplibs_error("Invalid pass");
}
std::istream &operator>>(std::istream &is, Pass &p) {
  std::string token;
  is >> token;

  if (token == "fwd") {
    p = Pass::FWD;
  } else if (token == "wu") {
    p = Pass::WU;
  } else if (token == "both") {
    p = Pass::BOTH;
  } else {
    throw poputil::poplibs_error("Invalid token for pass: " + token);
  }

  return is;
}
bool passEnabled(const Pass opt, const Pass pass) {
  return opt == pass || opt == Pass::BOTH;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  constexpr unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  boost::optional<std::string> profileDir;
  unsigned groups = 1, numEntries, embeddingSize, batchSize;
  double sparsityFactor;
  Type dataType = HALF;
  Type partialsType = FLOAT;
  ShapeOption<std::size_t> weightedAreaBegin;
  ShapeOption<std::size_t> weightedAreaEnd;
  ShapeOption<std::size_t> blockSize;

  weightedAreaBegin.val = weightedAreaEnd.val = {0, 0};
  double weightedAreaWeighting = 1.0;
  std::string matmulOptionsString;
  bool testMatmul = true;

  // Embedding options
  std::size_t numIndices;
  double scale = 1.0;
  Pass pass = Pass::BOTH;
  bool debugPrint = false;
  bool randomInput = true;

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
    ("data-type", po::value(&dataType)->default_value(dataType),
      "Data type of operands")
    ("groups", po::value(&groups)->default_value(groups),
      "Number of groups")
    ("num-entries", po::value(&numEntries)->required(),
      "Number of entries in embedding matrix")
    ("embedding-size", po::value(&embeddingSize)->required(),
      "Embedding size of embedding matrix")
    ("batch-size", po::value(&batchSize)->required(),
      "Batch size used for matmul")
    ("sparsity-factor", po::value(&sparsityFactor)->required(),
     "Proportion of elements of left-hand operand which are non-zero")
    ("block-size",
     po::value<ShapeOption<std::size_t>>(&blockSize)->default_value(1),
     "Block size as rows and columns (only square blocks are supported)")
    ("weighted-area-begin",
     po::value<ShapeOption<std::size_t>>
        (&weightedAreaBegin)->default_value(weightedAreaBegin),
     "Starting indices of an area of the sparse operand with a different "
     "level of sparsity to the rest")
    ("weighted-area-end",
     po::value<ShapeOption<std::size_t>>
        (&weightedAreaEnd)->default_value(weightedAreaEnd),
     "Ending indices of an area of the sparse operand with a different "
     "level of sparsity to the rest")
    ("weighted-area-weighting",
     po::value<double>
        (&weightedAreaWeighting)->default_value(weightedAreaWeighting),
     "Weighting for probability that a sparse element resides within the "
     "specified area (Not itself a probability - values > 1 are OK)")
    ("tiles-per-ipu", po::value(&tilesPerIPU), "Number of tiles per IPU")
    ("matmul-options", po::value<std::string>(&matmulOptionsString),
     "Options to use for the matrix multiplication, specified as a JSON "
     "string, e.g. {\"key\":\"value\"}")
    ("test-matmul", po::value(&testMatmul)->default_value(testMatmul),
     "Run the matmul and fetch/compare results.  Can be disabled to test "
     "block sizes that are allowed for embedding but not currently for matmul")
    // Embedding options
    ("num-indices",
     po::value<std::size_t>(&numIndices)->required(),
     "The number of indices to use for the embedding layer")
   ("pass",
     po::value<Pass>(&pass)->default_value(pass),
     "Which pass of the embedding layer to perform: fwd | wu | both")
    ("scale",
      po::value<double>(&scale)->default_value(scale),
      "The scale to use in applying a weight update")
    ("debug-print",
     po::value<bool>(&debugPrint)->default_value(debugPrint),
     "Print out debug information")
    ("random-input",
     po::value<bool>(&randomInput)->default_value(randomInput),
     "Use random, or counting input")
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

  if (numEntries % blockRows) {
    throw poputil::poplibs_error("--num-entries must be an integer multiple of "
                                 "rows in a block");
  }

  if (embeddingSize % blockCols) {
    throw poputil::poplibs_error(
        "--embedding-size must be an integer multiple of "
        "columns in a block");
  }

  // align weighted area to a block size grid
  weightedAreaBegin.val[0] = roundDown(weightedAreaBegin.val[0], blockRows);
  weightedAreaBegin.val[1] = roundDown(weightedAreaBegin.val[1], blockCols);
  weightedAreaEnd.val[0] = roundDown(weightedAreaEnd.val[0], blockRows);
  weightedAreaEnd.val[1] = roundDown(weightedAreaEnd.val[1], blockCols);

  PlanningCache cache;

  OptionFlags matmulOptions;
  // Plan and generate meta info for the gradW pass if we want to run the
  // embedding update.  This will produce the metainfo for the fully connected
  // gradW pass, which is used in embedding update vertex to divide work.
  // However we only run `fullyConnectedFwd`
  matmulOptions.set("doGradWPass",
                    passEnabled(pass, Pass::WU) ? "true" : "false");
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
  const auto params = FullyConnectedParams::createWithNzRatio(
      sparsityParams, sparsityFactor, batchSize, groups, embeddingSize,
      numEntries);
  // No support for groups yet
  assert(groups == 1);
  const SparseTensor embeddingMatrix = createFullyConnectedWeights(
      graph, dataType, params, "embeddingMatrix", matmulOptions, &cache);

  const auto mulInput = createFullyConnectedInput(
      graph, dataType, params, "mulInput", matmulOptions, &cache);
  Tensor mulOutput;
  if (testMatmul) {
    mulOutput = fullyConnectedFwd(graph, embeddingMatrix, mulInput, params,
                                  prog, "multiply", matmulOptions, &cache);
  }
  std::vector<std::pair<std::string, char *>> tmap;
  auto rawMetaInfo = allocateHostMemoryForTensor(
      embeddingMatrix.getMetaInfoTensor(), "embedding.meta", graph, uploadProg,
      downloadProg, tmap);
  auto rawNzInfo = allocateHostMemoryForTensor(
      embeddingMatrix.getNzValuesTensor(), "embedding.values", graph,
      uploadProg, downloadProg, tmap);
  auto rawMulInput = allocateHostMemoryForTensor(
      mulInput, "mulInput", graph, uploadProg, downloadProg, tmap);

  std::unique_ptr<char[]> rawOut;
  if (testMatmul) {
    rawOut = allocateHostMemoryForTensor(mulOutput, "mulOutput", graph,
                                         uploadProg, downloadProg, tmap);
  }
  using EType = float;
  Partitioner<EType> partitioner(params, dataType, target, matmulOptions,
                                 &cache);

  // Create indices for the slice
  std::mt19937 randomEngine;
  auto indices =
      createIndicesTensor(graph, params, numIndices, matmulOptions, "indices");
  std::vector<unsigned> hostIndices(indices.numElements());
  writeRandomValues(target, UNSIGNED_INT, hostIndices, 0u, numEntries - 1,
                    randomEngine);

  boost::multi_array<double, 2> hostSlicedInput(
      boost::extents[indices.dim(0)][embeddingSize]);
  if (randomInput) {
    writeRandomBinaryValues(target, dataType, hostSlicedInput, -1.0, 1.0,
                            randomEngine);
  } else {
    // Easier to debug with unique values for slice
    double count = -1.0;
    for (unsigned i = 0; i < indices.dim(0); i++) {
      for (unsigned j = 0; j < embeddingSize; j++) {
        hostSlicedInput[i][j] = count--;
      }
    }
  }

  Tensor slicedResult;
  if (passEnabled(pass, Pass::FWD)) {
    slicedResult = embeddingSlice(graph, embeddingMatrix, indices, prog, params,
                                  "sliceEmbedding", matmulOptions, &cache);
  }
  Tensor updateSlices;
  if (passEnabled(pass, Pass::WU)) {
    updateSlices = createSliceTensor(graph, dataType, params, indices.dim(0),
                                     "sliceTensor", matmulOptions, &cache);
    auto scaleT = graph.addConstant(FLOAT, {}, scale);
    graph.setTileMapping(scaleT, 0);
    embeddingUpdateAdd(graph, embeddingMatrix, updateSlices, indices, scaleT,
                       prog, params, "updateEmbedding", matmulOptions, &cache);
  }

  auto rawIndices = allocateHostMemoryForTensor(indices, "indices", graph,
                                                uploadProg, downloadProg, tmap);
  std::unique_ptr<char[]> rawSlicedResult, rawUpdateSlices;
  if (passEnabled(pass, Pass::FWD)) {
    rawSlicedResult = allocateHostMemoryForTensor(
        slicedResult, "slicedResult", graph, uploadProg, downloadProg, tmap);
  }
  if (passEnabled(pass, Pass::WU)) {
    rawUpdateSlices = allocateHostMemoryForTensor(
        updateSlices, "updateSlices", graph, uploadProg, downloadProg, tmap);
  }
  Sequence controlProg({std::move(uploadProg), std::move(prog)});
  if (!ignoreData) {
    controlProg.add(downloadProg);
  }

  OptionFlags engineOptions;
  if (profilingEnabled) {
    engineOptions.set("debug.instrument", "true");
    if (profileDir) {
      engineOptions.set("autoReport.all", "true");
      engineOptions.set("autoReport.directory", *profileDir);
    }
  }
  Engine engine(graph, std::move(controlProg), engineOptions);

  if (vm.count("compile-only"))
    return 0;

  attachStreams(engine, tmap);

  const bool floatingPointCouldRepresentMaxAccum = [&] {
    const auto maxVal = maxContiguousInteger(dataType);

    double weightedThreshold, remainingThreshold;
    std::tie(weightedThreshold, remainingThreshold) =
        poplibs_test::sparse::calculateWeightedVsRemainingSparsityFactor(
            {numEntries / blockRows, embeddingSize / blockCols}, sparsityFactor,
            {weightedAreaBegin[0] / blockRows,
             weightedAreaBegin[1] / blockCols},
            {weightedAreaEnd[0] / blockRows, weightedAreaEnd[1] / blockCols},
            weightedAreaWeighting);
    const auto numWeightedEmbeddingSize =
        (weightedAreaEnd[1] - weightedAreaBegin[1]);
    const auto numWeightedEntries = (weightedAreaEnd[0] - weightedAreaBegin[0]);
    std::size_t maxEmbeddingSize =
        numWeightedEmbeddingSize * weightedThreshold +
        (embeddingSize - numWeightedEmbeddingSize) * remainingThreshold;
    std::size_t maxEntries =
        numWeightedEntries * weightedThreshold +
        (numEntries - numWeightedEntries) * remainingThreshold;
    maxEntries = roundDown(maxEntries, blockRows);
    maxEmbeddingSize = roundDown(maxEmbeddingSize, blockCols);
    const auto getOpsPerOutputElementEstimate = [&](const Pass &pass) -> int {
      // TODO: number of accumulations in weight update should be
      // given by the likely maximum number of indices that will be
      // equal.
      const auto numAccumulations = pass == Pass::FWD ? maxEmbeddingSize : 0;
      return numAccumulations;
    };
    // We use a modifier to account for the unlikeliness of picking all positive
    // or negative 1s which would actually get us to the max precisely
    // represented integer.
    constexpr int modifier = 10;
    // We use another modifier to account for the chance that sparsity is not
    // perfectly evenly spread in this instant.
    constexpr double wiggleRoom = 1.3;
    if (passEnabled(pass, Pass::FWD) &&
        wiggleRoom * getOpsPerOutputElementEstimate(Pass::FWD) >
            maxVal * modifier) {
      return false;
    }
    if (passEnabled(pass, Pass::WU) &&
        wiggleRoom * getOpsPerOutputElementEstimate(Pass::WU) >
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
          randomEngine, {numEntries, embeddingSize}, {blockRows, blockCols},
          sparsityFactor, weightedAreaBegin, weightedAreaEnd,
          weightedAreaWeighting, useBipolarDistribution);
  boost::multi_array<double, 2> hostMulInput(
      boost::extents[mulInput.dim(0)][mulInput.dim(1)]);
  if (useBipolarDistribution) {
    writeRandomBinaryValues(target, dataType, hostMulInput, -1.0, 1.0,
                            randomEngine);
  } else {
    writeRandomValues(target, dataType, hostMulInput, -3.0, 3.0, randomEngine);
  }
  boost::multi_array<double, 2> hostMulOut(
      boost::extents[batchSize][numEntries]);
  if (!randomInput) {
    // Create unique values for debug in the embedding layer
    for (unsigned i = 0; i < csrMatrix.nzValues.size(); i++) {
      csrMatrix.nzValues[i] = i + 1;
    }
  }
  const auto buckets = partitioner.createSparsityDataImpl(csrMatrix);

  // Matmul input
  copy(target, hostMulInput, dataType, rawMulInput.get());
  // Sparse data input to matmul and to embedding. Nz values are updated
  // if we use update
  copy(target, buckets.metaInfo, UNSIGNED_SHORT, rawMetaInfo.get());
  copy(target, buckets.nzValues, dataType, rawNzInfo.get());
  // Embedding inputs
  copy(target, hostIndices, UNSIGNED_INT, rawIndices.get());
  if (passEnabled(pass, Pass::WU)) {
    copy(target, hostSlicedInput, dataType, rawUpdateSlices.get());
  }
  device.bind([&](const Device &d) { engine.loadAndRun(d); });
  bool matchesModel = true;
  if (!ignoreData) {
    const double relTolerance = dataType == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
    if (testMatmul) {
      copy(target, mulOutput.elementType(), rawOut.get(), hostMulOut);
    }
    boost::multi_array<double, 2> hostDenseEmbeddingMatrix(
        boost::extents[numEntries][embeddingSize]);
    boost::multi_array<double, 2> modelMulOut(
        boost::extents[batchSize][numEntries]);
    hostDenseEmbeddingMatrix = poplibs_test::sparse::csrToDenseMatrix(
        csrMatrix.nzValues.data(), csrMatrix.columnIndices.data(),
        csrMatrix.rowIndices.data(), csrMatrix.nzValues.size(), numEntries,
        embeddingSize, blockRows, blockCols);
    poplibs_test::gemm::generalMatrixMultiply(
        hostMulInput, hostDenseEmbeddingMatrix, modelMulOut, false, true);

    // To check, for embedding slice, fetch the Slice values
    boost::multi_array<double, 2> hostSlicedResult(
        boost::extents[indices.dim(0)][embeddingSize]);
    if (passEnabled(pass, Pass::FWD)) {
      copy(target, dataType, rawSlicedResult.get(), hostSlicedResult);
    }
    // To check, for embedding update, fetch the Nz values.  We can check they
    // haven't changed in other cases.
    std::vector<EType> hostNzResult(buckets.nzValues.size());
    copy(target, dataType, rawNzInfo.get(), &hostNzResult[0],
         hostNzResult.size());
    // Interpret the NZ values, along with the original meta info to create a
    // dense representation of the IPU result
    auto ipuCSR = partitioner.sparsityDataImplToCSRMatrix(
        {std::move(buckets.metaInfo), std::move(hostNzResult)});
    auto ipuDenseEmbeddingMatrix = poplibs_test::sparse::csrToDenseMatrix(
        ipuCSR.nzValues.data(), csrMatrix.columnIndices.data(),
        ipuCSR.rowIndices.data(), ipuCSR.nzValues.size(), numEntries,
        embeddingSize, blockRows, blockCols);

    // So we can track which values are genuinely populated, make a
    // representation of the NZ values
    std::vector<float> nzFlags(ipuCSR.nzValues.size(), 1.0);
    auto denseEmbeddingMatrixFlags = poplibs_test::sparse::csrToDenseMatrix(
        nzFlags.data(), csrMatrix.columnIndices.data(),
        ipuCSR.rowIndices.data(), ipuCSR.nzValues.size(), numEntries,
        embeddingSize, blockRows, blockCols);

    // Model sliced result
    boost::multi_array<double, 2> modelSlicedResult(
        boost::extents[indices.dim(0)][embeddingSize]);
    if (passEnabled(pass, Pass::FWD)) {
      poplibs_test::embedding::multiSlice(hostDenseEmbeddingMatrix, hostIndices,
                                          modelSlicedResult);
    }
    if (passEnabled(pass, Pass::WU)) {
      for (unsigned i = 0; i < hostIndices.size(); i++) {
        for (unsigned j = 0; j < embeddingSize; j++) {
          // Zero out the slices where the data in the result doesn't exist
          if (denseEmbeddingMatrixFlags[hostIndices[i]][j] == 0) {
            hostSlicedInput[i][j] = 0;
          }
        }
      }
      poplibs_test::embedding::multiUpdateAdd(hostSlicedInput, hostIndices,
                                              scale, hostDenseEmbeddingMatrix);
    }
    if (debugPrint) {
      if (passEnabled(pass, Pass::FWD)) {
        std::cout << "Debug - sliced results";
        const auto printSize = std::min(embeddingSize, 70u);
        std::cout << "\nSliced results, " << printSize << " columns of "
                  << embeddingSize;
        for (unsigned i = 0; i < numIndices; i++) {
          std::cout << "\nRow:" << i << " index:" << hostIndices[i]
                    << "    ipu:";
          for (unsigned j = 0; j < printSize; j++) {
            if (j % 16 == 0)
              std::cout << " / ";
            std::cout << (hostSlicedResult[i][j] < 10 &&
                                  hostSlicedResult[i][j] != 0
                              ? " "
                              : "")
                      << hostSlicedResult[i][j] << ",";
          }
          std::cout << "\nRow:" << i << " index:" << hostIndices[i]
                    << "   host:";
          for (unsigned j = 0; j < printSize; j++) {
            if (j % 16 == 0)
              std::cout << " / ";
            std::cout << (modelSlicedResult[i][j] < 10 &&
                                  modelSlicedResult[i][j] != 0
                              ? " "
                              : "")
                      << modelSlicedResult[i][j] << ",";
          }
        }
        std::cout << "\n";
      }
      if (passEnabled(pass, Pass::WU)) {
        std::cout << "Debug - sparse input expanded to dense data, updated";
        for (unsigned i = 0; i < numEntries; i++) {
          std::cout << "\nRow host:" << i << "    ";
          for (unsigned j = 0; j < embeddingSize; j++) {
            std::cout << hostDenseEmbeddingMatrix[i][j] << ",";
          }
          std::cout << "\nRow  ipu:" << i << "    ";
          for (unsigned j = 0; j < embeddingSize; j++) {
            std::cout << ipuDenseEmbeddingMatrix[i][j] << ",";
          }
        }
        std::cout << "\n";
      }
    }

    if (passEnabled(pass, Pass::FWD)) {
      matchesModel &= checkIsClose("slicedResult", hostSlicedResult,
                                   modelSlicedResult, relTolerance);
    }
    if (passEnabled(pass, Pass::WU)) {
      matchesModel &= checkIsClose("updatedInput", ipuDenseEmbeddingMatrix,
                                   hostDenseEmbeddingMatrix, relTolerance);
    }
    if (testMatmul && randomInput) {
      // verify matmul result too
      matchesModel &=
          checkIsClose("mulOutput", hostMulOut, modelMulOut, relTolerance);
    }
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
  if (e.graphProfile.type() == ProfileValue::Type::MAP) {
    poplar::printGraphSummary(std::cerr, e.profilePath,
                              {{"showVarStorage", "true"}});
  }
  throw;
}
