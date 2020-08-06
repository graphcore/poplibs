// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <chrono>
#include <exception>
#include <fstream>
#include <istream>
#include <ostream>
#include <poplar/CycleCount.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/TestDevice.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Pass.hpp>
#include <poplibs_test/SparseMatrix.hpp>
#include <poplibs_test/Util.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/codelets.hpp>
#include <queue>
#include <random>
#include <ratio>

#include "../lib/popsparse/FullyConnectedOnTile.hpp"
#include "../lib/popsparse/FullyConnectedOptions.hpp"
#include "../lib/popsparse/FullyConnectedPlan.hpp"
#include "../lib/popsparse/SparsePartitionerImpl.hpp"
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/logging.hpp"
#include "popsparse/FullyConnected.hpp"
#include "popsparse/FullyConnectedParams.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_test::util;
using namespace poputil;
using poplibs_test::Pass;
using namespace poplibs_support;

using namespace popsparse;
using namespace popsparse::dynamic;

// Tolerances used when data is not ignored
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1

// TODO: This could be a program option if at all needed
using EType = float;

void print(const boost::multi_array_ref<double, 2> &matA) {
  std::cerr.precision(8);
  const auto numRows = matA.shape()[0];
  const auto numColumns = matA.shape()[1];
  for (std::size_t row = 0; row != numRows; ++row) {
    std::cerr << "\n";
    for (std::size_t col = 0; col != numColumns; ++col) {
      std::cerr << "   " << matA[row][col];
    }
  }
}

template <typename T>
static void logBucketStatistics(std::vector<PNBucket> &buckets,
                                const CSRMatrix<T> &csrMatrix) {
  if (buckets.empty()) {
    std::cerr << "   - No buckets found"
              << "\n";
    return;
  }

  std::size_t pnUsed = buckets.size();

  std::size_t maxNzElements = 0, maxMetaInfo = 0;
  std::size_t totalNzElements = 0, totalMetaInfo = 0;
  std::for_each(buckets.begin(), buckets.end(), [&](const PNBucket &b) {
    maxNzElements = std::max(maxNzElements, b.numNzElements);
    maxMetaInfo = std::max(maxMetaInfo, b.metaInfoElements);
    totalNzElements += b.numNzElements;
    totalMetaInfo += b.metaInfoElements;
  });

  std::cerr << "   - NZ entries " << csrMatrix.nzValues.size() << "\n";
  std::cerr << "   - NZ elements/PN max : " << maxNzElements;
  std::cerr << " avg : " << static_cast<double>(totalNzElements) / pnUsed;
  std::cerr << "\n";
  std::cerr << "   - Meta info elements/PN max : " << maxMetaInfo;
  std::cerr << " avg : " << static_cast<double>(totalMetaInfo) / pnUsed;
  std::cerr << "\n";
}

// TODO: Move this to a shared source file with SparsePartitionerTest which
// does the same exact validation.
template <typename T>
void validateBuckets(const PNBucketsImpl<T> &pnBucketsImpl,
                     const CSRMatrix<T> &csrMatrix, std::size_t numRows,
                     std::size_t numColumns) {
  // piece together information per row into a CSR format
  std::vector<std::size_t> colIndicesActual;
  std::vector<double> nzValuesActual;
  std::vector<std::size_t> rowIndicesActual;

  auto inInterval = [](const poplar::Interval range, std::size_t val) {
    return (val >= range.begin() && val < range.end());
  };

  std::cerr << "\nValidating partition ... ";
  std::size_t numValues = 0;
  for (std::size_t row = 0; row != numRows; ++row) {
    rowIndicesActual.push_back(numValues);
    for (std::size_t col = 0; col != numColumns; ++col) {
      // find tile partition that matched
      for (const auto &pn : pnBucketsImpl.pnBuckets) {

        for (const auto &p : pn.subGroups) {
          auto rowInterval = p.tile.getRows();
          auto colInterval = p.tile.getColumns();

          if (inInterval(rowInterval, row) && inInterval(colInterval, col)) {
            for (const auto &r : p.tileInfo) {
              if (r.rowNumber + rowInterval.begin() == row) {
                for (const auto &c : r.positionValues) {
                  if (c.first + colInterval.begin() == col) {
                    colIndicesActual.push_back(c.first + colInterval.begin());
                    nzValuesActual.push_back(
                        pnBucketsImpl.nzValues.at(c.second));
                    ++numValues;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  rowIndicesActual.push_back(numValues);
  bool success = true;

  for (std::size_t row = 0; row != csrMatrix.rowIndices.size(); ++row) {
    if (csrMatrix.rowIndices.at(row) != rowIndicesActual.at(row)) {
      std::cerr << "\n row indices at " << row << " incorrect";
      success = false;
    }
  }

  for (std::size_t col = 0; col != csrMatrix.columnIndices.size(); ++col) {
    if (csrMatrix.columnIndices.at(col) != colIndicesActual.at(col)) {
      std::cerr << "\n col indices at " << col << " incorrect";
      success = false;
    }
    if (csrMatrix.nzValues.at(col) != nzValuesActual.at(col)) {
      std::cerr << "\n nz values at " << col << " incorrect : ";
      std::cerr << csrMatrix.nzValues.at(col) << " " << nzValuesActual.at(col);
      success = false;
    }
  }
  std::cerr << " validation of partition completed : ";
  std::cerr << (success ? "true" : "false") << "\n";
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel;
  unsigned numGroups = 1;
  unsigned inputSize;
  unsigned outputSize;
  unsigned batchSize;
  bool reportPlan;
  std::string profileJsonPath;
  Type dataType;
  unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  Pass pass = Pass::FWD;
  std::string matmulOptionsString;
  double sparsityFactor;
  const auto partialsType = FLOAT;
  ShapeOption<std::size_t> weightedAreaBegin;
  ShapeOption<std::size_t> weightedAreaEnd;
  weightedAreaBegin.val = weightedAreaEnd.val = {0, 0};
  double weightedAreaWeighting = 1.0;
  bool denseGradWSerialSplits = false;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("device-type",
     po::value<DeviceType>(&deviceType)->default_value(deviceType),
     "Device type: Cpu | Sim | Hw | IpuModel")
    ("input-size", po::value<unsigned>(&inputSize)->required(),
     "Number of inputs")
    ("output-size", po::value<unsigned>(&outputSize)->required(),
     "Number of output channels")
    ("sparsity-factor", po::value<double>(&sparsityFactor)->required(),
     "Sparsity factor (ratio of number of non-zero values to total weight "
     "values")
    ("data-type",
     po::value<Type>(&dataType)->default_value(HALF),
     "Type of the input and output data")
    ("tiles-per-ipu",
     po::value(&tilesPerIPU),
     "Number of tiles per IPU")
    ("batch-size",
     po::value<unsigned>(&batchSize)->default_value(1),
     "Batch size")
    ("single-phase",
     po::value<Pass>(&pass)->default_value(pass),
     "Run phase all | fwd | bwd | wu")
    ("ignore-data", "When set, no upload/download or verification of "
     "results is performed")
    ("validate-partition", "validate partition created by partitioner ")
    ("plan-only", "Whether to perform planning only and skip creation "
     "and running of the program")
    ("profile", "Enable profiling and print profiling report")
    ("profile-json", po::value<std::string>(&profileJsonPath)->default_value(profileJsonPath),
     "Path to a file into which the profiling report will be output in json format")
    ("report-plan", po::value<bool>(&reportPlan)->default_value(false),
     "Display plan")
    ("report-total-cycle-counts", "Report total cycle count ignoring upload/download for "
     "each pass. Note not compatible with 'profile' option")
    ("variable-seed", "Use a variable seed based on clock, rather than a "
     "single fixed seed that does not change between runs of this tool")
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
    ("matmul-options", po::value<std::string>(&matmulOptionsString),
     "Options to use for the matrix multiplication, specified as a JSON "
     "string, e.g. {\"key\":\"value\"}")
    ("report-dense-gradw-serial-splits",
      po::value<bool>(&denseGradWSerialSplits)->
        default_value(denseGradWSerialSplits),
     "Report dense GradW splits when GradW pass is enabled")
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
  bool profilingEnabled = profile || !profileJsonPath.empty();
  bool reportTotalCycleCounts =
      vm.count("report-total-cycle-counts") && deviceType == DeviceType::Hw;
  bool ignoreData = vm.count("ignore-data");
  bool planOnly = vm.count("plan-only");
  bool variableSeed = vm.count("variable-seed");
  bool validatePartition = vm.count("validate-partition");

  if (reportTotalCycleCounts && profilingEnabled) {
    throw poputil::poplibs_error(
        "--report-total-cycle-counts and --profile or --profile-json specified "
        "at the same time. This is not allowed as one affects the other");
  }

  if (weightedAreaBegin[0] > weightedAreaEnd[0] ||
      weightedAreaBegin[1] > weightedAreaEnd[1]) {
    std::stringstream ss;
    ss << "Invalid weighted area specified: " << weightedAreaBegin.val << ","
       << weightedAreaEnd.val;
    throw poputil::poplibs_error(ss.str());
  }

  if (weightedAreaEnd[0] > outputSize || weightedAreaEnd[1] > inputSize) {
    std::stringstream ss;
    ss << "Specified weighted area is out of bounds: Weighted area="
       << weightedAreaBegin.val << "," << weightedAreaEnd.val
       << " out of bounds {" << outputSize << "," << inputSize << "}";
    throw poputil::poplibs_error(ss.str());
  }

  PlanningCache cache;

  poplar::OptionFlags options;
  bool doBwdPass = pass == Pass::BWD || pass == Pass::ALL;
  bool doWuPass = pass == Pass::WU || pass == Pass::ALL;
  options.set("availableMemoryProportion", "1.0");
  options.set("doGradAPass", doBwdPass ? "true" : "false");
  options.set("doGradWPass", doWuPass ? "true" : "false");

  // User options specified via --matmul-options override defaults
  if (!matmulOptionsString.empty()) {
    poplar::readJSON(matmulOptionsString, options);
  }

  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, true)
                    : createTestDeviceFullSize(deviceType, numIPUs, true);
  const auto &target = device.getTarget();

  SparsityParams sparsityParams(SparsityType::Element,
                                SparsityStructure::Unstructured);

  const auto params = FullyConnectedParams::createWithNzRatio(
      std::move(sparsityParams), sparsityFactor, batchSize, numGroups,
      inputSize, outputSize);

  popsparse::fullyconnected::Plan plan;
  popsparse::fullyconnected::Cost planCost;

  // Always do forward
  std::tie(plan, planCost) = popsparse::fullyconnected::getPlan(
      target, dataType, params, options, &cache);

  if (reportPlan) {
    std::string str;
    if (doBwdPass || doWuPass) {
      str += "Joint : ";
      if (doBwdPass) {
        str += " GradA + ";
      }
      if (doWuPass) {
        str += " GradW + ";
      }
    }
    str += "Fwd Plan \n";
    std::cerr << str << plan << "\n" << planCost << "\n";
  }

  std::size_t fwdMetaInfoBucketSize = plan.fwdMetaInfoElemsPerBucket;
  std::size_t gradAMetaInfoBucketSize = plan.gradAMetaInfoElemsPerBucket;
  std::size_t nzElementBucketSize = plan.nzElemsPerBucket;

  std::cerr << "Using bucket sizes:"
            << "\n  meta info (forward): " << fwdMetaInfoBucketSize
            << "\n  meta info (grad-a): " << gradAMetaInfoBucketSize
            << "\n  nz element : " << nzElementBucketSize << "\n";

  Partitioner<EType> partitioner(params, dataType, target, options, &cache);
  std::mt19937 randomEngine;
  if (variableSeed) {
    using namespace std::chrono;
    using SeedDurationType = duration<std::mt19937::result_type, std::nano>;
    const auto now = high_resolution_clock::now();
    const auto seed =
        duration_cast<SeedDurationType>(now.time_since_epoch()).count();
    randomEngine.seed(seed);
  }

  const bool floatingPointCouldRepresentMaxAccum = [&] {
    const auto maxVal = maxContiguousInteger(dataType);

    double weightedThreshold, remainingThreshold;
    std::tie(weightedThreshold, remainingThreshold) =
        poplibs_test::sparse::calculateWeightedVsRemainingSparsityFactor(
            {outputSize, inputSize}, sparsityFactor, weightedAreaBegin,
            weightedAreaEnd, weightedAreaWeighting);
    const auto numWeightedInputChannels =
        (weightedAreaEnd[1] - weightedAreaBegin[1]);
    const auto numWeightedOutputChannels =
        (weightedAreaEnd[0] - weightedAreaBegin[0]);
    const auto maxInputChannels =
        numWeightedInputChannels * weightedThreshold +
        (params.getInputChannelsPerGroup() - numWeightedInputChannels) *
            remainingThreshold;
    const auto maxOutputChannels =
        numWeightedOutputChannels * weightedThreshold +
        (params.getOutputChannelsPerGroup() - numWeightedOutputChannels) *
            remainingThreshold;
    const auto getOpsPerOutputElementEstimate = [&](const Pass &pass) -> int {
      const auto numAccumulations = pass == Pass::FWD   ? maxInputChannels
                                    : pass == Pass::BWD ? maxOutputChannels
                                                        : params.getBatchSize();
      assert(numAccumulations < std::numeric_limits<int>::max());
      return numAccumulations;
    };
    // We use a modifier to account for the unlikeliness of picking all positive
    // or negative 1s which would actually get us to the max precisely
    // represented integer.
    constexpr int modifier = 10;
    // We use another modifier to account for the chance that sparsity is not
    // perfectly evenly spread in this instant.
    constexpr double wiggleRoom = 1.3;
    if (wiggleRoom * getOpsPerOutputElementEstimate(Pass::FWD) >
        maxVal * modifier) {
      return false;
    }
    if (doBwdPass && wiggleRoom * getOpsPerOutputElementEstimate(Pass::BWD) >
                         maxVal * modifier) {
      return false;
    }
    if (doWuPass && wiggleRoom * getOpsPerOutputElementEstimate(Pass::WU) >
                        maxVal * modifier) {
      return false;
    }
    return true;
  }();

  // create CSR matrix for the given sparsity factor
  const bool useBipolarDistribution = floatingPointCouldRepresentMaxAccum;
  CSRMatrix<EType> csrMatrix;
  std::tie(csrMatrix.nzValues, csrMatrix.columnIndices, csrMatrix.rowIndices) =
      poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
          randomEngine, {outputSize, inputSize}, sparsityFactor,
          weightedAreaBegin, weightedAreaEnd, weightedAreaWeighting,
          useBipolarDistribution);

  // Forward
  boost::multi_array<double, 2> hostInput(boost::extents[batchSize][inputSize]);
  if (useBipolarDistribution) {
    writeRandomBinaryValues(target, dataType, hostInput, -1.0, 1.0,
                            randomEngine);
  } else {
    writeRandomValues(target, dataType, hostInput, -3.0, 3.0, randomEngine);
  }
  boost::multi_array<double, 2> hostOutputActs(
      boost::extents[batchSize][outputSize]);

  // GradA
  boost::multi_array<double, 2> hostOutputGrad(
      boost::extents[batchSize][outputSize]);
  if (useBipolarDistribution) {
    writeRandomBinaryValues(target, dataType, hostOutputGrad, -1.0, 1.0,
                            randomEngine);
  } else {
    writeRandomValues(target, dataType, hostOutputGrad, -3.0, 3.0,
                      randomEngine);
  }
  boost::multi_array<double, 2> hostInputGrad(
      boost::extents[batchSize][inputSize]);

  auto pnBucketsImpl = partitioner.getImpl().createBuckets(csrMatrix);

  if (validatePartition) {
    validateBuckets(pnBucketsImpl, csrMatrix, outputSize, inputSize);
  }

  std::cerr << "Logging Forward pass bucket statistics: ";
  logBucketStatistics(pnBucketsImpl.pnBuckets, csrMatrix);

  if (planOnly) {
    return 0;
  }

  Graph graph(target);
  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  popsparse::addCodelets(graph);
  Sequence fwdProg, bwdProg, wuProg, uploadProg, downloadProg;

  // Build the graph
  std::cerr << "Constructing graph...\n";
  const SparseTensor weights = createFullyConnectedWeights(
      graph, dataType, params, "weights", options, &cache);
  const Tensor input = createFullyConnectedInput(graph, dataType, params,
                                                 "input", options, &cache);
  const Tensor outputActs = fullyConnectedFwd(graph, weights, input, params,
                                              fwdProg, "fwd", options, &cache);

  // GradW
  boost::multi_array<double, 1> hostWeightGrad(
      boost::extents[weights.getNzValuesTensor().numElements()]);

  Tensor outputGrad;
  if (doBwdPass || doWuPass) {
    outputGrad = graph.clone(outputActs, "outputGrad");
  }
  Tensor inputGrad;
  if (doBwdPass) {
    inputGrad = fullyConnectedGradA(graph, weights, outputGrad, params, bwdProg,
                                    "grada", options, &cache);
  }

  Tensor weightGrad;
  if (doWuPass) {
    weightGrad = fullyConnectedSparseGradW(graph, weights.getMetaInfoTensor(),
                                           outputGrad, input, params, wuProg,
                                           "wu", options, &cache);
  }
  std::cerr << "Done\n";

  std::vector<std::pair<std::string, char *>> tmap;
  auto rawMetaInfo =
      allocateHostMemoryForTensor(weights.getMetaInfoTensor(), "weights.meta",
                                  graph, uploadProg, downloadProg, tmap);
  auto rawNzInfo =
      allocateHostMemoryForTensor(weights.getNzValuesTensor(), "weights.nz",
                                  graph, uploadProg, downloadProg, tmap);
  auto rawInput = allocateHostMemoryForTensor(input, "input", graph, uploadProg,
                                              downloadProg, tmap);
  auto rawOutputActs = allocateHostMemoryForTensor(
      outputActs, "outputActs", graph, uploadProg, downloadProg, tmap);

  std::unique_ptr<char[]> rawOutputGrad;
  if (doBwdPass || doWuPass) {
    rawOutputGrad = allocateHostMemoryForTensor(outputGrad, "outputGrad", graph,
                                                uploadProg, downloadProg, tmap);
  }

  std::unique_ptr<char[]> rawInputGrad;
  if (!ignoreData && doBwdPass) {
    rawInputGrad = allocateHostMemoryForTensor(inputGrad, "inputGrad", graph,
                                               uploadProg, downloadProg, tmap);
  }

  std::unique_ptr<char[]> rawWeightGrad;
  if (!ignoreData && doWuPass) {
    rawWeightGrad = allocateHostMemoryForTensor(weightGrad, "weightGrad", graph,
                                                uploadProg, downloadProg, tmap);
  }

  Tensor fwdCycles, bwdCycles, wuCycles;
  if (reportTotalCycleCounts) {
    fwdCycles = cycleCount(graph, fwdProg, 0, "fwdCycles");
    graph.createHostRead("fwdCycles", fwdCycles);
    if (doBwdPass) {
      bwdCycles = cycleCount(graph, bwdProg, 0, "bwdCycles");
      graph.createHostRead("bwdCycles", bwdCycles);
    }
    if (doWuPass) {
      wuCycles = cycleCount(graph, wuProg, 0, "wuCycles");
      graph.createHostRead("wuCycles", wuCycles);
    }
  }
  Sequence controlProg(std::move(uploadProg), std::move(fwdProg),
                       std::move(bwdProg), std::move(wuProg));
  if (!ignoreData) {
    controlProg.add(std::move(downloadProg));
  }

  std::cerr << "Creating engine...\n";
  OptionFlags engineOptions;
  if (profilingEnabled) {
    engineOptions.set("debug.instrument", "true");
  }
  Engine engine(graph, std::move(controlProg), engineOptions);
  attachStreams(engine, tmap);

  std::cerr << "Done\n";

  std::cerr << "Running...\n";

  // Actual bucket info use by device graph
  auto buckets = partitioner.createSparsityDataImpl(csrMatrix);

  const auto &metaInfoFlat = buckets.metaInfo;
  const auto &nzValuesFlat = buckets.nzValues;
  // Overflow info is the same for all passes at time of writing.
  std::cerr << "overflowInfo = {" << metaInfoFlat.at(0) << ","
            << metaInfoFlat.at(1) << "," << metaInfoFlat.at(2) << "}\n";

  copy(target, hostInput, dataType, rawInput.get());
  copy(target, metaInfoFlat, UNSIGNED_SHORT, rawMetaInfo.get());
  copy(target, nzValuesFlat, dataType, rawNzInfo.get());
  if (!ignoreData && (doBwdPass || doWuPass)) {
    copy(target, hostOutputGrad, outputGrad.elementType(), rawOutputGrad.get());
  }

  device.bind([&](const Device &d) {
    engine.loadAndRun(d);
    if (reportTotalCycleCounts) {
      std::uint64_t cyclesBuffer;
      engine.readTensor("fwdCycles", &cyclesBuffer);
      std::cerr << "  Forward pass cycles: " << cyclesBuffer << "\n";
      if (doBwdPass) {
        engine.readTensor("bwdCycles", &cyclesBuffer);
        std::cerr << "  GradA pass cycles: " << cyclesBuffer << "\n";
      }
      if (doWuPass) {
        engine.readTensor("wuCycles", &cyclesBuffer);
        std::cerr << "  GradW pass cycles: " << cyclesBuffer << "\n";
      }
    }
  });

  bool matchesModel = true;
  if (!ignoreData) {
    const double relTolerance = dataType == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
    copy(target, outputActs.elementType(), rawOutputActs.get(), hostOutputActs);
    boost::multi_array<double, 2> hostDenseWeights(
        boost::extents[outputSize][inputSize]);
    boost::multi_array<double, 2> modelOutputActs(
        boost::extents[batchSize][outputSize]);
    hostDenseWeights = poplibs_test::sparse::csrToDenseMatrix(
        csrMatrix.nzValues.data(), csrMatrix.columnIndices.data(),
        csrMatrix.rowIndices.data(), csrMatrix.nzValues.size(), outputSize,
        inputSize);
    poplibs_test::gemm::generalMatrixMultiply(hostInput, hostDenseWeights,
                                              modelOutputActs, false, true);
    matchesModel &= checkIsClose("outputActs", hostOutputActs, modelOutputActs,
                                 relTolerance);
    if (doBwdPass) {
      copy(target, inputGrad.elementType(), rawInputGrad.get(), hostInputGrad);
      boost::multi_array<double, 2> modelInputGrad(
          boost::extents[batchSize][inputSize]);
      poplibs_test::gemm::generalMatrixMultiply(
          hostOutputGrad, hostDenseWeights, modelInputGrad, false, false);
      matchesModel &= checkIsClose("inputGrad", hostInputGrad, modelInputGrad,
                                   relTolerance);
    }
    if (doWuPass) {
      copy(target, weightGrad.elementType(), rawWeightGrad.get(),
           hostWeightGrad);
      boost::multi_array<double, 2> modelWeightGrad(
          boost::extents[outputSize][inputSize]);
      poplibs_test::gemm::generalMatrixMultiply(hostOutputGrad, hostInput,
                                                modelWeightGrad, true, false);

      std::vector<EType> modelNzValuesCSR;
      auto columnIdxIt = csrMatrix.columnIndices.begin();
      for (auto rowIt = std::next(csrMatrix.rowIndices.begin());
           rowIt != csrMatrix.rowIndices.end(); ++rowIt) {
        const auto nnzThisRow = *rowIt - *std::prev(rowIt);
        const auto rowIdx =
            std::distance(csrMatrix.rowIndices.begin(), rowIt) - 1;
        for (std::size_t i = 0; i < nnzThisRow; ++i) {
          const auto columnIdx = *columnIdxIt++;
          modelNzValuesCSR.emplace_back(modelWeightGrad[rowIdx][columnIdx]);
        }
      }

      assert(modelNzValuesCSR.size() == csrMatrix.columnIndices.size());

      SparsityDataImpl<EType> actualBuckets;
      std::vector<EType> actualWeightGrads;
      actualWeightGrads.reserve(hostWeightGrad.size());
      std::copy(hostWeightGrad.begin(), hostWeightGrad.end(),
                std::back_inserter(actualWeightGrads));

      actualBuckets.nzValues = std::move(actualWeightGrads);
      actualBuckets.metaInfo = std::move(buckets.metaInfo);
      auto actualCSR = partitioner.sparsityDataImplToCSRMatrix(actualBuckets);

      auto ait = actualCSR.nzValues.begin();
      auto mit = modelNzValuesCSR.begin();
      for (; ait != actualCSR.nzValues.end(); ++ait, ++mit) {
        bool elemMatch = checkIsClose(*mit, *ait, relTolerance);
        if (!elemMatch) {
          std::cerr << "mismatch at  WeightsGrad.nz[";
          std::cerr << std::distance(actualCSR.nzValues.begin(), ait);
          std::cerr << "]:" << *mit << "!=" << *ait << "\n";
        }
        matchesModel &= elemMatch;
      }
      auto columnsMatch = std::equal(
          actualCSR.columnIndices.begin(), actualCSR.columnIndices.end(),
          csrMatrix.columnIndices.begin(), csrMatrix.columnIndices.end());
      if (!columnsMatch) {
        std::cerr << "CSR columns indices do not match\n";
      }
      matchesModel &= columnsMatch;
    }
  }

  if (denseGradWSerialSplits && doWuPass) {
    auto serialSplits =
        fullyConnectedDenseGradWSerialSplits(graph, dataType, params, options);
    std::cerr << "Dense GradW serial splits : "
              << "   groups " << std::get<0>(serialSplits)
              << "   input channel " << std::get<1>(serialSplits)
              << "   output channel " << std::get<2>(serialSplits) << "\n";
  }

  std::cerr << "Done\n";

  if (profile) {
    engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }
  if (!profileJsonPath.empty()) {
    std::ofstream os(profileJsonPath, std::ios_base::out);
    const auto &pr = engine.getProfile();
    poplar::serializeToJSON(os, pr);
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  if (e.graphProfile.type() == ProfileValue::Type::MAP) {
    poplar::printGraphSummary(std::cerr, e.graphProfile,
                              {{"showVarStorage", "true"}});
  }
  throw;
}
