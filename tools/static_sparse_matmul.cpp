// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "poplibs_support/TestDevice.hpp"

#include <boost/functional/hash.hpp>
#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/program_options.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>

#include "poplibs_test/TempDir.hpp"
#include "poputil/exceptions.hpp"
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/SparseMatrix.hpp>
#include <poplibs_test/Util.hpp>
#include <popops/Rearrange.hpp>
#include <poputil/OptionParsing.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>

#include "../lib/popsparse/SparseStorageInternal.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "popops/codelets.hpp"
#include "popsparse/MatMul.hpp"
#include "popsparse/MatMulParams.hpp"
#include "popsparse/PlanningCache.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "popsparse/SparseTensor.hpp"
#include <poplar/CycleCount.hpp>
#include <popsparse/codelets.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <gccs/Algorithm.hpp>
#include <optional>

#include <fstream>

// Tolerances used
#define FLOAT_REL_TOL 0.01
#define HALF_REL_TOL 0.1

using namespace poplar;
using namespace poplar::program;

using namespace poplibs_test::util;
using namespace poplar_test;
using namespace poplibs_support;

using namespace popsparse;
using namespace popsparse::dynamic;
using namespace poputil;

using EType = float;
bool readHeaderFromMaskFile(const std::string &fileName,
                            boost::optional<unsigned> &numRows,
                            boost::optional<unsigned> &numColumns,
                            unsigned blockLength) {
  std::ifstream cin(fileName);
  if (!cin.is_open()) {
    std::cerr << "Cannot open sparsity mask file " << fileName << "\n";
    return false;
  }
  unsigned numRowsInFile, numColumnsInFile;
  cin >> numRowsInFile >> numColumnsInFile;
  if (numColumns) {
    if (*numColumns / blockLength != numColumnsInFile) {
      std::cerr << "Columns in sparsity file (" << numColumnsInFile
                << ") do not match command line "
                << "options (" << *numColumns / blockLength << ")\n";
      return false;
    }
  }
  *numColumns = numColumnsInFile * blockLength;

  if (numRows) {
    if (*numRows / blockLength != numRowsInFile) {
      std::cerr << "Rows in sparsity file (" << numRowsInFile
                << ") do not match command line options ("
                << *numRows / blockLength << ")\n";
      return false;
    }
  }
  *numRows = numRowsInFile * blockLength;
  return true;
}

bool createCSRMatrixFromMaskFile(const std::string &fileName,
                                 CSRMatrix<EType> &csrMatrix, std::mt19937 &rng,
                                 bool useBipolarDistribution) {
  std::ifstream cin(fileName);
  if (!cin.is_open()) {
    std::cerr << "Cannot open sparsity mask file " << fileName << "\n";
    return false;
  }
  unsigned numRows, numColumns;
  cin >> numRows >> numColumns;
  csrMatrix.nzValues.clear();
  csrMatrix.columnIndices.clear();
  csrMatrix.rowIndices.clear();
  const auto blockLength = csrMatrix.getBlockDimensions()[0];

  auto randNormal = boost::random::normal_distribution<EType>(0, 1.0);
  auto randBernoulli = boost::random::bernoulli_distribution<EType>{};

  unsigned nzCount = 0;
  csrMatrix.rowIndices.push_back(nzCount);
  for (unsigned r = 0; r != numRows; r++) {
    for (unsigned c = 0; c != numColumns; ++c) {
      unsigned isNz;
      cin >> isNz;
      if (isNz) {
        for (unsigned i = 0; i != blockLength * blockLength; ++i) {
          if (useBipolarDistribution) {
            csrMatrix.nzValues.push_back(randBernoulli(rng) ? 1.0 : -1.0);
          } else {
            csrMatrix.nzValues.push_back(randNormal(rng));
          }
        }
        csrMatrix.columnIndices.push_back(c * blockLength);
        nzCount += blockLength * blockLength;
      }
    }
    csrMatrix.rowIndices.push_back(nzCount);
  }

  return true;
}

bool writeMaskFileFromCSRMatrix(const std::string &fileName,
                                const CSRMatrix<EType> &csrMatrix) {
  std::ofstream out(fileName);
  if (!out) {
    std::cerr << "Cannot open sparsity mask output file " << fileName << "\n";
    return false;
  }

  const auto blockLength = csrMatrix.getBlockDimensions()[0];
  out << csrMatrix.numRows / blockLength << ' '
      << csrMatrix.numColumns / blockLength << ' ';

  // Output mask entry for every entry in the matrix
  for (std::size_t r = 0; r != csrMatrix.rowIndices.size() - 1; ++r) {
    auto columnIt = csrMatrix.columnIndices.begin() + csrMatrix.rowIndices[r];
    for (std::size_t c = 0; c != csrMatrix.numColumns; ++c) {
      unsigned val = 0;
      if (columnIt != csrMatrix.columnIndices.end()) {
        if (c == *columnIt / blockLength) {
          val = 1;
          columnIt++;
        }
      }
      out << val << ' ';
    }
  }

  return true;
}

int main(int argc, char **argv) try {
  namespace po = boost::program_options;

  DeviceType deviceType = DeviceType::IpuModel2;
  constexpr unsigned numIPUs = 1;
  boost::optional<unsigned> tilesPerIPU;
  boost::optional<unsigned> numBands;
  boost::optional<std::string> profileDir;
  boost::optional<unsigned> nSplit;
  boost::optional<double> availableMemoryProportion;
  unsigned groups = 1, n;
  boost::optional<unsigned> m, k;
  double sparsityFactor = 0.1;
  Type dataType = HALF;
  ShapeOption<std::size_t> weightedAreaBegin;
  ShapeOption<std::size_t> weightedAreaEnd;
  bool doDenseSparse = false;
  unsigned blockLength = 1;
  std::string sparsityFileName = "";
  std::string outputSparsityFileName = "";

  weightedAreaBegin.val = weightedAreaEnd.val = {0, 0};
  double weightedAreaWeighting = 1.0;

  const std::vector<unsigned> supportedBlockLengths = {1, 4, 8, 16};
  std::stringstream ss;
  ss << "Block lengths supported: ";
  for (const auto &x : supportedBlockLengths) {
    ss << " " << x;
  }
  const std::string supportedBlockLengthsStr = ss.str();

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
    ("data-type", po::value(&dataType)->default_value(dataType), "Data type of operands")
    ("ignore-data", 
      "Do not download data and compare against reference implementation")
    ("do-dense-sparse", 
      po::value(&doDenseSparse)->default_value(doDenseSparse), 
      "Do a dense * sparse rather then sparse * dense mutiplication")
    ("m", po::value(&m), "Rows in left-hand operand (optional if sparsity file is specified")
    ("k", po::value(&k), "Columns in left-hand operand/Rows in right-hand operand (optional if sparsity file is specified)")
    ("n", po::value(&n)->required(), "Columns in right-hand operand")
    ("sparsity-factor", po::value(&sparsityFactor)->default_value(sparsityFactor),
     "Proportion of elements of left-hand operand which are non-zero")
    ("block-length",  po::value(&blockLength)->default_value(blockLength), 
      supportedBlockLengthsStr.c_str())
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
    ("num-column-bands", po::value(&numBands), "An optional parameter that is "
     "planned when not set, and used for splitting the columns of the sparse "
     "matrix. Use this to override the planned split")
    ("n-split", po::value(&nSplit), "An optional parameter that is "
     "planned when not set, and used for splitting the columns of the dense "
     "matrix. Use this to override the planned split")
    ("available-memory-proportion", po::value(&availableMemoryProportion), 
     "An optional parameter that gives the percentage of tile memory that "
     "is available for planning the matmul operation")
    ("variable-seed", "Use a variable seed based on clock, rather than a "
     "single fixed seed that does not change between runs of this tool")
    ("verbose", "Do trace level logging")     
    ("sparsity-matrix-file",
      po::value<std::string>(&sparsityFileName)->default_value(sparsityFileName)
      , "The file name for the sparsity mask (first line is row column "
      "followed by row major ordering of 0/1")
    ("output-sparsity-matrix-file",
      po::value<std::string>(&outputSparsityFileName)->default_value(outputSparsityFileName),
      "The file name to output the sparsity mask to")
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

  if (std::find(supportedBlockLengths.begin(), supportedBlockLengths.end(),
                blockLength) == supportedBlockLengths.end()) {
    std::cerr << "\nBlock size not supported";
    return 1;
  }

  Type partialsType = dataType;

  const auto maskFileUsed = !sparsityFileName.empty();

  if (maskFileUsed) {
    if (!readHeaderFromMaskFile(sparsityFileName, doDenseSparse ? k : m,
                                doDenseSparse ? m : k, blockLength)) {
      return 1;
    }
  } else {
    if (!m) {
      std::cerr << "\nWhen sparsity mask file is not given, `m` must be set";
      return 1;
    }
    if (!k) {
      std::cerr << "\nWhen sparsity mask file is not given, `k` must be set";
      return 1;
    }
  }

  if (*m % blockLength != 0) {
    std::cerr << "\nRows of left-hand-side matrix must be an integral multiple "
                 "of block size";
    return 1;
  }

  if (*k % blockLength != 0) {
    std::cerr << "\nColumns of left-hand-side matrix must be an integral "
                 "multiple of block size";
    return 1;
  }

  // align weighted area to a block size grid
  weightedAreaBegin.val[0] =
      gccs::alignPrev(weightedAreaBegin.val[0], blockLength);
  weightedAreaBegin.val[1] =
      gccs::alignPrev(weightedAreaBegin.val[1], blockLength);
  weightedAreaEnd.val[0] = gccs::alignPrev(weightedAreaEnd.val[0], blockLength);
  weightedAreaEnd.val[1] = gccs::alignPrev(weightedAreaEnd.val[1], blockLength);

  if (doDenseSparse) {
    std::swap(weightedAreaBegin.val[0], weightedAreaBegin.val[1]);
    std::swap(weightedAreaEnd.val[0], weightedAreaEnd.val[1]);
  }
  std::size_t numSparseRows = doDenseSparse ? *k : *m;
  std::size_t numSparseColumns = doDenseSparse ? *m : *k;

  bool profile = vm.count("profile");
  bool profilingEnabled = profile || profileDir;
  bool variableSeed = vm.count("variable-seed");
  bool ignoreData = vm.count("ignore-data");
  bool verboseLogging = vm.count("verbose");

  auto device = tilesPerIPU
                    ? createTestDevice(deviceType, numIPUs, *tilesPerIPU, true)
                    : createTestDeviceFullSize(deviceType, numIPUs, true);
  const auto &target = device.getTarget();
  Graph graph(target);
  popops::addCodelets(graph);
  popsparse::addCodelets(graph);
  Sequence uploadProg, prog, downloadProg;
  const std::string debugString = "sparse-mm";

  std::mt19937 randomEngine;
  if (variableSeed) {
    using namespace std::chrono;
    using SeedDurationType = duration<std::mt19937::result_type, std::nano>;
    const auto now = high_resolution_clock::now();
    const auto seed =
        duration_cast<SeedDurationType>(now.time_since_epoch()).count();
    randomEngine.seed(seed);
  }

  std::array<std::size_t, 2> blockDims = {blockLength, blockLength};
  CSRMatrix<EType> csrMatrix(blockDims);
  csrMatrix.numRows = numSparseRows;
  csrMatrix.numColumns = numSparseColumns;

  bool useBipolarDistribution;
  if (maskFileUsed) {
    useBipolarDistribution = true;
    if (!createCSRMatrixFromMaskFile(sparsityFileName, csrMatrix, randomEngine,
                                     true)) {
      return 1;
    }
    sparsityFactor = static_cast<double>(csrMatrix.nzValues.size()) /
                     (numSparseRows * numSparseColumns);
  } else {
    useBipolarDistribution =
        poplibs_test::sparse::floatingPointCouldRepresentMaxAccum(
            {numSparseRows, numSparseColumns}, {blockLength, blockLength},
            weightedAreaBegin, weightedAreaEnd, dataType, sparsityFactor,
            weightedAreaWeighting);
    std::tie(csrMatrix.nzValues, csrMatrix.columnIndices,
             csrMatrix.rowIndices) =
        poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
            randomEngine, {numSparseRows, numSparseColumns},
            {blockLength, blockLength}, sparsityFactor, weightedAreaBegin,
            weightedAreaEnd, weightedAreaWeighting, useBipolarDistribution);
  }

  if (!outputSparsityFileName.empty()) {
    if (!writeMaskFileFromCSRMatrix(outputSparsityFileName, csrMatrix)) {
      return 1;
    }
  }

  poplar::OptionFlags sparseOptionFlags;
  if (numBands) {
    sparseOptionFlags.set("numBands", std::to_string(*numBands));
  }
  if (nSplit) {
    sparseOptionFlags.set("nSplit", std::to_string(*nSplit));
  }
  if (availableMemoryProportion) {
    sparseOptionFlags.set("availableMemoryProportion",
                          std::to_string(*availableMemoryProportion));
  }
  if (verboseLogging) {
    sparseOptionFlags.set("verboseLogging", "true");
  }
  static_::PlanningCache cache;
  static_::MatMulParams matMulParams =
      doDenseSparse
          ? static_::MatMulParams::createForDenseSparse(groups, n, *k, *m)
          : static_::MatMulParams::createForSparseDense(groups, *m, *k, n);

  // Create partitioner object
  auto partitioner =
      static_::Partitioner<EType>(matMulParams, dataType, graph.getTarget(),
                                  sparseOptionFlags, &cache, debugString);

  auto sparsityImpl = partitioner.createSparsityDataImpl(csrMatrix);

  auto sparse =
      doDenseSparse
          ? static_::createDenseSparseMatMulRHS(graph, dataType, matMulParams,
                                                csrMatrix, debugString,
                                                sparseOptionFlags, &cache)
          : static_::createSparseDenseMatMulLHS(graph, dataType, matMulParams,
                                                csrMatrix, debugString,
                                                sparseOptionFlags, &cache);

  auto dense =
      doDenseSparse
          ? static_::createDenseSparseMatMulLHS(graph, dataType, matMulParams,
                                                csrMatrix, debugString,
                                                sparseOptionFlags, &cache)
          : static_::createSparseDenseMatMulRHS(graph, dataType, matMulParams,
                                                csrMatrix, debugString,
                                                sparseOptionFlags, &cache);

  auto out =
      doDenseSparse
          ? static_::denseSparseMatMul(graph, dense, sparse, prog, false, false,
                                       debugString, sparseOptionFlags, &cache)
          : static_::sparseDenseMatMul(graph, sparse, dense, prog, false, false,
                                       debugString, sparseOptionFlags, &cache);

  std::vector<std::pair<std::string, HostMemory>> tmap;
  auto rawNzInfo =
      allocateHostMemoryForTensor(sparse.getNzValuesTensor(), "lhs.nz", graph,
                                  uploadProg, downloadProg, tmap);
  auto rawDense = allocateHostMemoryForTensor(dense, "rhs", graph, uploadProg,
                                              downloadProg, tmap);
  auto rawOut = allocateHostMemoryForTensor(out, "out", graph, uploadProg,
                                            downloadProg, tmap);

  const auto canUseCycleCountForDevice =
      isSimulator(deviceType) || isHw(deviceType);
  if (canUseCycleCountForDevice) {
    auto cycles = cycleCount(graph, prog, 0, SyncType::INTERNAL, "totalCycles");
    graph.createHostRead("cycles", cycles);
  }

  Sequence controlProg({std::move(uploadProg), std::move(prog)});
  controlProg.add(downloadProg);

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

  boost::multi_array<double, 2> hostDense(
      boost::extents[dense.dim(1)][dense.dim(2)]);
  if (useBipolarDistribution) {
    writeRandomBinaryValues(target, dataType, hostDense, -1.0, 1.0,
                            randomEngine);
  } else {
    writeRandomValues(target, dataType, hostDense, -1.0, 1.0, randomEngine);
  }

  boost::multi_array<double, 2> hostOut(boost::extents[out.dim(1)][out.dim(2)]);
  if (!ignoreData) {
    copy(target, hostDense, dataType, rawDense.get());
  }
  copy(target, sparsityImpl.nzValues, dataType, rawNzInfo.get());

  device.bind([&](const Device &d) {
    std::uint64_t cyclesBuffer;
    engine.loadAndRun(d);
    double actSparsity =
        static_cast<double>(csrMatrix.nzValues.size()) / (*m * *k);
    std::cerr << "\nStatic sparsity: m " << *m << ", k " << *k << ", n " << n
              << ", block length " << blockLength << ", nz blocks "
              << csrMatrix.columnIndices.size() << ", dType " << dataType
              << ", sparsity " << actSparsity;
    if (maskFileUsed) {
      std::cerr << "(mask file = " << sparsityFileName << ")";
    }
    if (canUseCycleCountForDevice) {
      engine.readTensor("cycles", &cyclesBuffer, &cyclesBuffer + 1);
      constexpr double freqGHz = 1.85;
      double tFlops =
          2 * csrMatrix.nzValues.size() * n * freqGHz * 1.0e9 / cyclesBuffer;
      std::cerr << ", total cycles: " << cyclesBuffer;
      std::cerr << ", TFlops/sec @" << freqGHz << "GHz = " << tFlops;
    }
    std::cerr << "\n";
  });

  bool matchesModel = true;
  if (!ignoreData) {
    const double relTolerance = dataType == HALF ? HALF_REL_TOL : FLOAT_REL_TOL;
    copy(target, out.elementType(), rawOut.get(), hostOut);
    boost::multi_array<double, 2> modelOut(
        boost::extents[out.dim(1)][out.dim(2)]);
    boost::multi_array<double, 2> hostDenseLHS(
        boost::extents[numSparseRows][numSparseColumns]);
    hostDenseLHS = poplibs_test::sparse::csrToDenseMatrix(
        csrMatrix.nzValues.data(), csrMatrix.columnIndices.data(),
        csrMatrix.rowIndices.data(), csrMatrix.nzValues.size(), numSparseRows,
        numSparseColumns, blockLength, blockLength);

    if (doDenseSparse) {
      poplibs_test::gemm::generalMatrixMultiply(hostDense, hostDenseLHS,
                                                modelOut, false, false);
    } else {
      poplibs_test::gemm::generalMatrixMultiply(hostDenseLHS, hostDense,
                                                modelOut, false, false);
    }
    matchesModel &= checkIsClose("out", hostOut, modelOut, relTolerance);
  }

  if (profile) {
    engine.printProfileSummary(std::cout, {{"showExecutionSteps", "true"}});
  }

  if (!matchesModel) {
    std::cerr << "Validation failed\n";
    return 1;
  } else {
    if (ignoreData) {
      std::cerr << "Validation not enabled\n";
    } else {
      std::cerr << "Validation success\n";
    }
  }

  return 0;
} catch (const poplar::graph_memory_allocation_error &e) {
  if (!e.profilePath.empty()) {
    poplar::printGraphSummary(std::cerr, e.profilePath,
                              {{"showVarStorage", "true"}});
  }
  throw;
}
