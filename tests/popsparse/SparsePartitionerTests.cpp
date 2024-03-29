// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE SparsePartionerTests
#include "../lib/popsparse/SparsePartitionerImpl.hpp"
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include "poplar/Type.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_test/SparseMatrix.hpp"
#include "poplibs_test/Util.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <random>

using namespace poplibs_support;
using namespace poplibs_test::util;
using namespace poplar_test;

// Used data types
// TODO: We could make the testing of actual bucket creation by passing
// this as a test option
const poplar::Type dataType = poplar::FLOAT;
const poplar::Type accumType = poplar::FLOAT;

// Build CSR matrix
static popsparse::CSRMatrix<float>
buildCSRMatrix(const std::vector<size_t> &dimensions, double sparsityFactor,
               const std::array<std::size_t, 2> &blockDimensions = {1, 1}) {

  assert(dimensions[0] % blockDimensions[0] == 0 &&
         dimensions[1] % blockDimensions[1] == 0);
  logging::popsparse::debug("Dimension in CSR Matrix generation {}",
                            dimensions);
  using EType = float;
  std::mt19937 randomEngine;
  auto [nzValues, columnIndices, rowIndices] =
      poplibs_test::sparse::buildCSRMatrix<EType, std::size_t>(
          randomEngine, dimensions, {blockDimensions[0], blockDimensions[1]},
          sparsityFactor, {0, 0}, {0, 0}, 0, false);
  logging::popsparse::debug("NZ Values {} : {}", nzValues.size(), nzValues);
  logging::popsparse::debug("Columns Indices {} : {}", columnIndices.size(),
                            columnIndices);
  logging::popsparse::debug("Row Indices {} : {}", rowIndices.size(),
                            rowIndices);

  return popsparse::CSRMatrix<EType>(
      dimensions[0], dimensions[1], std::move(nzValues),
      std::move(columnIndices), std::move(rowIndices), blockDimensions);
}

template <class T>
static void printVec(const std::vector<T> &vec, std::string name) {
  std::string res = "{";
  for (const auto val : vec) {
    res += std::to_string(val);
    res += ",";
  }
  res += "}";
  logging::popsparse::debug("{}: {}", name, res);
}

static bool validatePartition(const std::vector<std::size_t> &dimensions,
                              const std::vector<std::size_t> &grainSizes,
                              const std::vector<std::size_t> &xSplits,
                              const std::vector<std::size_t> &ySplits,
                              const std::vector<std::size_t> &zSplits,
                              double sparsityFactor,
                              std::size_t metaInfoBucketSize,
                              std::size_t nzElementsBucketSize,
                              std::size_t bucketsPerZ, bool useBlockMetaInfo,
                              bool includeGradA, bool includeGradW,
                              bool checkSparsityDataImpl, bool useDense) {

  const auto blockSizeX = grainSizes.at(0);
  const auto blockSizeY = grainSizes.at(1);
  const auto blockSize = blockSizeX * blockSizeY;
  const auto numRowBlocks = dimensions[0] / blockSizeX;

  auto csrMatrix =
      buildCSRMatrix(dimensions, sparsityFactor, {blockSizeX, blockSizeY});

  // Create partitioner object with plan information
  //
  // The following doesn't actually affect partition generation so just
  // set them to some value.
  const std::size_t metaInfoBucketSizeGradA = 0;
  // Set shared buckets so we don't have to deal with size of gradA impl
  // buckets.
  const bool sharedBuckets = true;
  // TODO: Test partitioner options
  const popsparse::PartitionerOptions options;
  popsparse::dynamic::PartitionerImpl partitioner(
      dimensions, grainSizes, {grainSizes.at(0), grainSizes.at(1)}, xSplits,
      ySplits, zSplits, metaInfoBucketSize, metaInfoBucketSizeGradA,
      nzElementsBucketSize, 6, bucketsPerZ, useBlockMetaInfo, includeGradA,
      includeGradW, sharedBuckets, dataType, accumType, options, useDense);

  auto pnBucketsImpl = partitioner.createBuckets(csrMatrix);

  if (checkSparsityDataImpl) {
    auto impl = partitioner.bucketImplAllPasses(pnBucketsImpl);
    auto csrMatrixRecovered =
        partitioner.bucketsToCSRMatrix(impl.first, impl.second);
    const bool nzIncorrect = csrMatrix.nzValues != csrMatrixRecovered.nzValues;
    const bool colIncorrect =
        csrMatrix.columnIndices != csrMatrixRecovered.columnIndices;
    const bool rowIncorrect =
        csrMatrix.rowIndices != csrMatrixRecovered.rowIndices;
    if (nzIncorrect || colIncorrect || rowIncorrect) {
      logging::popsparse::err("Failed nz {} col {} row {}", nzIncorrect,
                              colIncorrect, rowIncorrect);
      if (nzIncorrect) {
        printVec(csrMatrix.nzValues, "original nz");
        printVec(csrMatrixRecovered.nzValues, "New nz");
      }
      if (colIncorrect) {
        printVec(csrMatrix.columnIndices, "original cols");
        printVec(csrMatrixRecovered.columnIndices, "new cols");
      }
      if (rowIncorrect) {
        printVec(csrMatrix.rowIndices, "original rows");
        printVec(csrMatrixRecovered.rowIndices, "new rows");
      }
      return false;
    }
  }
  auto pnBuckets = pnBucketsImpl.pnBuckets;
  const auto &nzValues = pnBucketsImpl.nzValues;

  partitioner.overflowInfoForFwd(pnBuckets);

  // tile partitions must be less than the number of splits
  if (pnBuckets.size() !=
      xSplits.size() * ySplits.size() * zSplits.size() * bucketsPerZ) {
    return false;
  }

  // convert bucket information back to a CSR format
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>>
      perRowPositionNzIndexPairs(numRowBlocks);
  std::size_t totalNzBlocks = 0;
  for (const auto &pn : pnBuckets) {
    for (const auto &p : pn.subGroups) {
      const auto &rowInterval = p.tile.getRows();
      const auto &colInterval = p.tile.getColumns();

      for (const auto &r : p.tileInfo) {
        const auto row = r.rowNumber + rowInterval.begin();
        for (const auto &c : r.positionValues) {
          const auto col = c.first + colInterval.begin();
          perRowPositionNzIndexPairs.at(row / blockSizeX)
              .emplace_back(col, c.second);
        }
        totalNzBlocks += r.positionValues.size();
      }
    }
  }

  std::vector<std::size_t> colIndicesActual;
  std::vector<double> nzValuesActual;
  std::vector<std::size_t> rowIndicesActual;
  colIndicesActual.reserve(totalNzBlocks);
  nzValuesActual.reserve(totalNzBlocks * blockSize);
  rowIndicesActual.reserve(numRowBlocks + 1);

  std::size_t totalNzElems = 0;
  for (std::size_t rowBlock = 0; rowBlock < numRowBlocks; ++rowBlock) {
    rowIndicesActual.push_back(totalNzElems);

    auto &nzPositionAndIndex = perRowPositionNzIndexPairs.at(rowBlock);

    // Sort by column index. Column index should be unique
    // and so the default std::pair sort should be sufficient.
    std::sort(nzPositionAndIndex.begin(), nzPositionAndIndex.end());
    for (const auto &entry : nzPositionAndIndex) {
      colIndicesActual.push_back(entry.first);
      assert((entry.second + 1) * blockSize <= nzValues.size());
      nzValuesActual.insert(nzValuesActual.end(),
                            nzValues.begin() + entry.second * blockSize,
                            nzValues.begin() + (entry.second + 1) * blockSize);
      totalNzElems += blockSize;
    }
  }
  rowIndicesActual.push_back(totalNzElems);

  logging::popsparse::debug(" Actual nz values: {} ", nzValuesActual);
  logging::popsparse::debug(" expect nz values: {} ", csrMatrix.nzValues);
  logging::popsparse::debug(" Actual column values: {} ", colIndicesActual);
  logging::popsparse::debug(" expect column values: {} ",
                            csrMatrix.columnIndices);
  logging::popsparse::debug(" Actual row values: {} ", rowIndicesActual);
  logging::popsparse::debug(" expect row values: {} ", csrMatrix.rowIndices);

  // If using dense partition nzValues will not equal the CSR nz values
  if (!useDense && nzValuesActual.size() != csrMatrix.nzValues.size()) {
    return false;
  }
  if (!useDense &&
      !std::equal(nzValuesActual.begin(), nzValuesActual.end(),
                  csrMatrix.nzValues.begin(), csrMatrix.nzValues.end())) {
    return false;
  }

  if (colIndicesActual.size() != csrMatrix.columnIndices.size()) {
    return false;
  }
  if (!std::equal(colIndicesActual.begin(), colIndicesActual.end(),
                  csrMatrix.columnIndices.begin(),
                  csrMatrix.columnIndices.end())) {
    return false;
  }

  if (rowIndicesActual.size() != csrMatrix.rowIndices.size()) {
    return false;
  }

  if (!std::equal(rowIndicesActual.begin(), rowIndicesActual.end(),
                  csrMatrix.rowIndices.begin(), csrMatrix.rowIndices.end())) {
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  double sparsityLevel = 0.1;

  ShapeOption<std::size_t> matShape;
  ShapeOption<std::size_t> splitShape;
  ShapeOption<std::size_t> blockShape;
  blockShape.val = {1, 1};
  std::size_t minMultipleGrainSizeZ = dataType == poplar::FLOAT ? 2 : 4;
  std::size_t grainSizeZ = minMultipleGrainSizeZ;
  std::size_t numBucketsZ = 1;
  bool includeGradW = true;
  bool includeGradA = true;
  double excess = 0.1;
  bool useDense = false;

  po::options_description desc("Options");
  // clang-format off
  desc.add_options()
    ("help", "Produce help message")
    ("matmul-shape",
     po::value<ShapeOption<std::size_t>>(&matShape)->required(),
     "Triplet representing matmul {Rows, Columns, Batches}")
    ("split-shape",
     po::value<ShapeOption<std::size_t>>(&splitShape)->required(),
     "Triplet representing number of splits {Rows, Columns, Batches")
    ("sparsity-level",
     po::value<double>(&sparsityLevel)->default_value(sparsityLevel),
     "Level of sparsity")
    ("block-shape",
     po::value<ShapeOption<std::size_t>>(&blockShape)->default_value(blockShape),
     "Pair representing block-size of sparse elements {BlockRows, BlockColumns}")
    ("batch-grain-size",
     po::value<std::size_t>(&grainSizeZ)->default_value(grainSizeZ),
     "Number of grains in batch dimension (must be multiple of 2 for float, "
     "and multiple of 4 for half")
    ("excess",
      po::value<double>(&excess)->default_value(excess),
      "Excess bucket size")
    ("num-buckets-z",
     po::value<std::size_t>(&numBucketsZ)->default_value(numBucketsZ),
     "Number of buckets per Z")
    ("disable-sparsity-data-impl-checks", "Don't run checks that generate on-device sparsity data implementation")
    ("include-gradw",
     po::value<bool>(&includeGradW)->default_value(includeGradW),
     "Include GradW")
    ("include-grada",
     po::value<bool>(&includeGradA)->default_value(includeGradA),
     "Include GradA")
    ("use-dense",
     po::value<bool>(&useDense)->default_value(useDense),
     "use dense partitioner")
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

  if (matShape.val.size() != 3) {
    throw poputil::poplibs_error("shape of matrix must be 3-dimensional");
  }

  if (splitShape.val.size() != 3) {
    throw poputil::poplibs_error("shape of splits must be 3-dimensional");
  }

  if (matShape[0] % blockShape[0] != 0 || matShape[1] % blockShape[1] != 0) {
    throw poputil::poplibs_error(
        "sparse matrix dimensions (" + std::to_string(matShape[0]) + "," +
        std::to_string(matShape[1]) + ") are not divisible by block size (" +
        std::to_string(blockShape[0]) + "," + std::to_string(blockShape[1]));
  }

  if (grainSizeZ % minMultipleGrainSizeZ) {
    throw poputil::poplibs_error("Grain size must be a multiple of " +
                                 std::to_string(minMultipleGrainSizeZ) +
                                 " for data type");
  }

  if (matShape[2] % grainSizeZ) {
    throw poputil::poplibs_error("Batch must be a multiple of grain size");
  }

  if (splitShape[2] % grainSizeZ) {
    throw poputil::poplibs_error(
        "Batch split must be a multiple of grain size");
  }

  for (unsigned i = 0; i != matShape.val.size(); ++i) {
    if (splitShape[i] > matShape[i]) {
      throw poputil::poplibs_error("split for dim " + std::to_string(i) +
                                   " must be less than size");
    }
  }

  const auto blockSize = blockShape[0] * blockShape[1];
  const bool useBlockMetaInfoFormat = blockSize > 1;
  const unsigned numWorkers = 6;
  auto nzBlocksPerfectlyUniform = sparsityLevel *
                                  (matShape[0] / blockShape[0]) *
                                  (matShape[1] / blockShape[1]);
  auto nzBucketSize = nzBlocksPerfectlyUniform * blockSize * (1 + excess);
  auto metaInfoBucketSize =
      (popsparse::dynamic::fixedMetaInfoCost(useBlockMetaInfoFormat, numWorkers,
                                             includeGradW) *
           3.0 +
       (matShape[0] / blockShape[0]) * 2.0 + nzBlocksPerfectlyUniform) *
      (1 + excess);

  const bool disableSparsityDataImplCheck =
      vm.count("disable-sparsity-data-impl-checks");

  auto createSplit = [](unsigned size, unsigned partitionSize,
                        unsigned grainSize) {
    auto grains = gccs::ceildiv(size, grainSize);
    std::vector<std::size_t> split;
    const auto grainsPerPartition = gccs::ceildiv(grains, partitionSize);
    for (unsigned i = 0; i != partitionSize; ++i) {
      const auto tileBegin = i * grainsPerPartition * grainSize;
      split.push_back(tileBegin);
    }
    return split;
  };

  const std::vector<std::size_t> grainSizes = {blockShape[0], blockShape[1],
                                               numBucketsZ};
  std::vector<std::vector<std::size_t>> splits(3);
  for (unsigned i = 0; i != 3; ++i) {
    splits[i] = createSplit(matShape[i], splitShape[i], grainSizes[i]);
  }

  const bool checkSparsityDataImpl = !disableSparsityDataImplCheck;
  return !validatePartition(
      matShape.val, grainSizes, splits[0], splits[1], splits[2], sparsityLevel,
      metaInfoBucketSize, nzBucketSize, numBucketsZ, useBlockMetaInfoFormat,
      includeGradA, includeGradW, checkSparsityDataImpl, useDense);
}
