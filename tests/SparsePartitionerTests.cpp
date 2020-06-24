// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "../lib/popsparse/SparsePartitionerImpl.hpp"
#include "../lib/popsparse/SparseStorageInternal.hpp"
#include "poplar/Type.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_test/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <cmath>

using namespace poplibs_support;
using namespace poplibs_test::util;

static bool inInterval(const poplar::Interval range, std::size_t val) {
  return (val >= range.begin() && val < range.end());
}

// Build CSR matrix
popsparse::CSRMatrix<double>
buildCSRMatrix(const std::vector<size_t> &dimensions, double sparsityFactor) {
  boost::multi_array<double, 2> matrix(
      boost::extents[dimensions[0]][dimensions[1]]);
  logging::debug("Dimension in CSR Matrix generation {}", dimensions);

  std::vector<double> nzValues;
  std::vector<std::size_t> columnIndices;
  std::vector<std::size_t> rowIndices;

  boost::random::mt19937 rng;
  auto randUniform = boost::random::uniform_real_distribution<float>(0, 1.0);
  auto randNormal = boost::random::normal_distribution<float>(0, 1.0);

  std::size_t numNzRowElements = 0;
  rowIndices.push_back(0);
  for (std::size_t row = 0; row != dimensions[0]; ++row) {
    for (std::size_t col = 0; col != dimensions[1]; ++col) {
      auto x = randNormal(rng);
      if (randUniform(rng) < sparsityFactor) {
        nzValues.push_back(x);
        columnIndices.push_back(col);
        ++numNzRowElements;
      }
    }
    rowIndices.push_back(numNzRowElements);
  }

  logging::debug("NZ Values {} : {}", nzValues.size(), nzValues);
  logging::debug("Columns Indices {} : {}", columnIndices.size(),
                 columnIndices);
  logging::debug("Row Indices {} : {}", rowIndices.size(), rowIndices);

  return popsparse::CSRMatrix<double>(nzValues, columnIndices, rowIndices);
}

static bool validatePartition(const std::vector<std::size_t> &dimensions,
                              const std::vector<std::size_t> &grainSizes,
                              const std::vector<std::size_t> &xSplits,
                              const std::vector<std::size_t> &ySplits,
                              const std::vector<std::size_t> &zSplits,
                              double sparsityFactor,
                              std::size_t metaInfoBucketSize,
                              std::size_t nzElementsBucketSize,
                              std::size_t bucketsPerZ, bool transposed,
                              bool includeGradA, bool includeGradW) {

  const auto numRows = dimensions[0];
  const auto numColumns = dimensions[1];

  auto csrMatrix = buildCSRMatrix(dimensions, sparsityFactor);

  // Create partitioner object with plan information
  popsparse::PartitionerImpl<double> partitioner(
      dimensions, grainSizes, xSplits, ySplits, zSplits, metaInfoBucketSize,
      nzElementsBucketSize, 6, bucketsPerZ, includeGradA, includeGradW);

  auto pnBuckets = partitioner.createBuckets(csrMatrix);
  partitioner.overflowInfoForFwd(pnBuckets);

  // If transposed implementation, we do a transpose followed by a transpose
  if (transposed) {
    auto pnBucketsTransposed = partitioner.transposedBuckets(pnBuckets);
    pnBuckets = partitioner.transposedBuckets(pnBucketsTransposed);
  }

  // tile partitions must be less than the number of splits
  if (pnBuckets.size() !=
      xSplits.size() * ySplits.size() * zSplits.size() * bucketsPerZ) {
    return false;
  }

  // piece together information per row into a CSR format
  std::vector<std::size_t> colIndicesActual;
  std::vector<double> nzValuesActual;
  std::vector<std::size_t> rowIndicesActual;

  std::size_t numValues = 0;
  for (std::size_t row = 0; row != numRows; ++row) {
    rowIndicesActual.push_back(numValues);
    for (std::size_t col = 0; col != numColumns; ++col) {
      // find tile partition that matched
      for (const auto &pn : pnBuckets) {

        for (const auto &p : pn.subGroups) {
          auto rowInterval = p.tile.getRows();
          auto colInterval = p.tile.getColumns();

          if (inInterval(rowInterval, row) && inInterval(colInterval, col)) {
            for (const auto &r : p.tileInfo) {
              if (r.rowNumber + rowInterval.begin() == row) {
                for (const auto &c : r.positionValues) {
                  if (c.first + colInterval.begin() == col) {
                    colIndicesActual.push_back(c.first + colInterval.begin());
                    nzValuesActual.push_back(c.second);
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

  logging::debug(" Actual nz values: {} ", nzValuesActual);
  logging::debug(" expect nz values: {} ", csrMatrix.nzValues);
  logging::debug(" Actual column values: {} ", colIndicesActual);
  logging::debug(" expect column values: {} ", csrMatrix.columnIndices);
  logging::debug(" Actual row values: {} ", rowIndicesActual);
  logging::debug(" expect row values: {} ", csrMatrix.rowIndices);

  if (nzValuesActual.size() != csrMatrix.nzValues.size()) {
    return false;
  }
  if (!std::equal(nzValuesActual.begin(), nzValuesActual.end(),
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
  std::size_t grainSizeZ = 1;
  std::size_t numBucketsZ = 1;
  bool includeGradW = true;
  bool includeGradA = true;
  double excess = 0.1;

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
    ("batch-grain-size",
     po::value<std::size_t>(&grainSizeZ)->default_value(grainSizeZ),
     "Number of grains in batch dimension")
    ("excess", 
      po::value<double>(&excess)->default_value(excess),
      "Excess bucket size")
    ("num-buckets-z",
     po::value<std::size_t>(&numBucketsZ)->default_value(numBucketsZ),
     "Number of buckets per Z")
    ("include-gradw",
     po::value<bool>(&includeGradW)->default_value(includeGradW),
     "Include GradW")
    ("include-grada",
     po::value<bool>(&includeGradA)->default_value(includeGradA),
     "Include GradA")
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
  const unsigned numWorkers = 6;
  auto nzBucketSize = sparsityLevel * matShape[0] * matShape[1] * (1 + excess);
  auto metaInfoBucketSize =
      popsparse::fixedMetaInfoCost(numWorkers, includeGradW) * 3.0 +
      matShape[0] * 2.0 + nzBucketSize;

  auto createSplit = [](unsigned size, unsigned partitionSize,
                        unsigned grainSize) {
    auto grains = poplibs_support::ceildiv(size, grainSize);
    std::vector<std::size_t> split;
    const auto grainsPerPartition = ceildiv(grains, partitionSize);
    for (unsigned i = 0; i != partitionSize; ++i) {
      const auto tileBegin = i * grainsPerPartition * grainSize;
      split.push_back(tileBegin);
    }
    return split;
  };

  const std::vector<std::size_t> grainSizes = {1, 1, numBucketsZ};
  std::vector<std::vector<std::size_t>> splits(3);
  for (unsigned i = 0; i != 3; ++i) {
    splits[i] = createSplit(matShape[i], splitShape[i], grainSizes[i]);
  }

  return !validatePartition(matShape.val, grainSizes, splits[0], splits[1],
                            splits[2], sparsityLevel, metaInfoBucketSize,
                            nzBucketSize, numBucketsZ, false, includeGradA,
                            includeGradW);
}
