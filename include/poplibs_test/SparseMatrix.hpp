// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_SparseMatrix_hpp
#define poplibs_test_SparseMatrix_hpp

#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_test/Util.hpp"
#include <boost/multi_array.hpp>
#include <boost/random.hpp>
#include <gccs/Algorithm.hpp>

#include <queue>
#include <vector>

namespace poplibs_test {
namespace sparse {

template <typename ValueType, typename IndexType>
boost::multi_array<double, 2>
csrToDenseMatrix(const ValueType *nzValues, const IndexType *columnIndices,
                 const IndexType *rowIndices,
                 const std::size_t numNonZeroValues, const std::size_t numRows,
                 const std::size_t numColumns, const std::size_t blockRows,
                 const std::size_t blockCols) {
  boost::multi_array<double, 2> mat(boost::extents[numRows][numColumns]);
  const auto blockArea = blockRows * blockCols;
  std::size_t i = 0;
  for (std::size_t row = 0; row != numRows; row += blockRows) {
    const auto bRow = row / blockRows;
    std::size_t numColEntries =
        (rowIndices[bRow + 1] - rowIndices[bRow]) / blockArea;
    assert(numColEntries <= numColumns);
    for (std::size_t col = 0; col != numColEntries; ++col) {
      assert(columnIndices[i] < numColumns);
      // nzValues kept in row major order
      for (std::size_t r = 0; r != blockRows; ++r) {
        for (std::size_t c = 0; c != blockCols; ++c) {
          mat[row + r][columnIndices[i] + c] =
              nzValues[i * blockArea + r * blockCols + c];
        }
      }
      ++i;
    }
  }
  assert(i * blockArea == numNonZeroValues);
  return mat;
}

std::tuple<double, double> calculateWeightedVsRemainingSparsityFactor(
    const std::vector<std::size_t> &dimensions, double sparsityFactor,
    const std::vector<std::size_t> &weightedAreaBegin,
    const std::vector<std::size_t> &weightedAreaEnd,
    double weightedAreaSparsityWeight) {
  // Calculate the likelihood of generating an element in the denser area
  // vs. the rest of the matrix.
  const auto matrixArea = product(dimensions);
  const auto weightedAreaShape = [&] {
    auto v = weightedAreaEnd;
    assert(weightedAreaEnd.size() == weightedAreaBegin.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      v.at(i) -= weightedAreaBegin.at(i);
    }
    return v;
  }();
  const auto weightedArea = product(weightedAreaShape);
  if (weightedArea == 0) {
    return std::make_tuple(0.0, sparsityFactor);
  }

  // We want to weight the sparsity factor used in the weighted area
  // up to a maximum which is the most elements we can fit into the
  // weighted area.
  const auto totalExpectedElems = std::round(matrixArea * sparsityFactor);
  const auto maxElemsInWeightedArea =
      std::min<double>(totalExpectedElems, weightedArea);
  const auto uniformElemsInWeightedArea = weightedArea * sparsityFactor;
  const auto weightedElemsInWeightedArea =
      std::min(maxElemsInWeightedArea,
               uniformElemsInWeightedArea * weightedAreaSparsityWeight);

  const double weightedThreshold = weightedElemsInWeightedArea / weightedArea;
  const double remainingThreshold =
      (totalExpectedElems - weightedElemsInWeightedArea) /
      double(matrixArea - weightedArea);
  return std::make_tuple(weightedThreshold, remainingThreshold);
}

// Given sparsity and weighting, determine if a bipolar distribution could
// be used in the matrix multiplication
bool floatingPointCouldRepresentMaxAccum(
    const std::vector<std::size_t> &dimensions,
    const std::vector<std::size_t> &blockDimensions,
    const std::vector<std::size_t> &weightedAreaBegin,
    const std::vector<std::size_t> &weightedAreaEnd, const poplar::Type &type,
    double sparsityFactor, double weightedAreaWeighting) {
  const auto maxVal = poplibs_test::util::maxContiguousInteger(type);
  double weightedThreshold, remainingThreshold;
  std::tie(weightedThreshold, remainingThreshold) =
      poplibs_test::sparse::calculateWeightedVsRemainingSparsityFactor(
          {dimensions[0] / blockDimensions[0],
           dimensions[1] / blockDimensions[1]},
          sparsityFactor,
          {weightedAreaBegin[0] / blockDimensions[0],
           weightedAreaBegin[1] / blockDimensions[1]},
          {weightedAreaEnd[0] / blockDimensions[0],
           weightedAreaEnd[1] / blockDimensions[1]},
          weightedAreaWeighting);

  const auto numWeightedK = (weightedAreaEnd[1] - weightedAreaBegin[1]);
  const auto numWeightedM = (weightedAreaEnd[0] - weightedAreaBegin[0]);
  std::size_t maxK = numWeightedK * weightedThreshold +
                     (dimensions[1] - numWeightedK) * remainingThreshold;
  std::size_t maxM = numWeightedM * weightedThreshold +
                     (dimensions[0] - numWeightedM) * remainingThreshold;
  maxM = gccs::alignPrev(maxM, blockDimensions[0]);
  maxK = gccs::alignPrev(maxK, blockDimensions[1]);

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
  if (wiggleRoom * getOpsPerOutputElementEstimate(false) > maxVal * modifier) {
    return false;
  }
  return true;
}

// Build CSR matrix
template <typename ValueType, typename IndexType, typename RandomEngine>
std::tuple<std::vector<ValueType>, std::vector<IndexType>,
           std::vector<IndexType>>
buildCSRMatrix(RandomEngine &rng, const std::vector<size_t> &dimensions,
               const std::vector<std::size_t> &blockSize, double sparsityFactor,
               const std::vector<std::size_t> &weightedAreaBegin,
               const std::vector<std::size_t> &weightedAreaEnd,
               // Specified as a weight to make it easy to default
               // to an even distribution.
               double weightedAreaSparsityWeight,
               bool useBipolarValueDistribution) {

  const std::size_t blockRows = blockSize.at(0);
  const std::size_t blockCols = blockSize.at(1);
  const auto blockArea = blockRows * blockCols;
  std::vector<ValueType> nzValues;
  std::vector<IndexType> columnIndices;
  std::vector<IndexType> rowIndices;

  std::size_t capacityEstimate =
      product(dimensions) / blockArea * sparsityFactor * 1.25;

  // reserve sufficient data
  rowIndices.reserve(dimensions[0] / blockRows);
  columnIndices.reserve(capacityEstimate);
  nzValues.reserve(capacityEstimate * (blockArea));

  auto randUniform =
      boost::random::uniform_real_distribution<ValueType>(0, 1.0);
  auto randNormal = boost::random::normal_distribution<ValueType>(0, 1.0);
  auto randBernoulli = boost::random::bernoulli_distribution<ValueType>{};

  double weightedThreshold, remainingThreshold;
  std::tie(weightedThreshold, remainingThreshold) =
      calculateWeightedVsRemainingSparsityFactor(
          dimensions, sparsityFactor,
          {weightedAreaBegin[0] / blockRows, weightedAreaBegin[1] / blockCols},
          {weightedAreaEnd[0] / blockRows, weightedAreaEnd[1] / blockCols},
          weightedAreaSparsityWeight);

  // generating two random numbers takes time and hence if the numbers
  // are not important for any arithmetic, we use the ones already generated.
  const bool useNormal = false;

  IndexType numNzRowElements = 0;
  rowIndices.push_back(0);

  for (std::size_t row = 0; row != dimensions[0]; row += blockRows) {
    for (ValueType col = 0; col != dimensions[1]; col += blockCols) {
      const bool inWeightedArea =
          (row >= weightedAreaBegin.at(0) && row < weightedAreaEnd.at(0) &&
           col >= weightedAreaBegin.at(1) && col < weightedAreaEnd.at(1));
      auto u = randUniform(rng);
      const auto threshold =
          inWeightedArea ? weightedThreshold : remainingThreshold;
      if (u < threshold) {
        for (std::size_t bElem = 0; bElem != blockArea; ++bElem) {
          float y;
          if (useBipolarValueDistribution) {
            y = randBernoulli(rng) ? 1.0 : -1.0;
          } else {
            if (useNormal) {
              y = randNormal(rng);
            } else {
              y = randUniform(rng) * 2 - 1;
            }
          }
          nzValues.push_back(y);
        }
        columnIndices.push_back(col);
        numNzRowElements += blockArea;
      }
    }
    rowIndices.push_back(numNzRowElements);
  }
  return std::make_tuple(nzValues, columnIndices, rowIndices);
}

} // end namespace sparse
} // end namespace poplibs_test

#endif // poplibs_test_SparseMatrix_hpp
