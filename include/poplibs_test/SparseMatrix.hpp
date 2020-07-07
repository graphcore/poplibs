// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_SparseMatrix_hpp
#define poplibs_test_SparseMatrix_hpp

#include <boost/multi_array.hpp>
#include <boost/random.hpp>

#include "poplibs_support/VectorUtils.hpp"

#include <queue>
#include <vector>

namespace poplibs_test {
namespace sparse {

template <typename ValueType, typename IndexType>
boost::multi_array<double, 2>
csrToDenseMatrix(const ValueType *nzValues, const IndexType *columnIndices,
                 const IndexType *rowIndices,
                 const std::size_t numNonZeroValues, const std::size_t numRows,
                 const std::size_t numColumns) {
  boost::multi_array<double, 2> mat(boost::extents[numRows][numColumns]);

  std::size_t i = 0;
  for (std::size_t row = 0; row != numRows; ++row) {
    std::size_t numColEntries = rowIndices[row + 1] - rowIndices[row];
    assert(numColEntries <= numColumns);
    for (std::size_t col = 0; col != numColEntries; ++col) {
      assert(columnIndices[i] < numColumns);
      mat[row][columnIndices[i]] = nzValues[i];
      ++i;
    }
  }
  assert(i == numNonZeroValues);
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

// Build CSR matrix
template <typename ValueType, typename IndexType, typename RandomEngine>
std::tuple<std::vector<ValueType>, std::vector<IndexType>,
           std::vector<IndexType>>
buildCSRMatrix(RandomEngine &rng, const std::vector<size_t> &dimensions,
               double sparsityFactor,
               const std::vector<std::size_t> &weightedAreaBegin,
               const std::vector<std::size_t> &weightedAreaEnd,
               // Specified as a weight to make it easy to default
               // to an even distribution.
               double weightedAreaSparsityWeight,
               bool useBipolarValueDistribution) {

  std::vector<ValueType> nzValues;
  std::vector<IndexType> columnIndices;
  std::vector<IndexType> rowIndices;

  std::size_t capacityEstimate = product(dimensions) * sparsityFactor * 1.25;

  // reserve sufficient data
  rowIndices.reserve(dimensions[0]);
  columnIndices.reserve(capacityEstimate);
  nzValues.reserve(capacityEstimate);

  auto randUniform =
      boost::random::uniform_real_distribution<ValueType>(0, 1.0);
  auto randNormal = boost::random::normal_distribution<ValueType>(0, 1.0);
  auto randBernoulli = boost::random::bernoulli_distribution<ValueType>{};

  double weightedThreshold, remainingThreshold;
  std::tie(weightedThreshold, remainingThreshold) =
      calculateWeightedVsRemainingSparsityFactor(
          dimensions, sparsityFactor, weightedAreaBegin, weightedAreaEnd,
          weightedAreaSparsityWeight);

  // generating two random numbers takes time and hence if the numbers
  // are not important for any arithmetic, we use the ones already generated.
  const bool useNormal = false;

  IndexType numNzRowElements = 0;
  rowIndices.push_back(0);
  // A pool of weight vals to use rather than generating new ones
  // when !useNormal.
  std::queue<ValueType> weightVals;
  for (std::size_t row = 0; row != dimensions[0]; ++row) {
    for (ValueType col = 0; col != dimensions[1]; ++col) {
      const bool inWeightedArea =
          (row >= weightedAreaBegin.at(0) && row < weightedAreaEnd.at(0) &&
           col >= weightedAreaBegin.at(1) && col < weightedAreaEnd.at(1));
      auto u = randUniform(rng);
      if (!useNormal && !useBipolarValueDistribution) {
        weightVals.emplace(u);
      }
      const auto threshold =
          inWeightedArea ? weightedThreshold : remainingThreshold;
      if (u < threshold) {
        float y;
        if (useBipolarValueDistribution) {
          y = randBernoulli(rng) ? 1.0 : -1.0;
        } else {
          if (useNormal) {
            y = randNormal(rng);
          } else {
            y = weightVals.front() * 2 - 1;
            weightVals.pop();
          }
        }
        nzValues.push_back(y);
        columnIndices.push_back(col);
        ++numNzRowElements;
      }
    }
    rowIndices.push_back(numNzRowElements);
  }
  return std::make_tuple(nzValues, columnIndices, rowIndices);
}

} // end namespace sparse
} // end namespace poplibs_test

#endif // poplibs_test_SparseMatrix_hpp
