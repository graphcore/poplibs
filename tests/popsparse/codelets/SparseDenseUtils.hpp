// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef _popsparse_SparseDenseUtils_hpp
#define _popsparse_SparseDenseUtils_hpp

#include "poplibs_support/Algorithm.hpp"
#include "poputil/exceptions.hpp"
#include <boost/random.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <iostream>
#include <random>
#include <vector>

enum class VertexType {
  Forward,
  GradA,      // version where a separate meta information table is used
  Transposed, // Uses forward meta information table
  GradW,
  GradWAmp,
};

std::ostream &operator<<(std::ostream &os, const VertexType &vt);

std::istream &operator>>(std::istream &is, VertexType &vt);

template <typename RandomEngine>
static std::tuple<unsigned, std::vector<unsigned>>
generateSparseSubGroupIds(RandomEngine &randomEngine, std::size_t n,
                          unsigned min, unsigned max) {
  std::vector<unsigned> subGroupIds(max - min + 1);
  std::iota(subGroupIds.begin(), subGroupIds.end(), min);
  auto randomGen = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max);
    return dist(randomEngine);
  };
  boost::range::random_shuffle(subGroupIds, randomGen);
  const auto processedSubGroupId = subGroupIds.back();
  subGroupIds.resize(n - 1);
  return std::make_tuple(processedSubGroupId, subGroupIds);
}

template <typename RandomEngine>
static std::tuple<std::vector<std::vector<unsigned>>,
                  std::vector<std::vector<unsigned>>>
partitionSubGroupElems(RandomEngine &randomEngine, std::size_t numIndices,
                       unsigned numBuckets, unsigned numSplitsPerBucket,
                       unsigned numOtherSubGroups,
                       unsigned numOtherSubGroupElems) {
  std::vector<unsigned> numElemsOtherSubGroups(numOtherSubGroups);
  {
    auto randomDist = boost::random::uniform_int_distribution<unsigned>(
        1, numOtherSubGroupElems);
    unsigned sum = 0;
    for (auto &n : numElemsOtherSubGroups) {
      n = randomDist(randomEngine);
      sum += n;
    }
    unsigned newSum = 0;
    if (sum > 0) {
      for (auto &n : numElemsOtherSubGroups) {
        n = (n * numOtherSubGroupElems) / sum;
        newSum += n;
      }
    }
    // Round robin assign any rounding error to get exactly the total number we
    // asked for.
    for (std::size_t i = 0; numOtherSubGroupElems - newSum != 0;
         ++newSum, i = (i + 1) % numElemsOtherSubGroups.size()) {
      numElemsOtherSubGroups[i]++;
    }
  }
  std::vector<std::vector<unsigned>> processedSubGroupIndices(numBuckets);
  std::vector<std::vector<unsigned>> numElems(numBuckets);
  const auto randomGenForShuffle = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max - 1);
    return dist(randomEngine);
  };
  const auto maxIndicesPerBucket =
      poplibs_support::ceildiv(numIndices, numBuckets);
  for (std::size_t bucket = 0; bucket < numBuckets; ++bucket) {
    numElems[bucket].resize(numSplitsPerBucket + numOtherSubGroups);
    const auto bucketStartIndex = bucket * maxIndicesPerBucket;
    const auto bucketEndIndex =
        std::min((bucket + 1) * maxIndicesPerBucket, numIndices);
    const auto maxIndicesPerSplit = poplibs_support::ceildiv(
        bucketEndIndex - bucketStartIndex, numSplitsPerBucket);
    std::vector<unsigned> indices(numOtherSubGroups + numSplitsPerBucket);
    std::iota(indices.begin(), indices.end(), 0);
    boost::range::random_shuffle(indices, randomGenForShuffle);
    for (std::size_t splitIdx = 0; splitIdx < numSplitsPerBucket; ++splitIdx) {
      const auto subGroupIndex = indices.at(splitIdx);
      const auto startIndex = bucketStartIndex + splitIdx * maxIndicesPerSplit;
      const auto endIndex =
          std::min(bucketStartIndex + (splitIdx + 1) * maxIndicesPerSplit,
                   bucketEndIndex);
      numElems[bucket].at(subGroupIndex) = endIndex - startIndex;
    }
    for (std::size_t otherIdx = 0; otherIdx < numOtherSubGroups; ++otherIdx) {
      const auto subGroupIndex = indices.at(otherIdx + numSplitsPerBucket);
      numElems[bucket].at(subGroupIndex) = numElemsOtherSubGroups.at(otherIdx);
    }
    indices.resize(numSplitsPerBucket);
    std::sort(indices.begin(), indices.end());
    std::swap(processedSubGroupIndices[bucket], indices);
  }

  return std::make_tuple(processedSubGroupIndices, numElems);
}

#endif // _popsparse_SparseDenseUtils_hpp
