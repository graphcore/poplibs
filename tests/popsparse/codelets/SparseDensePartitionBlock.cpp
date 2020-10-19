// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <random>

#include <boost/optional.hpp>
#include <boost/random.hpp>

#include <poplar/Graph.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>

#include "../lib/popsparse/SparseMetaInfo.hpp"
#include "SparseDensePartitionBlock.hpp"
#include <popsolver/Model.hpp>
#include <poputil/Util.hpp>

// Test functions to generate sparse tensor data and metadata for codelet
// testing

using namespace poplar;
using namespace poputil;
using namespace poplibs_support;

bool isGradWVertex(VertexType vt) {
  return vt == VertexType::GradW || vt == VertexType::GradWAmp;
}

// generate sparse block start indices
template <typename RandomEngine>
std::vector<std::array<unsigned, 2>> generateBlockSparseIndices(
    RandomEngine &randomEngine, const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &blockSize, std::size_t n) {
  const std::vector<std::size_t> blockShape = {shape[0] / blockSize[0],
                                               shape[1] / blockSize[1]};
  // Generate n random indices that are within the flattened given shape.
  std::vector<unsigned> randomIndices(product(blockShape));
  std::iota(randomIndices.begin(), randomIndices.end(), 0);
  auto randomGen = [&](unsigned max) {
    boost::random::uniform_int_distribution<unsigned> dist(0, max - 1);
    return dist(randomEngine);
  };
  boost::range::random_shuffle(randomIndices, randomGen);
  randomIndices.resize(n);

  std::vector<std::array<unsigned, 2>> rowColumnIndices(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto unflattenedIndex =
        vectorConvert<unsigned>(unflattenIndex(blockShape, randomIndices[i]));
    rowColumnIndices[i] = {
        unflattenedIndex[0] * static_cast<unsigned>(blockSize[0]),
        unflattenedIndex[1] * static_cast<unsigned>(blockSize[1])};
  }
  return rowColumnIndices;
}

unsigned
getGradWWorkerPartition(const Target &target,
                        const std::vector<unsigned> &aRowColumnCounts) {
  // Much easier than forward partitions. Just partition the total number
  // of columns of A between workers.
  const auto totalAElems = std::accumulate(
      aRowColumnCounts.begin(), aRowColumnCounts.end(), std::size_t(0));

  const auto numWorkers = target.getNumWorkerContexts();
  const auto aElemsPerWorker = ceildiv(totalAElems, numWorkers);
  const auto numAPartitions = ceildiv(totalAElems, aElemsPerWorker);

  return numAPartitions;
}

// Split the batch dimension across workers
std::vector<unsigned int> getForwardWorkerPartition(const Target &target,
                                                    unsigned bColumns) {
  auto splits = poputil::splitRegionsBetweenWorkers(target, {{0, bColumns}}, 1);
  std::vector<unsigned int> worklist(target.getNumWorkerContexts() * 2);

  unsigned index = 0;
  for (const auto split : splits) {
    for (const auto interval : split) {
      worklist.at(index) = interval.begin();
      worklist.at(index + 1) = interval.size();
      index += 2;
    }
  }
  return worklist;
}

template <typename RandomEngine>
std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const std::vector<std::size_t> &blockSize, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType, unsigned xPartition, unsigned yPartition) {

  // Factor by which row and column offsets are scaled
  const auto blockElems = product(blockSize);
  const auto fillGradWInfo = isGradWVertex(vertexType);

  // Order indices of a by column then row
  std::sort(indices.begin(), indices.end());

  std::vector<std::vector<unsigned>> metaInfo(numBuckets);
  auto garbageDist =
      boost::random::uniform_int_distribution<unsigned>(0, 0xffff);
  std::size_t nzOffset = 0;
  for (unsigned bucket = 0; bucket < numBuckets; ++bucket) {
    std::size_t splitIdx = 0;
    std::size_t numSplits = processedSubGroupIndices.at(bucket).size();
    for (std::size_t i = 0; i < otherSubGroupIds.size() + numSplits; ++i) {
      if (splitIdx != numSplits &&
          i == processedSubGroupIndices[bucket][splitIdx]) {

        std::vector<unsigned> rows;
        std::vector<unsigned> rowColumnCounts;
        boost::optional<unsigned> lastRowIndex;
        for (std::size_t nzIdx = nzOffset;
             nzIdx < nzOffset + subGroupNumElems[bucket].at(i); ++nzIdx) {
          if (!lastRowIndex || *lastRowIndex != indices.at(nzIdx)[0]) {
            rows.emplace_back();
            rowColumnCounts.emplace_back();
          }
          rows.back() = indices.at(nzIdx)[0];
          ++rowColumnCounts.back();
          lastRowIndex = rows.back();
        }

        metaInfo[bucket].emplace_back(processedSubGroupId);
        const auto processedSubGroupNumElems = subGroupNumElems[bucket].at(i);
        using T = unsigned;
        const auto subgroupEntryElems =
            sizeof(popsparse::BlockMetaInfo<T>::SubGroupEntry) / sizeof(T);
        const auto outputEntryElems =
            sizeof(popsparse::BlockMetaInfo<T>::OutputEntry) / sizeof(T);
        const auto gradWEntryElems =
            sizeof(popsparse::BlockMetaInfo<T>::GradWWorkerEntry) / sizeof(T);

        unsigned gradWNumUsedWorkers = 0;
        if (fillGradWInfo) {
          gradWNumUsedWorkers =
              getGradWWorkerPartition(target, rowColumnCounts);
        }

        const auto totalMetaInfoElems =
            subgroupEntryElems + rows.size() * outputEntryElems +
            processedSubGroupNumElems + gradWNumUsedWorkers * gradWEntryElems;
        metaInfo[bucket].emplace_back(xPartition);
        metaInfo[bucket].emplace_back(yPartition);
        metaInfo[bucket].emplace_back(processedSubGroupNumElems * blockElems);
        metaInfo[bucket].emplace_back(totalMetaInfoElems);
        metaInfo[bucket].emplace_back(rows.size() - 1);
        metaInfo[bucket].emplace_back(gradWNumUsedWorkers);

        std::vector<unsigned> metaInfoGradWWorkerEntryIndices;
        if (fillGradWInfo) {
          metaInfoGradWWorkerEntryIndices.resize(gradWNumUsedWorkers);
          for (unsigned worker = 0; worker < gradWNumUsedWorkers; ++worker) {
            metaInfoGradWWorkerEntryIndices[worker] = metaInfo[bucket].size();
            for (std::size_t i = 0; i < gradWEntryElems; ++i) {
              metaInfo[bucket].emplace_back(~0u);
            }
          }
        }
        // Output row -> column list meta-info
        std::vector<unsigned> outputEntryMetaInfoIndices(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) {
          const auto aRow = indices.at(nzOffset)[0];
          // First entry is offset into output memory to process.
          // bColumns are inner-most dimension.
          const auto aRowOffsetInC = aRow;
          outputEntryMetaInfoIndices[r] = metaInfo[bucket].size();
          metaInfo[bucket].push_back(aRowOffsetInC);
          metaInfo[bucket].push_back(rowColumnCounts[r] - 1);
          for (unsigned c = 0; c < rowColumnCounts[r]; ++c) {
            metaInfo[bucket].push_back(indices.at(nzOffset)[1]);
            ++nzOffset;
          }
        }

        if (fillGradWInfo) {
          const auto totalAElems = std::accumulate(
              rowColumnCounts.begin(), rowColumnCounts.end(), std::size_t(0));
          const auto numAElemsPerPartition =
              ceildiv(totalAElems, gradWNumUsedWorkers);
          unsigned currRowIndex = 0;
          unsigned currRowColumnIndex = 0;
          for (unsigned worker = 0; worker < gradWNumUsedWorkers; ++worker) {
            const auto sparseStartIndex = worker * numAElemsPerPartition;
            const auto sparseEndIndex =
                std::min((worker + 1) * numAElemsPerPartition, totalAElems);
            const auto numSparseElems = sparseEndIndex - sparseStartIndex;

            unsigned startRowIndex = currRowIndex;
            unsigned startRowStartColumnIndex = currRowColumnIndex;
            const auto workerEntryIndex =
                metaInfoGradWWorkerEntryIndices[worker];
            metaInfo[bucket][workerEntryIndex + 0] =
                sparseStartIndex * blockElems;
            metaInfo[bucket][workerEntryIndex + 1] =
                outputEntryMetaInfoIndices[startRowIndex] - workerEntryIndex;
            metaInfo[bucket][workerEntryIndex + 2] = startRowStartColumnIndex;
            metaInfo[bucket][workerEntryIndex + 3] = numSparseElems;

            // Advance to next worker's work
            unsigned numRemainingElems = numSparseElems;
            while (numRemainingElems > 0) {
              const auto elemsThisRow =
                  std::min(numRemainingElems,
                           rowColumnCounts[currRowIndex] - currRowColumnIndex);
              numRemainingElems -= elemsThisRow;
              currRowColumnIndex += elemsThisRow;
              if (currRowColumnIndex >= rowColumnCounts[currRowIndex]) {
                currRowColumnIndex = 0;
                currRowIndex++;
              }
            }
          }
        }
        ++splitIdx;
      } else {
        const auto otherSubGroupIdx = i - splitIdx;
        const auto subGroupId = otherSubGroupIds[otherSubGroupIdx];
        const auto numElems = subGroupNumElems[bucket][i];
        metaInfo[bucket].emplace_back(subGroupId);
        // Enforce a large numbered partition to go with sub group IDs that
        // aren't to be used.
        metaInfo[bucket].emplace_back(0xffff);
        metaInfo[bucket].emplace_back(0xffff);
        metaInfo[bucket].emplace_back(numElems * blockElems);
        // We also just use this no. of sub-elements as garbage in the meta-info
        // for the other (unprocessed) sub-groups.
        metaInfo[bucket].emplace_back(numElems + 5);
        for (std::size_t i = 0; i < numElems; ++i) {
          metaInfo[bucket].emplace_back(garbageDist(randomEngine));
        }
      }
    }
    constexpr unsigned endSubGroupId = 0;
    metaInfo[bucket].push_back(endSubGroupId);
  }
  return metaInfo;
}

template std::vector<std::array<unsigned, 2>>
generateBlockSparseIndices<std::mt19937>(
    std::mt19937 &randomEngine, const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &blockSize, std::size_t n);

template std::vector<std::vector<unsigned>>
generateMetaInfoAndPartition<std::mt19937>(
    std::mt19937 &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape,
    const std::vector<std::size_t> &blockSize, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType, unsigned xPartition, unsigned yPartition);
