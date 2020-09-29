// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <random>

#include <boost/optional.hpp>
#include <boost/random.hpp>

#include <poplar/Graph.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>

#include "SparseDensePartition.hpp"
#include <popsolver/Model.hpp>
#include <poputil/Util.hpp>

// Test functions to generate sparse tensor data and metadata for codelet
// testing

using namespace poplar;
using namespace poputil;
using namespace poplibs_support;

template <typename RandomEngine>
std::vector<std::array<unsigned, 2>>
generateSparseIndices(RandomEngine &randomEngine,
                      const std::vector<std::size_t> &shape, std::size_t n) {
  // Generate n random indices that are within the flattened given shape.
  std::vector<unsigned> randomIndices(product(shape));
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
        vectorConvert<unsigned>(unflattenIndex(shape, randomIndices[i]));
    rowColumnIndices[i] = {unflattenedIndex[0], unflattenedIndex[1]};
  }
  return rowColumnIndices;
}

std::tuple<unsigned, unsigned, unsigned>
getForwardWorkerPartition(const Target &target, const Type &inputType,
                          const unsigned bColumns, const unsigned aRows,
                          const std::vector<unsigned> &aRowColumnCounts) {
  // Split rows of a and columns of b between workers.
  //
  // For this functional test we'll just first split columns of b, then
  // rows of a to try and utilise all workers.
  //
  // NOTE: A problem with this is that by needing a contiguous range of
  // columns to process, in the given order, we are restricting how work
  // can be split. We cannot change the order of columns because that
  // would make things difficult for other passes. We could do some more
  // heavy encoding to allow interleaved columns to be selected for
  // workers but it's more memory to encode.
  const auto bColumnGrainSize = target.getVectorWidth(inputType);
  const auto bColumnGrains = ceildiv(bColumns, bColumnGrainSize);

  popsolver::Model m;
  const auto mWorkerARowsPartition = m.addVariable(1, aRows);
  const auto mWorkerBColumnGrainsPartition = m.addVariable(1, bColumnGrains);

  const auto mNumARows = m.addConstant(aRows);
  const auto mNumBColumnGrains = m.addConstant(bColumnGrains);

  const auto mWorkerARows = m.ceildiv(mNumARows, mWorkerARowsPartition);
  const auto mWorkerBColumnGrains =
      m.ceildiv(mNumBColumnGrains, mWorkerBColumnGrainsPartition);

  const auto numWorkers = target.getNumWorkerContexts();
  const auto mMaxWorkerAElems = m.call<unsigned>(
      {mWorkerARows},
      [&](const std::vector<unsigned> &values) -> popsolver::DataType {
        const auto workerARows = values[0];
        unsigned maxWorkerAElems = 0;
        for (unsigned worker = 0; worker < numWorkers; ++worker) {
          const auto elems = std::accumulate(
              aRowColumnCounts.begin() +
                  std::min<unsigned>(worker * workerARows, aRows),
              aRowColumnCounts.begin() +
                  std::min<unsigned>((worker + 1) * workerARows, aRows),
              0u);
          maxWorkerAElems = std::max<unsigned>(maxWorkerAElems, elems);
        }
        return popsolver::DataType{maxWorkerAElems};
      });

  m.lessOrEqual(
      m.product({mWorkerARowsPartition, mWorkerBColumnGrainsPartition}),
      popsolver::DataType{numWorkers});

  const auto mMaxWorkerGrains =
      m.product({mMaxWorkerAElems, mWorkerBColumnGrains});

  const auto s = m.minimize(mMaxWorkerGrains);
  if (!s.validSolution()) {
    throw poplibs_error("Failed to find a plan to split work between workers!");
  }

  const auto numAPartitions = s[mWorkerARowsPartition].getAs<unsigned>();
  const auto numBPartitions =
      s[mWorkerBColumnGrainsPartition].getAs<unsigned>();
  return std::make_tuple(bColumnGrainSize, numAPartitions, numBPartitions);
}

std::tuple<unsigned, unsigned>
getGradWWorkerPartition(const Target &target, const Type &inputType,
                        const unsigned bColumns, const unsigned aRows,
                        const std::vector<unsigned> &aRowColumnCounts) {
  // Much easier than forward partitions. Just partition the total number
  // of columns of A between workers.
  const auto totalAElems = std::accumulate(
      aRowColumnCounts.begin(), aRowColumnCounts.end(), std::size_t(0));

  // Grain size of columns of b is the same as for forward for GradW codelet.
  const auto bColumnGrainSize = target.getVectorWidth(inputType);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto aElemsPerWorker = ceildiv(totalAElems, numWorkers);
  const auto numAPartitions = ceildiv(totalAElems, aElemsPerWorker);

  return std::make_tuple(bColumnGrainSize, numAPartitions);
}

template <typename RandomEngine>
std::vector<std::vector<unsigned>> generateMetaInfoAndPartition(
    RandomEngine &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType) {
  const auto generateGradWMetaInfo = vertexType == VertexType::GradW;
  const auto generateGradAMetaInfo = vertexType == VertexType::GradA;

  // Factor by which row and column offsets are scaled
  const auto yOffsetFactor = inputType == FLOAT ? 4 : 2;
  const auto xOffsetFactor = inputType == FLOAT ? 1 : 2;

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

        const auto bColumns = bShape[1];

        unsigned fwdBColumnGrainSize, fwdNumAPartitions, fwdNumBPartitions;
        std::tie(fwdBColumnGrainSize, fwdNumAPartitions, fwdNumBPartitions) =
            getForwardWorkerPartition(target, inputType, bColumns, rows.size(),
                                      rowColumnCounts);

        const auto fwdBColumnGrains = ceildiv(bColumns, fwdBColumnGrainSize);
        const auto fwdNumUsedWorkers = fwdNumAPartitions * fwdNumBPartitions;
        const auto fwdMaxPartitionARows =
            ceildiv(rows.size(), fwdNumAPartitions);
        const auto fwdMaxPartitionBColumnGrains =
            ceildiv(fwdBColumnGrains, fwdNumBPartitions);

        std::vector<unsigned> fwdPartitionAElemOffsets(fwdNumAPartitions, 0);
        for (unsigned partition = 1; partition < fwdNumAPartitions;
             ++partition) {
          const auto prevPartitionARowStart =
              (partition - 1) * fwdMaxPartitionARows;
          const auto prevPartitionARowEnd = partition * fwdMaxPartitionARows;
          fwdPartitionAElemOffsets[partition] =
              fwdPartitionAElemOffsets[partition - 1] +
              std::accumulate(rowColumnCounts.begin() + prevPartitionARowStart,
                              rowColumnCounts.begin() + prevPartitionARowEnd,
                              unsigned(0));
        }

        unsigned gradWNumUsedWorkers, gradWBColumnGrainSize,
            gradWNumAPartitions = 0;
        if (generateGradWMetaInfo) {
          std::tie(gradWBColumnGrainSize, gradWNumAPartitions) =
              getGradWWorkerPartition(target, inputType, bColumns, rows.size(),
                                      rowColumnCounts);
          gradWNumUsedWorkers = gradWNumAPartitions;
        }

        metaInfo[bucket].emplace_back(processedSubGroupId);
        const auto processedSubGroupNumElems = subGroupNumElems[bucket].at(i);
        const auto elemsPerNzElem = generateGradAMetaInfo ? 2 : 1;
        // No concept of a partition in codelet tests as yet
        const auto xPartition = 0;
        const auto yPartition = 0;
        metaInfo[bucket].emplace_back(xPartition);
        metaInfo[bucket].emplace_back(yPartition);
        metaInfo[bucket].emplace_back(processedSubGroupNumElems);
        const auto totalMetaInfoElems =
            9 + fwdNumUsedWorkers * 5 + rows.size() * 2 +
            processedSubGroupNumElems * elemsPerNzElem +
            (generateGradWMetaInfo ? 1 + 4 * gradWNumUsedWorkers : 0);
        const auto offsetToFirstOutputEntry =
            totalMetaInfoElems -
            (rows.size() * 2 + processedSubGroupNumElems * elemsPerNzElem);

        metaInfo[bucket].emplace_back(totalMetaInfoElems);
        metaInfo[bucket].emplace_back(bColumns);
        metaInfo[bucket].emplace_back(rows.size() - 1);
        metaInfo[bucket].emplace_back(offsetToFirstOutputEntry);
        metaInfo[bucket].emplace_back(fwdNumUsedWorkers);

        // Reserve space for worker entries
        std::vector<std::size_t> metaInfoFwdWorkerEntryIndices(
            fwdNumUsedWorkers);
        for (unsigned worker = 0; worker < fwdNumUsedWorkers; ++worker) {
          metaInfoFwdWorkerEntryIndices[worker] = metaInfo[bucket].size();
          for (std::size_t i = 0; i < 5; ++i) {
            metaInfo[bucket].emplace_back(~0u);
          }
        }

        // If needed reserve space for GradW worker entries
        std::vector<unsigned> metaInfoGradWWorkerEntryIndices;
        if (generateGradWMetaInfo) {
          metaInfo[bucket].emplace_back(gradWNumUsedWorkers);

          metaInfoGradWWorkerEntryIndices.resize(gradWNumUsedWorkers);
          for (unsigned worker = 0; worker < gradWNumUsedWorkers; ++worker) {
            metaInfoGradWWorkerEntryIndices[worker] = metaInfo[bucket].size();
            for (std::size_t i = 0; i < 4; ++i) {
              metaInfo[bucket].emplace_back(~0u);
            }
          }
        }

        // Output row -> column list meta-info
        std::vector<unsigned> outputEntryMetaInfoIndices(rows.size());
        std::size_t offsetGradA = 0;
        for (std::size_t r = 0; r < rows.size(); ++r) {
          const auto aRow = indices.at(nzOffset)[0];
          // First entry is offset into output memory to process.
          // bColumns are inner-most dimension.
          const auto aRowOffsetInC = aRow * bColumns;
          outputEntryMetaInfoIndices[r] = metaInfo[bucket].size();
          metaInfo[bucket].push_back(aRowOffsetInC * xOffsetFactor);
          // Use 1 less for float input type
          metaInfo[bucket].push_back(rowColumnCounts[r]);
          for (unsigned c = 0; c < rowColumnCounts[r]; ++c) {
            if (generateGradAMetaInfo) {
              metaInfo[bucket].push_back(yOffsetFactor * offsetGradA++);
            }
            metaInfo[bucket].push_back(indices.at(nzOffset)[1] * bColumns *
                                       yOffsetFactor);
            ++nzOffset;
          }
        }

        // Fill out worklist info for each worker
        for (unsigned worker = 0; worker < fwdNumUsedWorkers; ++worker) {
          const auto aPartitionIdx = worker % fwdNumAPartitions;
          const auto bPartitionIdx = worker / fwdNumAPartitions;
          const auto aRowIndex = aPartitionIdx * fwdMaxPartitionARows;
          const auto aRowEndIndex =
              std::min((aPartitionIdx + 1) * fwdMaxPartitionARows, rows.size());
          const auto bColumnIndex = bPartitionIdx *
                                    fwdMaxPartitionBColumnGrains *
                                    fwdBColumnGrainSize;
          const auto bColumnEndIndex =
              std::min((bPartitionIdx + 1) * fwdMaxPartitionBColumnGrains *
                           fwdBColumnGrainSize,
                       bColumns);
          const auto workerEntryIndex = metaInfoFwdWorkerEntryIndices[worker];
          metaInfo[bucket][workerEntryIndex + 0] =
              generateGradAMetaInfo ? 0
                                    : fwdPartitionAElemOffsets[aPartitionIdx];
          metaInfo[bucket][workerEntryIndex + 1] =
              bColumnEndIndex - bColumnIndex;
          metaInfo[bucket][workerEntryIndex + 2] = bColumnIndex;
          // Number is 1 less
          metaInfo[bucket][workerEntryIndex + 3] = aRowEndIndex - aRowIndex - 1;
          metaInfo[bucket][workerEntryIndex + 4] =
              outputEntryMetaInfoIndices[aRowIndex] - workerEntryIndex;
        }

        if (generateGradWMetaInfo) {
          const auto totalAElems = std::accumulate(
              rowColumnCounts.begin(), rowColumnCounts.end(), std::size_t(0));
          const auto numAElemsPerPartition =
              ceildiv(totalAElems, gradWNumAPartitions);
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
            metaInfo[bucket][workerEntryIndex + 0] = sparseStartIndex;
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
        metaInfo[bucket].emplace_back(numElems);
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
generateSparseIndices<std::mt19937>(std::mt19937 &randomEngine,
                                    const std::vector<std::size_t> &shape,
                                    std::size_t n);

template std::vector<std::vector<unsigned>>
generateMetaInfoAndPartition<std::mt19937>(
    std::mt19937 &randomEngine, std::vector<std::array<unsigned, 2>> &indices,
    const std::vector<std::size_t> &aShape,
    const std::vector<std::size_t> &bShape, unsigned numBuckets,
    unsigned processedSubGroupId, const std::vector<unsigned> &otherSubGroupIds,
    const std::vector<std::vector<unsigned>> processedSubGroupIndices,
    const std::vector<std::vector<unsigned>> &subGroupNumElems,
    const Target &target, const Type &inputType, const Type &partialType,
    VertexType vertexType);
