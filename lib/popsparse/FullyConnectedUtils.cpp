// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "FullyConnectedUtils.hpp"

#include "poplibs_support/Algorithm.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>

using namespace poplar;

namespace popsparse {

using namespace dynamic;

namespace fullyconnected {

unsigned calculateSubGroupId(unsigned numRowGroups, unsigned numSubRowGroups,
                             unsigned rowGroupIndex,
                             unsigned subRowGroupIndex) {
  return rowGroupIndex * numSubRowGroups + subRowGroupIndex + 1;
}

unsigned getNumSubGroups(unsigned numRowGroups, unsigned numSubRowGroups) {
  return numSubRowGroups * numRowGroups;
}

std::pair<unsigned, unsigned> getGroupIndices(unsigned subGroupId,
                                              unsigned numRowGroups,
                                              unsigned numSubRowGroups) {
  assert(subGroupId <= getNumSubGroups(numRowGroups, numSubRowGroups) &&
         subGroupId > 0);
  unsigned rowIndex = (subGroupId - 1) / numSubRowGroups;
  unsigned subRowIndex = (subGroupId - 1) % numSubRowGroups;
  return std::make_pair(rowIndex, subRowIndex);
}

// TODO: this could eventually be part of a class with a < operator
std::pair<std::size_t, std::size_t>
distanceToSubGroup(unsigned srcId, unsigned dstId, unsigned numRowGroups,
                   unsigned numSubRowGroups) {
  unsigned srcRowGroupIndex, srcSubRowGroupIndex;
  std::tie(srcRowGroupIndex, srcSubRowGroupIndex) =
      getGroupIndices(srcId, numRowGroups, numSubRowGroups);
  unsigned dstRowGroupIndex, dstSubRowGroupIndex;
  std::tie(dstRowGroupIndex, dstSubRowGroupIndex) =
      getGroupIndices(dstId, numRowGroups, numSubRowGroups);

  const auto rowGroupDist =
      (numRowGroups + dstRowGroupIndex - srcRowGroupIndex) % numRowGroups;

  const auto subRowGroupDist =
      (numSubRowGroups + dstSubRowGroupIndex - srcSubRowGroupIndex) %
      numSubRowGroups;
  return std::make_pair(rowGroupDist, subRowGroupDist);
}

// Factors outer-most dimensions of tensor t by the factors given in `factors`
// into 2 parts and makes factors inner dimensions compared to the original
// dims. e.g. A given tensor with shape {4,6,4} and factors {1,2}, we first
// divide shape into {4/1,1,6/2,2,4} and then shuffle to {4/1,6/2,1,2,4}.
Tensor factorDims(const Tensor &t, const std::vector<std::size_t> &fs,
                  unsigned startDim) {
  const auto numDims = fs.size();
  std::vector<std::size_t> factoredShape(numDims * 2);
  for (std::size_t d = 0; d < numDims; ++d) {
    assert(t.dim(startDim + d) % fs[d] == 0);
    factoredShape[d * 2 + 0] = t.dim(startDim + d) / fs[d];
    factoredShape[d * 2 + 1] = fs[d];
  }

  std::vector<unsigned> unInterleaveDimsDest(numDims * 2);
  std::iota(unInterleaveDimsDest.begin(), unInterleaveDimsDest.end(), startDim);
  std::vector<unsigned> unInterleaveDimsSource(unInterleaveDimsDest.size());
  // Shuffle source dims so they're interleaved
  for (std::size_t d = 0; d < numDims; ++d) {
    unInterleaveDimsSource[d] = unInterleaveDimsDest[d * 2 + 0];
    unInterleaveDimsSource[numDims + d] = unInterleaveDimsDest[d * 2 + 1];
  }
  return t.reshapePartial(startDim, startDim + numDims, factoredShape)
      .dimShufflePartial(unInterleaveDimsSource, unInterleaveDimsDest);
}

// Opposite of the above transformation. Does not need information per dim
// because it is present in the tensor. Just needs number of dims.
Tensor unfactorDims(const Tensor &t_, unsigned numDims, unsigned startDim) {
  auto t = t_;
  std::vector<unsigned> interleaveDimsSource(numDims * 2);
  std::vector<unsigned> interleaveDimsDest(numDims * 2);
  std::iota(interleaveDimsDest.begin(), interleaveDimsDest.end(), startDim);
  // Shuffle dest dims so they're interleaved
  for (std::size_t d = 0; d < numDims; ++d) {
    interleaveDimsSource[d * 2 + 0] = interleaveDimsDest[d];
    interleaveDimsSource[d * 2 + 1] = interleaveDimsDest[numDims + d];
  }
  t = t.dimShufflePartial(interleaveDimsSource, interleaveDimsDest);

  std::vector<std::size_t> unfactoredShape(numDims);
  for (std::size_t d = 0; d < numDims; ++d) {
    unfactoredShape[d] =
        t.dim(startDim + d * 2 + 0) * t.dim(startDim + d * 2 + 1);
  }
  return t.reshapePartial(startDim, startDim + numDims * 2, unfactoredShape);
}

Tensor inputExternalToInternalShape(const Tensor &t, std::size_t numGroups) {
  assert(t.rank() == 2);
  assert(t.dim(1) % numGroups == 0);
  return t.reshapePartial(1, 2, {numGroups, t.dim(1) / numGroups})
      .dimShuffle({1, 2, 0});
}

Tensor inputInternalToExternalShape(const Tensor &t, std::size_t numGroups) {
  assert(t.rank() == 3);
  assert(t.dim(0) == numGroups);
  return t.flatten(0, 2).dimShuffle({1, 0});
}

Tensor weightsExternalToInternalShape(const Tensor &t,
                                      std::size_t elemsPerBucket) {
  assert(t.rank() == 1);
  assert(t.numElements() % elemsPerBucket == 0);
  return t.reshape({t.numElements() / elemsPerBucket, elemsPerBucket});
}

Tensor weightsInternalToExternalShape(const Tensor &t,
                                      std::size_t elemsPerBucket) {
  assert(t.rank() == 2);
  assert(t.dim(1) == elemsPerBucket);
  return t.flatten();
}

SparseTensor weightsExternalToInternalShape(const SparseTensor &t,
                                            std::size_t metaInfoElemsPerBucket,
                                            std::size_t nzValuesPerBucket) {
  return SparseTensor(
      weightsExternalToInternalShape(t.getMetaInfoTensor(),
                                     metaInfoElemsPerBucket),
      weightsExternalToInternalShape(t.getNzValuesTensor(), nzValuesPerBucket));
}

SparseTensor weightsInternalToExternalShape(const SparseTensor &t,
                                            std::size_t metaInfoElemsPerBucket,
                                            std::size_t nzValuesPerBucket) {
  return SparseTensor(
      weightsInternalToExternalShape(t.getMetaInfoTensor(),
                                     metaInfoElemsPerBucket),
      weightsInternalToExternalShape(t.getNzValuesTensor(), nzValuesPerBucket));
}

Tensor weightsInternalSliceBuckets(const Tensor &t, std::size_t offset,
                                   std::size_t numElems) {
  assert(t.rank() == 2);
  assert(offset < t.dim(1));
  assert(offset + numElems <= t.dim(1));

  return t.slice(offset, offset + numElems, 1);
}

SparseTensor weightsInternalSliceBuckets(const SparseTensor &t,
                                         std::size_t offset,
                                         std::size_t numElems) {
  return SparseTensor(
      weightsInternalSliceBuckets(t.getMetaInfoTensor(), offset, numElems),
      t.getNzValuesTensor());
}

SparseTensor packWeights(const SparseTensor &buckets,
                         std::size_t metaInfoElemsPerBucket,
                         std::size_t nzValuesPerBucket,
                         const Tensor &overflowInfo) {
  const auto metaInfo =
      concat(overflowInfo.flatten(),
             weightsInternalToExternalShape(buckets.getMetaInfoTensor(),
                                            metaInfoElemsPerBucket));
  return SparseTensor(
      metaInfo, weightsInternalToExternalShape(buckets.getNzValuesTensor(),
                                               nzValuesPerBucket));
}

std::tuple<Tensor, Tensor> unpackWeights(const Tensor &metaInfo,
                                         std::size_t overflowInfoElems,
                                         std::size_t metaInfoElemsPerBucket) {
  const auto overflowInfo = metaInfo.slice(0, overflowInfoElems);
  return std::make_tuple(
      weightsExternalToInternalShape(
          metaInfo.slice(overflowInfoElems, metaInfo.numElements()),
          metaInfoElemsPerBucket),
      overflowInfo);
}

std::tuple<SparseTensor, Tensor>
unpackWeights(const SparseTensor &weights, std::size_t overflowInfoElems,
              std::size_t metaInfoElemsPerBucket,
              std::size_t nzValuesPerBucket) {
  const auto &metaInfo = weights.getMetaInfoTensor();
  const auto &nzValues = weights.getNzValuesTensor();
  const auto overflowInfo = metaInfo.slice(0, overflowInfoElems);
  return std::make_tuple(
      SparseTensor(
          weightsExternalToInternalShape(
              metaInfo.slice(overflowInfoElems, metaInfo.numElements()),
              metaInfoElemsPerBucket),
          weightsExternalToInternalShape(nzValues, nzValuesPerBucket)),
      overflowInfo);
}

std::size_t getNumOverflowInfoElems(std::size_t metaInfoTypeBits,
                                    std::size_t xSplits, std::size_t ySplits,
                                    std::size_t zSplits) {
  return 3 + poplibs_support::ceildiv(xSplits, metaInfoTypeBits);
}

std::vector<Tile>
splitTileBetweenWorkers(std::size_t numRows, std::size_t numColumns,
                        std::size_t numWorkers,
                        const std::vector<std::size_t> &rowWeights) {
  assert(rowWeights.empty() || rowWeights.size() == numRows);
  std::vector<Tile> workers;
  workers.reserve(numWorkers);

  // special case based on heuristic. Extend this to more
  // cases when numWorkers % numColumns == 0
  if (rowWeights.size() == numRows && numRows > 2 * numWorkers) {

    // find partials sums
    std::vector<std::size_t> partialSums;
    partialSums.resize(rowWeights.size());
    std::partial_sum(rowWeights.begin(), rowWeights.end(), partialSums.begin());

    // find average cost per worker
    std::size_t averageCost =
        poplibs_support::ceildiv(partialSums.back(), numWorkers);

    std::size_t begin = 0;
    for (std::size_t w = 0; w != numWorkers; ++w) {
      // it may be more accurate to do totalSum * (w + 1) / numWorkers;
      auto it = std::upper_bound(partialSums.begin(), partialSums.end(),
                                 (w + 1) * averageCost);
      auto dist =
          static_cast<std::size_t>(std::distance(partialSums.begin(), it));
      if (begin == dist) {
        continue;
      }
      const auto end = (w + 1 == numWorkers) ? numRows : dist;

      workers.push_back(
          Tile(poplar::Interval(begin, end), poplar::Interval(0, numColumns)));

      if (it == partialSums.end()) {
        break;
      }
      begin = end;
    }
    return workers;
  }
  auto grainsPerWorker = (numRows * numColumns + numWorkers - 1) / numWorkers;
  auto rowsPerWorker =
      std::min((grainsPerWorker + numColumns - 1) / numColumns, numRows);
  auto colsPerWorker = std::min(
      (grainsPerWorker + rowsPerWorker - 1) / rowsPerWorker, numColumns);

  const std::size_t numRowSplits = numRows / rowsPerWorker;
  const std::size_t numColSplits = numColumns / colsPerWorker;

  for (std::size_t row = 0; row != numRowSplits; ++row) {
    for (std::size_t col = 0; col != numColSplits; ++col) {
      const auto endRow =
          row == numRowSplits - 1 ? numRows : (row + 1) * rowsPerWorker;
      const auto endCol =
          col == numColSplits - 1 ? numColumns : (col + 1) * colsPerWorker;

      workers.push_back(
          Tile({row * rowsPerWorker, endRow}, {col * colsPerWorker, endCol}));
    }
  }

  std::size_t numRemaining = numWorkers - workers.size();

  // Allocate remaining if required by splitting largest first
  while (numRemaining) {
    std::sort(workers.begin(), workers.end(),
              [](const Tile &a, const Tile &b) { return a.size() > b.size(); });

    bool anySplit = false;
    auto &cand = workers.front();
    if (cand.getRows().size() > cand.getColumns().size()) {
      // split row
      std::size_t newRowSize = (cand.getRows().size() + 1) / 2;
      if (newRowSize != cand.getRows().size()) {
        auto rowStart = cand.getRows().begin();
        auto rowEnd = cand.getRows().begin() + newRowSize;
        workers.push_back(
            Tile(poplar::Interval(rowStart, rowEnd), cand.getColumns()));
        // update Tile
        cand = Tile(poplar::Interval(cand.getRows().begin() + newRowSize,
                                     cand.getRows().end()),
                    cand.getColumns());
        anySplit = true;
      }
    } else {
      // split column
      std::size_t newColSize = (cand.getColumns().size() + 1) / 2;
      if (newColSize != cand.getColumns().size()) {
        workers.push_back(
            Tile(cand.getRows(),
                 poplar::Interval(cand.getColumns().begin(),
                                  cand.getColumns().begin() + newColSize)));
        // update Tile
        cand = Tile(cand.getRows(),
                    poplar::Interval(cand.getColumns().begin() + newColSize,
                                     cand.getColumns().end()));
        anySplit = true;
      }
    }
    // There was nothing to split
    if (!anySplit) {
      break;
    }
    --numRemaining;
  }

  return workers;
}

} // end namespace fullyconnected
} // end namespace popsparse
