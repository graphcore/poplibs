// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "StaticMatMulPartitioner.hpp"
#include "MatMulOptions.hpp"
#include "PlanningCacheImpl.hpp"
#include "SparseMetaInfo.hpp"
#include "SparseStorageInternal.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/exceptions.hpp"
#include <cmath>
#include <gccs/Algorithm.hpp>
#include <limits>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poputil/OptionParsing.hpp>

using namespace poplar;
using namespace poplibs_support;
using namespace popsparse;

namespace {
// This is an estimate based on a strided reduce specialisation.
// TODO: Use common estimator functions for reduce
std::size_t reductionCycleEstimate(const Target &target, unsigned numOutputs,
                                   const Type &outputType,
                                   const Type &partialsType,
                                   unsigned reductionFactor) {
  // entry/exit + loop setup + result storage
  std::uint64_t cycles = 3 + 2 + 6 + 5;
  unsigned cyclesOuter;
  if (partialsType == HALF) {
    cyclesOuter = outputType == HALF ? 8 : 7;
  } else {
    cyclesOuter = 5;
  }
  unsigned outputsPerWorker =
      gccs::ceildiv(numOutputs, target.getNumWorkerContexts());
  unsigned accVectorWidth = partialsType == FLOAT ? 4 : 8;
  unsigned opCyclesPerVector = 1;
  cyclesOuter += 6 + opCyclesPerVector * (reductionFactor - 1);
  const auto numOuterLoops = gccs::ceildiv(outputsPerWorker, accVectorWidth);
  return (cycles + numOuterLoops * cyclesOuter) * target.getNumWorkerContexts();
}

static_::Partition::BandInfo
buildColumnBandInfo(const std::vector<unsigned> &columnCount,
                    unsigned numColumnBands, unsigned kBlock,
                    unsigned numNZBlocks) {
  static_::Partition::BandInfo bandInfo;
  unsigned averagePerBand = gccs::ceildiv(numNZBlocks, numColumnBands);
  bandInfo.boundaries.reserve(numColumnBands + 1);
  bandInfo.nzBlocksPerBand.reserve(numColumnBands);
  bandInfo.boundaries.push_back(0);
  auto it = columnCount.begin();
  bandInfo.maxBandSize = 0;
  for (unsigned c = 0; c != numColumnBands; ++c) {
    unsigned count = 0;
    while (count < averagePerBand && it != columnCount.end()) {
      count += *it++;
    }
    if (count == 0) {
      break;
    }
    bandInfo.boundaries.push_back(std::distance(columnCount.begin(), it));
    // Find if there are any zero columns in the range
    auto numZeroCols =
        std::accumulate(columnCount.begin() + bandInfo.boundaries[c],
                        columnCount.begin() + bandInfo.boundaries[c + 1], 0U,
                        [](unsigned a, unsigned b) { return a + (b == 0); });
    bandInfo.maxBandSize = std::max(bandInfo.maxBandSize,
                                    bandInfo.boundaries[c + 1] -
                                        bandInfo.boundaries[c] - numZeroCols);
    bandInfo.nzBlocksPerBand.push_back(count);
  }
  return bandInfo;
}

unsigned getNGrainSize(const Target &target, const Type &type) {
  return target.getVectorWidth(type);
}

unsigned worklistEntries(unsigned blockLength, unsigned numRows,
                         unsigned numBlocks, unsigned numWorkers) {
  using WorklistType = unsigned short;
  if (blockLength == 1) {
    return (sizeof(StaticMetaInfo<WorklistType>::WorkListEntry) * numWorkers +
            sizeof(StaticMetaInfo<WorklistType>::OutputEntry) * numRows +
            sizeof(StaticMetaInfo<WorklistType>::InputEntry)) /
           sizeof(WorklistType);
  } else {
    return (sizeof(StaticBlockMetaInfo<WorklistType>::WorkListEntry) *
                numWorkers +
            sizeof(StaticBlockMetaInfo<WorklistType>::OutputEntry) * numRows +
            sizeof(StaticBlockMetaInfo<WorklistType>::InputEntry)) /
           sizeof(WorklistType);
  }
}

// Given a number of splits, find a partition of `total` such that the grain
// size is respected. The partition can contain less than the number of splits
// if `fullPartition` is false, and guaranteed to always have zero length
// intervals. If `fullPartition` is true, the partition contains as many entries
// as the number of splits but some of them may be zero-length.
std::vector<Interval> buildPartition(std::size_t total, std::size_t numSplits,
                                     std::size_t grainSize,
                                     bool fullPartition) {
  assert(grainSize != 0);
  assert(total % grainSize == 0);
  std::vector<Interval> partition;
  partition.reserve(numSplits);
  unsigned numGrains = total / grainSize;
  unsigned grainsPerSplit = numSplits ? gccs::ceildiv(numGrains, numSplits) : 0;

  for (unsigned split = 0; split != numSplits; ++split) {
    unsigned splitBegin = std::min(grainsPerSplit * split, numGrains);
    unsigned splitEnd = std::min(grainsPerSplit * (split + 1), numGrains);
    if (splitBegin == splitEnd && !fullPartition)
      break;
    partition.emplace_back(splitBegin * grainSize, splitEnd * grainSize);
  }
  return partition;
}

std::set<std::pair<std::size_t, std::size_t>>
getUniquePartitionSizes(const std::vector<Interval> &partitionOfN,
                        const std::vector<Interval> &tilePartitionOfN) {
  assert(partitionOfN.size() == tilePartitionOfN.size());
  std::set<std::pair<std::size_t, std::size_t>> candidates;
  for (unsigned i = 0; i != partitionOfN.size(); ++i) {
    if (tilePartitionOfN[i].size() == 0) {
      return {};
    }
    candidates.emplace(partitionOfN[i].size(), tilePartitionOfN[i].size());
  }
  return candidates;
}

// Get permutation for the dimension given the dimension counts. Prune the
// dimension if pre-processing is enabled.
std::pair<std::vector<unsigned>, unsigned>
getPermutationAndPrunedSize(std::vector<unsigned> &counts, unsigned numNZBlocks,
                            unsigned dimSize, double rejectThreshold,
                            bool enablePreprocessing) {
  // sort row and column indices
  std::vector<unsigned> permutations;
  permutations.resize(counts.size());
  unsigned prunedDimSize = counts.size();
  std::iota(permutations.begin(), permutations.end(), 0);
  if (enablePreprocessing) {
    std::stable_sort(
        permutations.begin(), permutations.end(),
        [&](unsigned a, unsigned b) { return counts[a] > counts[b]; });
    // Take the expectation of number of non-zero blocks per-row/column if they
    // were uniformly distributed and reject rows/columns with fewer non-zero
    // blocks than some proportion of this expectation (rejectThreshold).
    auto it = std::lower_bound(permutations.begin(), permutations.end(),
                               numNZBlocks, [&](unsigned i, unsigned blocks) {
                                 return static_cast<double>(counts[i]) >
                                        rejectThreshold * blocks / dimSize;
                               });
    if (it != permutations.end()) {
      prunedDimSize = std::max(
          static_cast<unsigned>(std::distance(permutations.begin(), it)), 1U);
    }
    // indices are unique and hence stable sort is not needed. We could
    // randomise these so that indices with high weights get randomised.
    // But given typical placement of input data, there could be slightly higher
    // exchange memory/cycle cost.
    std::sort(permutations.begin(), permutations.begin() + prunedDimSize);
    std::vector<unsigned> newCounts(counts.size());
    for (unsigned i = 0; i != counts.size(); ++i) {
      newCounts[i] = counts[permutations[i]];
    }
    counts = std::move(newCounts);
  }
  return std::make_pair(permutations, prunedDimSize);
}
} // unnamed namespace

namespace popsparse {
namespace static_ {

std::size_t zeroingCycles(const Target &target, unsigned m, unsigned n,
                          unsigned blockLength, const Type &type) {
  unsigned bytesToZero = m * blockLength * n * target.getTypeSize(type);
  unsigned perWorker =
      gccs::ceildiv(bytesToZero * 8 / target.getDataPathWidth(),
                    target.getNumWorkerContexts());
  return (12 + perWorker) * target.getNumWorkerContexts();
}

// If we wanted to pass a set of columns per row, we would need to pass
// std::vector<std::vector<unsigned>> instead of m and k
// m and k are after dividing by block lengths
std::size_t matmulComputeCycles(const Target &target, unsigned m, unsigned k,
                                unsigned n, unsigned blockLength,
                                const Type &dataType,
                                const Type & /*partialsType*/) {
  unsigned typeSize = dataType == FLOAT ? 4 : 2;
  unsigned numWorkers = target.getNumWorkerContexts();
  unsigned cweiLoadBytesPerCycle = 8;
  auto coeffLoad = [&]() {
    return blockLength * blockLength * typeSize / cweiLoadBytesPerCycle;
  };
  std::size_t cycles = 0;
  if (blockLength == 1) {
    if (dataType == HALF) {
      // n == 4 and n == 8 are specialsed
      if (n == 4) {
        cycles += 31 + m * (7 + (k - 1) * 2);
      } else if (n == 8) {
        cycles += 31 + m * (8 + (k - 1) * 3);
      } else {
        cycles += 33 + m * (11 + ((n / 8) * (14 + (k - 1) * 3)) +
                            ((n % 8 != 0) * (9 + (k - 1) * 2)));
      }
    } else if (dataType == FLOAT) {
      if (n == 2) {
        cycles += 29 + m * (5 + (k - 1) * 2);
      } else if (n == 4) {
        cycles += 30 + m * (6 + (k - 1) * 3);
      } else {
        cycles += 31 + m * (12 + ((n / 4) * (10 + (k - 1) * 3)) +
                            ((n % 4 != 0) * (10 + (k - 1) * 2)));
      }
    }
    cycles *= numWorkers;
  } else {
    // block sizes other than those here will use the C++ codelet
    std::size_t headerRetainedCycles = 0;
    // For half, xRetained cycles removes the effect of implicit zeroing of
    // partials
    std::size_t xRetainedCycles = 0;
    std::size_t yCycles = 0;
    if (blockLength == 4) {
      if (dataType == HALF) {
        headerRetainedCycles = 17;
        xRetainedCycles = 1;
        if (n <= 2) {
          yCycles = 7 + n;
        } else {
          yCycles = 13 + n;
        }
      } else {
        headerRetainedCycles = 13;
        xRetainedCycles = 1;
        yCycles = 12 + (n - 1) * 4;
      }
    } else if (blockLength == 8) {
      if (dataType == HALF) {
        headerRetainedCycles = 13;
        xRetainedCycles = 1;
        yCycles = 14 + (n - 1) * 2;
      } else {
        headerRetainedCycles = 13;
        xRetainedCycles = 4;
        yCycles = 19 + (n - 1) * 8;
      }
    } else if (blockLength == 16) {
      if (dataType == HALF) {
        headerRetainedCycles = 13;
        xRetainedCycles = 1;
        yCycles = 19 + (n - 1) * 4;
      } else {
        headerRetainedCycles = 16;
        xRetainedCycles = 4;
        if (n == 1) {
          yCycles = 29 * 2;
        } else {
          yCycles = (37 + (n - 1) * 8) * 2;
        }
      }
    } else {
      assert(0);
    }
    cycles += 30 + headerRetainedCycles * numWorkers +
              m * (xRetainedCycles * numWorkers + 17 +
                   k * (16 + coeffLoad() + yCycles * numWorkers));
  }
  return cycles;
}

std::vector<Interval> buildPartitionForN(std::size_t n, std::size_t numSplits,
                                         std::size_t grainSize) {
  return buildPartition(n, numSplits, grainSize, false);
}

std::vector<Interval>
buildTilePartitionForN(const std::vector<Interval> &partitionOfN,
                       std::size_t numTiles, std::size_t tileGrain) {
  // Allocate maximum tiles based on each split size rather than number of
  // splits in the partition.
  constexpr bool allocateTilesBasedOnEachSplitSize = true;
  if (allocateTilesBasedOnEachSplitSize) {
    auto totalSize = std::accumulate(
        partitionOfN.begin(), partitionOfN.end(), std::size_t(1),
        [](std::size_t count, const Interval &i) { return count + i.size(); });
    const auto numSplits = partitionOfN.size();
    const auto numTileGrains = gccs::ceildiv(numTiles, tileGrain);
    std::vector<Interval> tilePartition(numSplits);
    std::size_t tileStart = 0;
    std::size_t minGrainsPerSplit = gccs::ceildiv(numTileGrains, numSplits);
    for (unsigned i = 0; i != numSplits; ++i) {
      auto maxGrainsPerSplit =
          gccs::ceildiv(numTileGrains * partitionOfN[i].size(), totalSize);
      // maxGrainsPerSplit is guaranteed to be >= 1. Deduct 1 because otherwise
      // the rounding effect reduces tiles allocated to the last split.
      auto numThisSplit = std::max(maxGrainsPerSplit - 1, minGrainsPerSplit);
      auto tileEnd = std::min(tileStart + numThisSplit * tileGrain, numTiles);
      if (i == numSplits - 1) {
        tileEnd = numTiles;
      }
      tilePartition[i] = Interval(tileStart, tileEnd);
      tileStart += tilePartition[i].size();
    }
    return tilePartition;
  } else {
    return buildPartition(numTiles, partitionOfN.size(), tileGrain, true);
  }
}

unsigned getTileGrainSize(const poplar::Target &target) {
  return target.getNumTiles() < target.getTilesPerSharedExchangeBus()
             ? 1
             : target.getTilesPerSharedExchangeBus();
}

// Allocate tiles per band respecting the grain size requested.
std::vector<unsigned>
allocateTilesForBands(const std::vector<unsigned> &nzBlocksPerColumnBand,
                      unsigned numTiles, unsigned tileGrain) {
  const unsigned numColumnBands = nzBlocksPerColumnBand.size();
  const unsigned totalNZBlocks = std::accumulate(
      nzBlocksPerColumnBand.begin(), nzBlocksPerColumnBand.end(), 0U);
  std::vector<unsigned> tilesPerBand;
  tilesPerBand.resize(numColumnBands);

  unsigned tilesUsed = 0;
  for (unsigned b = 0; b != numColumnBands; ++b) {
    unsigned tilesThisBand = std::floor(
        (double(nzBlocksPerColumnBand[b]) * numTiles) / totalNZBlocks);
    // Allocate the minimum number of tiles possible. If there are insufficient
    // tiles to allocate then we will return an empty allocation.
    if (tilesThisBand == 0) {
      tilesThisBand = std::min(numTiles, tileGrain);
    }
    // how many tiles are actually required?
    const unsigned blocksPerTile =
        gccs::ceildiv(nzBlocksPerColumnBand[b], tilesThisBand);

    const unsigned tilesForAllBlocks =
        gccs::ceildiv(nzBlocksPerColumnBand[b], blocksPerTile);
    tilesThisBand =
        std::min(gccs::alignNext(tilesForAllBlocks, tileGrain), tilesThisBand);

    if (b == numColumnBands - 1) {
      // allocate all remaining tiles to the last band. The grain size is
      // guaranteed after
      tilesThisBand = numTiles - tilesUsed;
    }
    tilesThisBand = gccs::alignPrev(tilesThisBand, tileGrain);

    auto tilesAvailable =
        std::min(std::max(tilesThisBand, tileGrain), numTiles - tilesUsed);

    // We want at least tileGrain tiles per band
    tilesAvailable = std::min(tilesAvailable,
                              numTiles - (numColumnBands - 1 - b) * tileGrain);
    if (tilesAvailable == 0) {
      return {};
    }

    tilesPerBand[b] = tilesAvailable;
    tilesUsed += tilesAvailable;
  }
  return tilesPerBand;
}

std::tuple<unsigned, unsigned, unsigned>
findBandWithMaxNzPerTile(const static_::Partition::BandInfo &bandInfo,
                         const std::vector<unsigned> &tilesPerBand) {
  assert(bandInfo.nzBlocksPerBand.size() == tilesPerBand.size());
  std::vector<double> nzBlocksPerTile;
  nzBlocksPerTile.reserve(tilesPerBand.size());
  for (unsigned i = 0; i != tilesPerBand.size(); ++i) {
    nzBlocksPerTile.push_back(
        tilesPerBand[i]
            ? static_cast<double>(bandInfo.nzBlocksPerBand[i]) / tilesPerBand[i]
            : 0);
  }
  auto maxIt = std::max_element(nzBlocksPerTile.begin(), nzBlocksPerTile.end());
  auto bestIndex = std::distance(nzBlocksPerTile.begin(), maxIt);
  return std::make_tuple(
      bandInfo.nzBlocksPerBand[bestIndex], tilesPerBand[bestIndex],
      bandInfo.boundaries[bestIndex + 1] - bandInfo.boundaries[bestIndex]);
}

unsigned getMGrainSize(const poplar::Target &target, const poplar::Type &type,
                       unsigned blockLength) {
  return blockLength == 1 ? 1 : getNGrainSize(target, type);
}

unsigned getKGrainSize(const poplar::Target &target, const poplar::Type &type,
                       unsigned blockLength) {
  return blockLength == 1 ? 1 : getNGrainSize(target, type);
}

template <typename T>
Partition getPartition(const CSRMatrix<T> &csr_, const MatMulParams &params,
                       const Target &target, const Type &dataType,
                       const MatMulOptions &options, PlanningCache *cache) {
  const unsigned m = params.getM();
  const unsigned k = params.getK();
  const unsigned n = params.getN();
  const unsigned blockLength = csr_.getBlockDimensions()[0];

  auto csr = csr_;
  canonicalizeCSR(csr);
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  PlanningCacheImpl::Key key(params, options, blockLength,
                             std::move(csr.rowIndices),
                             std::move(csr.columnIndices));
  if (cacheImpl) {
    auto &plans = cacheImpl->plans;
    auto match = plans.find(key);
    if (match != plans.end()) {
      logging::popsparse::debug("-Got cached plan for static sparse");
      return match->second;
    }
  }

  const unsigned numTiles = target.getNumTiles();
  const unsigned blockArea = blockLength * blockLength;
  const unsigned numNZBlocks = csr_.nzValues.size() / (blockArea);
  const unsigned numWorkers = target.getNumWorkerContexts();
  const unsigned mBlocks = m / blockLength;
  const unsigned kBlocks = k / blockLength;
  const unsigned nGrainSize = getNGrainSize(target, dataType);
  const auto dataTypeSize = target.getTypeSize(dataType);
  const auto partialsTypeSize = target.getTypeSize(options.partialsType);
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  const unsigned kGrainSize = getKGrainSize(target, dataType, blockLength);
  const unsigned mGrainSize = getMGrainSize(target, dataType, blockLength);

  logging::popsparse::debug("-Get plan for static sparse [{}, {}] * [{}, {}] ",
                            m, k, k, n);
  logging::popsparse::debug(" options {}", options);

  // Planner always assumes n is a multiple of grain size
  const unsigned nPadded = gccs::alignNext(n, nGrainSize);

  constexpr double rejectThreshold = 0.1;
  // Column and row counts are the gross metrics used to determine pruning
  // and row/column permutations in the pre-processing. Other hamming distance
  // based metrics could be used but are costlier in host compile time and
  // coould be resorted to in the future.
  std::vector<unsigned> colCount(kBlocks);
  for (const auto c : csr_.columnIndices) {
    ++colCount[c / blockLength];
  }

  // Row count is available in the CSR format
  std::vector<unsigned> rowCount;
  rowCount.reserve(mBlocks);
  assert(csr_.rowIndices.size() == mBlocks + 1);
  for (unsigned r = 0; r != mBlocks; ++r) {
    rowCount.push_back((csr_.rowIndices[r + 1] - csr_.rowIndices[r]) /
                       blockArea);
  }

  // Find row and column permutations if pre-processing is enabled. A linearly
  // ordered set when pre-processing is not enabled.
  auto [columnPermutations, kPlan] =
      getPermutationAndPrunedSize(colCount, numNZBlocks, kBlocks,
                                  rejectThreshold, options.enablePreprocessing);
  auto [rowPermutations, mPlan] =
      getPermutationAndPrunedSize(rowCount, numNZBlocks, mBlocks,
                                  rejectThreshold, options.enablePreprocessing);

  if (options.enablePreprocessing) {
    logging::popsparse::debug("  Pruning rows {} -> {}, columns {} -> {}",
                              mBlocks, mPlan, kBlocks, kPlan);
  }

  // Limit to 1.0 because pruning could result in an estimate greater than 1
  auto density =
      std::min(static_cast<double>(numNZBlocks) / (mPlan * kPlan), 1.0);

  // Assuming the probability of an element being non-zero is iid we can use a
  // binomial distribution to compute the probability and expectations.
  // Probability that at least one element is present in numElems given
  // probability of each element.
  auto probabilityAtLeastOneElement = [](double prob, unsigned numElems) {
    return 1.0 - std::pow(1.0 - prob, numElems);
  };

  // Find the mean of a binomial distribution
  auto mean = [](double prob, unsigned numElems) { return prob * numElems; };

  // Scaled standard deviation of a binomial distribution. When numElems is
  // large, we could approximate this as a normal.
  auto scaledStdDeviation = [](double prob, unsigned numElems,
                               double numSigmas = 1.0) {
    return std::ceil(std::pow(prob * (1 - prob) * numElems, 0.5) * numSigmas);
  };

  // These should ideally be increased to the number of tile grains. Set
  // to limit the search space.
  constexpr unsigned maxNSplit = 32;
  constexpr unsigned maxKSplit = 128;
  unsigned bestNSplit = 1;
  unsigned bestKSplit = 1;
  constexpr auto maxCost = std::numeric_limits<std::size_t>::max();

  // The best cycles and memory that have been found by the search
  std::size_t bestCycles = maxCost;
  std::size_t bestMemory = maxCost;

  // Default search bounds
  unsigned nSplitBegin = 1;
  unsigned nSplitEnd = std::min(maxNSplit, gccs::ceildiv(nPadded, nGrainSize));
  unsigned kSplitBegin = 1;
  unsigned kSplitEnd = std::min(maxKSplit, gccs::ceildiv(kPlan, kGrainSize));

  // Partition constraints if any
  if (options.numBands) {
    kSplitBegin = kSplitEnd = options.numBands;
  }
  if (options.nSplit) {
    nSplitBegin = nSplitEnd = options.nSplit;
  }

  const unsigned tileGrain = getTileGrainSize(target);
  assert(numTiles % tileGrain == 0);
  const unsigned numTileGrains = numTiles / tileGrain;
  std::size_t memoryBound =
      options.availableMemoryProportion * target.getBytesPerTile();
  for (unsigned sn = nSplitBegin; sn <= nSplitEnd; ++sn) {
    auto partitionN = buildPartitionForN(nPadded, sn, nGrainSize);
    auto tilePartitionN =
        buildTilePartitionForN(partitionN, numTiles, tileGrain);

    // Unbalanced allocation can occur due to fractions used and the need
    // to maintain grains. Find candidates of n to try in the search.
    auto uniquePartitionSizes =
        getUniquePartitionSizes(partitionN, tilePartitionN);
    if (options.verboseLogging) {
      logging::popsparse::debug("  Starting nSplit {}", sn);
      logging::popsparse::debug("  - partitionN : {}", partitionN);
      logging::popsparse::debug("  - tilePartitionN : {}", tilePartitionN);
    }
    if (uniquePartitionSizes.empty()) {
      continue;
    }

    for (unsigned sk = kSplitBegin; sk <= kSplitEnd; ++sk) {
      if (sn * sk > numTileGrains)
        continue;
      const auto bandInfo =
          buildColumnBandInfo(colCount, sk, kBlocks, numNZBlocks);

      if (options.verboseLogging) {
        logging::popsparse::debug("  - Candidate: nsplit {}, ksplit {}, "
                                  "sub-candidates {}",
                                  sn, sk, uniquePartitionSizes.size());
      }
      std::size_t candidateMemory = 0;
      std::size_t candidateCycles = 0;
      bool validCandidate = true;

      // Try each candidate of N. We are interested in the maximum memory and
      // cycles as execution across these candidates occurs in parallel
      for (auto [nPerSplit, tilesPerNSplit] : uniquePartitionSizes) {
        if (options.verboseLogging) {
          logging::popsparse::debug(
              "  - Trying sub-candidate with nPerSplit {}, tilesPerNSplit {}",
              nPerSplit, tilesPerNSplit);
        }
        auto tilesPerBand = allocateTilesForBands(bandInfo.nzBlocksPerBand,
                                                  tilesPerNSplit, tileGrain);

        // A valid allocation is not found
        if (tilesPerBand.empty()) {
          // Cases when number of NZ blocks is zero is a valid candidate that
          // is just a zero matrix.
          validCandidate = numNZBlocks == 0;
          break;
        }
        // The column partition/bands (split by sk) can result in uneven
        // tile allocation. Find the band that has the max NZ per tile to
        // find the cycles/memory over.
        auto [numBlocksKSplit, tilesInBand, kBand] =
            findBandWithMaxNzPerTile(bandInfo, tilesPerBand);
        unsigned nzBlocksPerTile = gccs::ceildiv(numBlocksKSplit, tilesInBand);
        unsigned mTile = gccs::ceildiv(mPlan, tilesInBand);

        // When a row doesn't fill the full tile only a proportion of the
        // columns in the band are used
        unsigned kTile = mTile == 1 ? std::min(static_cast<unsigned>(std::ceil(
                                                   numBlocksKSplit / density)),
                                               kPlan)
                                    : kBand;

        // Find how many of the rows and tiles will actually be present given
        // uniform sparsity. We find the probablity for columns and rows. Add
        // a spread on the estimate of 1*stdDeviation.
        double rowProb = probabilityAtLeastOneElement(density, kTile);
        unsigned mSparse = std::ceil(mean(rowProb, mTile) +
                                     scaledStdDeviation(rowProb, mTile));
        mSparse = std::min(mSparse, mTile);
        double colProb = probabilityAtLeastOneElement(density, mTile);
        unsigned kSparse = std::ceil(mean(colProb, kTile) +
                                     scaledStdDeviation(colProb, kTile));
        kSparse = std::min(kSparse, kTile);

        if (options.verboseLogging) {
          logging::popsparse::debug(
              "    kTile {}, mTile {}, mSparse {}, kSparse {}, nzBlocksPerTile "
              "{}, row prob {}, col prob {}",
              kTile, mTile, mSparse, kSparse, nzBlocksPerTile, rowProb,
              colProb);
        }
        // The offsets represented for element sparsity are different because of
        // the way the offsets are encoded for the codelets.
        const unsigned representationFactor =
            blockLength == 1 ? 1 : (8 * blockLength);

        // we must meet the constraints to fit the offset in Z
        // TODO: Also add one for the row offsets
        if (bandInfo.maxBandSize * blockLength * nPerSplit * dataTypeSize >
            target.getTypeLimitsMaxAs<std::uint64_t>(UNSIGNED_SHORT) *
                representationFactor) {
          if (options.verboseLogging) {
            logging::popsparse::debug("      yOffset exceeds metadata type");
          }
          validCandidate = false;
          break;
        }

        // This is not accurate because we assume an expected value of the
        // number of sparse rows processed on a tile
        if (worklistEntries(blockLength, mSparse, nzBlocksPerTile, numWorkers) >
            target.getTypeLimitsMaxAs<std::uint64_t>(UNSIGNED_SHORT)) {
          if (options.verboseLogging) {
            logging::popsparse::debug("      size of metadata exceeds max");
          }
          validCandidate = false;
          break;
        }

        // rudimentary work split
        // TODO: Use the same split between the partitioner and graph
        // construction
        auto tileWorkSplit = [&]() {
          if (blockLength == 1) {
            return std::make_tuple(gccs::ceildiv(mSparse, numWorkers),
                                   gccs::ceildiv(nzBlocksPerTile, mSparse),
                                   nPerSplit);
          } else {
            return std::make_tuple(mSparse,
                                   gccs::ceildiv(nzBlocksPerTile, mSparse),
                                   gccs::ceildiv(nPerSplit, numWorkers));
          }
        };

        auto [mWorker, kWorker, nWorker] = tileWorkSplit();

        // The default scheduling mode for the exchange scheduler is to use
        // aggressive multi-cast. Reducing the gap between average and max
        // allows exchange to use multicast more efficiently. Also as the
        // number of messages increases where more tiles are involved, the
        // scheduling becomes much slower, increasing compile times
        // significantly. Use the heuristic below to bias exchange cost.
        auto exchangeBias = [&](unsigned averageNumMessages,
                                unsigned maxMessages) {
          return static_cast<double>(maxMessages) / averageNumMessages;
        };

        auto rhsBias = exchangeBias(kSparse, kTile);
        if (options.verboseLogging) {
          logging::popsparse::debug(
              "      selected exchange bias of {} for {}/{}"
              " rhs sources ",
              rhsBias, kSparse, kTile);
        }
        std::size_t rhsDenseRowsBytes =
            nPerSplit * kSparse * blockLength * dataTypeSize;
        std::size_t exchangeRhsDenseRowsCycles =
            gccs::ceildiv(rhsDenseRowsBytes, 2 * exchangeBytesPerCycle) *
            rhsBias;
        std::size_t nzDataBytes = nzBlocksPerTile * blockArea * dataTypeSize;
        std::size_t exchangeNZDataCycles =
            (sn > 1) ? gccs::ceildiv(nzDataBytes, exchangeBytesPerCycle) : 0;
        std::size_t mmComputeCycles =
            matmulComputeCycles(target, mWorker, kWorker, nWorker, blockLength,
                                dataType, options.partialsType);
        std::size_t mmPartialsZeroingCycles =
            (blockLength == 1 || options.partialsType == HALF ||
             (options.partialsType == FLOAT && blockLength == 4))
                ? 0
                : zeroingCycles(target, mSparse, nPerSplit, blockLength,
                                options.partialsType);
        std::size_t mmComputeOutputBytes =
            mSparse * nPerSplit * partialsTypeSize * blockLength;
        // We get the expected value considering there may be (sk - 1) sources.
        // The expectation is (sk - 1) * rowProb. Scale by a factor of the
        // standard deviation and a single row split across tiles.
        unsigned reductionFactor = std::ceil(
            (rowProb * sk + scaledStdDeviation(rowProb, sk)) * kBand / kTile);
        std::size_t partialsBytes = gccs::ceildiv(
            mmComputeOutputBytes * (reductionFactor - 1), reductionFactor);

        // We end up moving the data to the final destination after reduction.
        // This could be done by moving the partials data where the output is
        // and doing the reduction there. The cost of zeroing output rows that
        // need to be zero are included as part of this cost.
        // TODO: Split the cost of zeroing out of the compounded value.
        std::size_t moveFinalOutput =
            gccs::ceildiv(m * nPadded * dataTypeSize, numTiles);
        std::size_t partialsExchangeCycles =
            gccs::ceildiv(partialsBytes, exchangeBytesPerCycle);
        // The reduction compute estimate is wrong in some cases where the
        // reduceMany doesn't consider all the reductions that occur in
        // parallel. The result is a poor tile balance (see T67345).
        // TODO: use a reduce estimate and the reduction factor to get bettr
        // cost estimate.
        const unsigned reductionOutputElems = gccs::ceildiv(
            mmComputeOutputBytes / partialsTypeSize, reductionFactor);
        std::size_t reductionComputeCycles =
            reductionFactor == 1
                ? 0
                : reductionCycleEstimate(target, reductionOutputElems, dataType,
                                         options.partialsType, reductionFactor);
        std::size_t totalCycles =
            exchangeRhsDenseRowsCycles + exchangeNZDataCycles +
            mmComputeCycles + mmPartialsZeroingCycles + partialsExchangeCycles +
            reductionComputeCycles + moveFinalOutput;

        auto memoryMMComputeBytes =
            rhsDenseRowsBytes + nzDataBytes + mmComputeOutputBytes;
        auto memoryAtReductionBytes = partialsBytes + mmComputeOutputBytes;
        auto totalMemory =
            std::max(memoryMMComputeBytes, memoryAtReductionBytes);

        if (options.verboseLogging) {
          logging::popsparse::debug("    Cycles:");
          logging::popsparse::debug("        exchangeRhsDenseRows {} ",
                                    exchangeRhsDenseRowsCycles);
          logging::popsparse::debug("        exchangeNZData {} ",
                                    exchangeNZDataCycles);
          logging::popsparse::debug("        mmCompute {} ", mmComputeCycles);
          logging::popsparse::debug("        zeroPartials {} ",
                                    mmPartialsZeroingCycles);
          logging::popsparse::debug("        partialsExchange {} ",
                                    partialsExchangeCycles);
          logging::popsparse::debug("        reductionCompute {}, factor {} ",
                                    reductionComputeCycles, reductionFactor);
          logging::popsparse::debug("        moveFinalOutput {} ",
                                    moveFinalOutput);

          logging::popsparse::debug("        total {}", totalCycles);
          logging::popsparse::debug("    Memory(bytes):");
          logging::popsparse::debug("        rhsDenseRows {} ",
                                    rhsDenseRowsBytes);
          logging::popsparse::debug("        nzData {} ", nzDataBytes);
          logging::popsparse::debug("        mmComputeOutput {} ",
                                    mmComputeOutputBytes);
          logging::popsparse::debug("        partials {} ", partialsBytes);
          logging::popsparse::debug("        Max {} ", totalMemory);
        }

        // find max of memory and cycles
        candidateCycles = std::max(candidateCycles, totalCycles);
        candidateMemory = std::max(candidateMemory, totalMemory);
      }

      if (validCandidate && candidateCycles < bestCycles &&
          candidateMemory <= memoryBound) {
        bestCycles = candidateCycles;
        bestMemory = candidateMemory;
        bestNSplit = sn;
        bestKSplit = sk;
        if (options.verboseLogging) {
          logging::popsparse::debug("      New partition sn {}, sk {} with "
                                    "cycles {}, memory {}",
                                    sn, sk, candidateCycles, candidateMemory);
        }
      }
    }
  }

  if (bestCycles == maxCost) {
    throw poputil::poplibs_error("Could not find valid partition (try "
                                 "increasing memory proprtion)");
  }

  logging::popsparse::debug("    partition: nSplit {}, kSplit {}, cycles {} "
                            "memory {} bytes",
                            bestNSplit, bestKSplit, bestCycles, bestMemory);

  Partition partition;
  partition.nGrainSize = nGrainSize;
  partition.kGrainSize = kGrainSize;
  partition.mGrainSize = mGrainSize;
  partition.nSplit = bestNSplit;
  auto bandInfo =
      buildColumnBandInfo(colCount, bestKSplit, kBlocks, numNZBlocks);
  partition.bandInfo = std::move(bandInfo);
  partition.rowPermutations = std::move(rowPermutations);
  partition.columnPermutations = std::move(columnPermutations);

  logging::popsparse::debug("Selected Plan: {}", partition);

  if (cacheImpl) {
    cacheImpl->plans.emplace(key, partition);
  }
  return partition;
}

template Partition getPartition(const CSRMatrix<float> &csr_,
                                const MatMulParams &params,
                                const Target &target, const Type &dataType,
                                const MatMulOptions &options,
                                PlanningCache *cache);
template Partition getPartition(const CSRMatrix<double> &csr_,
                                const MatMulParams &params,
                                const Target &target, const Type &dataType,
                                const MatMulOptions &options,
                                PlanningCache *cache);

} // namespace static_
} // namespace popsparse
