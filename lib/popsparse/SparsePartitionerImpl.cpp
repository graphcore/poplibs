// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "SparsePartitionerImpl.hpp"
#include "FullyConnectedUtils.hpp"
#include "SparseFormatsValidate.hpp"
#include "SparseMetaInfo.hpp"
#include "SparsePartitionerImpl.hpp"
#include "SparseStorageInternal.hpp"
#include "poplibs_support/logging.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "poputil/Util.hpp"

#include <gccs/Algorithm.hpp>

#include <algorithm>
#include <boost/multi_array.hpp>
#include <limits>
#include <unordered_map>

using namespace poplibs_support;
// Here only to account for sizes
using MetaInfoType = unsigned short;
using MI = popsparse::MetaInfo<MetaInfoType>;
using BMI = popsparse::BlockMetaInfo<MetaInfoType>;
using namespace popsparse::fullyconnected;

// to compute number of elements
#define miElems(x) (sizeof(x) / sizeof(MetaInfoType))

namespace {
using namespace popsparse;
using namespace popsparse::dynamic;

// get tile index from pnId
std::tuple<std::size_t, std::size_t, std::size_t>
getTileIndexFromPnId(std::size_t pnId, const std::vector<std::size_t> &numXYZ) {
  const auto z = pnId % numXYZ[2];
  const auto y = (pnId / numXYZ[2]) % numXYZ[1];
  const auto x = pnId / (numXYZ[1] * numXYZ[2]);
  return std::make_tuple(x, y, z);
}

// Number of non-zero values in a partition
std::size_t numNonZeroValues(const TilePartition &partition) {
  return std::accumulate(
      partition.tileInfo.begin(), partition.tileInfo.end(), 0UL,
      [](std::size_t a, const RowPositionValues &row) -> std::size_t {
        return a + row.positionValues.size();
      });
}

// convert tile partition to a CSR representation
CSRInternal tilePartitionToCsrMatrix(const TilePartition &partition) {
  std::vector<std::size_t> rowIndices;
  std::vector<std::size_t> columnIndices;
  std::vector<ValueType> nzValues;

  const auto numNzValues = numNonZeroValues(partition);
  columnIndices.resize(numNzValues);
  nzValues.resize(numNzValues);
  rowIndices.resize(partition.tile.getRows().size() + 1);

  std::size_t index = 0;
  for (std::size_t row = 0; row != partition.tile.getRows().size(); ++row) {
    rowIndices[row] = index;
    auto match = std::find_if(
        partition.tileInfo.begin(), partition.tileInfo.end(),
        [&](const RowPositionValues &r) { return r.rowNumber == row; });
    if (match != partition.tileInfo.end()) {
      for (const auto &p : match->positionValues) {
        columnIndices[index] = p.first;
        nzValues[index++] = p.second;
      }
    }
  }
  rowIndices.back() = index;
  return CSRInternal(nzValues, columnIndices, rowIndices);
}

// Create a tile representation from a CSR matrix
TilePartition csrMatrixToTilePartition(const CSRInternal &csrMatrix,
                                       const Tile &tile,
                                       const TileIndex &tileIndex) {
  std::vector<RowPositionValues> tileInfo;
  const auto numEntries = csrMatrix.rowIndices.size();
  for (std::size_t row = 1; row != numEntries; ++row) {
    const auto numValues =
        csrMatrix.rowIndices[row] - csrMatrix.rowIndices[row - 1];
    if (numValues) {
      std::vector<std::pair<std::size_t, ValueType>> rowEntry;
      auto posIt =
          csrMatrix.columnIndices.begin() + csrMatrix.rowIndices[row - 1];
      auto valIt = csrMatrix.nzValues.begin() + csrMatrix.rowIndices[row - 1];
      for (std::size_t i = 0; i != numValues; ++i) {
        rowEntry.emplace_back(*posIt++, *valIt++);
      }
      tileInfo.push_back(RowPositionValues(row - 1, rowEntry));
    }
  }
  return TilePartition(tileIndex, tile, tileInfo);
}

// Virtual mapping of tile to PN
std::size_t getPNId(const std::vector<std::size_t> &xyz,
                    const std::vector<std::size_t> &numXYZ) {
  return xyz[0] * numXYZ[1] * numXYZ[2] + xyz[1] * numXYZ[2] + xyz[2];
}

std::vector<TilePartition> static getTilePartition(
    const CSRInternal matrix, std::size_t numX, std::size_t numY,
    std::size_t numZ, std::size_t blockSizeX, std::size_t blockSizeY,
    const std::vector<std::size_t> &xSplits,
    const std::vector<std::size_t> &ySplits,
    const std::vector<std::size_t> &zSplits, std::size_t bucketsPerZ) {
  const auto &csr = matrix;

  const std::vector<std::size_t> numXYZ = {xSplits.size(), ySplits.size(),
                                           zSplits.size() * bucketsPerZ};

  // Each tile is spread over a group of PNs and hence we split it for a given
  // grain size
  std::size_t numPNs = std::accumulate(numXYZ.begin(), numXYZ.end(), 1,
                                       std::multiplies<std::size_t>());
  logging::popsparse::trace("  Creating tile partitions for {} PNs", numPNs);

  std::vector<TilePartition> tilePartitions(numPNs);

  for (std::size_t row = 0; row != xSplits.size(); ++row) {
    for (std::size_t column = 0; column != ySplits.size(); ++column) {
      const auto rowStart = xSplits[row];
      const auto rowEnd = row + 1 == xSplits.size() ? numX : xSplits[row + 1];
      const auto columnStart = ySplits[column];
      const auto columnEnd =
          column + 1 == ySplits.size() ? numY : ySplits[column + 1];

      poplar::Interval rowInterval(rowStart, rowEnd);
      poplar::Interval columnInterval(columnStart, columnEnd);
      std::size_t rowIndex = row, columnIndex = column;

      Tile tile(rowInterval, columnInterval);
      auto tp = getPositionValuePairsPerRow(csr, blockSizeX, blockSizeY, tile);
      logging::popsparse::trace("    Tile X={} Y={} number of rows {} ",
                                tile.getRows(), tile.getColumns(), tp.size());

      // Split intervals over Z-dimension
      std::vector<std::size_t> rowElements;
      std::vector<poplar::Interval> intervals;
      std::size_t numCols = 0;
      for (const auto &r : tp) {
        rowElements.push_back(numCols);
        const auto colsThisRow = r.positionValues.size();
        intervals.emplace_back(0, colsThisRow);
        numCols += colsThisRow;
      }
      rowElements.push_back(numCols);
      auto splits =
          poputil::splitRegions(intervals, 1, zSplits.size() * bucketsPerZ);

      auto it = std::next(rowElements.begin());
      std::size_t rIndex = 0, cIndex = 0, elementsUsed = 0;
      for (std::size_t z = 0; z != splits.size(); ++z) {
        const auto pn = getPNId({row, column, z}, numXYZ);
        std::vector<RowPositionValues> rowPosValues;
        logging::popsparse::trace("      z={}, pn={} : z splits={}", z, pn,
                                  splits[z]);
        auto splitIt = splits[z].begin();
        do {
          assert(!tp[rIndex].positionValues.empty());
          std::vector<std::pair<std::size_t, ValueType>> positionValues;
          for (std::size_t col = 0; col != splitIt->size(); ++col, ++cIndex) {
            positionValues.push_back(tp[rIndex].positionValues[cIndex]);
          }
          logging::popsparse::trace("        row : {} = {} ",
                                    tp[rIndex].rowNumber, positionValues);
          RowPositionValues rpEntry(tp[rIndex].rowNumber, positionValues);
          rowPosValues.push_back(rpEntry);
          elementsUsed += splitIt->size();
          ++splitIt;
          if (*it == elementsUsed) {
            ++rIndex;
            ++it;
            cIndex = 0;
          }
        } while (splitIt != splits[z].end());
        tilePartitions[pn] = TilePartition(
            std::make_tuple(rowIndex, columnIndex, z), tile, rowPosValues);
      }
    }
  }
  return tilePartitions;
}

// find amount of information kept on tile which is a sub-tile of the
// partition
std::size_t numMetaInfoElementsForWorker(const TilePartition &partition,
                                         const Tile &tile, bool includeGradW) {
  bool rowInTile = tile.getRows().begin() <
                   std::min(partition.tileInfo.size(), tile.getRows().end());

  // Only add in worker state if there's any output on the tile.
  std::size_t numElements = 0;
  if (rowInTile) {
    numElements += miElems(MI::WorkerEntry);
    logging::popsparse::trace("        --WI : {}", miElems(MI::WorkerEntry));
    if (includeGradW) {
      numElements += miElems(MI::GradWWorkerEntry);
      logging::popsparse::trace("        --GWI : {}",
                                miElems(MI::GradWWorkerEntry));
    }
  }
  return numElements;
}

std::pair<std::size_t, std::size_t>
sizesForTilePartition(const TilePartition &partition, std::size_t numZGrains,
                      std::size_t numWorkers, bool useBlockMetaInfoFormat,
                      bool useWorkerSplits, bool includeGradW) {

  // We don't duplicate rows
  auto numNZElements = numNonZeroValues(partition);
  std::size_t metaInfoElements = 0;

  if (useBlockMetaInfoFormat) {
    const auto outputEntriesElements =
        partition.tileInfo.size() * miElems(BMI::OutputEntry);
    const auto nzOffsetEntries = numNZElements;
    metaInfoElements +=
        outputEntriesElements + nzOffsetEntries + miElems(BMI::SubGroupEntry);
    if (includeGradW) {
      metaInfoElements += miElems(BMI::GradWWorkerEntry) * numWorkers;
    }
  } else {
    if (useWorkerSplits) {
      // we split the rows on the tile to be split. The partition is only
      // on the output rows and columns. We could also account for the number
      // of columns in each of the sparse rows, but that is an optimisation.
      auto workers = splitTileBetweenWorkers(partition.tileInfo.size(),
                                             numZGrains, numWorkers);

      for (const auto &worker : workers) {
        metaInfoElements +=
            numMetaInfoElementsForWorker(partition, worker, includeGradW);
      }
    } else {
      metaInfoElements += miElems(MI::WorkerEntry) * numWorkers;
      logging::popsparse::trace("        --WI : {}", metaInfoElements);
      if (includeGradW) {
        metaInfoElements += miElems(MI::GradWWorkerEntry) * numWorkers + 1;
      }
      logging::popsparse::trace("        --WI : {}", metaInfoElements);
    }

    std::size_t outputEntriesElements =
        partition.tileInfo.size() * miElems(MI::OutputEntry);

    std::size_t nzOffsetEntries = numNZElements;

    if (outputEntriesElements) {
      logging::popsparse::trace("        --Output entries : {}",
                                outputEntriesElements);
    }

    if (nzOffsetEntries) {
      logging::popsparse::trace("        --offset entries : {}",
                                nzOffsetEntries);
    }

    // include output entries in  meta info
    metaInfoElements +=
        outputEntriesElements + nzOffsetEntries + miElems(MI::SubGroupEntry);
  }

  return std::make_pair(metaInfoElements, numNZElements);
}

// Get the number of grains of Z in a tile
std::size_t getNumZGrains(const TileIndex &tileIndex,
                          const std::vector<std::size_t> &zSplits,
                          std::size_t numZ, std::size_t bucketsPerZ,
                          std::size_t grainSizeZ) {
  const auto zIndex = std::get<2>(tileIndex) / bucketsPerZ;
  const auto zBegin = zSplits.at(zIndex);
  const auto zEnd =
      (zIndex + 1 == zSplits.size()) ? numZ : zSplits.at(zIndex + 1);
  return (zEnd - zBegin + grainSizeZ - 1) / grainSizeZ;
}

// Given a bucket, computes the exact size in elements required for meta info
// and NZ values. The size information is then filled into the bucket structure.
static void fillBucketSizes(PNBucket &bucket,
                            const std::vector<std::size_t> &zSplits,
                            const std::size_t numZ,
                            const std::size_t grainSizeZ, bool useWorkerSplits,
                            std::size_t numWorkers, std::size_t bucketsPerZ,
                            bool useBlockMetaInfoFormat, bool includeGradW,
                            const std::string &str) {
  logging::popsparse::trace("      Determining sizes for PN bucket " + str);
  std::size_t nzElements = 0;
  std::size_t metaInfoElements = 0;
  if (!bucket.subGroups.empty()) {
    for (const auto &subgroup : bucket.subGroups) {
      // The bucket could be empty if all rows are removed
      if (subgroup.empty()) {
        continue;
      }
      const auto numGrains = getNumZGrains(subgroup.tileIndex, zSplits, numZ,
                                           bucketsPerZ, grainSizeZ);
      auto bucketSizes = sizesForTilePartition(subgroup, numGrains, numWorkers,
                                               useBlockMetaInfoFormat,
                                               useWorkerSplits, includeGradW);
      logging::popsparse::trace(
          "        Bucket group size : metainfo {}   nz elements {}",
          bucketSizes.first, bucketSizes.second);
      nzElements += bucketSizes.second;
      metaInfoElements += bucketSizes.first;
    }
  }
  bucket.numNzElements = nzElements;
  bucket.metaInfoElements = metaInfoElements;
}

// TODO: This should take the option to do actual worker splits
// Remove partitions which are full rows and/or part of a row (i.e columns) if
// enableColumnSplit = true. If splitting a single row is enabled, it will
// always be the last row in the partition.
std::vector<std::pair<std::size_t, std::size_t>>
findPartitionsToRemove(const std::vector<std::size_t> &rowWeights,
                       const std::pair<std::size_t, std::size_t> &target,
                       bool useBlockMetaInfoFormat, std::size_t numWorkers,
                       bool gradWEnabled, bool enableColumnSplit) {
  logging::popsparse::trace(
      "    -- find partitions for target {}, row weights {}", target,
      rowWeights);
  std::size_t miCost =
      fixedMetaInfoCost(useBlockMetaInfoFormat, numWorkers, gradWEnabled);
  logging::popsparse::trace("       initial MI costs : {}", miCost);
  std::size_t nzElems = 0;

  std::vector<std::pair<std::size_t, std::size_t>> partition;

  for (auto it = rowWeights.begin(); it != rowWeights.end(); ++it) {
    // Check number of elements that can fit
    std::size_t remainingElems =
        std::min(target.first - miCost, target.second - nzElems);
    std::size_t elemsToAlloc =
        (*it > remainingElems && enableColumnSplit) ? remainingElems : *it;
    const auto miCostUpdate = elemsToAlloc + miElems(MI::OutputEntry);
    const auto nzElemsUpdate = elemsToAlloc;
    if (miCost + miCostUpdate <= target.first &&
        nzElems + nzElemsUpdate <= target.second && elemsToAlloc) {
      miCost += miCostUpdate;
      nzElems += nzElemsUpdate;
      partition.emplace_back(std::distance(rowWeights.begin(), it),
                             elemsToAlloc);
    } else {
      break;
    }
  }

  logging::popsparse::trace(
      "   -- cost for selected partition : {} {} , partition {} ", miCost,
      nzElems, partition);
  return partition;
}

// Removes rows until target is reached
TilePartition
removeRows(PNBucket &bucket, const std::vector<std::size_t> &zSplits,
           std::size_t numZ, std::size_t grainSizeZ, std::size_t numWorkers,
           std::size_t metaInfoElementsTarget, std::size_t nzElementsTarget,
           std::size_t bucketsPerZ, bool useBlockMetaInfoFormat,
           bool useWorkerSplits, bool includeGradW) {
  TilePartition removedPartition;

  if (bucket.metaInfoElements <= metaInfoElementsTarget &&
      bucket.numNzElements <= nzElementsTarget) {
    return removedPartition;
  }

  logging::popsparse::trace(
      "  -removing rows: available  {} : target Elements  {} {} ",
      bucket.subGroups[0].tileInfo.size(), metaInfoElementsTarget,
      nzElementsTarget);

  // For now just go through row vectors and remove them.
  // TODO: Do this better
  std::vector<RowPositionValues> rowsRemoved;
  for (std::size_t i = bucket.subGroups[0].tileInfo.size(); i > 0; --i) {
    auto index = i - 1;
    rowsRemoved.push_back(bucket.subGroups[0].tileInfo[index]);
    bucket.subGroups[0].tileInfo.erase(bucket.subGroups[0].tileInfo.begin() +
                                       index);

    fillBucketSizes(bucket, zSplits, numZ, grainSizeZ, useWorkerSplits,
                    numWorkers, bucketsPerZ, useBlockMetaInfoFormat,
                    includeGradW, ": after removing row");
    logging::popsparse::trace(
        "  --removed index {}, size of bucket after {}, {}", index,
        bucket.metaInfoElements, bucket.numNzElements);
    if (bucket.metaInfoElements <= metaInfoElementsTarget &&
        bucket.numNzElements <= nzElementsTarget) {
      break;
    }
  }

  if (!rowsRemoved.empty()) {
    removedPartition = TilePartition(bucket.subGroups[0].tileIndex,
                                     bucket.subGroups[0].tile, rowsRemoved);
  }

  return removedPartition;
}

// The partition is described in terms of the row number and the end column as
// the start position is always zero
TilePartition
removeIntervals(TilePartition &tilePartition,
                std::vector<std::pair<std::size_t, std::size_t>> &intervals) {
  std::vector<RowPositionValues> rowPositionValues;

  // Sort to erase from largest index
  std::sort(
      intervals.begin(), intervals.end(),
      [](std::pair<std::size_t, std::size_t> &a,
         std::pair<std::size_t, std::size_t> &b) { return a.first > b.first; });

  for (const auto &p : intervals) {
    assert(p.first < tilePartition.tileInfo.size());
    auto &row = tilePartition.tileInfo[p.first];
    const std::size_t numColElems =
        tilePartition.tileInfo[p.first].positionValues.size();
    if (numColElems == p.second) {
      // remove whole row
      rowPositionValues.emplace_back(row.rowNumber, row.positionValues);
      tilePartition.tileInfo.erase(tilePartition.tileInfo.begin() + p.first);
    } else {
      // remove elements from the end
      std::vector<std::pair<std::size_t, ValueType>> posValues;
      posValues.insert(posValues.end(),
                       row.positionValues.begin() + numColElems - p.second,
                       row.positionValues.end());
      rowPositionValues.emplace_back(row.rowNumber, std::move(posValues));
      row.positionValues.resize(numColElems - p.second);
    }
  }
  return TilePartition(tilePartition.tileIndex, tilePartition.tile,
                       std::move(rowPositionValues));
}

void dumpBucketStatus(const std::vector<PNBucket> &mainBuckets,
                      const std::vector<PNBucket> &overflowBuckets = {}) {
  const auto numBuckets = mainBuckets.size();
  if (numBuckets == overflowBuckets.size()) {
    auto empty = std::all_of(overflowBuckets.begin(), overflowBuckets.end(),
                             [](const PNBucket &b) { return b.empty(); });
    logging::popsparse::trace("  - buckets overflown ? {}", !empty);
    for (std::size_t p = 0; p != numBuckets; ++p) {
      auto &bucket = mainBuckets[p];
      auto oBucket = overflowBuckets[p];
      logging::popsparse::trace(
          "  -PN {} groups {} : metainfo elems {} [{}]  nz {} [{}] ", p,
          bucket.numSubgroups(), bucket.metaInfoElements,
          oBucket.metaInfoElements, bucket.numNzElements,
          oBucket.numNzElements);
    }
  } else {

    for (std::size_t p = 0; p != numBuckets; ++p) {
      auto &bucket = mainBuckets[p];
      logging::popsparse::trace(
          "  -Main PN  {} groups {} : nz {}  metainfo elems {}", p,
          bucket.numSubgroups(), bucket.numNzElements, bucket.metaInfoElements);
    }
  }
}

std::vector<PNBucket>
createBucketsForPN(const std::vector<TilePartition> &tilePartitions,
                   const std::vector<std::size_t> &zSplits, std::size_t numZ,
                   std::size_t grainSizeZ, bool useWorkerSplits,
                   std::size_t numWorkers, std::size_t bucketsPerZ,
                   bool useBlockMetaInfoFormat, bool includeGradW) {
  const auto numPNs = tilePartitions.size();
  std::vector<PNBucket> buckets(tilePartitions.size());
  // The initial buckets contain one tile partition
  for (std::size_t p = 0; p != numPNs; ++p) {
    if (!tilePartitions[p].empty()) {
      buckets[p].subGroups.push_back(tilePartitions[p]);
      // fill in size information
      fillBucketSizes(buckets[p], zSplits, numZ, grainSizeZ, useWorkerSplits,
                      numWorkers, bucketsPerZ, useBlockMetaInfoFormat,
                      includeGradW, "create-" + std::to_string(p));
    }
  }
  return buckets;
}

std::size_t countNonEmpty(const std::vector<PNBucket> &bucket) {
  return std::accumulate(
      bucket.begin(), bucket.end(), (std::size_t)0,
      [](std::size_t prior, const PNBucket &b) -> std::size_t {
        return prior + (b.metaInfoElements != 0 || b.numNzElements != 0);
      });
}

static void logBucket(const PNBucket &b, const std::string &str) {
  logging::popsparse::trace("   - Logging Bucket : {} : [{}, {}]", str,
                            b.metaInfoElements, b.numNzElements);
  for (const auto &sg : b.subGroups) {
    logging::popsparse::trace("     - subgroup ");

    logging::popsparse::trace("      + Tile:: {}", sg.tile);
    logging::popsparse::trace(
        "      + Tile index:: row {}, col {}. z {}", std::get<0>(sg.tileIndex),
        std::get<1>(sg.tileIndex), std::get<2>(sg.tileIndex));
    for (const auto &r : sg.tileInfo) {
      logging::popsparse::trace("       - row : {}, num columns : {}",
                                r.rowNumber, r.positionValues.size());
    }
  }
}

std::size_t formSubgroupId(const TileIndex &tileIndex,
                           const std::vector<std::size_t> &numSplits,
                           bool gradA) {
  auto rowGroupIndex = std::get<0>(tileIndex);
  auto subRowGroupIndex = std::get<1>(tileIndex);
  auto numRowGroups = numSplits[0];
  auto numSubRowGroups = numSplits[1];
  if (gradA) {
    std::swap(rowGroupIndex, subRowGroupIndex);
    std::swap(numRowGroups, numSubRowGroups);
  }
  return calculateSubGroupId(numRowGroups, numSubRowGroups, rowGroupIndex,
                             subRowGroupIndex);
}

// Once all the rebalancing is done, we look at the distance from the original
// grouping overflown elements have moved. As the bucketing is done
// hierarchicaly (i.e. within S-ORGs first, then within ORGs followed by
// between ORGs), we can just compute the distance information in a tile has
// moved.
// The distance as measured as how long along the cyclic path a bucket or data
// moves and they move in increasing ORGs. If 'moveBuckets' is set to true then
// buckets move, else data moves.
//
std::vector<std::size_t>
findOverflowDistance(const std::vector<PNBucket> &pnBuckets,
                     const std::vector<std::size_t> &numSplits,
                     bool genForGradA, bool genForGradW,
                     std::size_t bucketsPerZ) {
  assert(!genForGradA || !genForGradW);

  if (genForGradA || genForGradW) {
    // No specific overflow info for GradA/GradW at time of writing.
    return {};
  }

  const auto numBuckets = pnBuckets.size();
  const auto numRowGroups = numSplits[0];
  const auto numSubRowGroups = numSplits[1];
  const auto numSORGs = numSplits[2];
  auto numSplitsForPnId = numSplits;
  numSplitsForPnId.back() *= bucketsPerZ;

  // we needn't keep this as a running max is sufficient. Kept
  // only for debugging.
  std::vector<std::pair<std::size_t, std::size_t>> distances(numBuckets);
  std::vector<bool> orgConnectivity(numRowGroups);
  std::vector<bool> sorgConnectivity(numSubRowGroups);

  for (std::size_t b = 0; b != numBuckets; ++b) {
    // go through sub-groups
    const auto &subGroups = pnBuckets[b].subGroups;
    const auto pnTileIndex = getTileIndexFromPnId(b, numSplitsForPnId);
    const auto thisPnSubgroup = formSubgroupId(pnTileIndex, numSplits, false);

    for (std::size_t sg = 0; sg != subGroups.size(); ++sg) {
      if (subGroups[sg].empty()) {
        continue;
      }
      // The buckets are generated for Fwd and we are using if we are using it
      // for backward we swap the tile indices.
      const auto subGroupId =
          formSubgroupId(subGroups[sg].tileIndex, numSplits, false);
      auto srcId = thisPnSubgroup;
      auto dstId = subGroupId;
      auto dist =
          distanceToSubGroup(srcId, dstId, numRowGroups, numSubRowGroups);

      orgConnectivity[dist.first] = true;
      sorgConnectivity[dist.second] = true;
      unsigned rowIndex, subRowIndex;
      std::tie(rowIndex, subRowIndex) =
          getGroupIndices(subGroupId, numRowGroups, numSubRowGroups);
      auto dstPnId = getPNId({rowIndex, subRowIndex, 0}, numSplits);

      distances[dstPnId] = std::max(dist, distances[dstPnId]);
    }
  }

  logging::popsparse::debug(" ORG connectivity {}", orgConnectivity);
  logging::popsparse::debug(" SORG connectivity {}", sorgConnectivity);
  const auto connectivityElems =
      gccs::ceildiv(orgConnectivity.size(), sizeof(MetaInfoType));
  std::vector<std::size_t> orgConnectivityBitset(connectivityElems);
  for (std::size_t org = 0; org < orgConnectivity.size(); ++org) {
    orgConnectivityBitset[org / sizeof(MetaInfoType)] |=
        unsigned(orgConnectivity[org]) << (org % sizeof(MetaInfoType));
  }

  const auto maxIt = std::max_element(distances.begin(), distances.end());
  logging::popsparse::trace("  Distance metric for PN : {}", distances);
  auto maxX = maxIt->first + 1;
  auto maxY = maxIt->second + 1;
  auto maxZ = numSORGs;
  auto numY = numSubRowGroups;
  auto numZ = numSORGs;
  const auto x = maxX;
  const auto y = x == 1 ? maxY : numY;
  const auto z = x == 1 && y == 1 ? maxZ : numZ;
  logging::popsparse::trace("  - selected distance triplet: {} {} {}", x, y, z);

  std::vector<std::size_t> result = {x, y, z};
  std::move(orgConnectivityBitset.begin(), orgConnectivityBitset.end(),
            std::back_inserter(result));
  return result;
}

// Check that matrix dimensions match with which partitioner was construted
void checkBlockDimensionsMatch(std::array<std::size_t, 2> matrixDimensions,
                               std::array<std::size_t, 2> originalDimensions) {
  if (matrixDimensions != originalDimensions) {
    throw poputil::poplibs_error("Block dimensions of matrix do not match "
                                 "ones with which partitioner was created");
  }
}

template <typename ValueType, typename IndexType>
void csrToDenseMatrixImpl(boost::multi_array_ref<ValueType, 2> &mat,
                          const ValueType *nzValues,
                          const IndexType *columnIndices,
                          const IndexType *rowIndices,
                          const std::size_t numNonZeroValues,
                          const std::size_t blockRows,
                          const std::size_t blockCols) {
  const std::size_t numRows = mat.shape()[1];
  const std::size_t numColumns = mat.shape()[0];
  (void)numColumns;
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
          mat[columnIndices[i] + c][row + r] =
              nzValues[i * blockArea + r * blockCols + c];
        }
      }
      ++i;
    }
  }
  assert(i * blockArea == numNonZeroValues);
}

template <typename T>
static std::pair<std::vector<std::size_t>, std::vector<T>> bucketsImplInternal(
    const PNBucket &bucket, const std::vector<T> &nzValues,
    const std::vector<std::size_t> &xSplits,
    const std::vector<std::size_t> &ySplits,
    const std::vector<std::size_t> &zSplits, std::size_t numZ,
    std::size_t grainX, std::size_t grainY, std::size_t grainZ,
    bool useBlockMetaInfoFormat, bool includeGradW, bool genForGradA,
    const poplar::Type &dataType, const poplar::Type &accumType,
    std::size_t metaInfoBucketElements, std::size_t nzElemsBucketBlocks,
    std::size_t numWorkers, std::size_t bucketsPerZ,
    const poplar::DebugNameAndId &dnai) {
  const std::size_t yOffsetTypeFactor =
      popsparse::getYOffsetTypeScaleFactor(dataType == poplar::FLOAT);
  const auto blockSize = grainX * grainY;

  auto getSubgroupNzCountAndSparseOffsets = [](const TilePartition &sg) {
    const auto numRows = sg.tileInfo.size();
    std::size_t nzCount = 0;
    std::vector<std::size_t> sparseOffset(numRows + 1);
    for (std::size_t row = 0; row != numRows; ++row) {
      sparseOffset[row] = nzCount;
      nzCount += sg.tileInfo[row].positionValues.size();
    }
    sparseOffset.back() = nzCount;
    return sparseOffset;
  };

  // local structure from which entries for both element and block worker
  // entries are derived.
  struct GradWInternal {
    std::size_t sparseOffset;
    std::size_t totalNumY;
    std::size_t metaInfoOffsetToOffsetsYInSFirst;
    std::size_t metaInfoOffsetOutputEntry;
  };

  // get GradW worker split
  auto getGradWWorkerSplit = [](const std::vector<std::size_t> &sparseOffset,
                                std::size_t numWorkers) {
    std::vector<GradWInternal> gradWEntries;
    const auto workerGradW =
        splitTileBetweenWorkers(1, sparseOffset.back(), numWorkers);
    auto numGradWWorkers = workerGradW.size();
    gradWEntries.resize(numGradWWorkers);
    for (std::size_t i = 0; i != numGradWWorkers; ++i) {
      const auto &w = workerGradW[i];
      gradWEntries[i].sparseOffset = w.getColumns().begin();
      gradWEntries[i].totalNumY = w.getColumns().size();
      // Get the sparseOffset for the first row that contains this
      // sparse offset (upper_bound: first entry strictly greater than,
      // std::prev: last entry less than or equal);
      auto it = std::prev(std::upper_bound(
          sparseOffset.begin(), sparseOffset.end(), w.getColumns().begin()));
      gradWEntries[i].metaInfoOffsetToOffsetsYInSFirst =
          w.getColumns().begin() - *it;
      std::size_t rowOffset = std::distance(sparseOffset.begin(), it);
      gradWEntries[i].metaInfoOffsetOutputEntry = rowOffset;
    }
    return gradWEntries;
  };

  std::vector<std::size_t> group;
  std::vector<T> nzBucket;
  group.reserve(metaInfoBucketElements);
  nzBucket.reserve(nzElemsBucketBlocks * blockSize);

  if (useBlockMetaInfoFormat) {
    BlockMetaInfo<std::size_t>::SubGroupEntry sgEntry;
    for (const auto &sg : bucket.subGroups) {
      sgEntry.id = formSubgroupId(
          sg.tileIndex, {xSplits.size(), ySplits.size(), zSplits.size()},
          genForGradA);
      sgEntry.xPartition = std::get<0>(sg.tileIndex);
      sgEntry.yPartition = std::get<1>(sg.tileIndex);
      const auto numRows = sg.tileInfo.size();
      if (numRows == 0) {
        continue;
      }

      auto sparseOffset = getSubgroupNzCountAndSparseOffsets(sg);
      std::size_t nzCount = sparseOffset.back();

      // We may also need to include gradW if enabled
      std::vector<GradWInternal> gradWEntries;
      if (includeGradW) {
        gradWEntries = getGradWWorkerSplit(sparseOffset, numWorkers);
      }
      std::size_t numGradWWorkers = gradWEntries.size();

      const auto offsetToNextSubGroupSparseEntries = nzCount * blockSize;
      const auto offsetToNextSubGroupMetaInfo =
          (sizeof(BlockMetaInfo<std::size_t>::SubGroupEntry) +
           sizeof(BlockMetaInfo<std::size_t>::OutputEntry) * numRows +
           sizeof(BlockMetaInfo<std::size_t>::GradWWorkerEntry) *
               numGradWWorkers +
           sizeof(BlockMetaInfo<std::size_t>::InputEntry) * nzCount) /
          sizeof(std::size_t);

      // BMI::SubGroupEntry
      group.push_back(sgEntry.id);
      group.push_back(sgEntry.xPartition);
      group.push_back(sgEntry.yPartition);

      group.push_back(offsetToNextSubGroupSparseEntries);
      group.push_back(offsetToNextSubGroupMetaInfo);
      group.push_back(numRows - 1);
      group.push_back(numGradWWorkers);

      // BMI::GradWEntries
      if (includeGradW) {
        std::size_t workersRemaining = numGradWWorkers;
        std::vector<MetaInfo<std::size_t>::OutputEntry> outputEntries(numRows);

        for (const auto &w : gradWEntries) {
          group.push_back(w.sparseOffset * blockSize);
          std::size_t offset =
              w.sparseOffset - w.metaInfoOffsetToOffsetsYInSFirst +
              (workersRemaining *
                   sizeof(BlockMetaInfo<std::size_t>::GradWWorkerEntry) +
               w.metaInfoOffsetOutputEntry *
                   sizeof(BlockMetaInfo<std::size_t>::OutputEntry)) /
                  sizeof(std::size_t);
          group.push_back(offset);
          group.push_back(w.metaInfoOffsetToOffsetsYInSFirst);
          group.push_back(w.totalNumY);
          workersRemaining--;
        }
      }

      for (const auto &rowEntry : sg.tileInfo) {
        group.push_back(rowEntry.rowNumber);
        group.push_back(rowEntry.positionValues.size() - 1);

        for (const auto &colEntry : rowEntry.positionValues) {
          group.push_back(colEntry.first);
          assert((colEntry.second + 1) * blockSize <= nzValues.size());
          nzBucket.insert(nzBucket.end(),
                          nzValues.begin() + colEntry.second * blockSize,
                          nzValues.begin() + (colEntry.second + 1) * blockSize);
        }
      }
    }
    group.push_back(BlockMetaInfo<std::size_t>::endSubGroupId);
  } else {
    assert(blockSize == 1);
    MetaInfo<std::size_t>::SubGroupEntry sgEntry;
    for (const auto &sg : bucket.subGroups) {
      sgEntry.id = formSubgroupId(
          sg.tileIndex, {xSplits.size(), ySplits.size(), zSplits.size()},
          genForGradA);
      sgEntry.xPartition = std::get<0>(sg.tileIndex);
      sgEntry.yPartition = std::get<1>(sg.tileIndex);
      const auto numGrains =
          getNumZGrains(sg.tileIndex, zSplits, numZ, bucketsPerZ, grainZ);
      const auto numRows = sg.tileInfo.size();
      std::vector<std::size_t> rowWeights;
      // There may be empty rows as we don't delete subgroups but only tile
      // rows within the subgroup
      if (numRows == 0) {
        continue;
      }
      if (numRows != 1) {
        rowWeights.resize(numRows);
        for (std::size_t row = 0; row != numRows; ++row) {
          rowWeights[row] = sg.tileInfo[row].positionValues.size();
        }
      }
      const auto workers =
          splitTileBetweenWorkers(numRows, numGrains, numWorkers, rowWeights);
      if (workers.empty()) {
        continue;
      }
      const auto numFwdWorkers = workers.size();
      sgEntry.numWorkers = numFwdWorkers;

      auto sparseOffset = getSubgroupNzCountAndSparseOffsets(sg);
      std::size_t nzCount = sparseOffset.back();
      std::vector<MetaInfo<std::size_t>::WorkerEntry> workerEntries(
          numFwdWorkers);

      for (std::size_t wIndex = 0; wIndex != numFwdWorkers; ++wIndex) {
        const auto &worker = workers[wIndex];
        auto &entry = workerEntries[wIndex];
        entry.numXm1 = worker.getRows().size() - 1;
        entry.numZ = worker.getColumns().size() * grainZ;
        entry.sparseOffset =
            sparseOffset[worker.getRows().begin()] - sparseOffset[0];
        entry.offsetZ = worker.getColumns().begin() * grainZ;
        entry.metaInfoOffset = worker.getRows().begin();
      }

      // We may also need to include gradW if enabled
      std::vector<GradWInternal> gradWEntries;
      if (includeGradW) {
        gradWEntries = getGradWWorkerSplit(sparseOffset, numWorkers);
      }
      std::size_t numGradWWorkers = gradWEntries.size();

      std::vector<MetaInfo<std::size_t>::OutputEntry> outputEntries(numRows);
      for (std::size_t row = 0; row != numRows; ++row) {
        outputEntries[row].numY = sg.tileInfo[row].positionValues.size();

        // This must be elements and it is possible that if the same meta-info
        // is used for the forward and GradW pass, we may need to be use a max
        // of the numZ for this tile.
        outputEntries[row].offsetXInQ = sg.tileInfo[row].rowNumber;
      }

      // we keep an offset for R and and offset for Y for GradA
      const auto entriesPerNz = genForGradA ? 2 : 1;

      // Now we have all the information we need to fill in the meta information
      // tables
      std::size_t nzEntriesThisSubgroup = nzCount - sparseOffset.front();
      std::size_t offsetToNextSubGroup =
          (sizeof(sgEntry) + sizeof(workerEntries[0]) * numFwdWorkers +
           sizeof(MetaInfo<std::size_t>::GradWWorkerEntry) * numGradWWorkers +
           sizeof(outputEntries[0]) * numRows) /
              sizeof(std::size_t) +
          1 * includeGradW + nzEntriesThisSubgroup * entriesPerNz;

      std::size_t offsetToFirstOutputEntry =
          offsetToNextSubGroup - nzEntriesThisSubgroup * entriesPerNz -
          (sizeof(outputEntries[0]) * numRows) / sizeof(std::size_t);
      group.push_back(sgEntry.id);
      group.push_back(sgEntry.xPartition);
      group.push_back(sgEntry.yPartition);
      group.push_back(nzEntriesThisSubgroup);
      group.push_back(offsetToNextSubGroup);
      group.push_back(numGrains * grainZ);
      group.push_back(numRows - 1);
      group.push_back(offsetToFirstOutputEntry);
      group.push_back(numFwdWorkers);

      // add worker entries
      std::size_t workersRemaining = numFwdWorkers;
      for (const auto &w : workerEntries) {
        // GradA uses an offset of 0 in each subgroup as there is transposition
        // is fused.
        const auto thisWorkerSparseOffset = genForGradA ? 0 : w.sparseOffset;
        group.push_back(thisWorkerSparseOffset);
        group.push_back(w.numZ);
        group.push_back(w.offsetZ);
        group.push_back(w.numXm1);
        const auto metaInfoOffset =
            (w.metaInfoOffset * sizeof(outputEntries[0]) +
             workersRemaining * sizeof(w) +
             sizeof(MetaInfo<std::size_t>::GradWWorkerEntry) *
                 numGradWWorkers) /
                sizeof(std::size_t) +
            1 * includeGradW + w.sparseOffset * entriesPerNz;
        group.push_back(metaInfoOffset);
        --workersRemaining;
      }

      if (includeGradW) {
        group.push_back(numGradWWorkers);
        std::size_t workersRemaining = numGradWWorkers;
        for (const auto &w : gradWEntries) {
          group.push_back(w.sparseOffset);
          std::size_t offset =
              w.sparseOffset - w.metaInfoOffsetToOffsetsYInSFirst +
              (workersRemaining *
                   sizeof(MetaInfo<std::size_t>::GradWWorkerEntry) +
               w.metaInfoOffsetOutputEntry * sizeof(outputEntries[0])) /
                  sizeof(std::size_t);
          group.push_back(offset);
          group.push_back(w.metaInfoOffsetToOffsetsYInSFirst);
          group.push_back(w.totalNumY);
          workersRemaining--;
        }
      }

      // fill in output entries followed by Y-offsets
      for (std::size_t row = 0; row != numRows; ++row) {
        group.push_back(outputEntries[row].offsetXInQ);
        group.push_back(outputEntries[row].numY);

        const auto &rowPos = sg.tileInfo.at(row);
        for (const auto &colPair : rowPos.positionValues) {
          // This must be bytes and it is possible that if the same meta-info is
          // used for the forward and GradW pass, we may need to be use
          // a max of the numZ for this tile.
          if (genForGradA) {
            // the type size for offsets is the same as yTypeSize
            const std::size_t transposeOffset =
                static_cast<std::size_t>(colPair.second);
            group.push_back(transposeOffset * yOffsetTypeFactor);
          }
          group.push_back(colPair.first * yOffsetTypeFactor * numGrains *
                          grainZ);
          if (!genForGradA) {
            nzBucket.push_back(nzValues.at(colPair.second));
          }
        }
      }
    }
    // This is the specially encoded subgroup if to indicate the end of the
    // bucket.
    group.emplace_back(MetaInfo<std::size_t>::endSubGroupId);
  }

  if (!dnai.getPathName().empty()) {
    logging::popsparse::debug("{} : mi {} nz {}  ", dnai.getPathName(),
                              group.size(), nzBucket.size());
  }
  if (group.size() > metaInfoBucketElements) {
    throw poputil::poplibs_error("Meta info exceeds specified bucket size}");
  }
  if (nzBucket.size() > nzElemsBucketBlocks * blockSize) {
    throw poputil::poplibs_error("NZ elements exceeds specified bucket size");
  }

  // Check if bucket elements are within bounds defined by type
  auto outsideTypeBounds =
      std::find_if(group.begin(), group.end(), [](std::size_t a) {
        return a > std::numeric_limits<MetaInfoType>::max();
      });
  if (outsideTypeBounds != group.end()) {
    throw ::poputil::poplibs_error(
        "Metainfo bucket element exceeds type bound");
  }
  group.resize(metaInfoBucketElements);
  nzBucket.resize(nzElemsBucketBlocks * blockSize);
  return std::make_pair(group, nzBucket);
}

} // unnamed namespace

namespace popsparse {

namespace dynamic {

std::vector<RowPositionValues>
getPositionValuePairsPerRow(const CSRInternal &csr, std::size_t blockSizeX,
                            std::size_t blockSizeY, const Tile &tile) {
  const auto startRow = tile.getRows().begin();
  const auto endRow = tile.getRows().end();
  const auto startColumn = tile.getColumns().begin();
  const auto endColumn = tile.getColumns().end();

  if (startRow % blockSizeX) {
    throw poputil::poplibs_error(
        "Start row in tile is not divisible by the row block size");
  }

  if (endRow % blockSizeX) {
    throw poputil::poplibs_error(
        "End row in tile is not divisible by the row block size");
  }

  if (startRow >= (csr.rowIndices.size() - 1) * blockSizeX) {
    throw poputil::poplibs_error("Start row in tile doesn't match information "
                                 "in CSR");
  }

  if (endRow > (csr.rowIndices.size() - 1) * blockSizeX) {
    throw poputil::poplibs_error("End row in tile doesn't match information "
                                 "in CSR");
  }

  std::vector<RowPositionValues> rowValuePairs;
  const auto blockSize = blockSizeX * blockSizeY;
  std::size_t numBlocksInTile = 0;
  for (auto row = startRow; row != endRow; row += blockSizeX) {
    std::vector<std::pair<std::size_t, ValueType>> valuePairs;
    const auto rowIdx = row / blockSizeX;
    for (auto internalNzIdx = csr.rowIndices[rowIdx] / blockSize;
         internalNzIdx != csr.rowIndices[rowIdx + 1] / blockSize;
         ++internalNzIdx) {
      // This can be optimised if the columns are always sorted in increasing
      // order.
      if (csr.columnIndices[internalNzIdx] >= startColumn &&
          csr.columnIndices[internalNzIdx] < endColumn) {
        valuePairs.emplace_back(csr.columnIndices[internalNzIdx] - startColumn,
                                csr.nzValues[internalNzIdx]);
        ++numBlocksInTile;
      }
    }
    if (!valuePairs.empty()) {
      rowValuePairs.emplace_back(
          RowPositionValues(row - startRow, std::move(valuePairs)));
    }
  }
  logging::popsparse::debug("    - total elems/blocks {}", numBlocksInTile);
  return rowValuePairs;
}

PartitionerImpl::PartitionerImpl(
    const std::vector<std::size_t> &dimensions,
    const std::vector<std::size_t> &grainSizes,
    const std::array<std::size_t, 2> &blockDimensions_,
    const std::vector<std::size_t> &xSplits_,
    const std::vector<std::size_t> &ySplits_,
    const std::vector<std::size_t> &zSplits_,
    std::size_t metaInfoBucketElements_,
    std::size_t metaInfoBucketElementsGradA_,
    std::size_t nzElementsBucketElements_, std::size_t numWorkerContexts_,
    std::size_t bucketsPerZ_, bool useBlockMetaInfoFormat_, bool includeGradA_,
    bool includeGradW_, bool sharedBuckets_, const poplar::Type &dataType_,
    const poplar::Type &accumType_, const PartitionerOptions &options,
    bool useDense)
    : useDense(useDense) {

  auto verifySplit = [](std::size_t dimension, const std::vector<size_t> &split,
                        const std::string &str) {
    // entries are less then numX and numY?
    if (split.size() > dimension) {
      throw poputil::poplibs_error("There must be at most as many splits as "
                                   "the dimension " +
                                   str);
    }
    std::for_each(split.begin(), split.end(), [&](std::size_t dim) {
      if (dim >= dimension) {
        throw poputil::poplibs_error("An element in a split must be less than "
                                     "the dimension " +
                                     str);
      }
    });
  };

  verifySplit(dimensions.at(0), xSplits_, "X");
  verifySplit(dimensions.at(1), ySplits_, "Y");
  verifySplit(dimensions.at(2), zSplits_, "Z");

  numX = dimensions.at(0);
  numY = dimensions.at(1);
  numZ = dimensions.at(2);

  grainX = grainSizes.at(0);
  grainY = grainSizes.at(1);
  grainZ = grainSizes.at(2);

  blockDimensions = blockDimensions_;

  xSplits = xSplits_;
  ySplits = ySplits_;
  zSplits = zSplits_;

  std::sort(xSplits.begin(), xSplits.end());
  std::sort(ySplits.begin(), ySplits.end());
  std::sort(zSplits.begin(), zSplits.end());

  // keep meta info size in elements
  metaInfoBucketElements = metaInfoBucketElements_;
  metaInfoBucketElementsGradA = metaInfoBucketElementsGradA_;
  nzElemsBucketBlocks = nzElementsBucketElements_ / (grainX * grainY);
  numWorkerContexts = numWorkerContexts_;
  bucketsPerZ = bucketsPerZ_;
  useBlockMetaInfoFormat = useBlockMetaInfoFormat_;
  if (grainX * grainY > 1 && !useBlockMetaInfoFormat) {
    throw poputil::poplibs_error("Non block meta-info format does not support "
                                 "block size greater than 1x1");
  }
  gradWEnabled = includeGradW_;
  gradAEnabled = includeGradA_;
  sharedBuckets = sharedBuckets_;
  dataType = dataType_;
  accumType = accumType_;
  optimiseForSpeed = options.optimiseForSpeed;
  forceBucketSpills = options.forceBucketSpills;
  useActualWorkerSplitCosts = options.useActualWorkerSplitCosts;

  logging::popsparse::debug(
      "Created partitioner for sparse matrix mult [X,Y] x [Y,Z]: ");
  logging::popsparse::debug("  --X = {}, Y = {}, Z = {}", numX, numY, numZ);
  logging::popsparse::debug("  --Split X : {}", xSplits);
  logging::popsparse::debug("  --Split Y : {}", ySplits);
  logging::popsparse::debug("  --Split Z : {}", zSplits);
  logging::popsparse::debug("  --Buckets per Z dimension : {}", bucketsPerZ_);
  logging::popsparse::debug("  --Meta-info bucket size in elems (fwd) : {}",
                            metaInfoBucketElements);
  logging::popsparse::debug("  --Meta-info bucket size in elems (grad-a) : {}",
                            metaInfoBucketElementsGradA);
  logging::popsparse::debug("  --NZ bucket size in elements : {}",
                            nzElementsBucketElements_);
}

// creates tile partitions based purely on tiling of the matrix. The tiling is
// done given the splits the planner decides to split the matrix.
std::vector<TilePartition>
PartitionerImpl::getTilePartitions(const CSRInternal &matrix) const {
  return getTilePartition(matrix, numX, numY, numZ, grainX, grainY, xSplits,
                          ySplits, zSplits, bucketsPerZ);
}

// Fixed cost for meta info subgroup
std::size_t fixedMetaInfoCost(bool useBlockMetaInfoFormat,
                              std::size_t numWorkers, bool gradWEnabled) {
  std::size_t metaInfoCost = 0;
  if (useBlockMetaInfoFormat) {
    metaInfoCost += miElems(BMI::SubGroupEntry);
    if (gradWEnabled) {
      metaInfoCost += miElems(BMI::GradWWorkerEntry) * numWorkers;
    }
  } else {
    metaInfoCost +=
        miElems(MI::SubGroupEntry) + miElems(MI::WorkerEntry) * numWorkers;
    if (gradWEnabled) {
      metaInfoCost += miElems(MI::GradWWorkerEntry) * numWorkers + 1;
    }
  }
  return metaInfoCost;
}

void PartitionerImpl::balanceBuckets(std::vector<PNBucket> &pnBuckets) const {

  const auto numBuckets = pnBuckets.size();

  // log new parition info
  logging::popsparse::trace("Before rebalancing ... ");
  dumpBucketStatus(pnBuckets);

  auto overflown = [&](const PNBucket &bucket) {
    return (bucket.metaInfoElements > metaInfoBucketElements - 1 ||
            bucket.numNzElements > nzElemsBucketBlocks);
  };

  // The overflow is kept in this
  std::vector<PNBucket> overflowBuckets(numBuckets);

  // First determine the number of elements overflow and strip off rows
  for (std::size_t p = 0; p != numBuckets; ++p) {
    auto &bucket = pnBuckets[p];

    if (overflown(bucket) || forceBucketSpills) {
      // remove rows from partition given a target to remove
      logging::popsparse::trace(
          "  Attempting to remove rows from pn {} : sizes {} {}", p,
          bucket.metaInfoElements, bucket.numNzElements);
      const auto metaInfoElems =
          forceBucketSpills ? 0 : metaInfoBucketElements - 1;
      const auto nzInfoElems = forceBucketSpills ? 0 : nzElemsBucketBlocks;

      auto tp = removeRows(bucket, zSplits, numZ, grainZ, numWorkerContexts,
                           metaInfoElems, nzInfoElems, bucketsPerZ,
                           useBlockMetaInfoFormat, useActualWorkerSplitCosts,
                           gradWEnabled);
      overflowBuckets[p].subGroups.push_back(tp);
      fillBucketSizes(overflowBuckets[p], zSplits, numZ, grainZ,
                      useActualWorkerSplitCosts, numWorkerContexts, bucketsPerZ,
                      useBlockMetaInfoFormat, gradWEnabled,
                      " : overflow bucket for pn " + std::to_string(p));
    }
  }

  // log new parition info
  logging::popsparse::trace("After partitioning to overflown buckets ... ");
  dumpBucketStatus(pnBuckets, overflowBuckets);

  auto fits = [&](const PNBucket &target, const PNBucket &cand) {
    return (target.metaInfoElements + cand.metaInfoElements <=
            metaInfoBucketElements - 1) &&
           (target.numNzElements + cand.numNzElements <= nzElemsBucketBlocks);
  };

  auto rebalance = [&](std::size_t pnRange, bool splitColumns) {
    if (std::all_of(overflowBuckets.begin(), overflowBuckets.end(),
                    [](const PNBucket &b) { return b.empty(); })) {
      return;
    }

    std::vector<std::size_t> ovfOrder(numBuckets);
    std::iota(ovfOrder.begin(), ovfOrder.end(), 0);

    // Sort entries within range such that the biggest buckets are allocated
    // first
    assert(numBuckets % pnRange == 0);
    for (std::size_t i = 0; i != numBuckets / pnRange; ++i) {
      std::sort(ovfOrder.begin() + i * pnRange,
                ovfOrder.begin() + (i + 1) * pnRange,
                [&](std::size_t a, std::size_t b) {
                  return overflowBuckets[a] > overflowBuckets[b];
                });
    }

    // Go through candidates list to fill
    for (std::size_t x = 0; x != xSplits.size(); ++x) {
      for (std::size_t y = 0; y != ySplits.size(); ++y) {
        for (std::size_t z = 0; z != zSplits.size() * bucketsPerZ; ++z) {
          std::size_t ovfPN = getPNId(
              {
                  x,
                  y,
                  z,
              },
              {xSplits.size(), ySplits.size(), zSplits.size() * bucketsPerZ});
          std::size_t pnStart = ovfPN / pnRange * pnRange;
          std::size_t pnEnd = pnStart + pnRange;

          // selected first entry in the sorted list belonging to the range
          const auto thisPN = ovfOrder[ovfPN];
          auto &ovfBucket = overflowBuckets[thisPN];

          if (ovfBucket.empty()) {
            continue;
          }

          logging::popsparse::trace(
              "  ===== overflow for PN {} : sizes {} {} ===", thisPN,
              ovfBucket.metaInfoElements, ovfBucket.numNzElements);
          logging::popsparse::trace("   - checking range [{} {})", pnStart,
                                    pnEnd);

          // PN buckets in range sorted in increasing order of size as we
          // want the largest sized to be allocated in the largest gap first
          std::vector<std::size_t> pnOrder(pnRange);
          std::iota(pnOrder.begin(), pnOrder.end(), 0);
          std::sort(pnOrder.begin(), pnOrder.end(),
                    [&](std::size_t a, std::size_t b) {
                      return pnBuckets[pnStart + a] < pnBuckets[pnStart + b];
                    });

          // look into sorted list of PNs to fill in
          for (std::size_t i = 0; i != pnRange; ++i) {
            // order in the same direction as buckets are cycled. Ideally
            // we need some common definition that ties actual implementation
            // and what is done here.
            auto pn =
                pnStart + (optimiseForSpeed
                               ? (thisPN - pnStart + pnRange - i) % pnRange
                               : pnOrder[i]);

            // Move the maximum if buckets spills are forced
            if (forceBucketSpills) {
              pn = pnStart + (thisPN - pnStart + i) % pnRange;
            }

            // We remove whole rows to create overflow buckets as rows of large
            // size are efficient due to lower processing overheads. But when
            // rebalancing we can split rows. So we could add to the same PN
            if (pn == thisPN && forceBucketSpills) {
              continue;
            }
            auto &bucket = pnBuckets[pn];
            logBucket(bucket, "Before PN " + std::to_string(pn));
            logBucket(ovfBucket,
                      " Before Overflow PN  " + std::to_string(thisPN));
            if (fits(bucket, overflowBuckets[thisPN])) {
              bucket.move(ovfBucket);
              logging::popsparse::trace("   *+++* : moved {} -> {}", thisPN,
                                        pn);
              logBucket(bucket, "After PN " + std::to_string(pn));
              logBucket(ovfBucket,
                        " After Overflow PN " + std::to_string(thisPN));
              break;
            } else {
              const auto available = std::make_pair(
                  metaInfoBucketElements - 1 - bucket.metaInfoElements,
                  nzElemsBucketBlocks - bucket.numNzElements);
              std::vector<std::pair<std::size_t, std::size_t>> intervals;
              std::size_t rowsInSg = ovfBucket.subGroups[0].tileInfo.size();
              std::vector<std::size_t> rowWeights;
              rowWeights.resize(rowsInSg);
              for (std::size_t row = 0; row != rowsInSg; ++row) {
                rowWeights[row] =
                    ovfBucket.subGroups[0].tileInfo[row].positionValues.size();
              }
              intervals = findPartitionsToRemove(
                  rowWeights, available, useBlockMetaInfoFormat,
                  numWorkerContexts, gradWEnabled, splitColumns);
              if (intervals.empty()) {
                continue;
              }
              auto removedPartition =
                  removeIntervals(ovfBucket.subGroups[0], intervals);
              bucket.subGroups.push_back(std::move(removedPartition));
            }
            fillBucketSizes(bucket, zSplits, numZ, grainZ,
                            useActualWorkerSplitCosts, numWorkerContexts,
                            bucketsPerZ, useBlockMetaInfoFormat, gradWEnabled,
                            " : add to pn bucket" + std::to_string(pn));
            fillBucketSizes(overflowBuckets[thisPN], zSplits, numZ, grainZ,
                            useActualWorkerSplitCosts, numWorkerContexts,
                            bucketsPerZ, useBlockMetaInfoFormat, gradWEnabled,
                            " : after overflow rows removed " +
                                std::to_string(thisPN));
            logging::popsparse::trace("   *+* : rows PNs {} -> {}", thisPN, pn);
            logBucket(bucket, "After PN " + std::to_string(pn));
            logBucket(overflowBuckets[thisPN],
                      " After Overflow PN " + std::to_string(thisPN));
            // All information in overflow has been allocated
            if (ovfBucket.empty()) {
              break;
            }
          }
        }
      }
    }
  };

  std::vector<std::size_t> pnRanges = {
      zSplits.size() * bucketsPerZ,
      zSplits.size() * bucketsPerZ * ySplits.size(),
      zSplits.size() * bucketsPerZ * ySplits.size() * xSplits.size()};
  if (forceBucketSpills) {
    std::swap(pnRanges[1], pnRanges[2]);
  }
  logging::popsparse::info("Rebalance:");

  for (std::size_t pnRange : pnRanges) {
    for (bool splitColumns : {false, true}) {
      // rebalance
      logging::popsparse::info("    : range {}, split cols ? {} non empty ? {}",
                               pnRange, splitColumns,
                               countNonEmpty(overflowBuckets));
      rebalance(pnRange, splitColumns);
    }
  }

  logging::popsparse::info("After rebalancing : non empty {}",
                           countNonEmpty(overflowBuckets));

  for (auto it = pnBuckets.begin(); it != pnBuckets.end(); ++it) {
    logging::popsparse::debug(" bucket size for PN {} : mi : {} nz : {}",
                              std::distance(pnBuckets.begin(), it),
                              it->metaInfoElements, it->numNzElements);
  }

  dumpBucketStatus(pnBuckets, overflowBuckets);
  if (countNonEmpty(overflowBuckets)) {
    std::size_t maxMetaInfo = 0;
    std::size_t maxNzValues = 0;
    std::for_each(overflowBuckets.begin(), overflowBuckets.end(),
                  [&](const PNBucket &b) {
                    maxMetaInfo = std::max(maxMetaInfo, b.metaInfoElements);
                    maxNzValues = std::max(maxNzValues, b.numNzElements);
                  });
    logging::popsparse::warn("overflow metainfo {}/{}, nz values {}/{}",
                             maxMetaInfo, metaInfoBucketElements, maxNzValues,
                             nzElemsBucketBlocks);
    throw poputil::poplibs_error("Overflow in buckets");
  }
}

template <typename T>
PNBucketsImpl<T> PartitionerImpl::createBuckets(const CSRMatrix<T> &matrix_,
                                                bool checkDims) const {
  logging::popsparse::trace("Partitioner called with CSR representation");
  if (checkDims) {
    checkBlockDimensionsMatch(matrix_.getBlockDimensions(), blockDimensions);
  }
  validateCSR(numX, numY, matrix_.getBlockDimensions(), matrix_.nzValues.size(),
              matrix_.rowIndices, matrix_.columnIndices);
  std::array<std::size_t, 2> grainsXAndY = {grainX, grainY};
  const auto &matrixNewBlockSize =
      matrix_.getBlockDimensions() != grainsXAndY
          ? changeCSRBlockSize(matrix_, grainsXAndY)
          : matrix_;

  if (matrixNewBlockSize.getNumRowsInBlock() != grainX ||
      matrixNewBlockSize.getNumColumnsInBlock() != grainY) {
    throw poputil::poplibs_error(
        "Number of rows/columns in block does not match grain sizes "
        "partitioner was created with");
  }
  if (matrixNewBlockSize.rowIndices.size() != (numX / grainX) + 1) {
    throw poputil::poplibs_error(
        "Number of row indices must match number of matrix rows");
  }

  if (matrixNewBlockSize.nzValues.size() / matrixNewBlockSize.getBlockSize() !=
      matrixNewBlockSize.columnIndices.size()) {
    throw poputil::poplibs_error(
        "Number of column indices must match number of non zero values");
  }
  return this->createBucketsNoErrorCheck(matrix_, matrixNewBlockSize);
}

template <typename T>
std::vector<T>
PartitionerImpl::createDenseBuckets(const CSRMatrix<T> &matrix_) const {
  std::vector<T> result(this->numX * this->numY);
  boost::multi_array_ref<T, 2> view(result.data(),
                                    boost::extents[this->numY][this->numX]);
  csrToDenseMatrixImpl(view, matrix_.nzValues.data(),
                       matrix_.columnIndices.data(), matrix_.rowIndices.data(),
                       matrix_.nzValues.size(), this->blockDimensions[0],
                       this->blockDimensions[1]);

  return result;
}

template <typename T>
PNBucketsImpl<T> PartitionerImpl::createBucketsNoErrorCheck(
    const CSRMatrix<T> &matrix_, const CSRMatrix<T> &matrixNewBlockSize) const {

  // TODO: Avoid this copy, or at least do it in the internal format.
  auto matrix = matrixNewBlockSize;
  canonicalizeCSR(matrix);

  // Translate original matrix with typed data to a generic matrix with
  // std::size_t indices into actual data.
  std::vector<ValueType> nzOffsets;
  nzOffsets.resize(matrix.columnIndices.size());
  std::iota(nzOffsets.begin(), nzOffsets.end(), 0);
  auto csrInternal = CSRInternal(std::move(nzOffsets), matrix.columnIndices,
                                 matrix.rowIndices);
  auto tilePartitions = getTilePartitions(std::move(csrInternal));
  auto pnBuckets = createBucketsForPN(
      tilePartitions, zSplits, numZ, grainZ, useActualWorkerSplitCosts,
      numWorkerContexts, bucketsPerZ, useBlockMetaInfoFormat, gradWEnabled);
  balanceBuckets(pnBuckets);
  return {pnBuckets,
          this->useDense ? this->createDenseBuckets(matrix_) : matrix.nzValues};
}

template <typename T>
PNBucketsImpl<T>
PartitionerImpl::createBuckets(const CSCMatrix<T> &matrix_) const {
  logging::popsparse::trace("Partitioner called with CSC representation");
  checkBlockDimensionsMatch(matrix_.getBlockDimensions(), blockDimensions);
  validateCSC(numX, numY, matrix_.getBlockDimensions(), matrix_.nzValues.size(),
              matrix_.rowIndices, matrix_.columnIndices);
  std::array<std::size_t, 2> grainsXAndY = {grainX, grainY};
  const auto &matrix = matrix_.getBlockDimensions() != grainsXAndY
                           ? changeCSCBlockSize(matrix_, grainsXAndY)
                           : matrix_;

  if (matrix.getNumRowsInBlock() != grainX ||
      matrix.getNumColumnsInBlock() != grainY) {
    throw poputil::poplibs_error(
        "Number of rows/columns in block does not match grain sizes "
        "partitioner was created with");
  }
  if (matrix.columnIndices.size() != (numY / grainY) + 1) {
    throw poputil::poplibs_error(
        "Number of column indices must match number of matrix columns");
  }

  if (matrix.nzValues.size() / matrix.getBlockSize() !=
      matrix.rowIndices.size()) {
    throw poputil::poplibs_error(
        "Number of row indices must match number of non zero values");
  }

  // TODO: Transposing this full typed matrix with block size is
  // not as efficient as creating an internal CSC matrix and
  // bucketing this. This is just convenient for the timebeing.
  return createBuckets(cscToCSR(numX, numY, matrix), false);
}

template <typename T>
PNBucketsImpl<T>
PartitionerImpl::createBuckets(const COOMatrix<T> &matrix_) const {
  logging::popsparse::trace("Partitioner called with COO representation");
  checkBlockDimensionsMatch(matrix_.getBlockDimensions(), blockDimensions);
  validateCOO(numX, numY, matrix_.getBlockDimensions(), matrix_.nzValues.size(),
              matrix_.rowIndices, matrix_.columnIndices);
  std::array<std::size_t, 2> grainsXAndY = {grainX, grainY};
  const auto &matrix = matrix_.getBlockDimensions() != grainsXAndY
                           ? changeCOOBlockSize(matrix_, grainsXAndY)
                           : matrix_;

  if (matrix.nzValues.size() / matrix.getBlockSize() !=
          matrix.rowIndices.size() ||
      matrix.nzValues.size() / matrix.getBlockSize() !=
          matrix.columnIndices.size()) {
    throw poputil::poplibs_error("Number of non-zero values, row indices, and "
                                 "column indices must be equal");
  }
  return createBuckets(cooToCSR(numX, numY, matrix), false);
}

template <typename T>
std::pair<std::vector<std::size_t>, std::vector<T>>
PartitionerImpl::bucketForForward(const PNBucket &pnBucket,
                                  const std::vector<T> &nzValues,
                                  const poplar::DebugNameAndId &dnai) const {
  return bucketsImplInternal<T>(
      pnBucket, nzValues, xSplits, ySplits, zSplits, numZ, grainX, grainY,
      grainZ, useBlockMetaInfoFormat, gradWEnabled, false, dataType, accumType,
      metaInfoBucketElements, nzElemsBucketBlocks, numWorkerContexts,
      bucketsPerZ, {dnai});
}

template <typename T>
std::vector<std::size_t>
PartitionerImpl::bucketForGradA(const PNBucket &pnBucket,
                                const std::vector<T> &nzValues,
                                const poplar::DebugNameAndId &dnai) const {
  PNBucket indicesBucket;
  indicesBucket.metaInfoElements = pnBucket.metaInfoElements;
  indicesBucket.numNzElements = pnBucket.numNzElements;
  for (const auto &sg : pnBucket.subGroups) {
    ValueType index = 0;
    TilePartition tp;
    tp.tile = sg.tile;
    tp.tileIndex = sg.tileIndex;
    for (const auto &rowPos : sg.tileInfo) {
      std::vector<std::pair<std::size_t, ValueType>> positionValues;
      for (const auto &posVal : rowPos.positionValues) {
        positionValues.emplace_back(posVal.first, index++);
      }
      tp.tileInfo.emplace_back(
          RowPositionValues(rowPos.rowNumber, positionValues));
    }
    auto csr = tilePartitionToCsrMatrix(tp);
    auto transpose = csrTranspose<ValueType>(tp.tile.getRows().size(),
                                             tp.tile.getColumns().size(), csr);

    Tile tile(tp.tile.getColumns(), tp.tile.getRows());
    TileIndex tileIndex =
        std::make_tuple(std::get<1>(tp.tileIndex), std::get<0>(tp.tileIndex),
                        std::get<2>(tp.tileIndex));
    auto tpGradA = csrMatrixToTilePartition(transpose, tile, tileIndex);
    indicesBucket.subGroups.emplace_back(tpGradA);
  }
  return bucketsImplInternal<T>(
             indicesBucket, nzValues, ySplits, xSplits, zSplits, numZ, grainX,
             grainY, grainZ, useBlockMetaInfoFormat, false, true, dataType,
             accumType, metaInfoBucketElementsGradA, nzElemsBucketBlocks,
             numWorkerContexts, bucketsPerZ, {dnai})
      .first;
}

template <typename T>
std::pair<std::vector<std::vector<std::size_t>>, std::vector<std::vector<T>>>
PartitionerImpl::bucketsForForward(const PNBucketsImpl<T> &pnBucketsImpl,
                                   const poplar::DebugNameAndId &dnai) const {

  if (this->useDense) {
    throw poputil::poplibs_error(
        "Buckets for forward not implemented for dense plans");
  }
  const auto &pnBuckets = pnBucketsImpl.pnBuckets;
  const auto &nzValues = pnBucketsImpl.nzValues;
  const auto numBuckets = pnBuckets.size();
  std::vector<std::vector<std::size_t>> metaInfoBucket(numBuckets);
  std::vector<std::vector<T>> nzBucket(numBuckets);

  for (std::size_t b = 0; b != numBuckets; ++b) {
    auto pnImpl = bucketForForward(pnBuckets[b], nzValues, {dnai});
    metaInfoBucket[b] = std::move(pnImpl.first);
    nzBucket[b] = std::move(pnImpl.second);
  }
  return std::make_pair(metaInfoBucket, nzBucket);
}

template <typename T>
std::vector<std::vector<std::size_t>>
PartitionerImpl::bucketsForGradA(const PNBucketsImpl<T> &pnBucketsImpl,
                                 const poplar::DebugNameAndId &dnai) const {

  if (this->useDense) {
    throw poputil::poplibs_error(
        "Buckets for GradA not implemented for dense plans");
  }
  const auto &pnBuckets = pnBucketsImpl.pnBuckets;
  const auto &nzValues = pnBucketsImpl.nzValues;
  const auto numBuckets = pnBuckets.size();
  std::vector<std::vector<std::size_t>> metaInfoBucket(numBuckets);
  std::vector<std::vector<T>> nzBucket(numBuckets);

  for (std::size_t b = 0; b != numBuckets; ++b) {
    auto metaInfoBucketImpl = bucketForGradA(pnBuckets[b], nzValues, {dnai});
    metaInfoBucket[b] = std::move(metaInfoBucketImpl);
  }
  return metaInfoBucket;
}

std::vector<std::size_t> PartitionerImpl::overflowInfoForFwd(
    const std::vector<PNBucket> &pnBuckets) const {
  const std::vector<std::size_t> numXYZ = {xSplits.size(), ySplits.size(),
                                           zSplits.size()};
  return findOverflowDistance(pnBuckets, numXYZ, false, false, bucketsPerZ);
}

std::vector<std::size_t> PartitionerImpl::overflowInfoForGradA(
    const std::vector<PNBucket> &pnBuckets) const {
  const std::vector<std::size_t> numXYZ = {xSplits.size(), ySplits.size(),
                                           zSplits.size()};
  return findOverflowDistance(pnBuckets, numXYZ, true, false, bucketsPerZ);
}

std::vector<std::size_t> PartitionerImpl::overflowInfoForGradW(
    const std::vector<PNBucket> &pnBuckets) const {
  const std::vector<std::size_t> numXYZ = {xSplits.size(), ySplits.size(),
                                           zSplits.size()};
  return findOverflowDistance(pnBuckets, numXYZ, false, true, bucketsPerZ);
}

template <typename T>
std::pair<std::vector<std::size_t>, std::vector<T>>
PartitionerImpl::bucketImplAllPasses(const PNBucketsImpl<T> &pnBucketsImpl,
                                     const poplar::DebugNameAndId &dnai) const {

  const auto &pnBuckets = pnBucketsImpl.pnBuckets;
  const auto &nzValues = pnBucketsImpl.nzValues;
  // We use the same overflow info for all passes
  auto metaInfoBucket = overflowInfoForFwd(pnBuckets);

  std::vector<T> nzBucket;
  if (this->useDense) {
    nzBucket = pnBucketsImpl.nzValues;
  }
  for (std::size_t b = 0; b != pnBuckets.size(); ++b) {
    std::string str = "";
    if (logging::popsparse::shouldLog(logging::Level::Debug)) {
      str = "Real forward buckets for PN " + std::to_string(b);
    }
    auto bucketFwd = bucketForForward(pnBuckets[b], nzValues, {dnai, str});

    metaInfoBucket.insert(metaInfoBucket.end(), bucketFwd.first.begin(),
                          bucketFwd.first.end());
    if (!this->useDense) {
      // no need to compute Nzbuckets when using dense plan. can just copy
      // them
      // TODO could be a little more efficient by not creating the
      // nzvalues buckets at all when using dense plan inside bucketsForForward
      nzBucket.insert(nzBucket.end(), bucketFwd.second.begin(),
                      bucketFwd.second.end());
    }
    if (!sharedBuckets && gradAEnabled) {
      std::string str = "";
      if (!dnai.getPathName().empty() &&
          logging::popsparse::shouldLog(logging::Level::Debug)) {
        str = "Real forward buckets for PN " + std::to_string(b);
      }
      auto bucketGradA = bucketForGradA(pnBuckets[b], nzValues, {dnai, str});

      metaInfoBucket.insert(metaInfoBucket.end(), bucketGradA.begin(),
                            bucketGradA.end());
    }
  }
  return std::make_pair(metaInfoBucket, nzBucket);
}

template <typename T>
std::vector<T> PartitionerImpl::createCOONzValues(
    const std::vector<std::size_t> &cooNzOffsets,
    const std::vector<std::size_t> &cooColumnIndices,
    const std::vector<std::size_t> &cooRowIndices,
    const std::vector<T> &nzValues) const {

  const auto blockSize = this->grainX * this->grainY;
  std::vector<T> cooNzValues;
  cooNzValues.reserve(cooNzOffsets.size() * blockSize);
  if (!this->useDense) {
    for (const auto &nzOffset : cooNzOffsets) {
      if ((nzOffset + 1) * blockSize > nzValues.size()) {
        throw poputil::poplibs_error("Invalid meta-info: Meta-info refers to "
                                     "more non-zero elements than exist");
      }
      cooNzValues.insert(cooNzValues.end(),
                         nzValues.begin() + nzOffset * blockSize,
                         nzValues.begin() + (nzOffset + 1) * blockSize);
    }
  } else {
    assert(cooColumnIndices.size() == cooRowIndices.size());
    boost::const_multi_array_ref<T, 2> mat(
        nzValues.data(), boost::extents[this->numY][this->numX]);
    assert(nzValues.size() == mat.num_elements());
    for (std::size_t i = 0; i < cooRowIndices.size(); ++i) {
      for (std::size_t x = cooRowIndices[i];
           x < cooRowIndices[i] + this->grainX; ++x) {
        for (std::size_t y = cooColumnIndices[i];
             y < cooColumnIndices[i] + this->grainY; ++y) {
          cooNzValues.push_back(mat[y][x]);
        }
      }
    }
  }
  return cooNzValues;
}

template <typename T>
COOMatrix<T>
PartitionerImpl::bucketsToCOOMatrix(const std::vector<std::size_t> &metaInfo,
                                    const std::vector<T> &nzValues) const {

  using U = std::size_t;
  using MI_U = MetaInfo<U>;
  using BMI_U = BlockMetaInfo<U>;

  const auto blockSizeX = grainX;
  const auto blockSizeY = grainY;
  const auto blockSize = blockSizeX * blockSizeY;

  // We use metaInfo that is created for the combined passes but we only look at
  // the forward buckets to reconstruct the COO representation
  std::size_t miBucketElemsPerPN = metaInfoBucketElements;
  if (gradAEnabled && !sharedBuckets) {
    miBucketElemsPerPN += metaInfoBucketElementsGradA;
  }

  // Number of buckets
  const std::size_t numBuckets =
      xSplits.size() * ySplits.size() * zSplits.size() * bucketsPerZ;

  // exclude overflow info which is part of meta info
  std::size_t miIndex = getNumOverflowInfoElems(
      sizeof(MetaInfoType), xSplits.size(), ySplits.size(), zSplits.size());

  if (metaInfo.size() != miIndex + numBuckets * miBucketElemsPerPN) {
    throw poputil::poplibs_error("Metainfo flattened buckets size does not "
                                 "match partitioner in COO conversion");
  }
  if (this->useDense) {
    if (nzValues.size() != this->numX * this->numY) {
      throw poputil::poplibs_error("Nzvalues doens't equal size of matrix");
    }
  } else {
    if (nzValues.size() != numBuckets * nzElemsBucketBlocks * blockSize) {
      throw poputil::poplibs_error("NZ flattened buckets size does not match "
                                   "partitioner in COO conversion");
    }
  }

  std::vector<std::size_t> cooRowIndices;
  std::vector<std::size_t> cooColumnIndices;
  std::vector<std::size_t> cooNzOffsets;
  std::vector<std::size_t> flattenedIndex;

  if (useBlockMetaInfoFormat) {
    for (std::size_t b = 0, nzIndex = 0; b != numBuckets;
         ++b, miIndex += miBucketElemsPerPN, nzIndex += nzElemsBucketBlocks) {
      std::size_t miIndexThisPN = miIndex;
      std::size_t nzIndexThisPN = nzIndex;

      while (metaInfo[miIndexThisPN] != BMI_U::endSubGroupId) {
        const auto *sgEntry = reinterpret_cast<const BMI_U::SubGroupEntry *>(
            &metaInfo[miIndexThisPN]);
        auto groupIndices =
            getGroupIndices(sgEntry->id, xSplits.size(), ySplits.size());
        if (groupIndices.first >= xSplits.size() ||
            groupIndices.second >= ySplits.size()) {
          throw poputil::poplibs_error(
              "Invalid meta-info: Invalid subGroupId " +
              std::to_string(sgEntry->id));
        }

        const auto startRow = xSplits.at(groupIndices.first);
        const auto endRow = groupIndices.first + 1 == xSplits.size()
                                ? numX
                                : xSplits.at(groupIndices.first + 1);
        const auto startCol = ySplits.at(groupIndices.second);
        const auto endCol = groupIndices.second + 1 == ySplits.size()
                                ? numY
                                : ySplits.at(groupIndices.second + 1);

        const auto numRows = sgEntry->numXm1 + 1;
        if (numRows > endRow - startRow) {
          throw poputil::poplibs_error(
              "Invalid meta-info: Invalid number of rows (" +
              std::to_string(numRows) + ") for subGroup " +
              std::to_string(sgEntry->id));
        }

        std::size_t index = miIndexThisPN +
                            sizeof(BMI_U::SubGroupEntry) / sizeof(U) +
                            sgEntry->numGradWWorkers *
                                sizeof(BMI_U::GradWWorkerEntry) / sizeof(U);

        for (std::size_t rowIdx = 0; rowIdx != numRows; ++rowIdx) {
          const auto *outputEntry =
              reinterpret_cast<const BMI_U::OutputEntry *>(&metaInfo[index]);
          index += sizeof(BMI_U::OutputEntry) / sizeof(U);
          const auto thisRow = startRow + outputEntry->offsetXInQ;
          if (startRow > thisRow || thisRow >= endRow) {
            throw poputil::poplibs_error(
                "Invalid meta-info: Invalid row (" + std::to_string(thisRow) +
                ") for subGroup " + std::to_string(sgEntry->id));
          }
          const auto *inputEntries =
              reinterpret_cast<const BMI_U::InputEntry *>(&metaInfo[index]);
          const auto numCols = outputEntry->numYm1 + 1;
          if (numCols > endCol - startCol) {
            throw poputil::poplibs_error(
                "Invalid meta-info: Invalid number of columns (" +
                std::to_string(numCols) + ") for row " +
                std::to_string(thisRow) + " in subGroup " +
                std::to_string(sgEntry->id));
          }
          index += (sizeof(BMI_U::InputEntry) / sizeof(U)) * numCols;

          for (std::size_t colIdx = 0; colIdx < numCols; ++colIdx) {
            const auto thisCol = startCol + inputEntries[colIdx].offsetYInS;
            if (startCol > thisCol || thisCol >= endCol) {
              throw poputil::poplibs_error(
                  "Invalid meta-info: Invalid column (" +
                  std::to_string(thisCol) + ") for row " +
                  std::to_string(thisRow) + " in subGroup " +
                  std::to_string(sgEntry->id));
            }
            cooRowIndices.push_back(thisRow);
            cooColumnIndices.push_back(thisCol);
            flattenedIndex.push_back(thisRow * numY + thisCol);
            cooNzOffsets.push_back(nzIndexThisPN++);
          }
        }
        miIndexThisPN += sgEntry->offsetToNextSubGroupMetaInfo;
        if (sgEntry->offsetToNextSubGroupMetaInfo <
                sizeof(BMI_U::SubGroupEntry) / sizeof(U) ||
            (miIndexThisPN >= miIndex + miBucketElemsPerPN)) {
          throw poputil::poplibs_error(
              "Invalid meta-info: offset to next subGroup from subGroup " +
              std::to_string(sgEntry->id) + " not in valid range");
        }
      }
    }
  } else {

    // offsets are scaled depending on data type.
    const std::size_t yOffsetTypeFactor =
        popsparse::getYOffsetTypeScaleFactor(dataType == poplar::FLOAT);
    for (std::size_t b = 0, nzIndex = 0; b != numBuckets;
         ++b, miIndex += miBucketElemsPerPN, nzIndex += nzElemsBucketBlocks) {
      std::size_t miIndexThisPN = miIndex;
      std::size_t nzIndexThisPN = nzIndex;

      while (metaInfo[miIndexThisPN] != MI::endSubGroupId) {
        const auto *sgEntry = reinterpret_cast<const MI_U::SubGroupEntry *>(
            &metaInfo[miIndexThisPN]);
        auto groupIndices =
            getGroupIndices(sgEntry->id, xSplits.size(), ySplits.size());

        if (groupIndices.first >= xSplits.size() ||
            groupIndices.second >= ySplits.size()) {
          throw poputil::poplibs_error("possibly corrupt or invalid metaInfo");
        }

        // we can now get the indices of rows and columns
        const auto numRows = sgEntry->numXm1 + 1;
        const auto zScale = sgEntry->numZ;

        if (numRows > numX || zScale > numZ) {
          throw poputil::poplibs_error("possibly corrupt or invalid metaInfo");
        }

        std::size_t index =
            miIndexThisPN + sgEntry->offsetToFirstOutputEntryMetaInfo;
        for (std::size_t row = 0; row != numRows; ++row) {
          const auto *outputEntry =
              reinterpret_cast<const MI_U::OutputEntry *>(&metaInfo[index]);
          index += miElems(MI::OutputEntry);
          const auto thisRow =
              xSplits[groupIndices.first] + outputEntry->offsetXInQ;
          const auto *yOffset = reinterpret_cast<const U *>(&metaInfo[index]);
          if (outputEntry->numY > numY) {
            throw poputil::poplibs_error(
                "possibly corrupt or invalid metaInfo");
          }
          for (std::size_t col = 0; col != outputEntry->numY; ++col) {
            const auto yIdx = *yOffset++ / (yOffsetTypeFactor * zScale);
            const auto colIndex = yIdx + ySplits[groupIndices.second];
            cooRowIndices.push_back(thisRow);
            cooColumnIndices.push_back(colIndex);
            cooNzOffsets.push_back(nzIndexThisPN++);
            flattenedIndex.push_back(thisRow * numY + colIndex);
          }
          index += outputEntry->numY;
        }
        miIndexThisPN += sgEntry->offsetToNextSubGroupMetaInfo;
        // This is to catch abnormalities in the data
        if (sgEntry->offsetToNextSubGroupMetaInfo <
                sizeof(MI_U::SubGroupEntry) / sizeof(U) ||
            (miIndexThisPN >= miIndex + miBucketElemsPerPN)) {
          throw poputil::poplibs_error("possibly corrupt or invalid metaInfo");
        }
      }
    }
  }

  // Sort all indices + values by row-major order
  std::vector<std::size_t> index;
  index.resize(cooNzOffsets.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&](std::size_t a, std::size_t b) {
    return flattenedIndex[a] < flattenedIndex[b];
  });

  for (std::size_t i = 0; i != index.size(); ++i) {
    std::size_t j;
    while (i != (j = index[i])) {
      auto k = index[j];
      std::swap(cooColumnIndices[j], cooColumnIndices[k]);
      std::swap(cooRowIndices[j], cooRowIndices[k]);
      std::swap(cooNzOffsets[j], cooNzOffsets[k]);
      std::swap(index[i], index[j]);
    }
  }

  auto cooNzValues = createCOONzValues(cooNzOffsets, cooColumnIndices,
                                       cooRowIndices, nzValues);

  auto matrix =
      COOMatrix<T>(std::move(cooNzValues), std::move(cooColumnIndices),
                   std::move(cooRowIndices), {blockSizeX, blockSizeY});
  std::array<std::size_t, 2> grainXAndY = {blockSizeX, blockSizeY};
  if (blockDimensions != grainXAndY) {
    return changeCOOBlockSize(matrix, blockDimensions);
  }
  return matrix;
}

template <typename T>
CSRMatrix<T>
PartitionerImpl::bucketsToCSRMatrix(const std::vector<std::size_t> &metaInfo,
                                    const std::vector<T> &nzValues) const {
  auto cooMatrix = bucketsToCOOMatrix(metaInfo, nzValues);
  return cooToCSR(numX, numY, cooMatrix);
}

template <typename T>
CSCMatrix<T>
PartitionerImpl::bucketsToCSCMatrix(const std::vector<std::size_t> &metaInfo,
                                    const std::vector<T> &nzValues) const {
  auto csrMatrix = bucketsToCSRMatrix(metaInfo, nzValues);
  return csrToCSC(numX, numY, csrMatrix);
}

// Instantiations of templated member methods
template PNBucketsImpl<double>
PartitionerImpl::createBuckets<double>(const CSCMatrix<double> &) const;
template PNBucketsImpl<float>
PartitionerImpl::createBuckets<float>(const CSCMatrix<float> &) const;

template PNBucketsImpl<double>
PartitionerImpl::createBuckets<double>(const CSRMatrix<double> &, bool) const;
template PNBucketsImpl<float>
PartitionerImpl::createBuckets<float>(const CSRMatrix<float> &, bool) const;

template PNBucketsImpl<double>
PartitionerImpl::createBuckets<double>(const COOMatrix<double> &) const;
template PNBucketsImpl<float>
PartitionerImpl::createBuckets<float>(const COOMatrix<float> &) const;

template CSCMatrix<double>
PartitionerImpl::bucketsToCSCMatrix<double>(const std::vector<std::size_t> &,
                                            const std::vector<double> &) const;
template CSCMatrix<float>
PartitionerImpl::bucketsToCSCMatrix<float>(const std::vector<std::size_t> &,
                                           const std::vector<float> &) const;

template CSRMatrix<double>
PartitionerImpl::bucketsToCSRMatrix<double>(const std::vector<std::size_t> &,
                                            const std::vector<double> &) const;
template CSRMatrix<float>
PartitionerImpl::bucketsToCSRMatrix<float>(const std::vector<std::size_t> &,
                                           const std::vector<float> &) const;
template COOMatrix<double>
PartitionerImpl::bucketsToCOOMatrix<double>(const std::vector<std::size_t> &,
                                            const std::vector<double> &) const;
template COOMatrix<float>
PartitionerImpl::bucketsToCOOMatrix<float>(const std::vector<std::size_t> &,
                                           const std::vector<float> &) const;

template std::vector<std::vector<std::size_t>>
PartitionerImpl::bucketsForGradA<double>(const PNBucketsImpl<double> &,
                                         const poplar::DebugNameAndId &) const;
template std::vector<std::vector<std::size_t>>
PartitionerImpl::bucketsForGradA<float>(const PNBucketsImpl<float> &,
                                        const poplar::DebugNameAndId &) const;

template std::pair<std::vector<std::size_t>, std::vector<double>>
PartitionerImpl::bucketImplAllPasses<double>(
    const PNBucketsImpl<double> &, const poplar::DebugNameAndId &) const;

template std::pair<std::vector<std::size_t>, std::vector<float>>
PartitionerImpl::bucketImplAllPasses<float>(
    const PNBucketsImpl<float> &, const poplar::DebugNameAndId &) const;

} // namespace dynamic
} // namespace popsparse
