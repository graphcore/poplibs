// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ReductionPlan.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include <boost/icl/interval_map.hpp>

#include "poplibs_support/logging.hpp"
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_support/IclUtil.hpp>

#include "RegionWrapping.hpp"

using namespace poplar;
using namespace poplibs;
using namespace poplibs_support;

namespace popops {

// Get the maximum number of tiles that a single partial value is spread over.
std::size_t getMaxTileSpread(const Graph::TileToTensorMapping &mapping,
                             std::size_t outputSize) {

  boost::icl::interval_map<std::size_t, std::size_t> spread;
  using MapEntry =
      std::pair<boost::icl::right_open_interval<std::size_t>, std::size_t>;
  auto comp = [](const MapEntry &a, const MapEntry &b) {
    return a.second < b.second;
  };

  for (const auto &tileRegions : mapping) {
    boost::icl::interval_set<std::size_t> outputRegionsUsedOnTile;
    wrapRegions(tileRegions.begin(), tileRegions.end(), outputSize,
                [&](size_t begin, size_t end) {
                  outputRegionsUsedOnTile +=
                      boost::icl::interval<std::size_t>::right_open(begin, end);
                });
    // add in regions used by tile
    for (const auto &region : outputRegionsUsedOnTile) {
      spread.add(std::make_pair(region, 1));
    }
  }
  return std::max_element(spread.begin(), spread.end(), comp)->second;
}

// Get the maximum number of elements on any tile for a mapping.
//
// reducedMapping is a vector of tiles, each of which
// contains a vector of tensor regions. The length of the regions
// on each tile is summed and the maximum total is returned.
static unsigned getMaxElementsPerTile(
    const std::vector<std::vector<Interval>> &reducedMapping) {
  // The current maximum.
  unsigned maxElementsPerTile = 0;

  for (const auto &entry : reducedMapping) {
    unsigned tileElements =
        std::accumulate(entry.begin(), entry.end(), 0U,
                        [](unsigned sum, const Interval &region) {
                          return sum + region.end() - region.begin();
                        });
    maxElementsPerTile = std::max(maxElementsPerTile, tileElements);
  }
  return maxElementsPerTile;
}

// Get the maximum number of partial inputs for any output.
static unsigned getMaxPartialsPerElement(const IntermediatePartials &ipIn) {
  boost::icl::interval_map<std::size_t, unsigned> numPartials;

  for (auto tile : ipIn.tiles()) {
    for (const auto &re : ipIn.outputRegions(tile))
      numPartials += std::make_pair(re, 1U);
  }

  return std::max_element(numPartials.begin(), numPartials.end())->second;
}

// Estimate how many cycles the final reduction step will take if we send the
// partials to the tiles where the output is mapped and do the reduction there.
static unsigned
estimateReduceAtDstCost(const Target &target, const IntermediatePartials &ipIn,
                        const Graph::TileToTensorMapping &outMapping) {

  const auto partialType = ipIn.dataType();
  const auto partialTypeBytes = target.getTypeSize(partialType);
  const auto partialVectorWidth = target.getVectorWidth(partialType);

  // The maximum number of partials that are reduced to a single output.
  unsigned maxPartialsPerElement = getMaxPartialsPerElement(ipIn);

  // Get the maximum number of partials that need to be sent to each tile
  // for the reduction.
  std::size_t maxElementsPerTile = getMaxElementsPerTile(outMapping);

  // How many bytes we might have to send from each tile before computation.
  const auto preComputeExchangeBytes = maxElementsPerTile * partialTypeBytes;

  // How quickly exchange happens.
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();

  // Get the delay caused by syncing i.e. the time before when all tiles
  // are done, and when the last tile is released from sync.
  const auto syncCycles = target.getMaxIPUSyncDelay();

  unsigned cycles = 0;
  // Pre-exchange.
  cycles += (preComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  // Compute
  cycles +=
      maxPartialsPerElement *
      ((maxElementsPerTile + partialVectorWidth - 1) / partialVectorWidth);
  // Sync
  cycles += syncCycles;

  // There's no post-exchange step here.
  return cycles;
}

// Estimate how many cycles the reduce will cost if we send the partials to
// every tile, do the reduction, and then send the reduced values to the tiles
// where the output is mapped.
static unsigned estimateBalancedReduceCost(
    const Target &target, const IntermediatePartials &ipIn, Type reducedType,
    std::size_t numReducedElements,
    const Graph::TileToTensorMapping &outMapping, unsigned grainSize) {

  // The type of the elements we are summing.
  const auto partialType = ipIn.dataType();
  const auto partialTypeBytes = target.getTypeSize(partialType);
  const auto partialVectorWidth = target.getVectorWidth(partialType);
  const auto reducedTypeBytes = target.getTypeSize(reducedType);

  // Split the reduced elements into groups of size `grainSize` and
  // get the number of groups.
  unsigned numReducedGroups = (numReducedElements + grainSize - 1) / grainSize;
  const auto numTiles = target.getNumTiles();

  // Work out how many groups would go on each tile (at most).
  unsigned maxReducedGroups = (numReducedGroups + numTiles - 1) / numTiles;

  // The maximum number of elements on each tile.
  const auto maxElementsPerTile = maxReducedGroups * grainSize;

  // The maximum number of partials that are reduced to a single output.
  unsigned maxPartialsPerElement = getMaxPartialsPerElement(ipIn);

  // Worst case exchange to get data onto the tile.
  const auto preComputeExchangeBytes =
      maxElementsPerTile * maxPartialsPerElement * partialTypeBytes;
  // Worse case exchange to get data to the destination.
  const auto postComputeExchangeBytes =
      getMaxElementsPerTile(outMapping) * reducedTypeBytes;
  const auto exchangeBytesPerCycle = target.getExchangeBytesPerCycle();
  const auto syncCycles = target.getMaxIPUSyncDelay();
  unsigned cycles = 0;
  // Pre-exchange
  cycles += (preComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  // Sync.
  cycles += syncCycles;
  // Compute
  cycles +=
      maxPartialsPerElement *
      ((maxElementsPerTile + partialVectorWidth - 1) / partialVectorWidth);
  // Post-exchange
  cycles += (postComputeExchangeBytes + exchangeBytesPerCycle - 1) /
            exchangeBytesPerCycle;
  // Sync.
  cycles += syncCycles;
  return cycles;
}

bool shouldReduceAtDestination(const Target &target,
                               const IntermediatePartials &ipIn,
                               const Graph::TileToTensorMapping &outMapping,
                               Type reducedType,
                               std::size_t numReducedElements) {

  // Is it best to do the reduction spread out over lots of tiles, and
  // then copy the result to the destination, or do it at the destination?

  auto balancedCycles = estimateBalancedReduceCost(
      target, ipIn, reducedType, numReducedElements, outMapping, 8);
  auto destCycles = estimateReduceAtDstCost(target, ipIn, outMapping);

  return destCycles < balancedCycles;
}

boost::icl::split_interval_map<std::size_t, std::size_t>
calculateSplit(const IntermediatePartials &ir, std::size_t grainSize,
               std::size_t minPieceCols, std::size_t minPieceRows,
               std::size_t minPieceSize, unsigned numPieces) {

  // Calculate the pieces we should split this into. Basically, first we get
  // the entire set of output regions from every tile (preserving splits).
  //
  // For example if we have the following regions on each tile:
  //
  // Tile                Output regions on tile
  //
  //  0    |-------|      |-------------------|  |------|
  //  1    |-------|    |---------------------|  |------------|
  //  2    |--------------------------------------------------|
  //  3            |----------------|
  //
  // Then allOutputRegionsSplitIcl is
  //
  //       |-------|----|-|---------|---------|--|------|-----|

  // Get the entire set of output regions from every tile.
  boost::icl::split_interval_set<std::size_t> allOutputRegionsSplitIcl;
  for (auto tile : ir.tiles()) {
    allOutputRegionsSplitIcl += ir.outputRegions(tile);
  }

  // find minimum non-split output region to decide whether to increase
  // minimum number of columns in a piece.
  typedef decltype(allOutputRegionsSplitIcl)::interval_type IntervalType;
  auto getIntervalSize = [](const IntervalType &it) {
    return it.upper() - it.lower();
  };
  const auto minInterval = *std::min_element(
      allOutputRegionsSplitIcl.begin(), allOutputRegionsSplitIcl.end(),
      [&](const IntervalType &a, const IntervalType &b) {
        return getIntervalSize(a) < getIntervalSize(b);
      });
  auto minIntervalSize = getIntervalSize(minInterval);

  // Work out the total amount of data. This is the total number of partials
  // in the reduction.
  std::size_t totalDataSize = 0;
  for (auto tile : ir.tiles())
    totalDataSize += ir.data(tile).numElements();

  // Find the average partials per output in this intermediate reduction stage.
  // As this is the internediate stage, this gives the average number of
  // tiles which have partials to contribute to each output. We use the square
  // root of the average number of tiles as the number of output partials.
  const auto averageSplit =
      static_cast<std::size_t>(std::sqrt(totalDataSize / ir.outputSize()));

  // Use a different minimum columns only if the minimum interval size allows
  // one to be used. This has the effect of increasing the number of reductions
  // per output and also reducing the amount of exchange size as grain size
  // per tile is increased.
  const auto averageGrainFactor = (minIntervalSize / grainSize) / averageSplit;
  const auto minPieceColsToUse =
      std::max(minPieceCols, averageGrainFactor * grainSize);

  if (minPieceColsToUse != minPieceCols) {
    logging::debug("Intermediate stage minimum columns changed from {} -> {}",
                   minPieceCols, minPieceColsToUse);
  }

  // We should have an output for every element in the final tensor.
  assert(allOutputRegionsSplitIcl.size() == ir.outputSize());

  auto allOutputRegions = splitIntervalSetToPoplar(allOutputRegionsSplitIcl);

  // Add extra splits so that we have at least numPieces pieces.
  // splitRegions is used here rather than splitRegionsBetweenWorkers
  // because the former never merges regions, and the latter might merge them.
  auto split = poputil::splitRegions(allOutputRegions, grainSize, numPieces,
                                     minPieceColsToUse);

  // Finally if that is not enough, work out the total amount of data, divide
  // it by numPieces to give how much data should be in each piece.
  // Then go through the output intervals and set the number of splits so
  // that each has roughly that amount of data.
  std::size_t idealPieceSize = totalDataSize / numPieces;
  // Avoid division by zero later.
  if (idealPieceSize < 1)
    idealPieceSize = 1;

  // Work out the answer.
  boost::icl::split_interval_map<std::size_t, std::size_t> splitMap;

  const auto &tilesForOutput = ir.getTilesForOutput();

  // splitRegions() decides a partition (split is a vector of vectors) but
  // we ignore than and just loop over every region, ignoring which partition
  // it is in.
  for (const auto &partition : split) {
    for (const auto &re : partition) {
      auto numPartials = tilesForOutput(re.begin()).size();
      auto numCols = re.size();

      // TODO: T12969 This would be more elegant if we work out how many rows
      // each piece is rather than how many there are.

      auto N = (numPartials * numCols) / idealPieceSize;
      // Prevent divide by zero.
      if (N < 1)
        N = 1;

      if ((numPartials * numCols) / N < minPieceSize)
        N = (numPartials * numCols) / minPieceSize;
      if (N < 1)
        N = 1;

      if (numPartials / N < minPieceRows)
        N = numPartials / minPieceRows;
      if (N < 1)
        N = 1;

      auto iclRe =
          boost::icl::interval<std::size_t>::right_open(re.begin(), re.end());

      splitMap.insert(std::make_pair(iclRe, N));
    }
  }

  return splitMap;
}

NextStep calculateNextStep(const Target &target,
                           const IntermediatePartials &ir) {
  // Should we do another intermediate reduction stage, or go straight to the
  // destination reduction?
  unsigned maxSources = 0;
  for (const auto &t : ir.getTilesForOutput()) {
    auto numTileSources = t.second.size();
    if (maxSources < numTileSources)
      maxSources = numTileSources;
  }

  // If the outputs occupy most of the tiles and the fan-in is low enough go
  // straight to an output stage. This should require less exchange and control
  // code without being significantly slower.
  if ((maxSources < 2 * sqrt(target.getTilesPerIPU())) &&
      (ir.tiles().size() > target.getTilesPerIPU() * 3 / 4))
    return INTERMEDIATE_TO_OUTPUT;

  // Basically, see how much data is left. If it is a lot, do another step.

  std::size_t totalDataSize = 0;
  for (auto tile : ir.tiles())
    totalDataSize += ir.data(tile).numElements();

  // Optimisation: reductionFactorThresholdToAddMoreStages was found empirically
  // and hasn't been tested a lot. E.g. on different sizes of IPUs.
  if (totalDataSize > ir.outputSize() * reductionFactorThresholdToAddMoreStages)
    return INTERMEDIATE_TO_INTERMEDIATE;

  return INTERMEDIATE_TO_OUTPUT;
}

} // namespace popops
