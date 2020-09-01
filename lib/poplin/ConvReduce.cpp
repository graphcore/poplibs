// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ConvReduce.hpp"

#include "ConvOptions.hpp"
#include "ConvReducePlan.hpp"
#include "poplin/ConvUtil.hpp"
#include "popops/Cast.hpp"
#include "popops/Zero.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include "poplibs_support/logging.hpp"

#include <boost/icl/interval_map.hpp>
#include <cassert>
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <set>
#include <utility>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace poplin {

static void
reduce(Graph &graph, std::map<unsigned, unsigned> tileToRow,
       bool enableFastReduce, bool enableSingleInputReduce,
       unsigned thisStageRows, Tensor partials, Tensor reduced,
       const std::vector<std::vector<Interval>> &reduceVertexMapping,
       ComputeSet reduceCS) {
  const auto &target = graph.getTarget();
  assert(partials[0].shape() == reduced.shape());
  if (partials.dim(0) == 0) {
    popops::zero(graph, reduced, reduceVertexMapping, reduceCS);
    return;
  }
  if (partials.dim(0) == 1) {
    popops::cast(graph, partials[0], reduced, reduceCS);
    return;
  }
  const auto partialType = partials.elementType();
  const auto bytesPerPartialsElement = target.getTypeSize(partialType);
  const auto partialsGrain = partialType == HALF ? 8 : 4;
  const auto memoryElementOffsets = target.getMemoryElementOffsets();
  const auto reducedType = reduced.elementType();
  const auto tilesPerInZGroup = partials.dim(0);
  if (tilesPerInZGroup > target.getRptCountMax()) {
    // if this error is hit then reduction needs to be split into smaller
    // reductions first reduce then the output of each are reduced against
    // each other
    poplibs_error("Reduction too large for counter");
  }
  auto flatPartials = partials.reshape(
      {tilesPerInZGroup, partials.numElements() / tilesPerInZGroup});
  auto flatReduced = reduced.flatten();
  std::string vertexName = "poplin::ReduceAdd";

  // Accumulate the partial sums.
  const auto numUsedTiles = reduceVertexMapping.size();
  assert(numUsedTiles <= target.getNumTiles());
  unsigned tilesUsed = 0;
  std::size_t maxOutWidth = 0;
  for (unsigned tile = 0; tile != numUsedTiles; ++tile) {
    const auto &tileRegions = reduceVertexMapping[tile];

    if (tileRegions.empty())
      continue;
    tilesUsed++;
    auto concatFlatReduced = concat(flatReduced.slices(tileRegions));
    const auto exchangedPartialsBytes = concatFlatReduced.numElements() *
                                        (tilesPerInZGroup - 1) *
                                        bytesPerPartialsElement;
    bool singleInputReduceIsPossible =
        ((concatFlatReduced.numElements() % partialsGrain) == 0);
    bool singleInputReducePartialsSize =
        enableSingleInputReduce &&
        checkPartialsSizeForSingleInputReduce(exchangedPartialsBytes,
                                              memoryElementOffsets);
    bool useSingleInputReduce =
        singleInputReduceIsPossible &&
        (enableFastReduce || singleInputReducePartialsSize);
    const auto v = graph.addVertex(
        reduceCS, templateVertex(vertexName, reducedType, partialType,
                                 useSingleInputReduce, enableFastReduce));

    if (useSingleInputReduce) {
      auto thisRow = tileToRow[tile] % thisStageRows;
      auto thisTilePartial =
          flatPartials.slice(thisRow, thisRow + 1, 0).flatten();
      thisTilePartial = concat(thisTilePartial.slices(tileRegions)).flatten();

      std::vector<Interval> exchangedIntervals = {
          {0, thisRow}, {thisRow + 1, flatPartials.dim(0)}};

      auto exchangedPartials =
          concat(flatPartials.slices(exchangedIntervals, 0), 0);
      exchangedPartials =
          concat(exchangedPartials.slices(tileRegions, 1), 1).flatten();
      graph.connect(v["initialPartial"], thisTilePartial);
      graph.connect(v["partials"], exchangedPartials);
      // One less as we separated out a row
      graph.setInitialValue(v["numPartials"], tilesPerInZGroup - 1);
    } else {
      // concatenate inner dimension so that vertex has a single contiguous
      // region to divide work across. Partials have to brought on-tile for each
      // input partial except for one that may reside on-tile.
      auto concatFlatPartials = concat(flatPartials.slices(tileRegions, 1), 1);
      graph.connect(v["partials"], concatFlatPartials);
      graph.setInitialValue(v["numPartials"], tilesPerInZGroup);
    }
    graph.setInitialValue(v["numElems"], concatFlatReduced.numElements());
    graph.connect(v["out"], concatFlatReduced);
    graph.setTileMapping(v, tile);

    maxOutWidth = std::max(maxOutWidth, concatFlatReduced.numElements());
  }
  logging::poplin::debug("    Tiles used : {} max outWidth: {}", tilesUsed,
                         maxOutWidth);
}

static Tensor partialGroupedReduce(
    Graph &graph, const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &tileGroupRegions,
    std::map<unsigned, unsigned> tileToRow, bool enableFastReduce,
    bool enableSingleInputReduce, const Tensor &partials, unsigned outDepth,
    const Type &resultType, ComputeSet cs, const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out =
      graph.addVariable(resultType, outDims, debugPrefix + "/partialReduceOut");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const unsigned minGrainSize = target.getVectorWidth(resultType);
  const unsigned partialChansPerGroup = partials.dim(partials.rank() - 1);
  const auto grainSize = std::max(partialChansPerGroup, minGrainSize);
  const auto roundedGrainSize =
      (grainSize + minGrainSize - 1) / minGrainSize * minGrainSize;

  logging::poplin::debug("  Total reduction depth {} in {} sections",
                         partialsDepth, outDepth);
  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval>> outSubMapping(numTiles);
    for (unsigned tileGroup = 0; tileGroup != numTileGroups; ++tileGroup) {
      const auto tilesInGroup = tileGroups[tileGroup].size();
      const auto tileBegin = (i * tilesInGroup) / outDepth;
      const auto tileEnd =
          std::max(tileBegin + 1, ((i + 1) * tilesInGroup) / outDepth);
      const auto outSplitRegions = splitRegions(
          tileGroupRegions[tileGroup], roundedGrainSize, tileEnd - tileBegin);
      for (unsigned j = 0; j != outSplitRegions.size(); ++j) {
        const auto tileIndex = tileGroups[tileGroup][j + tileBegin];
        outSubMapping[tileIndex].insert(outSubMapping[tileIndex].end(),
                                        outSplitRegions[j].begin(),
                                        outSplitRegions[j].end());
      }
    }

    graph.setTileMapping(out[i], outSubMapping);
    logging::poplin::debug("    Reduction section with Depth {}, Height {}.",
                           end - begin, out[i].numElements());
    reduce(graph, tileToRow, enableFastReduce, enableSingleInputReduce,
           end - begin, partials.slice(begin, end), out[i], outSubMapping, cs);
  }
  return out;
}

static Tensor groupedReduce(
    Graph &graph, const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &tileGroupRegions,
    std::map<unsigned, unsigned> tileToRow, bool enableFastReduce,
    bool enableSingleInputReduce, const Tensor &partials,
    const Type &resultType, ComputeSet cs, const std::string &debugPrefix) {
  return partialGroupedReduce(graph, tileGroups, tileGroupRegions, tileToRow,
                              enableFastReduce, enableSingleInputReduce,
                              partials, 1, resultType, cs, debugPrefix)
      .reshape(partials[0].shape());
}

static Tensor multiStageGroupedReduce(
    Graph &graph, const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &tileGroupRegions,
    std::map<unsigned, unsigned> tileToRow, Tensor partials,
    const Type &resultType, std::vector<ComputeSet> &computeSets,
    const ConvOptions &options, const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  auto plan =
      getMultiStageReducePlan(partialsDepth, options.enableMultiStageReduce);
  for (unsigned i = computeSets.size(); i <= plan.size(); ++i) {
    computeSets.push_back(
        graph.addComputeSet(debugPrefix + "/Reduce" + std::to_string(i)));
  }
  const auto partialsType = partials.elementType();
  logging::poplin::debug("  Multistage: {} stages", plan.size() + 1);
  for (unsigned i = 0; i != plan.size(); ++i) {
    logging::poplin::debug("  Stage {}:", i);
    std::string stepDebugPrefix = debugPrefix;
    if (plan.size() > 1)
      stepDebugPrefix += "/Stage" + std::to_string(i);
    partials = partialGroupedReduce(
        graph, tileGroups, tileGroupRegions, tileToRow,
        options.enableFastReduce, options.enableSingleInputReduce, partials,
        plan[i], partialsType, computeSets[i], stepDebugPrefix);
  }
  logging::poplin::debug("  Last stage:");
  auto reduced = groupedReduce(
      graph, tileGroups, tileGroupRegions, tileToRow, options.enableFastReduce,
      options.enableSingleInputReduce, partials, resultType,
      computeSets[plan.size()], debugPrefix);
  return reduced;
}

Tensor multiStageGroupedReduce(Graph &graph, Tensor partials,
                               const Type &resultType,
                               std::vector<ComputeSet> &computeSets,
                               const ConvOptions &options,
                               const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  logging::poplin::debug("Creating poplin::reduce vertices, debugStr: {}",
                         debugPrefix);
  std::map<unsigned, unsigned> tileToRow;
  // Build a map from the output to the set of tiles that contribute partial
  // sums.
  boost::icl::interval_map<unsigned, std::set<unsigned>> outputToTiles;
  for (unsigned i = 0; i != partialsDepth; ++i) {
    const auto tileMapping = graph.getTileMapping(partials[i]);
    for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
      // for each tile remember which row's data it contains.  Some of these
      // tiles will be used to reduce later, and the row will be used to
      // reference the tile local data
      tileToRow[tile] = i;
      for (const auto &interval : tileMapping[tile]) {
        outputToTiles +=
            std::make_pair(boost::icl::interval<unsigned>::right_open(
                               interval.begin(), interval.end()),
                           std::set<unsigned>({tile}));
      }
    }
  }

  // Build a map from sets of tiles the outputs they contribute to.
  std::map<std::set<unsigned>, std::vector<Interval>> tilesToOutputs;
  for (const auto &entry : outputToTiles) {
    tilesToOutputs[entry.second].emplace_back(entry.first.lower(),
                                              entry.first.upper());
  }
  std::vector<std::vector<unsigned>> tileGroups;
  std::vector<std::vector<Interval>> tileGroupRegions;
  tileGroups.reserve(tilesToOutputs.size());
  tileGroupRegions.reserve(tilesToOutputs.size());
  for (const auto &entry : tilesToOutputs) {
    tileGroups.emplace_back(entry.first.begin(), entry.first.end());
    tileGroupRegions.push_back(std::move(entry.second));
  }
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions, tileToRow,
                                 partials, resultType, computeSets, options,
                                 debugPrefix);
}

} // namespace poplin
