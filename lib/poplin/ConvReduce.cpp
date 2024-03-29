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
       bool enableFastReduce, unsigned thisStageRows, const Tensor &partials,
       const Tensor &reduced,
       const std::vector<std::vector<Interval>> &reduceVertexMapping,
       ComputeSet reduceCS, const DebugNameAndId &dnai) {
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
    bool singleInputReducePartialsSize = checkPartialsSizeForSingleInputReduce(
        exchangedPartialsBytes, memoryElementOffsets);
    bool useSingleInputReduce =
        singleInputReduceIsPossible &&
        (enableFastReduce || singleInputReducePartialsSize);
    bool useFastReduce = singleInputReduceIsPossible && enableFastReduce;

    const auto v = graph.addVertex(
        reduceCS, templateVertex(vertexName, reducedType, partialType,
                                 useSingleInputReduce, useFastReduce));

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

static void partialGroupedReduce(
    Graph &graph, boost::optional<Tensor> &output,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &tileGroupRegions,
    std::map<unsigned, unsigned> tileToRow, bool enableFastReduce,
    const Tensor &partials, unsigned outDepth, const Type &resultType,
    unsigned startTile, bool ascendingMapping, boost::optional<ComputeSet &> cs,
    const DebugNameAndId &dnai) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;

  Tensor out =
      output != boost::none
          ? output.get()
          : graph.addVariable(resultType, outDims, {dnai, "partialReduceOut"});

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
        const auto tileIndex =
            invTransformTileIndex(tileGroups[tileGroup][j + tileBegin],
                                  numTiles, startTile, ascendingMapping);
        outSubMapping[tileIndex].insert(outSubMapping[tileIndex].end(),
                                        outSplitRegions[j].begin(),
                                        outSplitRegions[j].end());
      }
    }
    if (!output) {
      graph.setTileMapping(out[i], outSubMapping);
    }
    logging::poplin::debug("    Reduction section with Depth {}, Height {}.",
                           end - begin, out[i].numElements());
    if (cs) {
      reduce(graph, tileToRow, enableFastReduce, end - begin,
             partials.slice(begin, end), out[i], outSubMapping, *cs, {dnai});
    }
  }
  output = out;
}

static void
groupedReduce(Graph &graph, boost::optional<Tensor> &output,
              const std::vector<std::vector<unsigned>> &tileGroups,
              const std::vector<std::vector<Interval>> &tileGroupRegions,
              std::map<unsigned, unsigned> tileToRow, bool enableFastReduce,
              const Tensor &partials, const Type &resultType,
              unsigned startTile, bool ascendingMapping,
              boost::optional<ComputeSet &> cs, const DebugNameAndId &dnai) {
  bool outputProvided = output != boost::none;
  if (outputProvided) {
    // Insert an additional dimension to reduce over, assumed to be present
    // when passed to partialGroupedReduce
    output = output.get().expand({0});
  }
  partialGroupedReduce(graph, output, tileGroups, tileGroupRegions, tileToRow,
                       enableFastReduce, partials, 1, resultType, startTile,
                       ascendingMapping, cs, dnai);
  if (!outputProvided) {
    // Having reduced, return the output to the required shape
    output = output.get().reshape(partials[0].shape());
  }
}

static void multiStageGroupedReduce(
    Graph &graph, boost::optional<Tensor> &reduced,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &tileGroupRegions,
    std::map<unsigned, unsigned> tileToRow, const Tensor &partials,
    const Type &resultType,
    boost::optional<std::vector<ComputeSet> &> computeSets,
    const ConvOptions &options, unsigned startTile, bool ascendingMapping,
    const poplar::DebugNameAndId &dnai) {
  const auto partialsDepth = partials.dim(0);
  auto plan =
      getMultiStageReducePlan(partialsDepth, options.enableMultiStageReduce);
  if (computeSets) {
    for (unsigned i = computeSets.get().size(); i <= plan.size(); ++i) {
      computeSets.get().push_back(
          graph.addComputeSet({dnai, "Reduce" + std::to_string(i)}));
    }
  }
  const auto partialsType = partials.elementType();
  logging::poplin::debug("  Multistage: {} stages", plan.size() + 1);
  boost::optional<Tensor> partialsToReduce = partials;
  for (unsigned i = 0; i != plan.size(); ++i) {
    logging::poplin::debug("  Stage {}:", i);
    std::string stepDebugPrefix = "";
    if (plan.size() > 1)
      stepDebugPrefix += "Stage" + std::to_string(i);
    boost::optional<ComputeSet &> cs;
    if (computeSets) {
      cs = computeSets.get()[i];
    }
    // Provide an output variable for the partialGroupedReduce function,
    // which is an optional as in other cases an output tensor would be
    // provided. partialsToReduce is the output from the last planned stage
    // (pass of the loop)
    boost::optional<Tensor> partialsOut;
    partialGroupedReduce(
        graph, partialsOut, tileGroups, tileGroupRegions, tileToRow,
        options.enableFastReduce, partialsToReduce.get(), plan[i], partialsType,
        startTile, ascendingMapping, cs, {dnai, stepDebugPrefix});
    partialsToReduce = partialsOut;
  }
  logging::poplin::debug("  Last stage:");
  boost::optional<ComputeSet &> cs;
  if (computeSets) {
    cs = computeSets.get()[plan.size()];
  }
  groupedReduce(graph, reduced, tileGroups, tileGroupRegions, tileToRow,
                options.enableFastReduce, partialsToReduce.get(), resultType,
                startTile, ascendingMapping, cs, {dnai});
}

static void multiStageGroupedReduceInternal(
    Graph &graph, boost::optional<Tensor> &output, const Tensor &partials,
    const Type &resultType,
    boost::optional<std::vector<ComputeSet> &> computeSets,
    const ConvOptions &options, unsigned startTile, bool ascendingMapping,
    const DebugNameAndId &dnai) {
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto partialsDepth = partials.dim(0);

  std::map<unsigned, unsigned> tileToRow;
  // Build a map from the output to the set of tiles that contribute partial
  // sums.
  boost::icl::interval_map<unsigned, std::set<unsigned>> outputToTiles;
  for (unsigned i = 0; i != partialsDepth; ++i) {
    const auto tileMapping = graph.getTileMapping(partials[i]);
    for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
      if (tileMapping[tile].empty()) {
        continue;
      }
      // for each tile remember which row's data it contains.  Some of these
      // tiles will be used to reduce later, and the row will be used to
      // reference the tile local data
      tileToRow[tile] = i;
      for (const auto &interval : tileMapping[tile]) {
        unsigned tileRef =
            transformTileIndex(tile, numTiles, startTile, ascendingMapping);
        outputToTiles +=
            std::make_pair(boost::icl::interval<unsigned>::right_open(
                               interval.begin(), interval.end()),
                           std::set<unsigned>({tileRef}));
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
  multiStageGroupedReduce(graph, output, tileGroups, tileGroupRegions,
                          tileToRow, partials, resultType, computeSets, options,
                          startTile, ascendingMapping, {dnai});
}

Tensor multiStageGroupedReduce(Graph &graph, const Tensor &partials,
                               const Type &resultType,
                               std::vector<ComputeSet> &computeSets,
                               const ConvOptions &options, unsigned startTile,
                               bool ascendingMapping,
                               const DebugNameAndId &dnai) {
  logging::poplin::debug("multiStageGroupedReduce: "
                         "Creating poplin::reduce vertices, debugStr: {}",
                         dnai.getPathName());
  boost::optional<Tensor> optOut;
  multiStageGroupedReduceInternal(graph, optOut, partials, resultType,
                                  computeSets, options, startTile,
                                  ascendingMapping, dnai);
  return optOut.get();
}

Tensor createMultiStageGroupedReduceOutput(Graph &graph, const Tensor &partials,
                                           const Type &resultType,
                                           const ConvOptions &options,
                                           unsigned startTile,
                                           bool ascendingMapping,
                                           const DebugNameAndId &dnai) {
  logging::poplin::debug("CreateMultiStageGroupedReduceOutput: "
                         "Creating poplin::reduce output, debugStr: {}",
                         dnai.getPathName());
  boost::optional<std::vector<ComputeSet> &> optionalComputeSets;
  boost::optional<Tensor> optOut;
  multiStageGroupedReduceInternal(graph, optOut, partials, resultType,
                                  optionalComputeSets, options, startTile,
                                  ascendingMapping, dnai);
  return optOut.get();
}

void multiStageGroupedReduceWithOutput(
    Graph &graph, const Tensor &output, const Tensor &partials,
    const Type &resultType, std::vector<ComputeSet> &computeSets,
    const ConvOptions &options, unsigned startTile, bool ascendingMapping,
    const DebugNameAndId &dnai) {
  logging::poplin::debug("multiStageGroupedReduceWithOutput: Creating "
                         "poplin::reduce vertices, debugStr: {}",
                         dnai.getPathName());
  boost::optional<Tensor> out = output;
  multiStageGroupedReduceInternal(graph, out, partials, resultType, computeSets,
                                  options, startTile, ascendingMapping, dnai);
}

} // namespace poplin
