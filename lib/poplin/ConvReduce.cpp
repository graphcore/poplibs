#include "ConvReduce.hpp"

#include <cassert>
#include <map>
#include <set>
#include <utility>

#include <boost/icl/interval_map.hpp>

#include <poplar/Tensor.hpp>
#include <poplar/Graph.hpp>

#include <popops/Cast.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/VertexTemplates.hpp>

#include "poplin/ConvUtil.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace poplin {

static void reduce(Graph &graph,
                   Tensor partials,
                   Tensor reduced,
                   const std::vector<
                     std::vector<Interval>
                   > &reduceVertexMapping,
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
  const auto reducedType = reduced.elementType();
  const auto tilesPerInZGroup = partials.dim(0);
  if (tilesPerInZGroup > target.getRptCountMax()) {
    // if this error is hit then reduction needs to be split into smaller
    // reductions first reduce then the output of each are reduced against
    // each other
    poplibs_error("Reduction to large for counter");
  }
  auto flatPartials =
      partials.reshape({tilesPerInZGroup,
                        partials.numElements() / tilesPerInZGroup});
  auto flatReduced = reduced.flatten();
  std::string vertexName = "poplin::ReduceAdd";

  // Accumulate the partial sums.
  const auto numUsedTiles =  reduceVertexMapping.size();
  assert(numUsedTiles <= target.getNumTiles());
  for (unsigned tile = 0; tile != numUsedTiles; ++tile) {
    const auto &tileRegions = reduceVertexMapping[tile];

    if (tileRegions.empty())
      continue;
    // concatenate inner dimension so that vertex has a single contiguous
    // region to divide work across. Partials have to brought on-tile for each
    // input partial except for one that may reside on-tile.
    auto concatFlatReduced = concat(flatReduced.slices(tileRegions));
    auto concatFlatPartials = concat(flatPartials.slices(tileRegions, 1), 1);
    const auto v =
        graph.addVertex(reduceCS,
                        templateVertex(vertexName, reducedType, partialType));






    graph.setInitialValue(v["numPartials"], tilesPerInZGroup);
    graph.setInitialValue(v["numElems"], concatFlatReduced.numElements());
    graph.connect(v["out"], concatFlatReduced);
    graph.connect(v["partials"], concatFlatPartials);
    graph.setTileMapping(v, tile);
  }
}

static Tensor
partialGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &
        tileGroupRegions,
    const Tensor &partials,
    unsigned outDepth,
    const Type &resultType,
    ComputeSet cs,
    const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out = graph.addVariable(resultType,
                                 outDims,
                                 debugPrefix + "/partialReduceOut");
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const unsigned minGrainSize = target.getVectorWidth(resultType);
  const unsigned partialChansPerGroup = partials.dim(partials.rank()-1);
  const auto grainSize = std::max(partialChansPerGroup, minGrainSize);
  const auto roundedGrainSize = (grainSize + minGrainSize - 1) / minGrainSize
                                * minGrainSize;

  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval>>
        outSubMapping(numTiles);
    for (unsigned tileGroup = 0; tileGroup != numTileGroups; ++tileGroup) {
      const auto tilesInGroup = tileGroups[tileGroup].size();
      const auto tileBegin = (i * tilesInGroup) / outDepth;
      const auto tileEnd = std::max(tileBegin + 1,
                                    ((i + 1) * tilesInGroup) / outDepth);
      const auto outSplitRegions =
          splitRegions(tileGroupRegions[tileGroup], roundedGrainSize,
                       tileEnd - tileBegin);
      for (unsigned j = 0; j != outSplitRegions.size(); ++j) {
        const auto tileIndex = tileGroups[tileGroup][j + tileBegin];
        outSubMapping[tileIndex].insert(outSubMapping[tileIndex].end(),
                                        outSplitRegions[j].begin(),
                                        outSplitRegions[j].end());
      }
    }
    graph.setTileMapping(out[i], outSubMapping);

    reduce(graph, partials.slice(begin, end), out[i], outSubMapping, cs);
  }
  return out;
}

static Tensor
groupedReduce(Graph &graph,
              const std::vector<std::vector<unsigned>> &tileGroups,
              const std::vector<
                std::vector<Interval>
              > &tileGroupRegions,
              const Tensor &partials,
              const Type &resultType,
              ComputeSet cs,
              const std::string &debugPrefix) {
  return partialGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
         1, resultType, cs, debugPrefix).reshape(partials[0].shape());
}

/// Return the number of reduce stages to use for a reduction of the specified
/// reduction depth.
static unsigned getNumReduceStages(unsigned partialsDepth) {
  /// Using more reduce stages affects code size as follows.
  /// If the reduction depth is p then a single stage reduction requires each
  /// tile to receive p messages. If instead we break the reduction down into n
  /// stages then each stage involves a reduction of reduce p^(1/n) messages.
  /// The total number of messages is n*p^(1/n). For large p, increase n
  /// will reducing the total number of messages received which is turn likely
  /// to also reduce the exchange code size. The thresholds below have been
  /// chosen based on benchmarking.
  if (partialsDepth >= 125)
    return 3;
  if (partialsDepth >= 16)
    return 2;
  return 1;
}

/// Return a plan for how to split a reduction into multiple stages along with
/// an estimate of the cost of the plan. The first member of the pair is a
/// vector of the depth of each partials tensor in each intermediate stage.
/// If the vector is empty there are no intermediate stages and the reduction
/// is performed in a single step. The second member of the pair is an
/// estimated cost. The cost is an estimate of the average number of messages
/// required per tile.
static std::pair<std::vector<unsigned>, float>
getMultiStageReducePlanAndCost(unsigned partialsDepth, unsigned numStages) {
  if (numStages == 1) {
    return {{}, partialsDepth};
  }
  auto nextDepthRoundDown =
      static_cast<unsigned>(
        std::pow(static_cast<double>(partialsDepth),
                 (numStages - 1.0) / numStages)
      );
  std::vector<unsigned> roundDownPlan, roundUpPlan;
  float roundDownCost, roundUpCost;
  std::tie(roundDownPlan, roundDownCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundDown, numStages - 1);
  roundDownCost += static_cast<float>(partialsDepth) / nextDepthRoundDown;
  auto nextDepthRoundUp = nextDepthRoundDown + 1;
  std::tie(roundUpPlan, roundUpCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundUp, numStages - 1);
  roundUpCost += static_cast<float>(partialsDepth) / nextDepthRoundUp;
  if (roundDownCost < roundUpCost) {
    roundDownPlan.insert(roundDownPlan.begin(), nextDepthRoundDown);
    return {roundDownPlan, roundDownCost};
  }
  roundUpPlan.insert(roundUpPlan.begin(), nextDepthRoundUp);
  return {roundUpPlan, roundUpCost};
}

static std::vector<unsigned>
getMultiStageReducePlan(unsigned partialsDepth) {
  const auto numStages = getNumReduceStages(partialsDepth);
  return getMultiStageReducePlanAndCost(partialsDepth, numStages).first;
}

static Tensor
multiStageGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval>> &
        tileGroupRegions,
    Tensor partials,
    const Type &resultType,
    std::vector<ComputeSet> &computeSets,
    const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  auto plan = getMultiStageReducePlan(partialsDepth);
  for (unsigned i = computeSets.size(); i <= plan.size(); ++i) {
    computeSets.push_back(
      graph.addComputeSet(debugPrefix + "/Reduce" +
                             std::to_string(i))
    );
  }
  const auto partialsType = partials.elementType();
  for (unsigned i = 0; i != plan.size(); ++i) {
    std::string stepDebugPrefix = debugPrefix;
    if (plan.size() > 1)
      stepDebugPrefix += "/Stage" + std::to_string(i);
    partials = partialGroupedReduce(graph, tileGroups, tileGroupRegions,
                                    partials, plan[i], partialsType,
                                    computeSets[i], stepDebugPrefix);
  }
  auto reduced = groupedReduce(graph, tileGroups, tileGroupRegions, partials,
                               resultType, computeSets[plan.size()],
                               debugPrefix);
  return reduced;
}

Tensor
multiStageGroupedReduce(Graph &graph,
                        Tensor partials,
                        const Type &resultType,
                        std::vector<ComputeSet> &computeSets,
                        const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  // Build a map from the output to the set of tiles that contribute partial
  // sums.
  boost::icl::interval_map<unsigned, std::set<unsigned>> outputToTiles;
  for (unsigned i = 0; i != partialsDepth; ++i) {
    const auto tileMapping = graph.getTileMapping(partials[i]);
    for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
      for (const auto &interval : tileMapping[tile]) {
        outputToTiles += std::make_pair(
                           boost::icl::interval<unsigned>::right_open(
                             interval.begin(),
                             interval.end()),
                           std::set<unsigned>({tile}));
      }
    }
  }
  // Build a map from sets of tiles the outputs they contribute to.
  std::map<std::set<unsigned>, std::vector<Interval>>
      tilesToOutputs;
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
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
                                 resultType, computeSets, debugPrefix);
}

}
