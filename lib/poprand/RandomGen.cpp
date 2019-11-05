#include "poprand/RandomGen.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include "poplar/RandomSeed.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional.hpp>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

namespace poprand {

// flatten 2D vector of intervals to a 1D vector
static std::vector<Interval>
flatten(const std::vector<std::vector<Interval>> &intervals2D) {
  std::vector<Interval> flattenedIntervals;
  for (const auto &intervals1D : intervals2D) {
    flattenedIntervals.insert(flattenedIntervals.end(), std::begin(intervals1D),
                              std::end(intervals1D));
  }
  return flattenedIntervals;
}

static void seedTensorChecks(const Tensor *seed) {
  if (seed) {
    if (seed->rank() != 1) {
      // We could allow seed of any shape as long as it has the required number
      // of elements. For now, impose the stricter condition
      throw poputil::poplibs_error("seed tensor must have rank 1");
    }
    if (seed->numElements() != 2) {
      throw poputil::poplibs_error("seed tensor must have 2 elements");
    }
    if (seed->elementType() != poplar::UNSIGNED_INT) {
      throw poputil::poplibs_error("seed tensor must be of type UNSIGNED_INT");
    }
  }
}

// Convert a range [minVal, maxVal] for uniform number generation into a
// scale and offset used internally by the uniform random number generator
static std::pair<double, double>
uniformScaleAndOffset(double minVal, double maxVal, const Type &dType) {
  double scale = maxVal - minVal;
  if (dType != INT) {
    double offset = scale / 2 + minVal;
    return std::make_pair(scale, offset);
  } else {
    if (minVal < std::numeric_limits<int32_t>::min() ||
        maxVal > static_cast<double>(std::numeric_limits<int32_t>::max())) {
      throw poputil::poplibs_error("range for uniform distribution invalid");
    }
    scale += 1.0;
    if (scale ==
        static_cast<double>(std::numeric_limits<uint32_t>::max()) + 1) {
      scale = 0;
    }
    return std::make_pair(scale, minVal);
  }
}

// If master seed tensor is not null then read hw seeds tensor and
// program master seed
// TODO: T12982 To avoid creating vertex state for each worker within the random
// generator codelets we add the getHwSeeds and setSeed program followed by the
// setHwSeeds program. This is not efficient in both cycles and memory but
// is an expedient solution. We can revisit this if memory and performance
// becomes an issue.
static boost::optional<Tensor>
maybeSaveHwSeedsAndSetSeeds(Graph &graph, const Tensor *masterSeed,
                            uint32_t seedModifier, Sequence &prog,
                            const std::string &debugPrefix) {
  if (masterSeed) {
    auto hwSeeds = getHwSeeds(graph, prog, debugPrefix);
    setSeed(graph, *masterSeed, seedModifier, prog, debugPrefix);
    return hwSeeds;
  }
  return boost::none;
}

// Restore Hw seeds
static void maybeRestoreHwSeeds(Graph &graph,
                                const boost::optional<Tensor> &hwSeeds,
                                Sequence &prog,
                                const std::string &debugPrefix) {
  if (hwSeeds != boost::none) {
    setHwSeeds(graph, *hwSeeds, prog, debugPrefix);
  }
}

Tensor uniform(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
               const Tensor &reference, const Type &outType, double minVal,
               double maxVal, Sequence &prog, const std::string &debugPrefix) {
  seedTensorChecks(masterSeed);
  auto fnPrefix = debugPrefix + "/uniform";
  auto out = graph.clone(outType, reference, fnPrefix + "/out");

  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, fnPrefix);
  auto cs = graph.addComputeSet(fnPrefix);
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {});
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  double scale, offset;
  std::tie(scale, offset) = uniformScaleAndOffset(minVal, maxVal, outType);

  unsigned int shift = 31;
  if (outType == INT) {
    unsigned tmpScale = (scale < 1.0) ? 1.0 : scale;
    shift = 31 - std::log2(tmpScale);
    int shiftR = (shift < 24) ? (24 - shift) : 0;
    int shiftL = (shift > 24) ? (shift - 24) : 0;

    tmpScale = scale;
    tmpScale += (1 << shiftR) - 1;
    tmpScale >>= shiftR;
    tmpScale <<= shiftL;
    scale = (tmpScale < 255) ? tmpScale : 255;
  }

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);

    const auto vertexTemplate =
        templateVertex("poprand::UniformSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["scale"], scale);
    graph.setInitialValue(v["offset"], offset);
    graph.setInitialValue(v["shift"], shift);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, fnPrefix);
  return out;
}

Tensor bernoulli(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
                 const Tensor &reference, const Type &outType, double prob,
                 Sequence &prog, const std::string &debugPrefix) {
  seedTensorChecks(masterSeed);

  auto fnPrefix = debugPrefix + "/bernoulli";
  auto out = graph.clone(outType, reference, fnPrefix + "/out");
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, fnPrefix);

  auto cs = graph.addComputeSet(fnPrefix);
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {});
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::BernoulliSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(prob * 65536.0));
    graph.setTileMapping(v, tile);
  }

  prog.add(Execute(cs));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, fnPrefix);
  return out;
}

Tensor normal(Graph &graph, const Tensor *masterSeed, uint32_t seedModifier,
              const Tensor &reference, const Type &outType, double mean,
              double stdDev, Sequence &prog, const std::string &debugPrefix) {
  seedTensorChecks(masterSeed);
  auto fnPrefix = debugPrefix + "/normal";
  auto out = graph.clone(outType, reference, fnPrefix + "/out");
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, fnPrefix);

  auto cs = graph.addComputeSet(fnPrefix);
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {});
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::NormalSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, fnPrefix);
  return out;
}

Tensor truncatedNormal(Graph &graph, const Tensor *masterSeed,
                       uint32_t seedModifier, const Tensor &reference,
                       const Type &outType, double mean, double stdDev,
                       double alpha, Sequence &prog,
                       const std::string &debugPrefix) {
  seedTensorChecks(masterSeed);
  auto fnPrefix = debugPrefix + "/truncatedNormal";
  auto out = graph.clone(outType, reference, fnPrefix + "/out");
  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, fnPrefix);
  auto cs = graph.addComputeSet(fnPrefix);
  auto outFlat = out.flatten();
  graph.reorderToSimplify(&outFlat, {});
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  const float logProb = -4.0;
  const unsigned iterations =
      std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::TruncatedNormalSupervisor", outType);
    auto v = graph.addVertex(cs, vertexTemplate,
                             {{"out", concat(outFlat.slices(intervals))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setInitialValue(v["alpha"], alpha);
    graph.setInitialValue(v["iterations"], iterations);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, fnPrefix);
  return out;
}

Tensor dropout(Graph &graph, const Tensor *masterSeed,
               const uint32_t seedModifier, const Tensor &in,
               const Tensor &reference, double dropoutProbability, double scale,
               Sequence &prog, const std::string &debugPrefix) {
  seedTensorChecks(masterSeed);
  auto fnPrefix = debugPrefix + "/dropout";
  if (in.shape() != reference.shape()) {
    throw poputil::poplibs_error("Input and reference shapes must match in "
                                 "dropout");
  }

  auto hwSeeds = maybeSaveHwSeedsAndSetSeeds(graph, masterSeed, seedModifier,
                                             prog, fnPrefix);
  auto out = graph.clone(in.elementType(), reference, fnPrefix + "/out");

  auto cs = graph.addComputeSet(fnPrefix);
  auto outFlat = out.flatten();
  auto inFlat = in.flatten();
  graph.reorderToSimplify(&inFlat, {&outFlat});

  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap = outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    const auto intervals = flatten(tileContiguousRegions);
    const auto vertexTemplate =
        templateVertex("poprand::DropoutSupervisor", in.elementType());
    auto inTile = concat(inFlat.slices(intervals));
    const auto &target = graph.getTarget();
    if (inTile.numElements() > target.getRptCountMax() *
                                   target.getNumWorkerContexts() *
                                   (inTile.elementType() == FLOAT ? 2 : 4)) {
      throw poputil::poplibs_error("Elements on tile exceed number that can "
                                   "be processed by codelet");
    }
    auto v = graph.addVertex(
        cs, vertexTemplate,
        {{"in", inTile}, {"out", concat(outFlat.slices(intervals))}});
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(dropoutProbability * 65536.0));
    graph.setInitialValue(v["numElems"], inTile.numElements());
    graph.setInitialValue(v["scale"], scale);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  maybeRestoreHwSeeds(graph, hwSeeds, prog, fnPrefix);
  return out;
}

void setSeed(poplar::Graph &graph, const poplar::Tensor &masterSeed,
             uint32_t seedModifier, poplar::program::Sequence &prog,
             const std::string &debugPrefix) {
  seedTensorChecks(&masterSeed);
  auto cs = graph.addComputeSet(debugPrefix + "/setMasterSeed");
  const auto &target = graph.getTarget();
  auto numTiles = target.getNumTiles();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto v = graph.addVertex(cs, "poprand::SetSeedSupervisor",
                             {{"seed", masterSeed}});
    graph.setInitialValue(v["seedModifierUser"], seedModifier ^ 0x55555555U);
    // guarantee that even tile id 0 will have at least one bit set
    graph.setInitialValue(v["seedModifierHw"], (tile << 4) ^ 0xAAAAAAA0U);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
}

} // namespace poprand
