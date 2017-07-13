#include "popstd/Util.hpp"
#include "popstd/VertexTemplates.hpp"
#include "poprand/RandomGen.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Program.hpp"
#include "popstd/exceptions.hpp"
#include <cmath>
#include <cstdint>

using namespace popstd;
using namespace poplar;
using namespace poplar::program;

namespace poprand {

static uint64_t createSeed64(uint64_t callCount, uint64_t tile,
                           uint64_t vertexCount) {
  return ((callCount & ((1ULL << 16) - 1)) << 32)
         | ((tile & ((1ULL << 16) - 1)) << 48)
         | (vertexCount & ((1ULL << 32) - 1));
}

static void check(RandomGenMode mode) {
  if (mode == ALWAYS_REPEATABLE) {
    throw popstd::poplib_error("ALWAYS_REPEATABLE random generator mode "
                               "not supported");
  }
}

// Convert a range [minVal, maxVal] for uniform number generation into a
// scale and offset used internally by the uniform random number generator
static std::pair<float, float>
uniformScaleAndOffset(double minVal, double maxVal) {
  double scale = maxVal - minVal;
  double offset = scale /  2 + minVal;
  return std::make_pair(scale, offset);
}

void uniform(Graph &graph, Tensor &A, float minVal, float maxVal,
             uint64_t seed, RandomGenMode mode,
             Sequence &prog, const std::string &debugPrefix) {
  check(mode);

  // This generates randomness across calls
  static uint32_t callCount;
  float scale, offset;
  const auto dType = A.elementType();
  std::tie(scale, offset) = uniformScaleAndOffset(minVal, maxVal);

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/uniform");

  auto aFlat = A.flatten();
  ++callCount;

  const auto grainSize = deviceInfo.getVectorWidth(dType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    uint32_t vertexCount = 0;
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poprand::Uniform", dType),
                               {{"out", aFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["scale"], scale);
      graph.setInitialValue(v["offset"], offset);
      graph.setInitialValue(v["seedL"], seed);
      graph.setInitialValue(v["seedH"],
                            createSeed64(callCount, tile, vertexCount));
      graph.setTileMapping(v, tile);
      ++vertexCount;
    }
  }
  prog.add(Execute(cs));
}

void uniform(Graph &graph, Tensor &A, float minVal, float maxVal,
             Sequence &prog, const std::string &debugPrefix) {
  uniform(graph, A, minVal, maxVal, ~0, PLATFORM_REPEATABLE, prog, debugPrefix);
}


void bernoulli(Graph &graph, Tensor &A, float prob,
               uint64_t seed,  RandomGenMode mode,
               Sequence &prog, const std::string &debugPrefix) {
  check(mode);

  // This generates randomness across calls
  static uint32_t callCount;
  const auto dType = A.elementType();

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/bernoulli");

  auto aFlat = A.flatten();
  ++callCount;

  const auto grainSize = deviceInfo.getVectorWidth(dType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    uint32_t vertexCount = 0;
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poprand::Bernoulli", dType),
                               {{"out", aFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["prob"], prob);
      graph.setInitialValue(v["seedL"], seed);
      graph.setInitialValue(v["seedH"],
                            createSeed64(callCount, tile, vertexCount));
      graph.setTileMapping(v, tile);
      ++vertexCount;
    }
  }
  prog.add(Execute(cs));
}


void bernoulli(Graph &graph, Tensor &A, float prob,
               Sequence &prog, const std::string &debugPrefix) {
  bernoulli(graph, A, prob, ~0, PLATFORM_REPEATABLE, prog, debugPrefix);
}

void normal(Graph &graph, Tensor &A, float mean, float stdDev,
            uint64_t seed,  RandomGenMode mode,
            Sequence &prog, const std::string &debugPrefix) {
  check(mode);

  // This generates randomness across calls
  static uint32_t callCount;
  const auto dType = A.elementType();

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/normal");

  auto aFlat = A.flatten();
  ++callCount;

  const auto grainSize = deviceInfo.getVectorWidth(dType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    uint32_t vertexCount = 0;
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poprand::Normal", dType),
                               {{"out", aFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["mean"], mean);
      graph.setInitialValue(v["stdDev"], stdDev);
      graph.setInitialValue(v["seedL"], seed);
      graph.setInitialValue(v["seedH"],
                            createSeed64(callCount, tile, vertexCount));
      graph.setTileMapping(v, tile);
      ++vertexCount;
    }
  }
  prog.add(Execute(cs));
}

void normal(Graph &graph, Tensor &A, float mean, float stdDev,
            Sequence &prog, const std::string &debugPrefix) {
  normal(graph, A, mean, stdDev, ~0, PLATFORM_REPEATABLE, prog, debugPrefix);
}


void truncatedNormal(Graph &graph, Tensor &A, float mean, float stdDev,
                     float alpha, uint64_t seed, RandomGenMode mode,
                     Sequence &prog, const std::string &debugPrefix) {
  check(mode);

  // This generates randomness across calls
  static uint32_t callCount;
  const auto dType = A.elementType();

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/truncatedNormal");

  auto aFlat = A.flatten();
  ++callCount;

  // compute scale
  if (alpha < 1) {
    throw popstd::poplib_error("Alpha less than 1.0 not supported yet");
  }

  // select number of iterations such that probability that the number events
  // exceeding [+alpha, -alpha] is 10^-prob. Those events are then replaced
  // by uniform probability
  const float logProb = -4.0;
  const unsigned iterations =
    std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));


  const auto grainSize = deviceInfo.getVectorWidth(dType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    auto vertexRegions =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    uint32_t vertexCount = 0;
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poprand::TruncatedNormal",
                                              dType),
                               {{"out", aFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setInitialValue(v["mean"], mean);
      graph.setInitialValue(v["stdDev"], stdDev);
      graph.setInitialValue(v["alpha"], alpha);
      graph.setInitialValue(v["iterations"], iterations);
      graph.setInitialValue(v["seedL"], seed);
      graph.setInitialValue(v["seedH"],
                            createSeed64(callCount, tile, vertexCount));
      graph.setTileMapping(v, tile);
      ++vertexCount;
    }
  }
  prog.add(Execute(cs));
}

void truncatedNormal(Graph &graph, Tensor &a, float mean, float stdDev,
                     float alpha, Sequence &prog,
                     const std::string &debugPrefix) {
  truncatedNormal(graph, a, mean, stdDev, alpha, ~0, PLATFORM_REPEATABLE,
                  prog, debugPrefix);
}


} // namespace poprand
