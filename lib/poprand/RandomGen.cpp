#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poprand/RandomGen.hpp"
#include "poplar/Graph.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Program.hpp"
#include "poplar/exceptions.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/TileMapping.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

using namespace poputil;
using namespace poplar;
using namespace poplar::program;

namespace poprand {

enum Module {
  UNIFORM,
  BERNOULLI,
  NORMAL,
  TRUNCATED_NORMAL,
  NUM_MODULES
};

// Module IDs for generators
#define UNIFORM_MODULEID     0xAAAA
#define BERNOULLI_MODULEID   0x5555
#define NORMAL_MODULEID      0x3333
#define TRUNCNORMAL_MODULEID 0xCCCC

struct VertexInfo {
  poplar::Interval region;
};

using ParamsList = std::vector<std::pair<const std::string, double>>;

// Convert a range [minVal, maxVal] for uniform number generation into a
// scale and offset used internally by the uniform random number generator
static std::pair<double, double>
uniformScaleAndOffset(double minVal, double maxVal, const Type &dType) {
  double scale = maxVal - minVal;
  if (dType != INT) {
    double offset = scale /  2 + minVal;
    return std::make_pair(scale, offset);
  } else {
    if (minVal < std::numeric_limits<int32_t>::min() ||
        maxVal > static_cast<double>(std::numeric_limits<int32_t>::max())) {
      throw poputil::poplibs_error("range for uniform distribution invalid");
    }
    scale += 1.0;
    if (scale == static_cast<double>(std::numeric_limits<uint32_t>::max())
        + 1) {
      scale = 0;
    }
    return std::make_pair(scale, minVal);
  }
}

Random::Random() {
  callCount.resize(NUM_MODULES, 0);
}

Random::Random(RandomGenMode mode_) : mode(mode_) {
  callCount.resize(NUM_MODULES, 0);
}

Random::Random(RandomGenMode mode_, uint64_t seed_) : mode(mode_), seed(seed_) {
  callCount.resize(NUM_MODULES, 0);
}

Tensor
uniform(Graph &graph,
        const Tensor &reference,
        Type  inType,
        double minVal,
        double maxVal,
        Sequence &prog,
        const std::string &debugPrefix) {
  auto out = graph.clone(reference, debugPrefix + "/uniform/out");

  graph.setTileMapping(out, graph.getTileMapping(reference));

  auto cs = graph.addComputeSet(debugPrefix + "/uniform");
  auto outFlat = out.flatten();
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  double scale, offset;
  std::tie(scale, offset) = uniformScaleAndOffset(minVal, maxVal,
                                                  inType);

  unsigned int shift = 31;
  if (inType == INT) {
    unsigned tmpScale = (scale < 1.0) ? 1.0 : scale;
    shift = 31 - std::log2(tmpScale);
    int shiftR = (shift < 24) ? (24 - shift) : 0;
    int shiftL = (shift > 24) ? (shift - 24) : 0;

    tmpScale   = scale;
    tmpScale  += (1 << shiftR) - 1;
    tmpScale >>= shiftR;
    tmpScale <<= shiftL;
    scale      = (tmpScale < 255) ? tmpScale : 255;
  }

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap =  outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto vertexTemplate =
      templateVertex("poprand::UniformSupervisor", inType);
    auto v =
      graph.addVertex(cs, vertexTemplate,
                      {{"out", concat(outFlat.slices(thisTileMap))}});
    graph.setInitialValue(v["scale"], scale);
    graph.setInitialValue(v["offset"], offset);
    graph.setInitialValue(v["shift"], shift);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return out;
}

Tensor
bernoulli(Graph &graph,
          const Tensor &reference,
          Type  inType,
          double prob,
          Sequence &prog,
          const std::string &debugPrefix) {
  auto out = graph.clone(reference, debugPrefix + "/bernoulli/out");

  graph.setTileMapping(out, graph.getTileMapping(reference));

  auto cs = graph.addComputeSet(debugPrefix + "/bernoulli");
  auto outFlat = out.flatten();
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap =  outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto vertexTemplate =
      templateVertex("poprand::BernoulliSupervisor", inType);
    auto v =
      graph.addVertex(cs, vertexTemplate,
                      {{"out", concat(outFlat.slices(thisTileMap))}});
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(prob * 65536.0));
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return out;
}

Tensor
normal(Graph &graph,
       const Tensor &reference,
       Type  inType,
       double mean,
       double stdDev,
       Sequence &prog,
       const std::string &debugPrefix) {
  auto out = graph.clone(reference, debugPrefix + "/normal/out");

  graph.setTileMapping(out, graph.getTileMapping(reference));

  auto cs = graph.addComputeSet(debugPrefix + "/normal");
  auto outFlat = out.flatten();
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap =  outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto vertexTemplate =
      templateVertex("poprand::NormalSupervisor", inType);
    auto v =
      graph.addVertex(cs, vertexTemplate,
                      {{"out", concat(outFlat.slices(thisTileMap))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return out;
}

Tensor
truncatedNormal(Graph &graph,
                const Tensor &reference,
                Type  inType,
                double mean,
                double stdDev,
                double alpha,
                Sequence &prog,
                const std::string &debugPrefix) {
  auto out = graph.clone(reference, debugPrefix + "/truncatedNormal/out");

  graph.setTileMapping(out, graph.getTileMapping(reference));

  auto cs = graph.addComputeSet(debugPrefix + "/truncatedNormal");
  auto outFlat = out.flatten();
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  const float logProb = -4.0;
  const unsigned iterations =
    std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap =  outFlatTileMap[tile];
    if (thisTileMap.empty())
      continue;
    const auto vertexTemplate =
      templateVertex("poprand::TruncatedNormalSupervisor", inType);
    auto v =
      graph.addVertex(cs, vertexTemplate,
                      {{"out", concat(outFlat.slices(thisTileMap))}});
    graph.setInitialValue(v["mean"], mean);
    graph.setInitialValue(v["stdDev"], stdDev);
    graph.setInitialValue(v["alpha"], alpha);
    graph.setInitialValue(v["iterations"], iterations);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return out;
}

Tensor
dropout(Graph &graph,
        const Tensor &in,
        const Tensor &reference,
        double dropoutProbability,
        double scale,
        Sequence &prog,
        const std::string &debugPrefix) {
  auto out = graph.clone(reference, debugPrefix + "/dropout/out");

  if (in.shape() != reference.shape()) {
    throw poputil::poplibs_error("Input and reference shapes must match in "
                                 "dropout");
  }

  graph.setTileMapping(out, graph.getTileMapping(reference));

  auto cs = graph.addComputeSet(debugPrefix + "/dropout");
  auto outFlat = out.flatten();
  auto inFlat = in.flatten();
  const auto outFlatTileMap = graph.getTileMapping(outFlat);

  for (auto tile = 0U; tile != outFlatTileMap.size(); ++tile) {
    const auto thisTileMap =  outFlatTileMap[tile];
    if (thisTileMap.empty()) continue;
    const auto vertexTemplate =
      templateVertex("poprand::DropoutSupervisor", in.elementType());
    auto inTile = concat(inFlat.slices(thisTileMap));
    auto v =
      graph.addVertex(cs, vertexTemplate,
                      { { "in", inTile },
                        { "out", concat(outFlat.slices(thisTileMap)) } });
    // The probability used by f16v4rmask/f32v2rmask is the bottom 17-bits of
    // the 2nd input operand. Hence the scaling by 2^16.
    graph.setInitialValue(v["prob"], (unsigned)(dropoutProbability * 65536.0));
    graph.setInitialValue(v["scale"], scale);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return out;
}

void Random::
uniform(Graph &graph, Tensor &A, double minVal, double maxVal,
        Sequence &prog, const std::string &debugPrefix) {
  auto B = poprand::uniform(graph, A, A.elementType(), minVal, maxVal, prog,
                            debugPrefix);
  prog.add(Copy(B, A));
}

void Random::
bernoulli(Graph &graph, Tensor &A, double prob, Sequence &prog,
          const std::string &debugPrefix) {
  auto B = poprand::bernoulli(graph, A, A.elementType(), prob, prog,
                              debugPrefix);
  prog.add(Copy(B, A));
}

void Random::
normal(Graph &graph, Tensor &A, double mean, double stdDev, Sequence &prog,
       const std::string &debugPrefix) {
  auto B = poprand::normal(graph, A, A.elementType(), mean, stdDev, prog,
                           debugPrefix);
  prog.add(Copy(B, A));
}

void Random::
truncatedNormal(Graph &graph, Tensor &A, double mean, double stdDev,
                double alpha, Sequence &prog, const std::string &debugPrefix) {
  auto B = poprand::truncatedNormal(graph, A, A.elementType(), mean, stdDev,
                                    alpha, prog, debugPrefix);
  prog.add(Copy(B, A));
}

void Random::
dropout(Graph &graph, Tensor &A, double dropoutProbability, double scale,
        Sequence &prog, const std::string &debugPrefix) {
  auto B = poprand::dropout(graph, A, A, dropoutProbability, scale, prog,
                            debugPrefix);
  prog.add(Copy(B, A));
}

void setSeed(poplar::Graph &graph,
             const poplar::Tensor &masterSeed,
             uint32_t seedModifier,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix) {
  if (masterSeed.rank() != 1) {
    throw poputil::poplibs_error(
            "Master seed tensor must be of rank one");
  }
  if (masterSeed.elementType()  != poplar::UNSIGNED_INT) {
    throw poputil::poplibs_error(
            "Master seed tensor must be of type UNSIGNED_INT");
  }
  if (masterSeed.numElements() != 2) {
      throw poputil::poplibs_error(
              "Master seed tensor must have two elements of type UNSIGNED_INT");
  }

  auto cs = graph.addComputeSet(debugPrefix + "/setSeed");
  const auto &target = graph.getTarget();
  auto numTiles = target.getTilesPerIPU();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto v = graph.addVertex(cs,
                             "poprand::SetSeedSupervisor",
                             { { "seed", masterSeed } });
    graph.setInitialValue(v["seedModifierUser"], seedModifier ^ 0x55555555U);
    // guarantee that even tile id 0 will have at least one bit set
    graph.setInitialValue(v["seedModifierHw"], (tile << 4) ^ 0xAAAAAAA0U);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
}


poplar::Tensor getHwSeeds(poplar::Graph &graph,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix) {
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto numWorkerContexts = graph.getTarget().getNumWorkerContexts();

  auto seeds =
      graph.addVariable(poplar::UNSIGNED_INT, {numTiles, numWorkerContexts, 4},
                        debugPrefix + "getSeeds/seeds");
  auto cs = graph.addComputeSet(debugPrefix + "/getSeeds");

  for (auto tile = 0U; tile != numTiles; ++tile) {
    auto seedsThisTile = seeds[tile].flatten();
    auto v = graph.addVertex(cs,
                             "poprand::GetSeedsSupervisor",
                             { { "seeds", seedsThisTile } });
    graph.setTileMapping(seedsThisTile, tile);
    graph.setTileMapping(v, tile);
  }
  prog.add(Execute(cs));
  return seeds;
}



} // namespace poprand
