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

static uint64_t createSeedU64(uint64_t callCount, uint64_t tile,
                              uint64_t vertexCount) {
  return ((callCount & ((1ULL << 16) - 1)) << 32)
         | ((tile & ((1ULL << 16) - 1)) << 48)
         | (vertexCount & ((1ULL << 32) - 1));
}

// If seed is not provided, create one for the lower 64 bits. A more
// complicated generation could be used but for now use a simple one.
static uint64_t colourSeedL64(uint64_t seed, uint16_t moduleId) {
  return seed ^ moduleId;
}

// colour part of seedU64
static uint16_t colouredIdU64(uint16_t id, uint16_t callCount,
                              RandomGenMode mode) {
  id += callCount + mode;
  return id;
}

struct VertexInfo {
  poplar::Interval<std::size_t> region;
};

std::vector<VertexInfo> buildVertices(Graph &graph, const Tensor &t) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  // Maximum number of vertices is a function of number of tiles / IPU and
  // number of worker contexts per tile
  // Note that deviceInfo.getNumTiles() returns the number of tiles in all
  // IPUs
  const auto maxVertices =
    (deviceInfo.getNumTiles() * deviceInfo.numWorkerContexts)
    / deviceInfo.numIPUs;
  unsigned numElements = t.numElements();
  const unsigned minGrainSize = t.elementType() == "half" ? 2 : 4;
  unsigned elemsPerVertex = (numElements + maxVertices - 1) / maxVertices;

  if (elemsPerVertex < minGrainSize) {
    elemsPerVertex = minGrainSize;
  }

  std::vector<VertexInfo> v;

  // loop around to create vertices
  unsigned vertexId = 0;
  while (numElements) {
    const auto elemsThisVertex = std::min(numElements, elemsPerVertex);
    const auto startIdx = vertexId * elemsPerVertex;
    v.push_back({{startIdx, startIdx + elemsThisVertex}});
    ++vertexId;
    numElements -= elemsThisVertex;
  }
  return v;
}

static void
buildProgram(Graph &graph, const Tensor &A, uint64_t seed, uint16_t colouredId,
             RandomGenMode mode, const std::string &vertexName,
             const std::vector<std::pair<const std::string, double>> &params,
             Sequence &prog, const std::string &debugPrefix) {
  const auto dType = A.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/" + vertexName);
  const bool saveRestoreSeed = mode != NOT_REPEATABLE;
  auto aFlat = A.flatten();

  if (mode == ALWAYS_REPEATABLE) {
    std::vector<VertexInfo> vertexTab = buildVertices(graph, A);
    unsigned numVertices = vertexTab.size();
    unsigned verticesPerTile = (numVertices + numTiles  - 1) / numTiles;
    unsigned tile = 0;
    unsigned vertexId = 0;
    while (numVertices) {
      unsigned verticesThisTile = std::min(verticesPerTile, numVertices);
      for (auto i = vertexId; i < vertexId + verticesThisTile; ++i) {
        auto v = graph.addVertex(cs, templateVertex("poprand::" + vertexName,
                                                    dType));
        graph.connect(v["out"][0], aFlat.slice(vertexTab[i].region));
        graph.setFieldSize(v["out"], 1);
        for (const auto &p : params) {
          graph.setInitialValue(v[p.first],
                                dType == "int" && vertexName == "Uniform" ?
                                static_cast<int>(p.second) :
                                static_cast<float>(p.second));
        }
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["seedL"], seed);
        graph.setInitialValue(v["seedH"], createSeedU64(colouredId, 0, i));
        graph.setInitialValue(v["saveRestoreSeed"], saveRestoreSeed);
        graph.setTileMapping(v, tile);
      }
      vertexId += verticesThisTile;
      numVertices -= verticesThisTile;
      ++tile;
    }
  } else {
    const auto mapping = graph.getTileMapping(A);
    const auto grainSize = deviceInfo.getVectorWidth(dType);

    // In an actual system, a separate Compute Set or a tile power up routine
    // could initialise the seeds for each tile. We need a flag to indicate
    // whether this code runs on a hardware target or not.
    const bool simulateNonRepeatable = true;

    for (auto tile = 0U; tile != numTiles; ++tile) {
      const auto tileContiguousRegions =
          graph.getSortedContiguousRegions(aFlat, mapping[tile]);
      auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
      uint32_t vertexCount = 0;
      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(cs,
                                 templateVertex("poprand::" + vertexName,
                                                dType),
                                 {{"out", aFlat.slices(regions)}});
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        for (const auto &p : params) {
          graph.setInitialValue(v[p.first],
                                dType == "int" && vertexName == "Uniform" ?
                                static_cast<int>(p.second):
                                static_cast<float>(p.second));
        }
        if (simulateNonRepeatable || mode != NOT_REPEATABLE) {
          graph.setInitialValue(v["seedL"], seed);
          graph.setInitialValue(v["seedH"],  createSeedU64(colouredId, tile,
                                                           vertexCount));
        }
        graph.setInitialValue(v["saveRestoreSeed"], saveRestoreSeed);
        graph.setTileMapping(v, tile);
        ++vertexCount;
      }
    }
  }
  prog.add(Execute(cs));
}

// Convert a range [minVal, maxVal] for uniform number generation into a
// scale and offset used internally by the uniform random number generator
static std::pair<double, double>
uniformScaleAndOffset(double minVal, double maxVal, const std::string dType) {
  double scale = maxVal - minVal;
  if (dType != "int") {
    double offset = scale /  2 + minVal;
    return std::make_pair(scale, offset);
  } else {
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

void Random::
uniform(Graph &graph, Tensor &A, double minVal, double maxVal,
        Sequence &prog, const std::string &debugPrefix) {
  double scale, offset;

  if (minVal >= maxVal) {
    throw popstd::poplib_error("range for uniform distribution invalid");
  }
  std::tie(scale, offset) = uniformScaleAndOffset(minVal, maxVal,
                                                  A.elementType());
  const auto derivedSeed = colourSeedL64(seed, UNIFORM_MODULEID);
  const auto colouredId =
      colouredIdU64(UNIFORM_MODULEID, callCount[UNIFORM]++, mode);
  buildProgram(graph, A, derivedSeed, colouredId, mode, "Uniform",
               {{"scale", scale}, {"offset", offset}}, prog, debugPrefix);
}

void Random::
bernoulli(Graph &graph, Tensor &A, double prob, Sequence &prog,
          const std::string &debugPrefix) {
  if (prob < 0 || prob > 1.0) {
    throw popstd::poplib_error("invalid bernoulli probability");
  }
  const auto derivedSeed = colourSeedL64(seed, BERNOULLI_MODULEID);
  const auto colouredId =
      colouredIdU64(BERNOULLI_MODULEID, callCount[BERNOULLI]++, mode);
  buildProgram(graph, A, derivedSeed, colouredId, mode, "Bernoulli",
               {{"prob", prob}}, prog, debugPrefix);
}

void Random::
normal(Graph &graph, Tensor &A, double mean, double stdDev, Sequence &prog,
       const std::string &debugPrefix) {
  const auto derivedSeed = colourSeedL64(seed, NORMAL_MODULEID);
  const auto colouredId =
      colouredIdU64(NORMAL_MODULEID, callCount[NORMAL]++, mode);
  buildProgram(graph, A, derivedSeed, colouredId, mode, "Normal",
               {{"mean", mean}, {"stdDev", stdDev}}, prog, debugPrefix);
}

void Random::
truncatedNormal(Graph &graph, Tensor &A, double mean, double stdDev,
                double alpha, Sequence &prog, const std::string &debugPrefix) {
  const auto derivedSeed = colourSeedL64(seed, TRUNCNORMAL_MODULEID);
  const auto colouredId =
    colouredIdU64(TRUNCNORMAL_MODULEID, callCount[TRUNCATED_NORMAL]++,
                  mode);

  if (alpha < 1) {
    throw popstd::poplib_error("Alpha less than 1.0 not supported yet");
  }

  // select number of iterations such that probability that the number events
  // exceeding [+alpha, -alpha] is at the most 10^-prob. Those events are then
  // replaced by uniform/triangular probability
  const float logProb = -4.0;
  const unsigned iterations =
    std::ceil(logProb / std::log10(std::erfc(alpha / std::sqrt(2.0))));

  buildProgram(graph, A, derivedSeed, colouredId, mode, "TruncatedNormal",
               {{"mean", mean}, {"stdDev", stdDev}, {"alpha", alpha},
                {"iterations", iterations}}, prog, debugPrefix);
}

} // namespace poprand
