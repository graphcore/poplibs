#include "popops/ElementWise.hpp"
#include "popops/Cast.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/OptionParsing.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

namespace {

// Check if we can use a supervisor vertex, or if the regions to process
// prevent it.  This can be due to either having multiple regions or if the
// region is too large.
bool validateRegionSizeForSupervisorVertex(
    const std::vector<std::vector<Interval>> &intervals,
    unsigned maxRegionSize) {
  if (maxRegionSize == UINT_MAX) {
    return true;
  }
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    const auto &regions = intervals[i];
    const unsigned regionElements = std::accumulate(
        regions.begin(), regions.end(), 0,
        [](std::size_t total, const Interval &i) { return total + i.size(); });
    if (regionElements > maxRegionSize) {
      return false;
    }
  }
  return true;
}

struct ScaledAddOptions {
  bool optimizeForSpeed = false;
};

ScaledAddOptions parseOptionFlags(const OptionFlags &options) {
  ScaledAddOptions scaledAddOpts;
  const poplibs::OptionSpec scaledAddSpec{
      {"optimizeForSpeed",
       poplibs::OptionHandler::createWithBool(scaledAddOpts.optimizeForSpeed)}};
  for (const auto &entry : options) {
    scaledAddSpec.parse(entry.first, entry.second);
  }
  return scaledAddOpts;
}

void scaledArithmeticConstImpl(Graph &graph, Tensor A, float scaleA, Tensor B,
                               float scaleB, Sequence &prog,
                               const std::string &debugPrefix,
                               const poplar::OptionFlags &options) {
  auto opts = parseOptionFlags(options);

  const auto addConstraints =
             (A.elementType() == HALF || A.elementType() == FLOAT) &&
             opts.optimizeForSpeed;
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  const auto &target = graph.getTarget();
  const auto dType = A.elementType();
  const auto dataBType = B.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto numWorkers = target.getNumWorkerContexts();

  const auto codeletName2D = scaleA != 1.0f ?
             templateVertex("popops::aXPlusbY2D", dType, true, addConstraints) :
             templateVertex("popops::ScaledAdd2D", dType, true, addConstraints);

  const auto codeletNameSupervisor = scaleA == 1.0f ?
             templateVertex("popops::ScaledAddSupervisor", dType, dataBType,
                            true, addConstraints) :
             templateVertex("popops::aXPlusbYSupervisor", dType, true,
                            addConstraints);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "A", 1),
    target.getRptCountMax() * vectorWidth);

  const auto maxSupervisorElements = std::min<std::size_t>(
    graph.getMaxVertexFieldValue(codeletNameSupervisor, "size"),
    target.getRptCountMax() * vectorWidth * numWorkers);

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);

    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1 &&
        validateRegionSizeForSupervisorVertex(tileContiguousRegions,
                                              maxSupervisorElements)) {
      const auto &region = tileContiguousRegions[0][0];
      auto v =
            graph.addVertex(cs, codeletNameSupervisor,
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)}});
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["size"], aFlat.slice(region).numElements());
      if(scaleA == 1.0f) {
        graph.setInitialValue(v["scaleB"], scaleB);
      }
      else {
        graph.setInitialValue(v["scaleA"], scaleA);
        graph.setInitialValue(v["scaleB"], scaleB);
      }
    } else {
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize, UINT32_MAX,
                                   max2DInnerElements);

      for (const auto &regions : vertexRegions) {
        auto v = graph.addVertex(cs, codeletName2D,
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)}});

        graph.setTileMapping(v, tile);
        if(scaleA == 1.0f) {
          graph.setInitialValue(v["scaleB"], scaleB);
        }
        else {
          graph.setInitialValue(v["scaleA"], scaleA);
          graph.setInitialValue(v["scaleB"], scaleB);
        }
      }
    }
  }
  prog.add(Execute(cs));
}


void scaledArithmeticTensorImpl(Graph &graph, Tensor A, Tensor scaleA, Tensor B,
           Tensor scaleB,
           const bool doSubtract,
           const bool doaXPlusbY,
           Sequence &prog,
           const std::string &debugPrefix,
           const poplar::OptionFlags &options) {
  auto opts = parseOptionFlags(options);
  const auto addConstraints =
             (A.elementType() == HALF || A.elementType() == FLOAT) &&
             opts.optimizeForSpeed;
  if (!A.isParallelWriteable())
    throw poputil::poplibs_error("Trying to accumulate to tensor that cannot be"
                                 " written in parallel");
  if (A.shape() != B.shape())
    throw poputil::poplibs_error("Input Tensors for scaled arithmetic must"
                                 " have the same shape");
  const auto &target = graph.getTarget();
  const auto dType = A.elementType();
  const auto dataBType = B.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto numWorkers = target.getNumWorkerContexts();

  const auto codeletName2D = doSubtract ?
          templateVertex("popops::ScaledSubtract2D", dType, addConstraints) :
          templateVertex("popops::ScaledAdd2D", dType, false, addConstraints);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max2DInnerElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "A", 1),
    target.getRptCountMax() * vectorWidth);

   const auto codeletNameSupervisorForSizingOnly =
    templateVertex("popops::ScaledAddSupervisor", dType, dataBType,
                   true, addConstraints);

  const auto maxSupervisorElements = std::min<std::size_t>(
    graph.getMaxVertexFieldValue(codeletNameSupervisorForSizingOnly, "size"),
    target.getRptCountMax() * vectorWidth * numWorkers);

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
   // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    graph.setTileMapping(scaleB, tile);

    if (tileContiguousRegions.size() == 1 &&
        tileContiguousRegions[0].size() == 1  &&
        validateRegionSizeForSupervisorVertex(tileContiguousRegions,
                                              maxSupervisorElements)) {
      const auto &region = tileContiguousRegions[0][0];

      const auto v = doSubtract ?
          graph.addVertex(cs, templateVertex("popops::ScaledSubtractSupervisor",
                                              dType, dataBType, addConstraints),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleB", scaleB.reshape({1})}}):
          doaXPlusbY ?
          graph.addVertex(cs, templateVertex("popops::aXPlusbYSupervisor",
                                              dType, false, addConstraints),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleA", scaleA.reshape({1})},
                                              {"scaleB", scaleB.reshape({1})}}):
          graph.addVertex(cs, templateVertex("popops::ScaledAddSupervisor",
                                              dType, dataBType,
                                              false, addConstraints),
                                             {{"A", aFlat.slice(region)},
                                              {"B", bFlat.slice(region)},
                                              {"scaleB", scaleB.reshape({1})}});
      graph.setInitialValue(v["size"], aFlat.slice(region).numElements());
      graph.setTileMapping(v, tile);
    } else {
      auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize, UINT32_MAX,
                                   max2DInnerElements);
      for (const auto &regions : vertexRegions) {
        auto v = doaXPlusbY ?
              graph.addVertex(cs, templateVertex("popops::aXPlusbY2D",
                                  dType, false, addConstraints),
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)},
                                  {"scaleA", scaleA},
                                  {"scaleB", scaleB}}) :
              graph.addVertex(cs, codeletName2D,
                                 {{"A", aFlat.slices(regions)},
                                  {"B", bFlat.slices(regions)},
                                  {"scaleB", scaleB}});
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(cs));
}

}


void scaledAddTo(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto castPrefix = debugPrefix + "/scaledAdd";
  auto cs = graph.addComputeSet(castPrefix + "/cast");
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, cs, castPrefix + "/B");
  }
  if (scaleB.elementType() != targetType) {
    scaleB = cast(graph, scaleB, targetType, cs, castPrefix + "/scaleB");
  }
  prog.add(Execute(cs));
  scaledArithmeticTensorImpl(graph, A, scaleB, B, scaleB, false, false,
                             prog, debugPrefix, options);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor B, float scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, prog, debugPrefix + "/scaledAdd/B");
  }
  scaledArithmeticConstImpl(graph, A, 1.0, B, scaleB,
                            prog, debugPrefix, options);
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, Tensor scaleB,
                        Sequence &prog, const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto castPrefix = debugPrefix + "/scaledSub";
  auto cs = graph.addComputeSet(castPrefix + "/cast");
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, cs, castPrefix + "/B");
  }
  if (scaleB.elementType() != targetType) {
    scaleB = cast(graph, scaleB, targetType, cs, castPrefix + "/scaleB");
  }
  prog.add(Execute(cs));
  scaledArithmeticTensorImpl(graph, A, scaleB, B, scaleB, true, false,
                             prog, debugPrefix, options);
}

void scaledSubtractFrom(Graph &graph, Tensor A, Tensor B, float scaleB,
                        Sequence &prog, const std::string &debugPrefix,
                        const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, prog, debugPrefix + "/ScaledSub/B");
  }
  scaledArithmeticConstImpl(graph, A, 1.0, B, -scaleB, prog, debugPrefix,
                            options);
}

void scaledAddTo(Graph &graph, Tensor A, Tensor scaleA, Tensor B, Tensor scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto fnPrefix = debugPrefix + "/scaledAdd";
  bool axpby = true;

  auto cs = graph.addComputeSet(fnPrefix + "/cast");
  if (scaleA.elementType() != targetType) {
    scaleA = cast(graph, scaleA, targetType, prog, fnPrefix+ "/scaleA");
  }
  // We don't support float axpby vertex. Synthesize using mul and scaledAdd
  if (A.elementType() == FLOAT) {
    mulInPlace(graph, A, scaleA, prog, fnPrefix);
    axpby = false;
  }
  if (targetType != B.elementType()) {
    B = cast(graph, B, targetType, cs, fnPrefix + "/B");
  }
  if (scaleB.elementType() != targetType) {
    scaleB = cast(graph, scaleB, targetType, cs, fnPrefix + "/scaleB");
  }
  prog.add(Execute(cs));
  scaledArithmeticTensorImpl(graph, A, scaleA, B, scaleB, false, axpby,
                             prog, debugPrefix, options);
}

void scaledAddTo(Graph &graph, Tensor A, float scaleA, Tensor B,  float scaleB,
                 Sequence &prog, const std::string &debugPrefix,
                 const poplar::OptionFlags &options) {
  const auto targetType = A.elementType();
  const auto fnPrefix = debugPrefix + "/scaledAdd";

  // we do not support float axpby. Synthesize using mul and scaledAdd
  if (A.elementType() == FLOAT && scaleA != 1.0f) {
    mulInPlace(graph, A, scaleA, prog, fnPrefix);
    scaleA = 1.0f;
  }
  if (B.elementType() != targetType) {
    B = cast(graph, B, targetType, prog, fnPrefix + "/B");
  }
  scaledArithmeticConstImpl(graph, A, scaleA, B, scaleB, prog,
                            debugPrefix, options);
}

}
