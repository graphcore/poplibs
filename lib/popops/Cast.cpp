// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popops/Cast.hpp"

#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <poplar/Graph.hpp>
#include <poplibs_support/logging.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace popops {

Program cast(Graph &graph, Tensor src, Tensor dst,
             const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  // Casting one type into itself, or int<->unsigned, is just a copy.
  // We use the '.reinterpret(dstType)' to bypass type checking in Copy for the
  // int<->unsigned case
  auto srcType = src.elementType();
  auto dstType = dst.elementType();
  if ((srcType == dstType) || ((srcType == INT) && (dstType == UNSIGNED_INT)) ||
      ((srcType == UNSIGNED_INT) && (dstType == INT))) {
    logging::popops::trace("Cast is just a copy");
    return Copy(src.reinterpret(dstType), dst);
  }
  auto cs = graph.addComputeSet(debugPrefix + "/Cast");
  cast(graph, src, dst, cs);
  return Execute(cs);
}

void cast(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
  assert(src.shape() == dst.shape());
  src = src.flatten();
  dst = dst.flatten();
  graph.reorderToSimplify(&dst, {&src}, false);
  const auto srcType = src.elementType();
  const auto dstType = dst.elementType();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getFloatVectorWidth();
  std::vector<std::vector<Interval>> mapping = graph.getTileMapping(dst);
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(dst, mapping[tile]);
    // We use the supervisor vertex only if we have a single contiguous region
    // on the tile.
    if (tileContiguousRegions.size() == 1) {
      VertexRef v;
      v = graph.addVertex(
          cs, templateVertex("popops::CastSupervisor", srcType, dstType));
      const auto numElems = intervalSequenceNumElements(tileContiguousRegions);
      graph.connect(v["src"], concat(src.slices(tileContiguousRegions)));
      graph.connect(v["dst"], concat(dst.slices(tileContiguousRegions)));
      // The supervisor vertex will partition work to each worker in multiples
      // of 4 elements. This ensures alignment of at least 8 bytes. Needed
      // because the worker vertex requires 8 byte alignment.
      unsigned grainSize = 4;
      // Computing the bitfields for the 'partitionParams' word. See the codelet
      // C++ definition for the meaning of the fields.
      unsigned numGrains = (numElems + grainSize - 1) / grainSize;
      unsigned numWorkerContexts = target.getNumWorkerContexts();
      unsigned workerCount = numWorkerContexts;
      unsigned grainsPerWorker = 1;
      unsigned workerLast = numWorkerContexts - 1;
      if (numGrains <= numWorkerContexts) {
        workerCount = numGrains;
        workerLast = workerCount - 1;
      } else {
        grainsPerWorker = numGrains / workerCount;
        unsigned rem = numGrains % workerCount;
        if (rem > 0) {
          workerCount = rem;
          grainsPerWorker += 1;
        }
      }
      unsigned workerElems = grainsPerWorker * grainSize;
      unsigned deltaLast =
          workerCount * workerElems +
          (numWorkerContexts - workerCount) * (workerElems - grainSize) -
          numElems;
      unsigned partitionParams = (workerElems << 9) | (workerCount << 6) |
                                 (workerLast << 3) | deltaLast;
      graph.setInitialValue(v["partitionParams"], partitionParams);
      graph.setTileMapping(v, tile);
    } else {
      auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, vectorWidth, 2 * vectorWidth);
      for (const auto &regions : vertexRegions) {
        const auto numRegions = regions.size();
        assert(numRegions != 0);
        VertexRef v;
        if (numRegions == 1) {
          const auto numElems = intervalSequenceNumElements(regions);
          v = graph.addVertex(cs,
                              templateVertex("popops::Cast", srcType, dstType));
          graph.connect(v["src"], concat(src.slices(regions)));
          graph.connect(v["dst"], concat(dst.slices(regions)));
          graph.setInitialValue(v["numElems"], numElems);
        } else {
          v = graph.addVertex(
              cs, templateVertex("popops::Cast2d", srcType, dstType));
          graph.connect(v["src"], src.slices(regions));
          graph.connect(v["dst"], dst.slices(regions));
        }
        graph.setTileMapping(v, tile);
      }
    }
  }
}

Tensor cast(Graph &graph, Tensor src, const Type &dstType, ComputeSet cs,
            const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  auto dst = graph.clone(dstType, src, debugPrefix + "/cast");
  cast(graph, src, dst, cs);
  return dst;
}

poplar::Tensor cast(Graph &graph, const Tensor &src, const Type &dstType,
                    Sequence &prog, const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  auto dst = graph.clone(dstType, src, debugPrefix + "/cast");
  prog.add(cast(graph, src, dst, debugPrefix));
  return dst;
}

poplar::Tensor checkAccuracyWhenCast(Graph &graph, const Tensor &input,
                                     Type outputType, double tolerance,
                                     poplar::program::Sequence &prog,
                                     const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  if ((input.elementType() != FLOAT && outputType != HALF) ||
      input.numElements() != 1) {
    throw poputil::poplibs_error(
        "Can only check the accuracy when casting"
        " single element tensors with data type float to half or half"
        " to float");
  }

  auto cs = graph.addComputeSet(debugPrefix + "/checkAccuracyWhenCast");
  auto v = graph.addVertex(cs, templateVertex("popops::CheckAccuracyWhenCast",
                                              input.elementType(), outputType));
  auto isAccurate =
      graph.addVariable(BOOL, {}, debugPrefix + "/checkAccuracyWhenCast");
  const auto tile = std::min(graph.getTarget().getNumTiles(), 4u) - 1;
  graph.setTileMapping(isAccurate, tile);

  graph.connect(v["input"], input.reshape({}));
  graph.setInitialValue(v["tolerance"], tolerance);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs, isAccurate));
  return isAccurate;
}

} // end namespace popops
