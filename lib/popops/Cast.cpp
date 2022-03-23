// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popops/Cast.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "poputil/DebugInfo.hpp"
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

constexpr unsigned maxDivisibleValue = (UINT_MAX / 0xAAAB) - 5;
constexpr unsigned elementsPerLoop = 4;

static bool validateRegionSizeForMultiVertex(
    const std::vector<std::vector<Interval>> &intervals, unsigned maxRepeatSize,
    unsigned numWorkers) {

  const auto numElems = intervalSequenceNumElements(intervals);
  if (numElems > maxDivisibleValue) {
    return false;
  }
  if (numElems > maxRepeatSize * numWorkers) {
    return false;
  }
  return true;
}

static void castImpl(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
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

  const unsigned maxElemsForRpt = target.getRptCountMax() * elementsPerLoop;
  const auto numWorkers = target.getNumWorkerContexts();

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (mapping[tile].empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(dst, mapping[tile]);
    // We use the 1D MultiVertex only if we have a single contiguous region
    // on the tile.
    if (tileContiguousRegions.size() == 1 &&
        validateRegionSizeForMultiVertex(tileContiguousRegions, maxElemsForRpt,
                                         numWorkers)) {
      VertexRef v;
      v = graph.addVertex(cs,
                          templateVertex("popops::Cast1D", srcType, dstType));
      const auto numElems = intervalSequenceNumElements(tileContiguousRegions);
      graph.connect(v["src"], concat(src.slices(tileContiguousRegions)));
      graph.connect(v["dst"], concat(dst.slices(tileContiguousRegions)));
      graph.setInitialValue(v["numElems"], numElems);
      graph.setTileMapping(v, tile);
    } else {
      auto vertexRegions =
          splitRegionsBetweenWorkers(target, tileContiguousRegions, vectorWidth,
                                     2 * vectorWidth, UINT_MAX, maxElemsForRpt);
      for (const auto &regions : vertexRegions) {
        const auto numRegions = regions.size();
        assert(numRegions != 0);
        VertexRef v;
        if (numRegions == 1) {
          const auto numElems = intervalSequenceNumElements(regions);
          v = graph.addVertex(cs, templateVertex("popops::Cast1DSingleWorker",
                                                 srcType, dstType));
          graph.connect(v["src"], concat(src.slices(regions)));
          graph.connect(v["dst"], concat(dst.slices(regions)));
          graph.setInitialValue(v["numElems"], numElems);
        } else {
          v = graph.addVertex(
              cs, templateVertex("popops::Cast2D", srcType, dstType));
          graph.connect(v["src"], src.slices(regions));
          graph.connect(v["dst"], dst.slices(regions));
        }
        graph.setTileMapping(v, tile);
      }
    }
  }
}

Program cast(Graph &graph, Tensor src, Tensor dst,
             const PoplibsOpDebugInfo &di) {
  // Casting one type into itself, or int<->unsigned, is just a copy.
  // We use the '.reinterpret(dstType)' to bypass type checking in Copy for the
  // int<->unsigned case
  // Casting between fp8 types is never just a copy as the metadata is not
  // known until runtime
  auto srcType = src.elementType();
  auto dstType = dst.elementType();
  if ((srcType == dstType && srcType != QUARTER) ||
      ((srcType == INT) && (dstType == UNSIGNED_INT)) ||
      ((srcType == UNSIGNED_INT) && (dstType == INT)) ||
      ((srcType == UNSIGNED_LONGLONG) && (dstType == LONGLONG)) ||
      ((srcType == LONGLONG) && (dstType == UNSIGNED_LONGLONG))) {
    logging::popops::trace("Cast is just a copy");
    return Copy(src.reinterpret(dstType), dst, false, {di});
  }
  auto cs = graph.addComputeSet({di, "Cast1DSingleWorker"});
  castImpl(graph, src, dst, cs);
  return Execute(cs, {di});
}

Program cast(Graph &graph, Tensor src, Tensor dst,
             const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dst));
  return cast(graph, src, dst, di);
}

Tensor cast(Graph &graph, Tensor src, const Type &dstType, ComputeSet cs,
            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dstType, cs));
  if (dstType == QUARTER) {
    throw poputil::poplibs_error("Metadata is required when creating a"
                                 " tensor of type quarter and casting to it");
  }
  auto dst = graph.clone(dstType, src, {di, "cast"});
  castImpl(graph, src, dst, cs);
  di.addOutput(dst);
  return dst;
}

Tensor cast(Graph &graph, Tensor src, const Type &dstType,
            const Tensor &metadata, ComputeSet cs,
            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(src, dstType, metadata, cs));

  if (dstType != QUARTER) {
    throw poputil::poplibs_error("Metadata only required when creating a"
                                 " tensor of type quarter and casting to it");
  }
  auto dst = graph.clone(dstType, &metadata, src, {di, "cast"});
  castImpl(graph, src, dst, cs);
  di.addOutput(dst);
  return dst;
}

void cast(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
  castImpl(graph, src, dst, cs);
}

Tensor cast(Graph &graph, const Tensor &src, const Type &dstType,
            Sequence &prog, const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  if (dstType == QUARTER) {
    throw poputil::poplibs_error("Metadata is required when creating a"
                                 " tensor of type quarter and casting to it");
  }
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dstType));
  auto dst = graph.clone(dstType, src, {di, "cast"});
  prog.add(cast(graph, src, dst, {di}));
  di.addOutput(dst);
  return dst;
}

Tensor cast(Graph &graph, const Tensor &src, const Type &dstType,
            const Tensor &metadata, Sequence &prog,
            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dstType, metadata));

  if (dstType != QUARTER) {
    throw poputil::poplibs_error("Metadata only required when creating a "
                                 " tensor of type quarter and casting to it");
  }
  auto dst = graph.clone(dstType, &metadata, src, {di, "cast"});
  prog.add(cast(graph, src, dst, {di}));
  di.addOutput(dst);
  return dst;
}

poplar::Tensor checkAccuracyWhenCast(Graph &graph, const Tensor &input,
                                     Type outputType, double tolerance,
                                     poplar::program::Sequence &prog,
                                     const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, outputType, tolerance));

  if ((input.elementType() != FLOAT && outputType != HALF) ||
      input.numElements() != 1) {
    throw poputil::poplibs_error(
        "Can only check the accuracy when casting"
        " single element tensors with data type float to half or half"
        " to float");
  }

  auto cs = graph.addComputeSet({di, "checkAccuracyWhenCast"});
  auto v = graph.addVertex(cs, templateVertex("popops::CheckAccuracyWhenCast",
                                              input.elementType(), outputType));
  auto isAccurate = graph.addVariable(BOOL, {}, {di, "checkAccuracyWhenCast"});
  const auto tile = std::min(graph.getTarget().getNumTiles(), 4u) - 1;
  graph.setTileMapping(isAccurate, tile);

  graph.connect(v["input"], input.reshape({}));
  graph.setInitialValue(v["tolerance"], tolerance);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs, isAccurate, {di}));
  return isAccurate;
}

} // end namespace popops
