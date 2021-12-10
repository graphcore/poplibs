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

#include <boost/optional.hpp>

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

static void cast(Graph &graph, Tensor src, Tensor dst,
                 boost::optional<Tensor> metaData, ComputeSet cs) {
  assert(src.shape() == dst.shape());
  src = src.flatten();
  dst = dst.flatten();
  graph.reorderToSimplify(&dst, {&src}, false);
  const auto srcType = src.elementType();
  const auto dstType = dst.elementType();

  // TODO - revise when a type is properly available
  std::string nameExtension = "";
  if (srcType == QUARTER || dstType == QUARTER) {
    nameExtension = "Fp8";
    assert(metaData);
  }
  // TODO end

  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getFloatVectorWidth();
  std::vector<std::vector<Interval>> mapping = graph.getTileMapping(dst);
  const auto numTiles = target.getNumTiles();

  const unsigned maxElemsForRpt = target.getRptCountMax() * elementsPerLoop;
  const auto numWorkers = target.getNumWorkerContexts();

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(dst, mapping[tile]);
    // We use the 1D MultiVertex only if we have a single contiguous region
    // on the tile.
    if (tileContiguousRegions.size() == 1 &&
        validateRegionSizeForMultiVertex(tileContiguousRegions, maxElemsForRpt,
                                         numWorkers)) {
      VertexRef v;
      v = graph.addVertex(cs, templateVertex("popops::Cast1D" + nameExtension,
                                             srcType, dstType));
      const auto numElems = intervalSequenceNumElements(tileContiguousRegions);
      graph.connect(v["src"], concat(src.slices(tileContiguousRegions)));
      graph.connect(v["dst"], concat(dst.slices(tileContiguousRegions)));
      if (metaData) {
        graph.connect(v["metaData"], metaData.get());
      }
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
          v = graph.addVertex(
              cs, templateVertex("popops::Cast1DSingleWorker" + nameExtension,
                                 srcType, dstType));
          graph.connect(v["src"], concat(src.slices(regions)));
          graph.connect(v["dst"], concat(dst.slices(regions)));
          if (metaData) {
            graph.connect(v["metaData"], metaData.get());
          }
          graph.setInitialValue(v["numElems"], numElems);
        } else {
          v = graph.addVertex(cs,
                              templateVertex("popops::Cast2D" + nameExtension,
                                             srcType, dstType));
          graph.connect(v["src"], src.slices(regions));
          graph.connect(v["dst"], dst.slices(regions));
          if (metaData) {
            graph.connect(v["metaData"], metaData.get());
          }
        }
        graph.setTileMapping(v, tile);
      }
    }
  }
}

Program cast(Graph &graph, Tensor src, Tensor dst,
             const boost::optional<Tensor> metaData,
             const PoplibsOpDebugInfo &di) {
  // Casting one type into itself, or int<->unsigned, is just a copy.
  // We use the '.reinterpret(dstType)' to bypass type checking in Copy for the
  // int<->unsigned case
  // Casting between fp8 types is never just a copy as the meta data is not
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
  cast(graph, src, dst, metaData, cs);
  return Execute(cs, {di});
}

Program cast(Graph &graph, Tensor src, Tensor dst,
             const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dst));
  return cast(graph, src, dst, boost::none, di);
}

Program cast(Graph &graph, Tensor src, Tensor dst, Tensor metaData_,
             const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dst, metaData_));
  boost::optional<Tensor> metaData = metaData_;
  return cast(graph, src, dst, metaData, di);
}

Tensor cast(Graph &graph, Tensor src, const Type &dstType, ComputeSet cs,
            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dstType, cs));
  auto dst = graph.clone(dstType, src, {di, "cast"});
  cast(graph, src, dst, boost::none, cs);
  di.addOutput(dst);
  return dst;
}

Tensor cast(Graph &graph, Tensor src, const Type &dstType, Tensor metaData,
            ComputeSet cs, const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(src, dstType, metaData, cs));
  auto dst = graph.clone(dstType, src, {di, "cast"});
  cast(graph, src, dst, metaData, cs);
  di.addOutput(dst);
  return dst;
}

void cast(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
  cast(graph, src, dst, boost::none, cs);
}

void cast(Graph &graph, Tensor src, Tensor dst, Tensor metaData_,
          ComputeSet cs) {
  boost::optional<Tensor> metaData = metaData_;
  cast(graph, src, dst, metaData, cs);
}

Tensor cast(Graph &graph, const Tensor &src, const Type &dstType,
            Sequence &prog, const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dstType));
  auto dst = graph.clone(dstType, src, {di, "cast"});
  prog.add(cast(graph, src, dst, boost::none, {di}));
  di.addOutput(dst);
  return dst;
}

Tensor cast(Graph &graph, const Tensor &src, const Type &dstType,
            const Tensor metaData_, Sequence &prog,
            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(src, dstType, metaData_));
  auto dst = graph.clone(dstType, src, {di, "cast"});
  boost::optional<Tensor> metaData = metaData_;
  prog.add(cast(graph, src, dst, metaData, {di}));
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
