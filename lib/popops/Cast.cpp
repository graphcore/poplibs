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

// This is based on the macros that reference `CastVertexName` in
// elemwiseMiscCodelets.cpp
static void validateCastVertexTypes(const Type &srcType, const Type &dstType,
                                    bool supportFloat) {
  if (srcType == QUARTER || dstType == QUARTER) {
    // If either type is quarter we can cast to/from a limited range of types
    const std::array<Type, 5> allowed = {HALF, CHAR, UNSIGNED_CHAR, SIGNED_CHAR,
                                         QUARTER};
    auto otherType = srcType == QUARTER ? dstType : srcType;
    if (std::find(allowed.begin(), allowed.end(), otherType) != allowed.end()) {
      return;
    }
    if (srcType == FLOAT || dstType == FLOAT) {
      if (supportFloat) {
        return;
      }
      // This error is API specific, make that clear in the message
      throw poputil::poplibs_error("Casting from " + srcType.toString() +
                                   " to " + dstType.toString() +
                                   " is not supported using computeSet APIs."
                                   " Use a Program API instead.");
    }
  } else if (srcType == dstType) {
    // Any type other than QUARTER, a cast to the same type is just a copy
    // so all combinations are possible
    return;
  } else if (dstType == UNSIGNED_LONGLONG || dstType == LONGLONG) {
    // We can cast to LONG types from other integral types, including just a
    // copy when the type is another LONG type
    const std::array<Type, 10> allowed = {
        SHORT, UNSIGNED_SHORT, BOOL,        UNSIGNED_INT,      INT,
        CHAR,  UNSIGNED_CHAR,  SIGNED_CHAR, UNSIGNED_LONGLONG, LONGLONG};
    if (std::find(allowed.begin(), allowed.end(), srcType) != allowed.end()) {
      return;
    }
  } else {
    // We can cast to/from all of these types
    const std::array<Type, 10> allowed = {
        CHAR, SIGNED_CHAR, UNSIGNED_CHAR, FLOAT,          HALF,
        INT,  SHORT,       UNSIGNED_INT,  UNSIGNED_SHORT, BOOL};
    bool srcFound =
        std::find(allowed.begin(), allowed.end(), srcType) != allowed.end();
    bool dstFound =
        std::find(allowed.begin(), allowed.end(), dstType) != allowed.end();
    if (srcFound && dstFound) {
      return;
    }
  }
  throw poputil::poplibs_error("Casting from " + srcType.toString() + " to " +
                               dstType.toString() + " is not supported");
}

static void castImpl(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {

  const auto srcType = src.elementType();
  const auto dstType = dst.elementType();
  validateCastVertexTypes(
      srcType, dstType, graph.getTarget().getNumConvUnits(QUARTER, HALF) != 0);
  if (src.shape() != dst.shape()) {
    throw poplibs_error(
        "Attempting to cast between tensors with different shapes");
  }

  src = src.flatten();
  dst = dst.flatten();
  graph.reorderToSimplify(&dst, {&src}, false);

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

static Tensor
doIntermediateCastIfRequired(Graph &graph, const Tensor &src,
                             const Type &dstType, ComputeSet &cs,
                             const poputil::PoplibsOpDebugInfo &di) {
  // Support cast float to/from quarter using an intermediate half tensor
  if ((src.elementType() == FLOAT && dstType == QUARTER) ||
      (src.elementType() == QUARTER && dstType == FLOAT)) {
    if (graph.getTarget().getNumConvUnits(QUARTER, HALF) != 0) {
      return src;
    } else {
      auto intermediate = graph.clone(HALF, src, {di, "castIntermediateHalf"});
      castImpl(graph, src, intermediate, cs);
      return intermediate;
    }
  } else {
    return src;
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
  auto cs1 = graph.addComputeSet({di, "Cast"});
  auto inter =
      doIntermediateCastIfRequired(graph, src, dst.elementType(), cs1, di);
  auto cs2 = graph.addComputeSet({di, "Cast"});
  castImpl(graph, inter, dst, cs2);
  return Sequence({Execute(cs1), Execute(cs2)}, {di});
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
  auto dst = graph.clone(dstType, metadata, src, {di, "cast"});
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
  auto dst = graph.clone(dstType, metadata, src, {di, "cast"});
  prog.add(cast(graph, src, dst, {di}));
  di.addOutput(dst);
  return dst;
}

void castWithOutput(Graph &graph, const Tensor &src, const Tensor &dst,
                    Sequence &prog, const DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, dst));
  prog.add(cast(graph, src, dst, {di}));
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
  graph.connect(v["output"], isAccurate);

  graph.setInitialValue(v["tolerance"], tolerance);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs, {di}));
  return isAccurate;
}

} // end namespace popops
