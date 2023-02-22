// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popops/Cast.hpp"
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

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

// Number of cycles overhead in a MultiVertex fpor work division over
// a single worker launch.
constexpr unsigned multiVertexCyclesForWorkDivision = 6;

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

// For single regions, it may be beneficial to select a single worker over a
// multivertex because of the overhead in dividing work.
// The cycles cost to cast is a simplistic model of assuming the
// cycles it takes for loading/storing the larger of the src/dst types.
static bool useSingleWorkerOverMultiVertex(unsigned numElems,
                                           unsigned srcElemsPerVector,
                                           unsigned dstElemsPerVector) {
  const auto elemsPerVector = std::min(srcElemsPerVector, dstElemsPerVector);
  return numElems <= elemsPerVector * multiVertexCyclesForWorkDivision;
}

// This is based on the macros that reference `CastVertexName` in
// elemwiseMiscCodelets.cpp
static void validateCastVertexTypes(const Type &srcType, const Type &dstType) {
  if (srcType == QUARTER || dstType == QUARTER) {
    // If either type is quarter we can cast to/from a limited range of types
    const std::array<Type, 6> allowed = {HALF,          FLOAT,       CHAR,
                                         UNSIGNED_CHAR, SIGNED_CHAR, QUARTER};
    auto otherType = srcType == QUARTER ? dstType : srcType;
    if (std::find(allowed.begin(), allowed.end(), otherType) != allowed.end()) {
      return;
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
  validateCastVertexTypes(srcType, dstType);
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
      const auto numElems = intervalSequenceNumElements(tileContiguousRegions);
      const auto vertexName = useSingleWorkerOverMultiVertex(
                                  numElems, target.getVectorWidth(srcType),
                                  target.getVectorWidth(dstType))
                                  ? "popops::Cast1DSingleWorker"
                                  : "popops::Cast1D";
      auto v =
          graph.addVertex(cs, templateVertex(vertexName, srcType, dstType));

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

// This function takes a metadata tensor as input and returns the
// following two tensors.
//  - Metadata tensor with scale bias of 0 (i.e., scale of 1.0) and
//  - A tensor that contains scale factor as float.
//      scaling = pow(2.0f, scaleBias) if negateScaling = false
//      scaling = pow(2.0f, -scaleBias) if negateScaling = true
static std::pair<Tensor, Tensor>
getQuarterUnitScaleMetadata(Graph &graph, const Tensor &metadata,
                            Sequence &prog, bool negateScaling) {
  if (metadata.elementType() != QUARTER_METADATA) {
    throw poputil::poplibs_error(
        fmt::format("The {} tensor must be of type QUARTER_METADATA",
                    metadata.getDebugStr()));
  }
  namespace pe = popops::expr;
  auto signedBit = map(
      graph,
      pe::Cast(pe::BitwiseAnd(pe::Cast(pe::_1, INT), pe::Const(0x20)), BOOL),
      {metadata.reinterpret(SIGNED_CHAR)}, prog);

  float negate = negateScaling ? -1.0f : 1.0f;
  constexpr unsigned numScaleBits = 6U;
  constexpr unsigned shiftOutFormatBits = 32U - numScaleBits;
  auto scaling =
      map(graph,
          pe::Exp2(pe::Cast(pe::ShrSE(pe::Cast(pe::_1, INT)
                                          << pe::Const(shiftOutFormatBits),
                                      pe::Const(shiftOutFormatBits)),
                            FLOAT) *
                   pe::Const(negate)),
          {metadata.reinterpret(SIGNED_CHAR), signedBit}, prog);

  // Create quarter tensor with unit metadata scale bias.
  auto mdUnitScale = [](Graph &g, const Tensor &metadata, Sequence &prog) {
    auto md = g.clone(metadata);
    mapInPlace(g,
               pe::Cast(pe::BitwiseAnd(pe::Cast(pe::_2, INT), pe::Const(0x80)),
                        SIGNED_CHAR),
               {md.reinterpret(SIGNED_CHAR), metadata.reinterpret(SIGNED_CHAR)},
               prog);
    return md;
  }(graph, metadata, prog);

  return {mdUnitScale, scaling};
}

static Program castQuarterToFloat(Graph &graph, Tensor src, Tensor dst,
                                  const PoplibsOpDebugInfo &di) {
  namespace pe = popops::expr;
  Sequence prog;
  auto [mdUnitScale, scaling] =
      getQuarterUnitScaleMetadata(graph, src.getMetadata(), prog, false);
  auto srcUnitScale = graph.clone(QUARTER, mdUnitScale, src);
  prog.add(Copy(src.reinterpret(UNSIGNED_CHAR),
                srcUnitScale.reinterpret(UNSIGNED_CHAR)));

  // Accurate quarter to float cast.
  // -------------------------------
  // Machine instructions do not exist to directly convert from quarter to
  // float, but can be done in the following steps.
  //  1. convert from quarter to half (either f143 or f152 format) with unit
  //     metadata scale. The range of quarter is a subset of the range of
  //     half as shown below.
  //       half ~ [2^-24, 2^15]
  //       quarter f143 ~ [2^-10, 2^7]
  //       quarter f152 ~ [2^-17, 2^15]
  //  2. convert from half to float.
  //  3. Scale float by pow(2.0f, metadataScale)
  mapInPlace(graph, pe::Mul(pe::Cast(pe::Cast(pe::_2, HALF), FLOAT), pe::_3),
             {dst, srcUnitScale, scaling}, prog);
  return prog;
}

static Program castFloatToQuarter(Graph &graph, Tensor src, Tensor dst,
                                  const PoplibsOpDebugInfo &di) {
  namespace pe = popops::expr;
  Sequence prog;
  auto [mdUnitScale, scaling] =
      getQuarterUnitScaleMetadata(graph, dst.getMetadata(), prog, true);

  // Accurate float to quarter cast, rounding to nearest, ties to even.
  // ------------------------------------------------------------------
  // In the following explanation quarter-f143 format is used without loss of
  // generality. The same principle should apply to quarter-f152. Machine
  // instructions do not exist to directly convert from float to quarter,
  // but can be done in the following steps.
  //  1. Scale float by pow(2.0f, -metadataScale)
  //  2. convert from float to half, taking rounding into account.
  //  3. convert from half to quarter (either f143 or f152 format) with unit
  //     metadata scale.
  //
  // A naive zeroing of the lowest float mantissa bits in Step 2 could
  // cause the production of values that on casting to half mistakenly
  // appear to be equidistant from adjacent quarter representations.
  // Under this condition called a "tie" the result is rouneded to the
  // nearest even valued bit representation. The false tie condition is avoided
  // by ensuring that bit 15 of the float bit representation is set if any of
  // the lowest 15 bits are non-zero.
  auto f32 = mul(graph, src, scaling, prog, {di});
  auto nonZeroLSBs = map(
      graph,
      pe::Cast(pe::Cast(pe::BitwiseAnd(pe::_1, pe::Const(0x00007fff)), BOOL),
               UNSIGNED_INT),
      {f32.reinterpret(UNSIGNED_INT)}, prog, {di});
  mapInPlace(graph,
             pe::BitwiseOr(pe::BitwiseAnd(pe::_1, pe::Const(0xffff8000)),
                           pe::Shl(pe::_2, pe::Const(15))),
             {f32.reinterpret(UNSIGNED_INT), nonZeroLSBs}, prog, {di});
  auto half = cast(graph, f32, HALF, prog, {di});
  auto quart = cast(graph, half, QUARTER, mdUnitScale, prog, {di});
  prog.add(Copy(quart.reinterpret(SIGNED_CHAR), dst.reinterpret(SIGNED_CHAR)));
  return prog;
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
  if (graph.getTarget().getNumConvUnits(QUARTER, HALF) == 0) {
    if (srcType == QUARTER && dstType == FLOAT) {
      return castQuarterToFloat(graph, src, dst, di);
    } else if (srcType == FLOAT && dstType == QUARTER) {
      return castFloatToQuarter(graph, src, dst, di);
    }
  }
  auto cs = graph.addComputeSet({di, "Cast"});
  castImpl(graph, src, dst, cs);
  return Sequence({Execute(cs)}, {di});
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
