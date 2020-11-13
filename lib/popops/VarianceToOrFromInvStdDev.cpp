// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "ExprOpUtil.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/ExprOp.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <poplar/Graph.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace logging = poplibs_support::logging;

namespace popops {

Program convertVariance(Graph &graph, Tensor src, Tensor dst,
                        const Tensor &epsilon, expr::BinaryOpType op,
                        const DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai});
  src = src.flatten();
  dst = dst.flatten();

  logging::popops::info(
      "convertVariance src={}, dst={}, epsilon={}, op={}, name={}", src.shape(),
      dst.shape(), epsilon.shape(), expr::binaryOpTypeToString(op),
      dnai.getPathName());

  if (epsilon.numElements() != 1) {
    throw poputil::poplibs_error("Epsilon must be a tensor with a single "
                                 "element for invStdDev to/from variance "
                                 "conversion.");
  }

  graph.reorderToSimplify(&dst, {&src}, false);
  const auto srcType = src.elementType();
  const auto dstType = dst.elementType();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getFloatVectorWidth();

  const auto mapping = graph.getTileMapping(src);
  const auto numTiles = target.getNumTiles();

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(src, mapping[tile]);
    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, vectorWidth, 2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      const auto numRegions = regions.size();
      VertexRef v;
      if (numRegions == 1) {
        const auto vertexName =
            srcType == dstType
                ? templateVertex("popops::BroadcastScalar1DSupervisor", op,
                                 srcType)
                : templateVertex("popops::BroadcastScalar2Types1DSupervisor",
                                 op, srcType, dstType);
        v = graph.addVertex(cs, vertexName);
        const auto &region = regions.front();
        graph.connect(v["data"], concat(src.slices(region)));
        graph.connect(v["out"], concat(dst.slices(region)));
        graph.connect(v["B"], epsilon.reshape({}));
      } else {
        const auto vertexName =
            srcType == dstType
                ? templateVertex("popops::BroadcastScalar2DData", op, srcType)
                : templateVertex("popops::BroadcastScalar2Types2DData", op,
                                 srcType, dstType);
        v = graph.addVertex(cs, vertexName);
        graph.connect(v["data"], src.slices(regions));
        graph.connect(v["out"], dst.slices(regions));
        graph.connect(v["B"], epsilon.reshape({}));
      }
      graph.setTileMapping(v, tile);
    };
  }
  return Execute(cs, {dnai});
}

Tensor varianceToInvStdDev(Graph &graph, const Tensor &src,
                           const Tensor &epsilon, Sequence &prog,
                           const Type dstType,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, epsilon, dstType));

  auto srcType = src.elementType();
  if (dstType != srcType && dstType != HALF) {
    throw poputil::poplibs_error("Cannot convert variance to inverse standard "
                                 "deviation using the data types provided.");
  }
  auto dst = graph.clone(dstType, src, {di, "varianceToInvStdDev"});
  prog.add(convertVariance(graph, src, dst, epsilon,
                           expr::BinaryOpType::VARIANCE_TO_INV_STD_DEV,
                           {di, "varianceToInvStdDev"}));
  di.addOutput(dst);
  return dst;
}

Tensor invStdDevToVariance(Graph &graph, const Tensor &src,
                           const Tensor &epsilon, Sequence &prog,
                           const Type dstType,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, epsilon, dstType));

  auto srcType = src.elementType();
  if (dstType != srcType && dstType != FLOAT) {
    throw poputil::poplibs_error("Cannot convert inverse standard deviation to"
                                 "variance using the data types provided.");
  }
  auto dst = graph.clone(dstType, src, {di, "invStdDevToVariance"});
  prog.add(convertVariance(graph, src, dst, epsilon,
                           expr::BinaryOpType::INV_STD_DEV_TO_VARIANCE,
                           {di, "invStdDevToVariance"}));
  di.addOutput(dst);
  return dst;
}

Tensor varianceToInvStdDev(Graph &graph, const Tensor &src, const float epsilon,
                           Sequence &prog, const Type dstType,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, epsilon, dstType));

  auto eps = graph.addConstant(src.elementType(), {}, epsilon, {di});
  graph.setTileMapping(eps, 0);
  auto output = varianceToInvStdDev(graph, src, eps, prog, dstType, {di});
  di.addOutput(output);
  return output;
}

Tensor invStdDevToVariance(Graph &graph, const Tensor &src, const float epsilon,
                           Sequence &prog, const Type dstType,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src, epsilon, dstType));

  auto eps = graph.addConstant(src.elementType(), {}, epsilon, {di});
  graph.setTileMapping(eps, 0);
  auto output = invStdDevToVariance(graph, src, eps, prog, dstType, {di});
  di.addOutput(output);
  return output;
}

} // end namespace popops
