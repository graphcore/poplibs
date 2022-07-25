// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/NaN.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "popops/Zero.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

namespace popops {

using namespace poputil;

namespace logging = poplibs_support::logging;

static poplar::Tensor hasNaNOrInf(poplar::Graph &graph,
                                  const poplar::Tensor &src,
                                  poplar::program::Sequence &prog,
                                  bool hasNaNOrInf,
                                  poputil::PoplibsOpDebugInfo &di) {
  const auto &dType = src.elementType();
  const auto csClear = graph.addComputeSet({di, "hasNaNOrInf-clear"});
  const auto csEval = graph.addComputeSet({di, "hasNaNOrInf-eval"});

  auto srcFlat = src.flatten();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(dType);

  graph.reorderToSimplify(&srcFlat, {}, false);
  const auto srcFlatTileMap = graph.getTileMapping(srcFlat);

  const auto &tileMapping = graph.getTileMapping(srcFlat);
  unsigned nMappedTiles = 0;
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    if (!tileMapping[tile].empty()) {
      ++nMappedTiles;
    }
  }

  // We generate a Tensor of FLOAT(s) per tile then reduce to a single
  // value. This avoids the padding insertion and removal that would be
  // needed if we were to exchange bool values.
  std::vector<poplar::Tensor> allTileResults;
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    if (tileMapping[tile].empty()) {
      continue;
    }

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(srcFlat, tileMapping[tile]);

    const std::string vertexNamePrefix = "popops::HasNaNOrInf";
    if (tileContiguousRegions.size() == 1) {
      auto tileResultT =
          graph.addVariable(poplar::FLOAT, {1}, {di, "tileHasNaNOrInf"});
      allTileResults.emplace_back(tileResultT);
      graph.setTileMapping(tileResultT, tile);
      zero(graph, tileResultT, tile, csClear);
      const auto vertexName =
          templateVertex(vertexNamePrefix + "1D", dType, hasNaNOrInf);
      // Divide work
      const auto grainSize = 8 / target.getTypeSize(dType);
      const auto numWorkers = target.getNumWorkerContexts();
      auto t = concat(srcFlat.slices(tileContiguousRegions));
      const auto numElements = t.numElements();
      const auto numGrains = numElements / grainSize;
      auto extras = numElements - numGrains * grainSize;
      const auto vertex = graph.addVertex(
          csEval, vertexName, {{"in", t}, {"outSetIfFound", tileResultT[0]}});
      graph.setInitialValue(vertex["sizeIn8BytesPerWorker"],
                            numGrains / numWorkers);
      graph.setInitialValue(vertex["remWorkerId"], numGrains % numWorkers);
      graph.setInitialValue(vertex["remWorkerExtras"], extras);
      graph.setTileMapping(vertex, tile);
    } else {
      const auto vertexName =
          templateVertex(vertexNamePrefix + "2D", dType, hasNaNOrInf);
      const auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, vectorWidth, 2 * vectorWidth);
      auto tileResultT = graph.addVariable(
          poplar::FLOAT, {vertexRegions.size()}, {di, "tileHasNaNOrInf"});
      allTileResults.emplace_back(tileResultT);
      graph.setTileMapping(tileResultT, tile);
      auto regionIdx = 0;
      for (const auto &regions : vertexRegions) {
        assert(!regions.empty());

        const auto vertex =
            graph.addVertex(csEval, vertexName,
                            {{"in", srcFlat.slices(regions)},
                             {"out", tileResultT[regionIdx++]}});
        graph.setTileMapping(vertex, tile);
      }
    }
  }

  prog.add(poplar::program::Execute(csClear, {di}));
  prog.add(poplar::program::Execute(csEval, {di}));

  const auto max =
      popops::reduce(graph, concat(allTileResults), poplar::FLOAT, {0},
                     {popops::Operation::MAX}, prog, {di, "reduce"});
  const auto out = popops::cast(graph, max, poplar::BOOL, prog, di);

  di.addOutput(out);
  return out;
}

poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext) {
  logging::popops::info("hasNaN src={}, name={}", src.shape(),
                        debugContext.getPathName());
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src));
  return hasNaNOrInf(graph, src, prog, false, di);
}

poplar::Tensor hasNaNOrInf(poplar::Graph &graph, const poplar::Tensor &src,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  logging::popops::info("hasNaNOrInf src={}, name={}", src.shape(),
                        debugContext.getPathName());
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src));
  return hasNaNOrInf(graph, src, prog, true, di);
}

} // namespace popops
