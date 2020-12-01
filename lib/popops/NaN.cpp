// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/NaN.hpp"

#include "poplibs_support/logging.hpp"
#include "popops/ElementWise.hpp"
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
  const auto cs = graph.addComputeSet({di, "hasNaNOrInf"});

  auto srcFlat = src.flatten();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(dType);

  graph.reorderToSimplify(&srcFlat, {}, false);
  const auto srcFlatTileMap = graph.getTileMapping(srcFlat);

  const auto &tileMapping = graph.getTileMapping(srcFlat);
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    if (tileMapping[tile].empty()) {
      continue;
    }

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(srcFlat, tileMapping[tile]);

    const std::string vertexNamePrefix = "popops::HasNaNOrInf";
    if (tileContiguousRegions.size() == 1) {
      const auto vertexName =
          templateVertex(vertexNamePrefix + "Supervisor", dType, hasNaNOrInf);
      // Divide work
      const auto grainSize = 8 / target.getTypeSize(dType);
      const auto numWorkers = target.getNumWorkerContexts();
      auto t = concat(srcFlat.slices(tileContiguousRegions));
      const auto numElements = t.numElements();
      const auto numGrains = numElements / grainSize;
      auto extras = numElements - numGrains * grainSize;
      const auto vertex = graph.addVertex(cs, vertexName,
                                          {
                                              {"in", t},
                                          });
      graph.setInitialValue(vertex["sizeIn8BytesPerWorker"],
                            numGrains / numWorkers);
      graph.setInitialValue(vertex["remWorkerId"], numGrains % numWorkers);
      graph.setInitialValue(vertex["remWorkerExtras"], extras);
      graph.setTileMapping(vertex, tile);
    } else {
      const auto vertexName =
          templateVertex(vertexNamePrefix, dType, hasNaNOrInf);
      const auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, vectorWidth, 2 * vectorWidth);
      for (const auto &regions : vertexRegions) {
        assert(!regions.empty());

        const auto vertex = graph.addVertex(cs, vertexName,
                                            {
                                                {"in", srcFlat.slices(regions)},
                                            });
        graph.setTileMapping(vertex, tile);
      }
    }
  }

  const auto out = graph.addVariable(poplar::BOOL, {1}, {di});
  graph.setTileMapping(out, 0);

  prog.add(poplar::program::Execute(cs, out[0], {di}));

  // TODO: T12949 Improve efficiency. This could be achieved by inverting this
  // function (ie. change it to `hasNoNaNs`); but a more intuitive solution is
  // to add support to the Execute program to invert the consensus bit before
  // writing it to the out tensor.
  popops::logicalNotInPlace(graph, out, prog, {di, "hasNaN"});
  di.addOutput(out);
  return out;
}

poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext) {
  logging::popops::info("hasNaN src={}, name={}", src.shape(),
                        debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src));
  return hasNaNOrInf(graph, src, prog, false, di);
}

poplar::Tensor hasNaNOrInf(poplar::Graph &graph, const poplar::Tensor &src,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  logging::popops::info("hasNaNOrInf src={}, name={}", src.shape(),
                        debugContext.getPathName());
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(src));
  return hasNaNOrInf(graph, src, prog, true, di);
}

} // namespace popops
