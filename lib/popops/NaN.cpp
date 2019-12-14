// Copyright (c) Graphcore Ltd, All rights reserved.
#include "popops/NaN.hpp"

#include "poplibs_support/logging.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

namespace popops {

using namespace poputil;

namespace logging = poplibs_support::logging;

poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix) {
  logging::info("hasNaN src={}, name={}", src.shape(), debugPrefix);

  const auto cs = graph.addComputeSet(debugPrefix + "/hasNaN");
  const auto vertexName = templateVertex("popops::HasNaN", src.elementType());

  const auto srcFlat = src.flatten();

  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(src.elementType());

  const auto &tileMapping = graph.getTileMapping(src);
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    if (tileMapping[tile].empty()) {
      continue;
    }

    const auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileMapping[tile], vectorWidth, 2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      assert(!regions.empty());

      const auto vertex = graph.addVertex(cs, vertexName,
                                          {
                                              {"in", srcFlat.slices(regions)},
                                          });
      graph.setTileMapping(vertex, tile);
    }
  }

  const auto out = graph.addVariable(poplar::BOOL, {1});
  graph.setTileMapping(out, 0);

  prog.add(poplar::program::Execute(cs, out[0]));

  // TODO: T12949 Improve efficiency. This could be achieved by inverting this
  // function (ie. change it to `hasNoNaNs`); but a more intuitive solution is
  // to add support to the Execute program to invert the consensus bit before
  // writing it to the out tensor.
  popops::logicalNotInPlace(graph, out, prog, debugPrefix + "/hasNaN");

  return out;
}

} // namespace popops
