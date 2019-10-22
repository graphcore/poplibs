#include "popops/NaN.hpp"

#include "popops/ElementWise.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

namespace popops {

using namespace poputil;

poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix) {
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

  // TODO: this isn't very efficient, we could invert this function (ie. change
  // it to `hasNoNaNs`) but that feels unintuitive. A better solution would be
  // to add support to the Execute program to invert the consensus bit before
  // writing it to the out tensor.
  popops::logicalNotInPlace(graph, out, prog, debugPrefix + "/hasNaN");

  return out;
}

} // namespace popops
