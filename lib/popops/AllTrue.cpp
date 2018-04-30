#include "popops/AllTrue.hpp"

#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

Tensor allTrue(Graph &graph, Tensor in, Sequence &prog,
               const std::string &debugPrefix) {
  const auto inType = in.elementType();

  if (inType != BOOL) {
    throw poputil::poplib_error("Operation allTrue only takes boolean tensors");
  }
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix);

  auto inFlat = in.flatten();
  graph.reorderToSimplify(&inFlat, {});

  const auto mapping = graph.getTileMapping(inFlat);
  const auto grainSize = target.getVectorWidth(inType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
            graph.getSortedContiguousRegions(inFlat, mapping[tile]);
    auto vertexRegions =
            splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                       grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               "popops::AllTrue",
                               {{"in", inFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  auto predicate = graph.addVariable(BOOL, {}, debugPrefix + "/predicate");
  graph.setTileMapping(predicate, 0);
  prog.add(Execute(cs, predicate));
  return predicate;
}

};
