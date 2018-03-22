#include "popops/Add.hpp"

#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

void addTo(Graph &graph, Tensor A, Tensor B, float k,
           Sequence &prog, const std::string &debugPrefix) {
  if (!A.isParallelWriteable())
    throw poputil::poplib_error("Trying to accumulate to tensor that cannot be "
                               "written in parallel");
  const auto &target = graph.getTarget();
  const auto dType = A.elementType();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");

  auto aFlat = A.flatten();
  auto bFlat = B.flatten();
  graph.reorderToSimplify(&aFlat, {&bFlat});
  const auto mapping = graph.getTileMapping(aFlat);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(aFlat, mapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popops::ScaledAdd",
                                              dType),
                               {{"data", aFlat.slices(regions)},
                                {"deltas", bFlat.slices(regions)}});
      graph.setInitialValue(v["K"], k);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}


void addTo(Graph &graph, Tensor A, Tensor B,
           Sequence &prog, const std::string &debugPrefix) {
  addTo(graph, A, B, 1.0, prog, debugPrefix);
}

}
