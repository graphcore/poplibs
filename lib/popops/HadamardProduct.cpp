#include "popops/HadamardProduct.hpp"

#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace poputil {

void hadamardProduct(Graph &graph, Tensor A, Tensor B,
                     Sequence &prog, const std::string &debugPrefix) {
  if (!A.isParallelWriteable())
    throw poputil::poplib_error("Trying to write to tensor that cannot be "
                               "written in parallel");
  const auto &target = graph.getTarget();
  const auto dataPathWidth = target.getDataPathWidth();
  const auto dType = A.elementType();
  const auto numTiles = target.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/HadamardProd");

  const auto aFlat = A.flatten();
  const auto bFlat = B.flatten();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(A, mapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popops::HadamardProd",
                                              dType),
                               {{"A", aFlat.slices(regions)},
                                {"B", bFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}

}
