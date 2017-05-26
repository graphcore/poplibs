#include <popstd/HadamardProduct.hpp>


#include "popstd/Util.hpp"
#include "popstd/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

void hadamardProduct(Graph &graph, Tensor A, Tensor B,
                     Sequence &prog, const std::string &debugPrefix) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto dType = A.elementType();
  const auto numTiles = deviceInfo.getNumTiles();
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
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(A, mapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popstd::HadamardProd",
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
