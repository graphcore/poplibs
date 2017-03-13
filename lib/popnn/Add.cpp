#include "popnn/Add.hpp"

#include "Util.hpp"
#include "VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

void addTo(Graph &graph, Tensor A, Tensor B, float k,
           Sequence &prog, const std::string &debugPrefix) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto dType = graph.getTensorElementType(A);
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(A);
  const auto cs = graph.addComputeSet(debugPrefix + "/AddTo");

  const auto paramsFlat = A.flatten();
  const auto deltasFlat = B.flatten();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, mapping[tile],
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("popnn::ScaledAdd",
                                              dType),
                               {{"data", paramsFlat.slices(regions)},
                                {"deltas", deltasFlat.slices(regions)}});
      graph.setInitialValue(v["K"], k);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}



}
