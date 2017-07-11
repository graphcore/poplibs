#include "OperationsDef.hpp"
#include "popstd/Operations.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Util.hpp"
#include "popstd/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

void allTrue(Graph &graph, Tensor in, Sequence &prog,
             const std::string &debugPrefix) {

  const auto inType = in.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto mapping = graph.getTileMapping(in);
  const auto cs = graph.addComputeSet(debugPrefix);

  if (inType != "bool") {
    throw popstd::poplib_error("Operation allTrue only takes boolean tensors");
  }

  auto inFlat = in.flatten();

  const auto grainSize = deviceInfo.getVectorWidth(inType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
            graph.getSortedContiguousRegions(inFlat, mapping[tile]);
    auto vertexRegions =
            splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                       grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               "popstd::AllTrue",
                               {{"in", inFlat.slices(regions)}});
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}

};
