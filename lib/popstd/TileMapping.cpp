#include "popstd/TileMapping.hpp"

#include "popstd/Util.hpp"
#include "popstd/exceptions.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include "util/gcd.hpp"

namespace popstd {

std::vector<std::vector<poplar::Interval<std::size_t>>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile,
                      unsigned grainSize) {
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto numElements = std::accumulate(shape.begin(), shape.end(), 1UL,
                                           std::multiplies<std::size_t>());
  std::vector<poplar::Interval<std::size_t>> regions = {
    {0, numElements}
  };
  return splitRegions(regions, grainSize, numTiles, minElementsPerTile);
}

std::vector<std::vector<poplar::Interval<std::size_t>>>
calcLinearTileMapping(const poplar::Graph &graph,
                      const poplar::Tensor &t) {
  const auto dType = t.elementType();
  // TODO - this get the correct type size from Poplar when poplar allows
  // that type of introspection.
  const auto &target = graph.getTarget();
  const auto typeSize = dType == "half" ? 2 : 4;
  unsigned grainSize = dType == "half" ? target.getHalfVectorWidth() :
                                         target.getFloatVectorWidth();
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + typeSize - 1) / minBytesPerTile;
  return calcLinearTileMapping(graph, t.shape(), minElementsPerTile,
                               grainSize);
}

void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                  unsigned minElementsPerTile ,
                  unsigned grainSize) {
  graph.setTileMapping(t, calcLinearTileMapping(graph, t.shape(),
                                                minElementsPerTile, grainSize));
}

void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t) {
  graph.setTileMapping(t, calcLinearTileMapping(graph, t));
}

} // end namespace popstd
