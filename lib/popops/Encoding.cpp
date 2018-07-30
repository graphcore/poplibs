#include "popops/Encoding.hpp"

#include "popops/Zero.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/Util.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

Tensor
encodeOneHot(Graph &graph,
             const poplar::Type &encodedType,
             const poplar::Tensor &indices,
             unsigned length,
             Sequence &prog,
             const std::string &debugPrefix) {
  // Verify inputs
  const auto inputShape = indices.shape();
  if (inputShape.size() > 1) {
    throw poputil::poplib_error("Dimensions of tensor containing indices "
                                "for one-hot encoding exceed 1 dimension");
  }

  const auto indexType = indices.elementType();
  if (indexType != UNSIGNED_INT &&
      indexType != INT) {
    throw poputil::poplib_error("Index type must be integer type");
  }

  const std::string layerPrefix = debugPrefix + "/OneHot";
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numIndices = indices.dim(0);

  auto encoded = graph.addVariable(encodedType,
                                   {numIndices, length},
                                   layerPrefix + "/Encoded");

  // Vertices to encode '1' at correct indices.
  // Also map resulting tensor to tile the index is mapped to.
  auto cs = graph.addComputeSet(layerPrefix + "/Encode");
  const auto grainSize = target.getVectorWidth(encodedType);
  const auto mapping = graph.getTileMapping(indices);
  for (unsigned tile = 0; tile < numTiles; tile++) {
    const auto &tileRegions = mapping[tile];
    if (tileRegions.empty())
      continue;
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(indices, tileRegions);
    const auto perVertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, grainSize * 2);

    for (const auto &regions : perVertexRegions) {
      VertexRef v;
      // For the common case of just one index encoded by a particular
      // vertex on a tile, this is special cased to reduce memory footprint
      if (regions.size() == 1 &&
          regions[0].size() == 1 &&
          regions[0][0].size() == 1) {
        const auto index = regions[0][0].begin();
        auto vertexEncoded = encoded[index];
        v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot",
                                               indices.elementType(),
                                               encodedType),
                            {{"index", indices[index]},
                             {"out", vertexEncoded}});
        graph.setTileMapping(vertexEncoded, tile);
      } else {
        auto vertexEncoded = concat(encoded.slices(regions));
        v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot2D",
                                               indices.elementType(),
                                               encodedType),
                            {{"indices", concat(indices.slices(regions))},
                             {"out", vertexEncoded}});
        graph.setTileMapping(vertexEncoded, tile);
      }
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return encoded;
}

} // end namespace popops
