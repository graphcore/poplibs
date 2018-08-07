#include "popops/Encoding.hpp"

#include "popops/Zero.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/Util.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

void encodeOneHot(Graph &graph,
                  const Tensor &indices,
                  const Tensor &encoded,
                  Sequence &prog,
                  const std::string &debugPrefix) {
  // Verify inputs
  auto inFlat = indices.flatten();
  const auto encodedShape = encoded.shape();
  if (encodedShape.size() != 2) {
    throw poputil::poplib_error("Tensor taking one-hot encoded output must be "
                                "2 dimensional");
  }
  const auto inputShape = inFlat.shape();
  if (encodedShape[0] != inputShape[0]) {
    throw poputil::poplib_error("Tensor taking one-hot encoded output must "
                                "have same number of rows as indices tensor.");
  }

  const auto indexType = inFlat.elementType();
  if (indexType != UNSIGNED_INT &&
      indexType != INT) {
    throw poputil::poplib_error("Index type must be integer type");
  }

  const std::string layerPrefix = debugPrefix + "/OneHot";
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  // Vertices to encode '1' at correct indices.
  // Also map resulting tensor to tile the index is mapped to.
  auto cs = graph.addComputeSet(layerPrefix + "/Encode");
  const auto grainSize = target.getVectorWidth(encoded.elementType());
  const auto mapping = graph.getTileMapping(inFlat);
  for (unsigned tile = 0; tile < numTiles; tile++) {
    const auto &tileRegions = mapping[tile];
    if (tileRegions.empty())
      continue;
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(inFlat, tileRegions);
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
        v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot",
                                               inFlat.elementType(),
                                               encoded.elementType()),
                            {{"index", inFlat[index]},
                             {"out", encoded[index]}});
      } else {
        v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot2D",
                                               inFlat.elementType(),
                                               encoded.elementType()),
                            {{"indices", concat(inFlat.slices(regions))},
                             {"out", concat(encoded.slices(regions))}});
      }
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}

} // end namespace popops
