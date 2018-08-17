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

  // how many elements in the encoded tensor refer to each index.
  const auto sliceLength = encodedShape[1];

  // Vertices to encode '1' at correct indices.
  // Also map resulting tensor to tile the index is mapped to.
  auto cs = graph.addComputeSet(layerPrefix + "/Encode");
  const auto mapping = graph.getTileMapping(inFlat);
  for (unsigned tile = 0; tile < numTiles; tile++) {
    const auto &tileRegions = mapping[tile];
    if (tileRegions.empty())
      continue;

    // loop through each index on this tile and gather the associated encoded
    // tensors together.
    std::vector<Tensor> encodedTensors;
    for (const auto &interval : tileRegions) {
      for (auto index = interval.begin(); index != interval.end(); ++index) {
        encodedTensors.push_back(encoded[index]);
      }
    }

    auto v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot",
                                                indexType,
                                                encoded.elementType()),
                             {{"indices", concat(inFlat.slices(tileRegions))},
                              {"out", concat(encodedTensors)}});
    graph.setInitialValue(v["sliceLength"], sliceLength);
    graph.setTileMapping(v, tile);
  }

  prog.add(Execute(cs));
}

} // end namespace popops
