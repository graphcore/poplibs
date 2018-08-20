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
  const auto encodedShape = encoded.shape();
  if (encodedShape.size() != 2) {
    throw poputil::poplib_error("Tensor taking one-hot encoded output must be "
                                "2 dimensional");
  }
  const auto inputShape = indices.shape();
  if (encodedShape[0] != inputShape[0]) {
    throw poputil::poplib_error("Tensor taking one-hot encoded output must "
                                "have same number of rows as indices tensor.");
  }

  const auto indexType = indices.elementType();
  if (indexType != UNSIGNED_INT &&
      indexType != INT) {
    throw poputil::poplib_error("Index type must be integer type");
  }

  const std::string layerPrefix = debugPrefix + "/OneHot";
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  // how many elements in the encoded tensor refer to each index.
  const auto elemsPerBatch = encodedShape[1];
  const auto numIndices = encodedShape[0];

  const auto grainSize = target.getVectorWidth(encoded.elementType());
  // Have a minimum grains per tile in order for the overhead not to dominate
  // compute
  const auto minGrainsPerTile = 4UL;
  auto numBatchGrains = (elemsPerBatch + grainSize - 1) / grainSize;
  auto grainsPerTile =
      std::max((numBatchGrains * numIndices + numTiles - 1) / numTiles,
               minGrainsPerTile);

  unsigned encElem = 0;
  unsigned tile = 0U;
  auto cs = graph.addComputeSet(layerPrefix + "/OneHotEncode");

  while (encElem != encoded.numElements()) {
    auto tileEncElemStart = encElem;
    auto tileEncElemEnd = std::min(tileEncElemStart + grainsPerTile * grainSize,
                                   encoded.numElements());

    std::vector<Tensor> encThisTile;
    std::vector<Tensor> indicesThisTile;
    std::vector<unsigned> offsets;
    std::vector<unsigned> sliceLen;
    auto elemsThisTile = tileEncElemEnd - tileEncElemStart;
    // Elements in each slice must belong to the same index
    while (elemsThisTile) {
      const auto thisBatchStart = encElem % elemsPerBatch;
      const auto thisBatchEnd =
          std::min(thisBatchStart + elemsThisTile, elemsPerBatch);
      const auto thisIndex = encElem / elemsPerBatch;
      encThisTile.push_back(encoded[thisIndex]
                 .slice({thisBatchStart, thisBatchEnd}));
      indicesThisTile.push_back(indices[thisIndex].expand({0}));
      offsets.push_back(thisBatchStart);
      const auto elemsThisEntry = thisBatchEnd - thisBatchStart;
      sliceLen.push_back(elemsThisEntry);
      assert(elemsThisTile >= elemsThisEntry);
      elemsThisTile -= elemsThisEntry;
      encElem += elemsThisEntry;
    }

    auto offsetTensor =
          graph.addConstant(UNSIGNED_INT, {offsets.size()}, offsets.data());

    auto sliceLenTensor =
          graph.addConstant(UNSIGNED_INT, {sliceLen.size()}, sliceLen.data());
    auto outFlattened = concat(encThisTile);
    auto v = graph.addVertex(cs, templateVertex("popops::EncodeOneHot",
                                                 indexType,
                                                 encoded.elementType()),
                             {{"indices", concat(indicesThisTile)},
                              {"out", outFlattened},
                              {"offsets", offsetTensor},
                              {"sliceLength", sliceLenTensor}});
    // TODO: Note that outLength is the sum of the elements of vector
    // sliceLength and as an optimisation maybe removed
    graph.setInitialValue(v["outLength"], outFlattened.numElements());
    graph.setTileMapping(v, tile);

    if (++tile >= numTiles)
      tile = 0;
  }
  prog.add(Execute(cs));
}

} // end namespace popops
