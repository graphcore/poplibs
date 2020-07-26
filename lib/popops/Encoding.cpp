// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popops/Encoding.hpp"

#include "poplibs_support/logging.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Zero.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace logging = poplibs_support::logging;

namespace popops {

namespace {

// On and off are only needed by the "slow" path so they are set to null if we
// are on the normal path and we just ignore them in that case. If they are set
// we will go into the "slow" codelet.
void encodeOneHotBase(Graph &graph, const Tensor &indices,
                      const Tensor &encoded, Sequence &prog, const Tensor *on,
                      const Tensor *off, const std::string &debugPrefix) {
  const std::string layerPrefix = debugPrefix + "/OneHot";
  logging::info("encodeOneHot indices={}, encoded={}, name={}", indices.shape(),
                encoded.shape(), layerPrefix);

  // Verify inputs
  const auto encodedShape = encoded.shape();
  if (encodedShape.size() != 2) {
    throw poputil::poplibs_error("Tensor taking one-hot encoded output must be "
                                 "2 dimensional");
  }

  if (indices.rank() != 1) {
    throw poputil::poplibs_error("Tensor of one-hot indices must have rank 1");
  }

  const auto inputShape = indices.shape();
  if (encodedShape[0] != inputShape[0]) {
    throw poputil::poplibs_error("Tensor taking one-hot encoded output must "
                                 "have same number of rows as indices tensor.");
  }

  const auto indexType = indices.elementType();
  if (indexType != UNSIGNED_INT && indexType != INT) {
    throw poputil::poplibs_error("Index type must be integer type");
  }

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  // how many elements in the encoded tensor refer to each index.
  const auto elemsPerBatch = encodedShape[1];
  const auto numIndices = encodedShape[0];

  auto oneHotOutput =
      graph.addVariable(encoded.elementType(), {numIndices, elemsPerBatch},
                        debugPrefix + "/encodedOut");

  const auto grainSize = target.getVectorWidth(encoded.elementType());
  // Have a minimum grains per tile in order for the overhead not to dominate
  // compute
  const auto minGrainsPerTile = 4UL;
  auto numBatchGrains = (elemsPerBatch + grainSize - 1) / grainSize;
  auto grainsPerTile = std::max(numBatchGrains, minGrainsPerTile);

  unsigned encElem = 0;
  unsigned tile = 0U;
  auto cs = graph.addComputeSet(layerPrefix + "/OneHotEncode");

  while (encElem != elemsPerBatch) {
    auto tileEncElemStart = encElem;
    auto tileEncElemEnd =
        std::min(tileEncElemStart + grainsPerTile * grainSize, elemsPerBatch);
    auto elemsThisTile = tileEncElemEnd - tileEncElemStart;
    auto tileOutput =
        oneHotOutput.slice(tileEncElemStart, tileEncElemEnd, 1).flatten();

    poplar::VertexRef v;

    if (!on || !off) {
      // Normal path with On/Off hardcoded as 1/0.
      v = graph.addVertex(
          cs,
          templateVertex("popops::EncodeOneHot", indexType,
                         encoded.elementType()),
          {{"indices", indices.flatten()}, {"out", tileOutput}});
    } else {
      // "Slow" path which loads On/Off first then assigns them.
      v = graph.addVertex(cs,
                          templateVertex("popops::EncodeOneHotCustomValues",
                                         indexType, encoded.elementType()),
                          {{"indices", indices.flatten()},
                           {"out", tileOutput},
                           {"On", *on},
                           {"Off", *off}});
    }
    graph.setInitialValue(v["offset"], tileEncElemStart);
    graph.setInitialValue(v["sliceLength"], elemsThisTile);
    graph.setTileMapping(v, tile);
    graph.setTileMapping(tileOutput, tile);

    if (++tile >= numTiles) {
      tile = 0;
    }
    encElem += elemsThisTile;
  }
  if (!on || !off) { // Only zero out memory if we are using 0/1 encoding schema
    popops::zero(graph, oneHotOutput, prog, layerPrefix + "/zero");
  }
  prog.add(Execute(cs));
  auto oneHotOutputRegrouped = popops::rearrange::regroupIfBeneficial(
      graph, oneHotOutput, encoded, prog, debugPrefix + "/postRegroup");
  prog.add(Copy(oneHotOutputRegrouped, encoded));
}

} // Namespace

void encodeOneHot(Graph &graph, const Tensor &indices, const Tensor &encoded,
                  Sequence &prog, const std::string &debugPrefix) {

  // Mark "on" and "off" as null as we want to go down the default path which
  // has them hardcoded to 1 and 0 respectively.
  encodeOneHotBase(graph, indices, encoded, prog, nullptr, nullptr,
                   debugPrefix);
}

void encodeOneHot(Graph &graph, const Tensor &indices, const Tensor &encoded,
                  Sequence &prog, const Tensor &on, const Tensor &off,
                  const std::string &debugPrefix) {
  encodeOneHotBase(graph, indices, encoded, prog, &on, &off, debugPrefix);
}

template <typename T>
static void iotaCommon(Graph &graph, const Tensor &t, T startInteger,
                       Sequence &prog, const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/iota";
  const auto &dType = t.elementType();

  logging::info("iota t={}, start={}, name={}", t.shape(), startInteger,
                fnPrefix);

  // TODO: T12947 If the number of elements per tile is very small, it may be
  // better to construct a constant tensor and copy it.

  auto tFlat = t.flatten();
  auto numElements = t.numElements();

  // checks on tensor
  // Silently return for zero element tensor
  if (numElements == 0) {
    return;
  }

  if (t.rank() > 1) {
    throw poputil::poplibs_error("Rank of tensor must be <= 1");
  }

  auto tileMapping = graph.getTileMapping(tFlat);
  auto cs = graph.addComputeSet(fnPrefix);
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getVectorWidth(dType);

  for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
    if (tileMapping[tile].empty()) {
      continue;
    }

    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileMapping[tile], vectorWidth, 2 * vectorWidth);

    for (const auto &regions : vertexRegions) {
      if (regions.empty()) {
        continue;
      }

      // build index tensor
      const auto regionSize = regions.size();
      std::vector<unsigned> offsets(regionSize);
      for (unsigned i = 0; i != regionSize; ++i) {
        offsets[i] = regions[i].begin() + startInteger;
      }
      auto offsetTensor = graph.addConstant(dType, {regionSize}, offsets.data(),
                                            fnPrefix + "/offsets");
      graph.setTileMapping(offsetTensor, tile);
      auto v = graph.addVertex(
          cs, templateVertex("popops::Iota", dType),
          {{"out", tFlat.slices(regions)}, {"offsets", offsetTensor}});
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}

void iota(Graph &graph, const Tensor &t, unsigned startInteger, Sequence &prog,
          const std::string &debugPrefix) {
  if (t.elementType() != UNSIGNED_INT) {
    throw poputil::poplibs_error("Tensor element type doesn't match start "
                                 "integer type");
  }
  iotaCommon<unsigned>(graph, t, startInteger, prog, debugPrefix);
}

void iota(Graph &graph, const Tensor &t, int startInteger, Sequence &prog,
          const std::string &debugPrefix) {
  if (t.elementType() != INT) {
    throw poputil::poplibs_error("Tensor element type doesn't match start "
                                 "integer type");
  }
  iotaCommon<int>(graph, t, startInteger, prog, debugPrefix);
}

} // end namespace popops
