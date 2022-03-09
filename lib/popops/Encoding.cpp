// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popops/Encoding.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Zero.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace popops {

namespace {

// On and off are only needed by the "slow" path so they are set to null if we
// are on the normal path and we just ignore them in that case. If they are set
// we will go into the "slow" codelet.
void encodeOneHotBase(Graph &graph, const Tensor &indices,
                      const Tensor &encoded, Sequence &prog, const Tensor *on,
                      const Tensor *off, const DebugNameAndId &dnai) {
  const std::string layerPrefix = "OneHot";
  logging::popops::info("encodeOneHot indices={}, encoded={}, name={}",
                        indices.shape(), encoded.shape(),
                        dnai.getPathName() + "/" + layerPrefix);

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
  const std::size_t numTiles = target.getNumTiles();

  // how many elements in the encoded tensor refer to each index.
  const auto elemsPerBatch = encodedShape[1];
  const auto numIndices = encodedShape[0];

  auto oneHotOutput = graph.addVariable(
      encoded.elementType(), {numIndices, elemsPerBatch}, {dnai, "encodedOut"});

  // We want to maximise the innermost dimension sizes for one hot encoding
  // as we don't divide indices amongst workers on a tile. We form grains
  // for both the dimension to allow nice groupings to be formed if the
  // sizes in each dimension allow to do so. We could use a planning system
  // and use cost estimates but is not done here to avoid costs based on
  // introspection.
  std::size_t minGrainsPerTile = 4UL;
  std::size_t grainSize = target.getVectorWidth(encoded.elementType());

  // number of grains of indices
  const auto numIndicesGrains = gccs::ceildiv(numIndices, grainSize);

  // Form groups containing grains of indices. Spread these over as many
  // tiles as possible.
  const auto indicesGrainsPerGroup = gccs::ceildiv(numIndicesGrains, numTiles);

  // Number of indices groups that can be formed given the grain size
  const auto numIndicesGroups =
      gccs::ceildiv(numIndicesGrains, indicesGrainsPerGroup);

  // Use the unused tile factor to allocate per batch groups. We do a factored
  // allocation to keep the vertex simple as otherwise we would have to provide
  // more information on start of indices.
  const auto numPerBatchGroups = std::max(numTiles / numIndicesGroups, 1UL);

  // Number of per batch grains that can be formed
  const auto numPerBatchGrains = gccs::ceildiv(elemsPerBatch, grainSize);
  const auto perBatchGrainsPerGroup = std::max(
      gccs::ceildiv(numPerBatchGrains, numPerBatchGroups), minGrainsPerTile);

  auto cs = graph.addComputeSet({dnai, layerPrefix + "/OneHotEncode"});
  const bool nonCustomValues = !on || !off;

  logging::popops::debug(
      "encodeOneHot: Indices : grains {}, groups {}, per group {}",
      numIndicesGrains, numIndicesGroups, indicesGrainsPerGroup);
  logging::popops::debug(
      "            : per-batch : grains {}, groups {} per group {}",
      numPerBatchGrains, numPerBatchGroups, perBatchGrainsPerGroup);

  unsigned tile = 0U;
  for (unsigned group = 0; group != numIndicesGroups; ++group) {
    unsigned encElem = 0;
    const auto indicesStart =
        std::min(group * indicesGrainsPerGroup * grainSize, numIndices);
    const auto indicesEnd =
        std::min((group + 1) * indicesGrainsPerGroup * grainSize, numIndices);

    const auto numIndicesThisTile = indicesEnd - indicesStart;
    if (numIndicesThisTile == 0) {
      continue;
    }

    while (encElem != elemsPerBatch) {
      auto tileEncElemStart = encElem;
      auto tileEncElemEnd = std::min(
          tileEncElemStart + perBatchGrainsPerGroup * grainSize, elemsPerBatch);
      auto elemsThisTile = tileEncElemEnd - tileEncElemStart;
      logging::popops::debug("  tile : {}, indices [{}:{}), per-batch [{}:{})",
                             tile, indicesStart, indicesEnd, tileEncElemStart,
                             tileEncElemEnd);
      auto tileOutput = oneHotOutput
                            .slice({indicesStart, tileEncElemStart},
                                   {indicesEnd, tileEncElemEnd})
                            .flatten();
      auto tileIndices = indices.slice(indicesStart, indicesEnd).flatten();

      poplar::VertexRef v;

      if (nonCustomValues) {
        // Normal path with On/Off hardcoded as 1/0.
        v = graph.addVertex(cs,
                            templateVertex("popops::EncodeOneHot", indexType,
                                           encoded.elementType()),
                            {{"indices", tileIndices}, {"out", tileOutput}});
      } else {
        // "Slow" path which loads On/Off first then assigns them.
        v = graph.addVertex(cs,
                            templateVertex("popops::EncodeOneHotCustomValues",
                                           indexType, encoded.elementType()),
                            {{"indices", tileIndices},
                             {"out", tileOutput},
                             {"On", *on},
                             {"Off", *off}});
        // TODO: T12944 Note that outLength is the sum of the elements of vector
        // sliceLength and as an optimisation maybe removed.
        graph.setInitialValue(v["outLength"], tileOutput.numElements());
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
  }
  if (!on || !off) { // Only zero out memory if we are using 0/1 encoding schema
    popops::zero(graph, oneHotOutput, prog, {dnai, layerPrefix + "/zero"});
  }
  prog.add(Execute(cs, {dnai}));

  auto oneHotOutputRegrouped = popops::rearrange::regroupIfBeneficial(
      graph, oneHotOutput, encoded, prog, {dnai, "postRegroup"});

  prog.add(Copy(oneHotOutputRegrouped, encoded, false, {dnai}));
}

} // Namespace

void encodeOneHot(Graph &graph, const Tensor &indices, const Tensor &encoded,
                  Sequence &prog, const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(indices, encoded));
  // Mark "on" and "off" as null as we want to go down the default path which
  // has them hardcoded to 1 and 0 respectively.
  encodeOneHotBase(graph, indices, encoded, prog, nullptr, nullptr, {di});
}

void encodeOneHot(Graph &graph, const Tensor &indices, const Tensor &encoded,
                  Sequence &prog, const Tensor &on, const Tensor &off,
                  const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(indices, encoded, on, off));
  encodeOneHotBase(graph, indices, encoded, prog, &on, &off, {di});
}

template <typename T>
static void iotaCommon(Graph &graph, const Tensor &t, T startInteger,
                       Sequence &prog, const DebugNameAndId &dnai) {
  const std::string fnPrefix = "iota";
  const auto &dType = t.elementType();

  logging::popops::info("iota t={}, start={}, name={}", t.shape(), startInteger,
                        dnai.getPathName() + "/" + fnPrefix);

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
  auto cs = graph.addComputeSet({dnai, fnPrefix});
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
                                            {dnai, fnPrefix + "/offsets"});
      graph.setTileMapping(offsetTensor, tile);
      auto v = graph.addVertex(
          cs, templateVertex("popops::Iota", dType),
          {{"out", tFlat.slices(regions)}, {"offsets", offsetTensor}});
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs, {dnai}));
}

void iota(Graph &graph, const Tensor &t, unsigned startInteger, Sequence &prog,
          const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, startInteger));

  if (t.elementType() != UNSIGNED_INT) {
    throw poputil::poplibs_error("Tensor element type doesn't match start "
                                 "integer type");
  }
  iotaCommon<unsigned>(graph, t, startInteger, prog, {di});
}

void iota(Graph &graph, const Tensor &t, int startInteger, Sequence &prog,
          const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, startInteger));
  if (t.elementType() != INT) {
    throw poputil::poplibs_error("Tensor element type doesn't match start "
                                 "integer type");
  }
  iotaCommon<int>(graph, t, startInteger, prog, {di});
}

} // end namespace popops
