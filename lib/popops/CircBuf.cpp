// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <numeric>
#include <popops/CircBuf.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Pad.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

CircBuf::CircBuf(Graph &graph, const Type &dataType, unsigned size,
                 const std::vector<std::size_t> &shape,
                 const poplar::DebugContext &debugContext)
    : graph(graph), size_(size), shape(shape) {
  const auto debugPrefix = debugContext.getPathName();
  auto N = std::accumulate(shape.begin(), shape.end(), 1UL,
                           std::multiplies<std::size_t>());
  unsigned grainSize = 4; // to allow 64bits/cycle for half/short
  auto nGrains = (N + grainSize - 1) / grainSize;
  padElements = nGrains * grainSize - N;

  hist = graph.addVariable(dataType, {nGrains, size, grainSize},
                           debugPrefix + "/CircBuf");
  auto numTiles = graph.getTarget().getNumTiles();
  auto regions = splitRegions({{0, nGrains}}, 1, numTiles);
  for (unsigned tile = 0; tile < regions.size(); ++tile) {
    const auto &tileRegions = regions[tile];
    for (const auto &r : tileRegions) {
      graph.setTileMapping(hist.slice(r), tile);
    }
  }

  index = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/CircBufIndex");
  graph.setInitialValue(index[0], 0);
  graph.setTileMapping(index, 0);
}

Graph::TileToTensorMapping CircBuf::getTileMapping() {
  return getSliceMapping(graph, hist, {1}, {1});
}

Tensor CircBuf::prev(unsigned i, Sequence &seq,
                     const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  if (i >= size_)
    std::abort();
  // compute required offset into an internal Tensor, prevIdx
  // this is mapped onto the tile where index is located
  Tensor prevIdx =
      graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/Offset");
  auto indexMapping = graph.getTileMapping(index);
  graph.setTileMapping(prevIdx, indexMapping);
  auto cs = graph.addComputeSet(debugPrefix + "/CircBufPrev");
  auto v = graph.addVertex(cs, "popops::CircOffset",
                           {{"indexIn", index[0]}, {"indexOut", prevIdx[0]}});
  graph.setInitialValue(v["hSize"], size_);
  graph.setInitialValue(v["offset"], size_ - i);
  graph.setTileMapping(v, indexMapping[0][0].begin());
  seq.add(Execute(cs));
  auto t = dynamicSlice(graph, hist, prevIdx, {1}, {1}, seq,
                        debugPrefix + "/CircBuf");
  t = popops::pad(graph, t.flatten(), 0,
                  -static_cast<std::ptrdiff_t>(padElements), 0);
  return t.reshape(shape);
}

void CircBuf::add(Tensor in, Sequence &seq,
                  const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  assert(in.shape() == shape);
  ComputeSet csIndexIncr = graph.addComputeSet(debugPrefix + "/CircBufSet");
  auto v = graph.addVertex(csIndexIncr, "popops::CircBufIncrIndex",
                           {{"index", index[0]}});
  graph.setInitialValue(v["hSize"], size_);
  graph.setTileMapping(v, 0);
  seq.add(Execute(csIndexIncr));
  // Inserting data into the circular buffer requires extra padding elements.
  // They will be discarded when extracted from the buffer.
  in = popops::pad(graph, in.flatten(), 0, padElements, 0);
  dynamicUpdate(graph, hist, in.reshape({hist.dim(0), 1, hist.dim(2)}), index,
                {1}, {1}, seq, debugPrefix + "/CircBuf");
}

poplar::Tensor CircBuf::getIndex() const { return index; }
unsigned CircBuf::size() const { return size_; }

} // namespace popops
