#include <popstd/CircBuf.hpp>
#include <popstd/Util.hpp>
#include <popstd/VertexTemplates.hpp>
#include <popstd/Operations.hpp>
#include <popstd/DynamicSlice.hpp>
#include <popstd/Pad.hpp>
#include <numeric>
#include <algorithm>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popstd {

CircBuf::CircBuf(Graph &graph, const Type &dataType,
                 unsigned size, const std::vector<std::size_t> &shape,
                 const std::string &debugPrefix) :
  graph(graph), size_(size), shape(shape) {
  auto N = std::accumulate(shape.begin(), shape.end(), 1UL,
                          std::multiplies<std::size_t>());
  unsigned grainSize = 4; // to allow 64bits/cycle for half/short
  auto nGrains = (N + grainSize - 1) / grainSize;
  padElements = nGrains * grainSize - N;

  hist = graph.addTensor(dataType, {nGrains, size, grainSize},
                         debugPrefix + "/CircBuf");
  auto numTiles = graph.getTarget().getNumTiles();
  auto regions = splitRegions({{0, nGrains}}, 1, numTiles);
  for (unsigned tile = 0; tile < regions.size(); ++tile) {
    const auto &tileRegions = regions[tile];
    for (const auto &r : tileRegions) {
      graph.setTileMapping(hist.slice(r), tile);
    }
  }

  index = graph.addTensor(UNSIGNED_INT, {1}, debugPrefix + "/CircBufIndex");
  graph.setInitialValue(index[0], 0);
  graph.setTileMapping(index, 0);
}

Tensor CircBuf::prev(unsigned i, Sequence &seq,
                     const std::string &debugPrefix) {
  if (i >= size_)
    std::abort();
  // compute required offset into an internal Tensor, prevIdx
  // this is mapped onto the tile where index is located
  Tensor prevIdx = graph.addTensor(UNSIGNED_INT, {1}, debugPrefix + "/Offset");
  auto indexMapping = graph.getTileMapping(index);
  graph.setTileMapping(prevIdx, indexMapping);
  auto cs = graph.addComputeSet(debugPrefix + "/CircBufPrev");
  auto v = graph.addVertex(cs, "popstd::CircOffset",
                           {{"indexIn", index[0]},
                            {"indexOut", prevIdx[0]}});
  graph.setInitialValue(v["hSize"], size_);
  graph.setInitialValue(v["offset"], size_ - i);
  graph.setTileMapping(v, indexMapping[0][0].begin());
  seq.add(Execute(cs));
  auto t = dynamicSlice(graph, hist, prevIdx, {1}, {1}, seq,
                        debugPrefix + "/CircBuf");
  t = popstd::pad(graph, t.flatten(), 0, -padElements, 0);
  return t.reshape(shape);
}

void CircBuf::add(Tensor in, Sequence &seq, const std::string &debugPrefix) {
  assert(in.shape() == shape);
  ComputeSet csIndexIncr = graph.addComputeSet(debugPrefix + "/CircBufSet");
  auto v = graph.addVertex(csIndexIncr, "popstd::CircBufIncrIndex",
                          {{"index", index[0]}});
  graph.setInitialValue(v["hSize"], size_);
  graph.setTileMapping(v, 0);
  seq.add(Execute(csIndexIncr));
  // Inserting data into the circular buffer requires extra padding elements.
  // They will be discarded when extracted from the buffer.
  in = popstd::pad(graph, in.flatten(), 0, padElements, 0);
  dynamicUpdate(graph, hist, in.reshape({hist.dim(0), 1, hist.dim(2)}), index,
                {1}, {1}, seq, debugPrefix + "/CircBuf");
}

}
