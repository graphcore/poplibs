// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popops/Zero.hpp"

#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include <poplar/Graph.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

void zero(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet zeroCS) {
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  const auto tFlat = t.flatten();
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(t, tileRegions);

  auto width = target.getDataPathWidth() / ((dType == HALF) ? 16 : 32);
  auto vertexRegions = splitRegionsBetweenWorkers(
      target, tileContiguousRegions, vectorWidth, 2 * vectorWidth,
      target.getRptCountMax() * width);

  for (const auto &regions : vertexRegions) {
    const auto numRegions = regions.size();
    VertexRef v;
    if (numRegions == 1) {
      v = graph.addVertex(zeroCS, templateVertex("popops::Zero", dType));
      const auto &region = regions.front();
      auto out = concat(tFlat.slices(region));
      graph.connect(v["out"], out);
    } else {
      v = graph.addVertex(zeroCS, templateVertex("popops::Zero2d", dType));
      auto out = tFlat.slices(regions);
      graph.connect(v["out"], out);
    }
    graph.setTileMapping(v, tile);
  }
}

void zero(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet zeroCS) {
  return zero(graph, t, {{0, t.numElements()}}, tile, zeroCS);
}

void zero(Graph &graph, const Tensor &t,
          const std::vector<std::vector<Interval>> &mapping,
          ComputeSet zeroCS) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    zero(graph, t, mapping[tile], tile, zeroCS);
  }
}

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, const std::string &debugPrefix) {
  auto cs = graph.addComputeSet(debugPrefix + "/Zero");
  auto tFlat = t.flatten();
  graph.reorderToSimplify(&tFlat, {});
  zero(graph, tFlat, graph.getTileMapping(tFlat), cs);
  prog.add(Execute(cs));
}

} // end namespace popops
