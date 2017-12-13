#include "popstd/Zero.hpp"

#include <poplar/Graph.hpp>
#include "popstd/Util.hpp"
#include "popstd/VertexTemplates.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

void
zero(poplar::Graph &graph,
     poplar::Tensor t,
     const std::vector<poplar::Interval<std::size_t>> &tileRegions,
     unsigned tile,
     poplar::ComputeSet zeroCS) {
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  const auto tFlat = t.flatten();
  const auto vectorWidth = target.getVectorWidth(dType);
  const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(t, tileRegions);
  auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 vectorWidth, 2 * vectorWidth);
  for (const auto &regions : vertexRegions) {
    const auto numRegions = regions.size();
    VertexRef v;
    if (numRegions == 1) {
      v = graph.addVertex(zeroCS, templateVertex("popstd::Zero", dType));
      const auto &region = regions.front();
      auto out = concat(tFlat.slices(region));
      graph.connect(v["out"], out);
    } else {
      v = graph.addVertex(zeroCS, templateVertex("popstd::Zero2d", dType));
      auto out = tFlat.slices(regions);
      graph.connect(v["out"], out);
    }
    graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
    graph.setTileMapping(v, tile);
  }
}

void
zero(Graph &graph,
     const Tensor &t,
     const std::vector<
       std::vector<Interval<std::size_t>>
     > &mapping,
     ComputeSet zeroCS) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    zero(graph, t, mapping[tile], tile, zeroCS);
  }
}

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix) {
  auto cs = graph.addComputeSet(debugPrefix + "/Zero");
  auto tFlat = t.flatten();
  graph.reorderToSimplify(&tFlat, {});
  zero(graph, tFlat, graph.getTileMapping(tFlat), cs);
  prog.add(Execute(cs));
}

} // end namespace popstd
