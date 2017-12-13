#include "popstd/Cast.hpp"

#include <poplar/Graph.hpp>
#include "popstd/Util.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/TileMapping.hpp"

using namespace poplar;
using namespace poplar::program;

namespace popstd {

Program
cast(Graph &graph, Tensor src, Tensor dst, const std::string &debugPrefix) {
  auto srcType = src.elementType();
  auto dstType = dst.elementType();
  if (srcType == dstType)
    return Copy(src, dst);
  auto cs = graph.addComputeSet(debugPrefix + "/Cast");
  cast(graph, src, dst, cs);
  return Execute(cs);
}

#ifndef NDEBUG
static bool
mappingIsComplete(const Tensor &t,
                  const std::vector<
                    std::vector<Interval<std::size_t>>
                  > &mapping) {
  unsigned mappedElements = 0;
  for (const auto &regions : mapping) {
    for (const auto &region : regions) {
      mappedElements += region.end() - region.begin();
    }
  }
  return mappedElements == t.numElements();
}
#endif

void
cast(Graph &graph, Tensor src, Tensor dst, ComputeSet cs) {
  assert(src.shape() == dst.shape());
  src = src.flatten();
  dst = dst.flatten();
  graph.reorderToSimplify(&dst, {&src});
  const auto srcType = src.elementType();
  const auto dstType = dst.elementType();
  const auto &target = graph.getTarget();
  const auto vectorWidth = target.getFloatVectorWidth();
  std::vector<std::vector<Interval<std::size_t>>> mapping;
  Tensor t = srcType == FLOAT ? src : dst;
  mapping = graph.getTileMapping(t);
  assert(mappingIsComplete(t, mapping));
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(t, mapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   vectorWidth, 2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      const auto numRegions = regions.size();
      assert(numRegions != 0);
      VertexRef v;
      if (numRegions == 1) {
        v = graph.addVertex(cs, templateVertex("popstd::Cast", srcType,
                                               dstType));
        const auto &region = regions.front();
        graph.connect(v["src"], concat(src.slices(region)));
        graph.connect(v["dst"], concat(dst.slices(region)));
      } else {
        v = graph.addVertex(cs, templateVertex("popstd::Cast2d", srcType,
                                               dstType));
        graph.connect(v["src"], src.slices(regions));
        graph.connect(v["dst"], dst.slices(regions));
      }
      graph.setInitialValue(v["dataPathWidth"], target.getDataPathWidth());
      graph.setTileMapping(v, tile);
    };
  }
}

poplar::Tensor
cast(Graph &graph, const Tensor &src, const Type &dstType,
     Sequence &prog, const std::string &debugPrefix) {
  auto dst = graph.clone(dstType, src, "cast");
  prog.add(cast(graph, src, dst, debugPrefix));
  return dst;
}

} // end namespace popstd
