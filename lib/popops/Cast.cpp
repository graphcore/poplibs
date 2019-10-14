#include "popops/Cast.hpp"

#include <poplar/Graph.hpp>
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/TileMapping.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

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
  std::vector<std::vector<Interval>> mapping;
  Tensor t = srcType == FLOAT ? src : dst;
  mapping = graph.getTileMapping(t);
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
        v = graph.addVertex(cs, templateVertex("popops::Cast", srcType,
                                               dstType));
        const auto &region = regions.front();
        graph.connect(v["src"], concat(src.slices(region)));
        graph.connect(v["dst"], concat(dst.slices(region)));
        graph.setInitialValue(v["numElems"], region[0].size());
      } else {
        v = graph.addVertex(cs, templateVertex("popops::Cast2d", srcType,
                                               dstType));
        graph.connect(v["src"], src.slices(regions));
        graph.connect(v["dst"], dst.slices(regions));
      }
      graph.setTileMapping(v, tile);
    };
  }
}

Tensor
cast(Graph &graph, Tensor src, const Type &dstType, ComputeSet cs,
     const std::string &debugPrefix) {
  auto dst = graph.clone(dstType, src, debugPrefix + "/cast");
  cast(graph, src, dst, cs);
  return dst;
}

poplar::Tensor
cast(Graph &graph, const Tensor &src, const Type &dstType,
     Sequence &prog, const std::string &debugPrefix) {
  auto dst = graph.clone(dstType, src, debugPrefix + "/cast");
  prog.add(cast(graph, src, dst, debugPrefix));
  return dst;
}

poplar::Tensor
checkAccuracyInHalfPrecision(Graph &graph, const Tensor &value,
                             float tolerance, poplar::program::Sequence &prog,
                             const std::string &debugPrefix) {
  if (value.elementType() != FLOAT || value.numElements() != 1) {
    throw poputil::poplibs_error("Can only check the accuracy of single element"
                                 " tensors with data type float.");
  }

  auto cs = graph.addComputeSet(debugPrefix + "/checkAccuracyInHalfPrecision");
  auto v = graph.addVertex(cs, "popops::CheckAccuracyInHalfPrecision");
  auto result = graph.addVariable(UNSIGNED_INT, {}, debugPrefix +
                                  "/checkAccuracyInHalfPrecision");

  graph.connect(v["value"], value.reshape({1}));
  graph.setInitialValue(v["tolerance"], tolerance);
  graph.connect(v["halfIsAccurate"], result);
  graph.setTileMapping(v, 0);
  graph.setTileMapping(result, 0);
  prog.add(Execute(cs));
  return result;
}

} // end namespace popops
