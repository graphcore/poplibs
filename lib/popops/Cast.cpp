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

std::pair<poplar::Tensor, poplar::Tensor>
checkAccuracyWhenCast(Graph &graph, const Tensor &input, Type outputType,
                      double tolerance, poplar::program::Sequence &prog,
                      const std::string &debugPrefix) {
  if ((input.elementType() != FLOAT && outputType != HALF) ||
       input.numElements() != 1) {
    throw poputil::poplibs_error("Can only check the accuracy when casting"
          " single element tensors with data type float to half or half"
          " to float");
  }

  auto cs = graph.addComputeSet(debugPrefix + "/checkAccuracyWhenCast");
  auto v = graph.addVertex(cs, templateVertex(
                           "popops::CheckAccuracyWhenCast",
                           input.elementType(), outputType));
  auto isAccurate = graph.addVariable(BOOL, {}, debugPrefix +
                                      "/checkAccuracyWhenCast");
  auto output = graph.addVariable(outputType, {}, debugPrefix +
                                  "/checkAccuracyWhenCast");
  const auto tile = std::min(graph.getTarget().getNumTiles(), 4u) - 1;
  graph.setTileMapping(isAccurate, tile);
  graph.setTileMapping(output, tile);

  graph.connect(v["input"], input.reshape({}));
  graph.connect(v["output"], output.reshape({}));
  graph.setInitialValue(v["tolerance"], tolerance);
  graph.setTileMapping(v, tile);

  prog.add(Execute(cs, isAccurate));
  return {isAccurate, output};
}

} // end namespace popops
