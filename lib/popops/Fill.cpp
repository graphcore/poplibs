// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "popops/Fill.hpp"

#include "poputil/DebugInfo.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include <poplar/Graph.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

template <typename FillValueType>
void fill(poplar::Graph &graph, poplar::Tensor t,
          const std::vector<poplar::Interval> &tileRegions, unsigned tile,
          poplar::ComputeSet fillCS, FillValueType fillValue) {
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
      v = graph.addVertex(fillCS, templateVertex("popops::Fill", dType));
      const auto &region = regions.front();
      auto out = concat(tFlat.slices(region));
      graph.connect(v["out"], out);
      graph.setInitialValue<FillValueType>(v["in"], fillValue);
    } else {
      v = graph.addVertex(fillCS, templateVertex("popops::Fill2d", dType));
      auto out = tFlat.slices(regions);
      graph.connect(v["out"], out);
      graph.setInitialValue<FillValueType>(v["in"], fillValue);
    }
    graph.setTileMapping(v, tile);
  }
}

template <typename FillValueType>
void fill(poplar::Graph &graph, const poplar::Tensor &t, unsigned tile,
          poplar::ComputeSet fillCS, FillValueType fillValue) {
  fill<FillValueType>(graph, t, {{0, t.numElements()}}, tile, fillCS,
                      fillValue);
}

template <typename FillValueType>
void fill(Graph &graph, const Tensor &t,
          const std::vector<std::vector<Interval>> &mapping, ComputeSet fillCS,
          FillValueType fillValue) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    fill<FillValueType>(graph, t, mapping[tile], tile, fillCS, fillValue);
  }
}

template <typename FillValueType>
void fill(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog, FillValueType fillValue,
          const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, fillValue));

  auto cs = graph.addComputeSet({di, "Fill"});
  auto tFlat = t.flatten();
  graph.reorderToSimplify(&tFlat, {}, false);
  fill<FillValueType>(graph, tFlat, graph.getTileMapping(tFlat), cs, fillValue);
  prog.add(Execute(cs, {di}));
}

#define FILL_EXPLICIT_INSTANTIATIONS(Type)                                     \
  template void fill<Type>(poplar::Graph &, poplar::Tensor,                    \
                           const std::vector<poplar::Interval> &, unsigned,    \
                           poplar::ComputeSet, Type);                          \
  template void fill<Type>(poplar::Graph &, const poplar::Tensor &, unsigned,  \
                           poplar::ComputeSet, Type);                          \
  template void fill<Type>(poplar::Graph &, const poplar::Tensor &,            \
                           const std::vector<std::vector<Interval>> &,         \
                           poplar::ComputeSet, Type);                          \
  template void fill<Type>(poplar::Graph &, const poplar::Tensor &,            \
                           poplar::program::Sequence &, Type,                  \
                           const poplar::DebugContext &)

FILL_EXPLICIT_INSTANTIATIONS(float);
FILL_EXPLICIT_INSTANTIATIONS(int);
FILL_EXPLICIT_INSTANTIATIONS(unsigned);

} // namespace popops