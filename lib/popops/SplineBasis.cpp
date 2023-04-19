// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "popops/SplineBasis.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;

namespace popops {

void splineBasis(Graph &graph, const Tensor &pseudo, const Tensor &kernelSize,
                 const Tensor &isOpenSpline, const Tensor &basis,
                 const Tensor &weightIndex, unsigned degree, Sequence &prog,
                 const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  static constexpr const char *layerPrefix = "SplineBasis";

  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(pseudo, kernelSize, isOpenSpline, basis, weightIndex));

  logging::popops::info(
      "SplineBasis psuedo={}, kernelSize={}, isOpenSpline={}, basis={}, "
      "weightIndex={}, name={}",
      pseudo.shape(), kernelSize.shape(), isOpenSpline.shape(), basis.shape(),
      weightIndex.shape(), di.getPathName() + "/" + layerPrefix);

  // Verify inputs
  const auto pseudoShape = pseudo.shape();
  if (pseudoShape.size() != 2) {
    throw poputil::poplibs_error("Pseudo coordinates tensor must be "
                                 "2-dimensional");
  }

  const auto pseudoType = pseudo.elementType();
  if (pseudoType != FLOAT and pseudoType != HALF) {
    throw poputil::poplibs_error("Pseudo coordinates type must be float or "
                                 "half");
  }

  const auto kernelSizeShape = kernelSize.shape();
  if (kernelSizeShape.size() != 1) {
    throw poputil::poplibs_error("KernelSize tensor must be 1-dimensional");
  }

  const auto kernelSizeType = kernelSize.elementType();
  if (kernelSizeType != INT) {
    throw poputil::poplibs_error("KernelSize tensor type must be integer type");
  }

  const auto isOpenSplineShape = isOpenSpline.shape();
  if (isOpenSplineShape.size() != 1) {
    throw poputil::poplibs_error("IsOpenSpline tensor must be 1-dimensional");
  }

  const auto isOpenSplineType = isOpenSpline.elementType();
  if (isOpenSplineType != UNSIGNED_CHAR) {
    throw poputil::poplibs_error(
        "IsOpenSpline tensor type must be unsigned char");
  }

  if (pseudoShape[1] != kernelSizeShape[0]) {
    throw poputil::poplibs_error(
        "KernelSize tensor shape must match pseudo tensor shape");
  }

  if (pseudoShape[1] != isOpenSplineShape[0]) {
    throw poputil::poplibs_error(
        "IsOpenSpline tensor shape must match pseudo tensor shape");
  }

  if (degree < 1 || degree > 3) {
    throw poputil::poplibs_error("Degree must be 1, 2 or 3");
  }

  auto cs = graph.addComputeSet({di, layerPrefix});

  const auto numDims = pseudo.dim(1);
  const auto numSplines = (size_t)(std::pow(degree + 1, numDims) + 0.5);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto mapping = graph.getTileMapping(basis);
  auto basisFlat = basis.flatten();

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;
    const auto inType = pseudo.elementType();
    const auto grainSize = std::max<unsigned>(
        target.getVectorWidth(inType), target.getAtomicStoreGranularity());

    const auto vertexRegions = splitRegionsBetweenWorkers(
        target, thisTileMap, grainSize, 2 * grainSize);
    if (vertexRegions.size()) {
      logging::popops::trace("Tile: {} Producing: {} SplineBasis vertices",
                             tile, vertexRegions.size());
    }

    for (const auto &region : vertexRegions) {
      if (region.empty())
        continue;

      std::vector<unsigned> offsets;
      for (auto &interval : region) {
        offsets.push_back(interval.begin());
      }

      // Calculate which psuedo input slice is needed for this vertex.
      const auto lastSliceSize = region.back().size();
      const auto edgeOffset = offsets.front() / numSplines;
      const auto edgeEnd =
          (offsets.back() + lastSliceSize + numSplines - 1) / numSplines;

      const auto tilePseudo =
          pseudo.slice({edgeOffset, 0}, {edgeEnd, numDims}).flatten();

      VertexRef v = graph.addVertex(
          cs,
          poputil::templateVertex("popops::SplineBasis", pseudo.elementType(),
                                  degree),
          {{"pseudo", tilePseudo},
           {"kernelSize", kernelSize.flatten()},
           {"isOpenSpline", isOpenSpline.flatten()},
           {"basis", basis.flatten().slices(region)},
           {"weightIndex", weightIndex.flatten().slices(region)}});
      graph.setInitialValue(v["numSplines"], numSplines);
      graph.setInitialValue(v["numDims"], numDims);
      graph.setInitialValue(v["offsets"], offsets);
      graph.setInitialValue(v["edgeOffset"], edgeOffset);
      graph.setTileMapping(v, tile);
    }
  }

  prog.add(poplar::program::Execute(cs));
}

} // end namespace popops
