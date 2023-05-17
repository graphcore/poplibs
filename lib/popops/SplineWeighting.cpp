// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include "popops/SplineWeighting.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
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

poplar::Tensor splineWeighting(poplar::Graph &graph,
                               const poplar::Tensor &input,
                               const poplar::Tensor &weight,
                               const poplar::Tensor &basis,
                               const poplar::Tensor &weightIndex,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  static constexpr const char *layerPrefix = "SplineWeighting";

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, weight, basis, weightIndex));

  logging::popops::info("SplineWeighting input={}, weight={}, basis={}, "
                        "weightIndex={}, name={}",
                        input.shape(), weight.shape(), basis.shape(),
                        weightIndex.shape(),
                        di.getPathName() + "/" + layerPrefix);

  // Verify inputs
  const auto inputShape = input.shape();
  const auto inputType = input.elementType();
  if (inputShape.size() != 2) {
    throw poputil::poplibs_error("Input tensor must be 2-dimensional");
  }
  if (inputType != FLOAT and inputType != HALF) {
    throw poputil::poplibs_error("Pseudo coordinates type must be float or "
                                 "half");
  }

  const auto weightShape = weight.shape();
  const auto weightType = weight.elementType();
  if (weightShape.size() != 3) {
    throw poputil::poplibs_error("Weight tensor must be 3-dimensional");
  }
  if (inputType != weightType) {
    throw poputil::poplibs_error("Weight tensor type must match input tensor "
                                 "type");
  }

  const auto basisShape = basis.shape();
  const auto basisType = basis.elementType();
  if (basisShape.size() != 2) {
    throw poputil::poplibs_error("Basis tensor must be 2-dimensional");
  }
  if (inputType != basisType) {
    throw poputil::poplibs_error("Basis tensor type must match input tensor "
                                 "type");
  }

  const auto weightIndexShape = weightIndex.shape();
  const auto weightIndexType = weightIndex.elementType();
  if (weightIndexShape.size() != 2) {
    throw poputil::poplibs_error("WeightIndex tensor must be 2-dimensional");
  }
  if (weightIndexType != INT) {
    throw poputil::poplibs_error("WeightIndex tensor type must be integer");
  }

  // Validate numEdges dimension
  if (inputShape[0] != basisShape[0]) {
    throw poputil::poplibs_error("Basis tensor shape must match input tensor "
                                 "shape");
  }
  if (inputShape[0] != weightIndexShape[0]) {
    throw poputil::poplibs_error("WeightIndex tensor shape must match input "
                                 "tensor shape");
  }
  // Validate numSplines dimension
  if (basisShape[1] != weightIndexShape[1]) {
    throw poputil::poplibs_error("Basis tensor shape must match weightIndex "
                                 "tensor shape");
  }
  // Validate numInputChannel dimension
  if (weightShape[1] != inputShape[1]) {
    throw poputil::poplibs_error("Weight tensor shape must match input tensor "
                                 "shape");
  }

  auto cs = graph.addComputeSet({di, layerPrefix});
  const auto numEdges = input.dim(0);
  const auto numInCh = input.dim(1);
  const auto numOutCh = weight.dim(2);
  const auto numSplines = basis.dim(1);
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();

  auto output = graph.addVariable(inputType, {numEdges, numOutCh},
                                  VariableMappingMethod::LINEAR, "output");
  const auto mapping = graph.getTileMapping(output);

  auto outputFlat = output.flatten();

  auto addVertex = [&](auto &tile, auto &region) {
    assert(region.size() == 1);
    const auto &region0 = region.front();
    std::vector<unsigned> offsets;
    for (const auto &r : region0) {
      offsets.push_back(r.begin());
    }
    // Calculate which input tensors' slices are needed for this vertex.
    const auto lastSliceSize = region0.back().size();
    const auto edgeOffset = offsets.front() / numOutCh;
    const auto edgeEnd =
        (offsets.back() + lastSliceSize + numOutCh - 1) / numOutCh;

    const auto tileInput =
        input.slice({edgeOffset, 0}, {edgeEnd, numInCh}).flatten();
    const auto tileBasis =
        basis.slice({edgeOffset, 0}, {edgeEnd, numSplines}).flatten();
    const auto tileWeightIndex =
        weightIndex.slice({edgeOffset, 0}, {edgeEnd, numSplines}).flatten();

    auto v = graph.addVertex(
        cs,
        poputil::templateVertex("popops::SplineWeighting", input.elementType()),
        {{"input", tileInput},
         {"weight", weight.flatten()},
         {"basis", tileBasis},
         {"weightIndex", tileWeightIndex},
         {"output", outputFlat.slices(region0)}});
    graph.setInitialValue(v["numInCh"], numInCh);
    graph.setInitialValue(v["numOutCh"], numOutCh);
    graph.setInitialValue(v["numSplines"], numSplines);
    graph.setInitialValue(v["offsets"], offsets);
    graph.setInitialValue(v["edgeOffset"], edgeOffset);

    graph.setTileMapping(v, tile);
  };

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;

    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outputFlat, thisTileMap);

    if (tileContiguousRegions.size() == 1) {
      logging::popops::trace("Tile: {} Producing: 1 SplineWeighting vertex",
                             tile);
      addVertex(tile, tileContiguousRegions);
    } else {
      const auto inType = input.elementType();
      const auto grainSize = std::max<unsigned>(
          target.getVectorWidth(inType), target.getAtomicStoreGranularity());

      const auto vertexRegions = splitRegionsBetweenWorkers(
          target, tileContiguousRegions, grainSize, 2 * grainSize);
      if (vertexRegions.size()) {
        logging::popops::trace(
            "Tile: {} Producing: {} SplineWeighting vertices", tile,
            vertexRegions.size());
      }

      for (const auto &regions : vertexRegions) {
        if (regions.empty())
          continue;

        addVertex(tile, regions);
      }
    }
  }

  // Add step to execute the compute set
  prog.add(poplar::program::Execute(cs));
  di.addOutput(output);
  return output;
}

} // end namespace popops
