#include "popnn/NonLinearity.hpp"
#include "popnn/NonLinearityDefUtil.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "poputil/Util.hpp"
using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace {

// computes softmax along the innermost dimension
// This is not an optimal implementation in terms of precision and order of
// operations
Tensor softmaxImpl(Graph &graph, Tensor t, bool stableAlgo, bool inPlace,
                   Sequence &prog, const std::string &debugStr = "") {
  const auto fnStr = debugStr + "/SoftMax";

  // Switch innermost dimension to outer as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  const auto innerDimSize = t.dim(rank - 1);

  bool needsCopy = !inPlace;
  if (stableAlgo) {
    auto max = popops::reduce(graph, tShuf, {0}, popops::Operation::MAX, prog,
                              fnStr)
               .expand({0}).broadcast(innerDimSize, 0);
    if (needsCopy) {
      tShuf = popops::sub(graph, tShuf, max, prog, fnStr);
      needsCopy = false;
    } else {
      popops::subInPlace(graph, tShuf, max, prog, fnStr);
    }
  }

  if (needsCopy) {
    tShuf = popops::exp(graph, tShuf, prog, fnStr);
  } else {
    popops::expInPlace(graph, tShuf, prog, fnStr);
  }

  auto sum =
    popops::reduce(graph, tShuf, {0}, popops::Operation::ADD, prog, fnStr)
    .expand({0}).broadcast(innerDimSize, 0);
  popops::divInPlace(graph, tShuf, sum, prog, fnStr);

  // Shuffle dimensions back to original ordering and return.
  // If inPlace == true then this is the same as the original tensor.
  auto tRet = tShuf.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(tRet.shape() == t.shape());
  return tRet;
}

} // end anonymous namespace


namespace popnn {

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          ComputeSet &cs,
                          const std::string &debugPrefix) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE) {
    throw poputil::poplib_error("SOFTMAX gradient not implemented");
  }
  const auto dType = out.elementType();
  const auto &target = graph.getTarget();
  auto inGradient = graph.clone(out, debugPrefix + "/NonLinearityGrad");
  auto outFlat = out.flatten();
  auto outGradFlat = outGradient.flatten();
  auto inGradFlat = inGradient.flatten();
  graph.reorderToSimplify(&inGradFlat, {&outFlat, &outGradFlat});
  // Use mapping of the output activations as the forward pass retains
  // tile mapping of the input tensor. This is useful for example in batchnorm
  // where exchange for some operations is avoided by having the same mapping
  // for the gradients and activatons
  auto outMapping = graph.getTileMapping(outFlat);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto vectorWidth = target.getVectorWidth(dType);

  const auto codeletName2D =
    templateVertex("popnn::NonLinearityGrad2D",
                   dType, nonLinearityType);
  const auto codeletNameSupervisor =
    templateVertex("popnn::NonLinearityGradSupervisor",
                   dType, nonLinearityType);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto maxSupervisorElements = std::min<std::size_t>(
    graph.getMaxVertexFieldValue(codeletNameSupervisor, "n"),
    target.getRptCountMax() * numWorkers * vectorWidth);
  const auto max2DInnerElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "inGrad", 1),
    target.getRptCountMax() * vectorWidth);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = outMapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    // If mapping of outGrad tensor on this tile is only region(s) from a
    // single variable, gather all the inputs to the non-linearity into
    // a single edge
    if (tileContiguousRegions.size() == 1) {
      const auto outGradTile = concat(outGradFlat.slices(thisTileMap));
      const auto numElements = outGradTile.numElements();
      if (numElements <= maxSupervisorElements) {
        auto v =
          graph.addVertex(cs, codeletNameSupervisor,
                          {{"out", concat(outFlat.slices(thisTileMap))},
                           {"outGrad", outGradTile},
                           {"inGrad", concat(inGradFlat.slices(thisTileMap))}});
        graph.setInitialValue(v["n"], numElements);
        graph.setTileMapping(v, tile);
        continue;
      }
    }
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize,
                                   max2DInnerElements);
    for (const auto &regions : vertexRegions) {
      auto v =
          graph.addVertex(cs, codeletName2D,
                          {{"out", outFlat.slices(regions)},
                           {"outGrad", outGradFlat.slices(regions)},
                           {"inGrad", inGradFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  return inGradient;
}

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/NonLinearityGrad");
  auto t = nonLinearityInputGradient(graph, nonLinearityType, out, outGradient,
                                     cs, debugPrefix);
  prog.add(Execute(cs));
  return t;
}

void
nonLinearityInPlace(poplar::Graph &graph, NonLinearityType nonLinearityType,
                    poplar::Tensor t, ComputeSet &cs,
                    const std::string &debugPrefix) {
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE) {
    throw poputil::poplib_error("Compute set variant of softmax not "
                               "implemented");
  }
  if (!t.isParallelWriteable())
    throw poputil::poplib_error("Trying to update tensor that cannot be "
                               "written in parallel");
  t = t.flatten();
  graph.reorderToSimplify(&t, {});
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  auto mapping = graph.getTileMapping(t);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = target.getVectorWidth(dType);

  const auto codeletName2D =
    templateVertex("popnn::NonLinearity2D", dType, nonLinearityType);
  const auto codeletNameSupervisor =
    templateVertex("popnn::NonLinearitySupervisor", dType, nonLinearityType);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto maxSupervisorElements = std::min<std::size_t>(
    graph.getMaxVertexFieldValue(codeletNameSupervisor, "n"),
    target.getRptCountMax() * numWorkers * vectorWidth);
  const auto max2DElements = std::min<std::size_t>(
    graph.getMaxFieldDim(codeletName2D, "data", 1),
    target.getRptCountMax() * vectorWidth);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(t, thisTileMap);
    // If mapping of outGrad tensor on this tile is only region(s) from a
    // single variable, gather all the inputs to the non-linearity into
    // a single edge
    if (tileContiguousRegions.size() == 1) {
      const auto tThisTile = concat(tFlat.slices(thisTileMap));
      const auto numElements = tThisTile.numElements();
      if (numElements <= maxSupervisorElements) {
        auto v =
            graph.addVertex(cs, codeletNameSupervisor,
                            {{"data", tThisTile}});
        graph.setInitialValue(v["n"], numElements);
        graph.setTileMapping(v, tile);
        continue;
      }
    }
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    auto numElements = intervalSequenceNumElements(tileContiguousRegions);
    auto minVectors =
        numElements <= vectorWidth * target.getNumWorkerContexts() ? 1 : 2;
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   vectorWidth, minVectors * vectorWidth,
                                   max2DElements);

    for (const auto &regions : vertexRegions) {
      auto v =
          graph.addVertex(cs, codeletName2D,
                          {{"data", tFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
}

void
nonLinearityInPlace(Graph &graph, NonLinearityType nonLinearityType,
                    Tensor t, Sequence &prog, const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/Nonlinearity";
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE) {
    bool stableAlgo = nonLinearityType == NonLinearityType::SOFTMAX_STABLE;
    softmaxImpl(graph, t, stableAlgo, true, prog, fnPrefix);
    return;
  }
  ComputeSet cs = graph.addComputeSet(fnPrefix);
  nonLinearityInPlace(graph, nonLinearityType, t, cs, fnPrefix);
  prog.add(Execute(cs));
}

Tensor nonLinearity(Graph &graph, NonLinearityType nonLinearityType,
                    Tensor t, Sequence &prog, const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/Nonlinearity";
  if (nonLinearityType == NonLinearityType::SOFTMAX ||
      nonLinearityType == NonLinearityType::SOFTMAX_STABLE) {
    bool stableAlgo = nonLinearityType == NonLinearityType::SOFTMAX_STABLE;
    return softmaxImpl(graph, t, stableAlgo, false, prog, fnPrefix);
  }
  ComputeSet cs = graph.addComputeSet(fnPrefix);
  auto out = graph.clone(t.elementType(), t, fnPrefix + "/out");
  nonLinearityInPlace(graph, nonLinearityType, out, cs, fnPrefix);
  prog.add(Copy(t, out));
  prog.add(Execute(cs));
  return out;
}


} // end namespace popnn
