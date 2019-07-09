#include "popnn/NonLinearity.hpp"
#include "popnn/NonLinearityDefUtil.hpp"
#include "popnn/NonLinearityDef.hpp"
#include "NonLinearityInternal.hpp"
#include "poplin/MatMul.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/EncodingConstants.hpp"
#include "popops/Reduce.hpp"
#include "poputil/Util.hpp"
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace {
float getNonLinearityScaling(popnn::NonLinearityType nonLinearityType) {
  return nonLinearityType == popnn::NonLinearityType::SOFTMAX_SCALED ?
                             SOFTMAX_SCALING : 1.0f;
}

// computes softmax along the innermost dimension
// This is not an optimal implementation in terms of precision and order of
// operations
Tensor softmaxImpl(Graph &graph, Tensor t, bool stableAlgo, bool inPlace,
                   bool scaled, Sequence &prog,
                   const std::string &debugStr = "") {
  const auto fnStr = debugStr + "/SoftMax";

  if (t.rank() < 2) {
    throw poplibs_error("input tensor to softmax non-linearity must have "
                        "at least 2 dimensions");
  }

  // Switch innermost dimension to outer as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  const auto innerDimSize = t.dim(rank - 1);

  bool needsCopy = !inPlace;
  if (stableAlgo) {
    auto max = popops::reduce(graph, tShuf, {0}, popops::Operation::MAX, prog,
                              fnStr)
               .expand({0}).broadcast(innerDimSize, 0);
    // We want to increase the range of exponents, below, using:
    // tShuf = tShuf - max + log(scale)
    // So do tShuf = tShuf - (max - log(scale))
    auto maxPlus = popops::sub(graph, max, std::log(SOFTMAX_SCALING),
                               prog, fnStr);

    if (needsCopy) {
      tShuf = popops::sub(graph, tShuf, maxPlus, prog, fnStr);
      needsCopy = false;
    } else {
      popops::subInPlace(graph, tShuf, maxPlus, prog, fnStr);
    }
  }

  if (needsCopy) {
    tShuf = popops::exp(graph, tShuf, prog, fnStr);
  } else {
    popops::expInPlace(graph, tShuf, prog, fnStr);
  }
  // For half types we can improve accuracy by scaling the result so that the
  // sum of the values is max half instead of 1.0.  In this case it also makes
  // sense to retain the reduction result as a float
  auto sumF = popops::reduce(graph, tShuf, poplar::FLOAT,
                             {0}, popops::Operation::ADD, prog, fnStr);
  // As the divide is broadcast we compute 1/x first as there are a lot fewer
  // elements than the number in tShuf
  // TODO: Check if there needs to be an eps added especially for half
  popops::invInPlace(graph, sumF, prog, fnStr);
  if (scaled) {
    popops::mulInPlace(graph, sumF, SOFTMAX_SCALING, prog, fnStr);
  }
  auto sum = (t.elementType() == poplar::HALF) ?
              popops::cast(graph, sumF, poplar::HALF, prog, fnStr) : sumF;

  auto oneOverSum = sum.expand({0}).broadcast(innerDimSize, 0);
  popops::mulInPlace(graph, tShuf, oneOverSum, prog, fnStr);

  // Shuffle dimensions back to original ordering and return.
  // If inPlace == true then this is the same as the original tensor.
  auto tRet = tShuf.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(tRet.shape() == t.shape());
  return tRet;
}

// computes the gradient of softmax along the innermost dimension
Tensor softmaxInputGradientImpl(Graph &graph,
                                const Tensor &out,
                                const Tensor &outGradient,
                                Sequence &prog,
                                const std::string &debugPrefix = "") {
  const auto layerPrefix = debugPrefix + "/SoftMaxGradient";

  if (out.shape() != outGradient.shape()) {
    throw poplibs_error("out and outGradient tensors must have the same "
                        "shape for softmax gradient calculation");
  }

  auto y = out;
  auto g = outGradient;
  if (y.rank() < 2) {
    y = y.flatten().expand({0});
    g = g.flatten().expand({0});
  }

  // Flatten to dimension over which softmax is performed and other dimensions.
  y = y.flatten(0, y.rank() - 1);
  g = g.flatten(0, g.rank() - 1);
  auto gXy = popops::mul(graph, y, g, prog, layerPrefix);
  auto sumgXy = popops::reduce(graph, gXy, {1}, popops::Operation::ADD,
                               prog, layerPrefix).expand({1});
  auto yXsumgXy = popops::mul(graph, y, sumgXy, prog, layerPrefix);
  auto inGradientFlat = gXy;
  popops::subInPlace(graph, inGradientFlat, yXsumgXy, prog, layerPrefix);
  auto inGradient = inGradientFlat.reshape(outGradient.shape());
  return inGradient;
}


} // end anonymous namespace


namespace popnn {

bool isSoftMax(NonLinearityType nl) {
  return (nl == NonLinearityType::SOFTMAX ||
          nl == NonLinearityType::SOFTMAX_STABLE ||
          nl == NonLinearityType::SOFTMAX_SCALED);
}

bool isStableAlgorithm(NonLinearityType nl) {
  return (nl == NonLinearityType::SOFTMAX_STABLE ||
         nl == NonLinearityType::SOFTMAX_SCALED);
}

bool isScaled(NonLinearityType nl) {
  return (nl == NonLinearityType::SOFTMAX_SCALED);
}

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          ComputeSet &cs,
                          const std::string &debugPrefix) {
  if (isSoftMax(nonLinearityType)) {
    throw poputil::poplibs_error("Compute set variant of softmax gradient not "
                                 "implemented");
  }
  const auto dType = out.elementType();
  const auto &target = graph.getTarget();
  auto inGradient = graph.clone(out, debugPrefix + "/NonLinearityGrad");
  // Identify cases where out was broadcast and therefore inGradient will
  // require remapping
  mapOutputForElementWiseOp(graph, {out}, inGradient);
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
  if (isSoftMax(nonLinearityType)) {
    return softmaxInputGradientImpl(graph, out, outGradient, prog,
                                    debugPrefix);
  }
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
  if (isSoftMax(nonLinearityType)) {
    throw poputil::poplibs_error("Compute set variant of softmax not "
                               "implemented");
  }
  if (!t.isParallelWriteable())
    throw poputil::poplibs_error("Trying to update tensor that cannot be "
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
  if (isSoftMax(nonLinearityType)) {
    softmaxImpl(graph, t, isStableAlgorithm(nonLinearityType), true,
                isScaled(nonLinearityType), prog, fnPrefix);
    return;
  }
  ComputeSet cs = graph.addComputeSet(fnPrefix);
  nonLinearityInPlace(graph, nonLinearityType, t, cs, fnPrefix);
  prog.add(Execute(cs));
}

Tensor nonLinearity(Graph &graph, NonLinearityType nonLinearityType,
                    Tensor t, Sequence &prog, const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/Nonlinearity";
  if (isSoftMax(nonLinearityType)) {
    return softmaxImpl(graph, t, isStableAlgorithm(nonLinearityType),
                       false, isScaled(nonLinearityType), prog, fnPrefix);
  }
  ComputeSet cs = graph.addComputeSet(fnPrefix);
  auto out = graph.clone(t.elementType(), t, fnPrefix + "/out");
  // Identify cases where t was broadcast and therefore out will require
  // remapping
  mapOutputForElementWiseOp(graph, {t}, out);
  nonLinearityInPlace(graph, nonLinearityType, out, cs, fnPrefix);

  prog.add(Copy(t, out));
  prog.add(Execute(cs));
  return out;
}
// Functions with a reference to a float, which will return the scaling
// that is used by the nonLinearityType selected.

void
nonLinearityInPlace(Graph &graph, NonLinearityType nonLinearityType,
                    Tensor t, float &nonLinearityScaling,
                    Sequence &prog, const std::string &debugPrefix) {
  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  nonLinearityInPlace(graph, nonLinearityType, t, prog, debugPrefix);
}

void
nonLinearityInPlace(poplar::Graph &graph, NonLinearityType nonLinearityType,
                    poplar::Tensor t, ComputeSet &cs,
                    float &nonLinearityScaling,
                    const std::string &debugPrefix) {
  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  nonLinearityInPlace(graph, nonLinearityType, t, cs, debugPrefix);
}

Tensor nonLinearity(Graph &graph, NonLinearityType nonLinearityType,
                    Tensor t, float &nonLinearityScaling,
                    Sequence &prog, const std::string &debugPrefix) {
  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  return nonLinearity(graph, nonLinearityType, t, prog, debugPrefix);
}

} // end namespace popnn
