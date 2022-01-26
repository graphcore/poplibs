// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popnn/NonLinearity.hpp"
#include "../popops/ExprOpUtil.hpp"
#include "HardSigmoid.hpp"
#include "NonLinearityInternal.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/MatMul.hpp"
#include "popnn/NonLinearityDef.hpp"
#include "popnn/NonLinearityDefUtil.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ElementWiseUtil.hpp"
#include "popops/EncodingConstants.hpp"
#include "popops/Reduce.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <cmath>

namespace poputil {

template <>
poplar::ProfileValue toProfileValue(const popnn::NonLinearityType &t) {
  switch (t) {
  case popnn::NonLinearityType::SIGMOID:
    return poplar::ProfileValue("SIGMOID");
  case popnn::NonLinearityType::HARD_SIGMOID:
    return poplar::ProfileValue("HARD_SIGMOID");
  case popnn::NonLinearityType::RELU:
    return poplar::ProfileValue("RELU");
  case popnn::NonLinearityType::TANH:
    return poplar::ProfileValue("TANH");
  case popnn::NonLinearityType::GELU:
    return poplar::ProfileValue("GELU");
  case popnn::NonLinearityType::SOFTMAX:
    return poplar::ProfileValue("SOFTMAX");
  case popnn::NonLinearityType::SOFTMAX_STABLE:
    return poplar::ProfileValue("SOFTMAX_STABLE");
  case popnn::NonLinearityType::SOFTMAX_SCALED:
    return poplar::ProfileValue("SOFTMAX_SCALED");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}
} // namespace poputil

using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace poputil;

namespace logging = poplibs_support::logging;

namespace {
float getNonLinearityScaling(popnn::NonLinearityType nonLinearityType) {
  return nonLinearityType == popnn::NonLinearityType::SOFTMAX_SCALED
             ? SOFTMAX_SCALING
             : 1.0f;
}

// computes softmax along the innermost dimension
// This is not an optimal implementation in terms of precision and order of
// operations
Tensor softmaxImpl(Graph &graph, Tensor t, bool stableAlgo, bool inPlace,
                   bool scaled, Sequence &prog, const DebugNameAndId &dnai) {
  const std::string fnStr = "SoftMax";
  const auto dType = t.elementType();
  logging::popnn::info("softmax t={}, name={}", t.shape(),
                       dnai.getPathName() + "/" + fnStr);
  if (t.rank() < 1) {
    throw poplibs_error("input tensor to softmax non-linearity must have "
                        "at least 1 dimension");
  }

  const bool expandDimension = t.rank() == 1;
  if (expandDimension) {
    t = t.expand({0});
  }

  // Switch innermost dimension to outer as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  const auto innerDimSize = t.dim(rank - 1);

  bool needsCopy = !inPlace;
  if (stableAlgo) {
    auto max = popops::reduce(graph, tShuf, {0}, popops::Operation::MAX, prog,
                              {dnai, fnStr})
                   .expand({0})
                   .broadcast(innerDimSize, 0);

    // Split subtract from max and addition with std::log(SOFTMAX_SCALING)
    // to avoid rounding errors in expression max - std::log(SOFTMAX_SCALING)
    // causing exp to exceed max half.
    // Note: We could do max - std::log(SOFTMAX_SCALING) for float type but
    //       is not done here to keep the code clean.
    if (needsCopy) {
      tShuf = popops::sub(graph, tShuf, max, prog, {dnai, fnStr});
      needsCopy = false;
    } else {
      popops::subInPlace(graph, tShuf, max, prog, {dnai, fnStr});
    }

    if (scaled) {
      popops::addInPlace(graph, tShuf, std::log(SOFTMAX_SCALING), prog,
                         {dnai, fnStr});
    }
  }

  if (needsCopy) {
    tShuf = popops::exp(graph, tShuf, prog, {dnai, fnStr});
  } else {
    popops::expInPlace(graph, tShuf, prog, {dnai, fnStr});
  }

  // For half types we can improve accuracy by scaling the result so that the
  // sum of the values is max half instead of 1.0.  In this case it also makes
  // sense to retain the reduction result as a float
  auto sumF = popops::reduce(graph, tShuf, poplar::FLOAT, {0},
                             popops::Operation::ADD, prog, {dnai, fnStr});

  // As the divide is broadcast we compute 1/x first as there are a lot fewer
  // elements than the number in tShuf
  // TODO: T12913 Check if there needs to be an eps added especially for half.
  popops::invInPlace(graph, sumF, prog, {dnai, fnStr});
  if (scaled) {
    popops::mulInPlace(graph, sumF, SOFTMAX_SCALING, prog, {dnai, fnStr});
  }
  auto sum = (dType == poplar::HALF)
                 ? popops::cast(graph, sumF, poplar::HALF, prog, {dnai, fnStr})
                 : sumF;

  auto oneOverSum = sum.expand({0}).broadcast(innerDimSize, 0);
  popops::mulInPlace(graph, tShuf, oneOverSum, prog,
                     {dnai, fnStr + "/invScale"});

  // Shuffle dimensions back to original ordering and return.
  // If inPlace == true then this is the same as the original tensor.
  auto tRet = tShuf.dimShufflePartial({0, rank - 1}, {rank - 1, 0});
  assert(tRet.shape() == t.shape());
  return expandDimension ? tRet.squeeze({0}) : tRet;
}

// computes the gradient of softmax along the innermost dimension
Tensor softmaxInputGradientImpl(Graph &graph, const Tensor &out,
                                const Tensor &outGradient, Sequence &prog,
                                const DebugNameAndId &dnai) {
  const auto layerPrefix = "SoftMaxGradient";
  logging::popnn::info("softmaxInputGradient out={}, outGradient={}, name={}",
                       out.shape(), outGradient.shape(),
                       dnai.getPathName() + "/" + layerPrefix);

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
  auto gXy = popops::mul(graph, y, g, prog, {dnai, layerPrefix});
  auto sumgXy = popops::reduce(graph, gXy, {1}, popops::Operation::ADD, prog,
                               {dnai, layerPrefix})
                    .expand({1});
  auto yXsumgXy = popops::mul(graph, y, sumgXy, prog, {dnai, layerPrefix});
  auto inGradientFlat = gXy;
  popops::subInPlace(graph, inGradientFlat, yXsumgXy, prog,
                     {dnai, layerPrefix});
  auto inGradient = inGradientFlat.reshape(outGradient.shape());
  return inGradient;
}

// Build up the given program so that it performs max(0, min(1, 0.2*x + 0.5))
// with the given tensor
void hardSigmoidImpl(Graph &graph, Tensor t, Sequence &prog,
                     const DebugNameAndId &dnai) {
  const auto layerPrefix = "hardSigmoidActivation";
  logging::popnn::info("hardSigmoid t={}, name={}", t.shape(),
                       dnai.getPathName() + "/" + layerPrefix);

  auto activation = popnn::HardSigmoidExpr::activation();
  mapInPlace(graph, *activation, {t}, prog, {dnai, layerPrefix});
}

// Build up the given program so that it peforms
//
//   outGradient*0.2 when elements of out are within the range[-2.5, +2.5]
//   outGradient*0 when elements of out are outside of the above range.
Tensor hardSigmoidGradImpl(Graph &graph, Tensor out, const Tensor &outGradient,
                           Sequence &prog, const DebugNameAndId &dnai) {
  const auto layerPrefix = "HardSigmoidGradient";
  logging::popnn::info(
      "hardSigmoidInputGradient out={}, outGradient={}, name={}", out.shape(),
      outGradient.shape(), dnai.getPathName() + "/" + layerPrefix);

  auto gradient = popnn::HardSigmoidExpr::gradient(out.elementType());
  return map(graph, *gradient, {out, outGradient}, prog, {dnai, layerPrefix});
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

// Three of the non linearities are implemented as popops vertices; This checks
// if 'nl' is one of them and, in case, it returns the corresponding popops enum
std::optional<expr::UnaryOpType> isPopops(NonLinearityType nl) {
  std::optional<expr::UnaryOpType> unaryOp;
  switch (nl) {
  case NonLinearityType::TANH:
    unaryOp = expr::UnaryOpType::TANH;
    break;
  case NonLinearityType::SIGMOID:
    unaryOp = expr::UnaryOpType::SIGMOID;
    break;
  case NonLinearityType::RELU:
    unaryOp = expr::UnaryOpType::RELU;
    break;
  default:;
  }
  return unaryOp;
}

Tensor nonLinearityInputGradient(Graph &graph,
                                 NonLinearityType nonLinearityType, Tensor out,
                                 Tensor outGradient, ComputeSet &cs,
                                 const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(out, outGradient, nonLinearityType, cs));

  if (isSoftMax(nonLinearityType)) {
    throw poputil::poplibs_error("Compute set variant of softmax gradient not "
                                 "implemented");
  }

  const auto layerPrefix = "NonLinearityGrad";
  logging::popnn::info(
      "nonLinearityInputGradient type={}, out={}, outGradient={}, name={}",
      nonLinearityType, out.shape(), outGradient.shape(),
      debugContext.getPathName() + layerPrefix);

  const auto dType = out.elementType();
  const auto &target = graph.getTarget();
  auto inGradient = createOutputForElementWiseOp(
      graph, {out}, out.elementType(), {di, layerPrefix});
  auto outFlat = out.flatten();
  auto outGradFlat = outGradient.flatten();
  auto inGradFlat = inGradient.flatten();
  graph.reorderToSimplify(&inGradFlat, {&outFlat, &outGradFlat}, false);
  // Use mapping of the output activations as the forward pass retains
  // tile mapping of the input tensor. This is useful for example in batchnorm
  // where exchange for some operations is avoided by having the same mapping
  // for the gradients and activations
  auto outMapping = graph.getTileMapping(outFlat);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto vectorWidth = target.getVectorWidth(dType);

  const auto codeletName2D =
      templateVertex("popnn::NonLinearityGrad2D", dType, nonLinearityType);
  const auto codeletName1D =
      templateVertex("popnn::NonLinearityGrad1D", dType, nonLinearityType);

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto max1DElements =
      std::min<std::size_t>(graph.getMaxVertexFieldValue(codeletName1D, "n"),
                            target.getRptCountMax() * numWorkers * vectorWidth);
  const auto max2DInnerElements =
      std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, "inGrad", 1),
                            target.getRptCountMax() * vectorWidth);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = outMapping[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, thisTileMap);
    // If mapping of outGrad tensor on this tile is only region(s) from a
    // single variable, gather all the inputs to the non-linearity into
    // a single edge
    if (tileContiguousRegions.size() == 1) {
      const auto outGradTile =
          concat(outGradFlat.slices(tileContiguousRegions));
      const auto numElements = outGradTile.numElements();
      if (numElements <= max1DElements) {
        auto v = graph.addVertex(
            cs, codeletName1D,
            {{"out", concat(outFlat.slices(tileContiguousRegions))},
             {"outGrad", outGradTile},
             {"inGrad", concat(inGradFlat.slices(tileContiguousRegions))}});
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
        splitRegionsBetweenWorkers(target, tileContiguousRegions, grainSize,
                                   2 * grainSize, max2DInnerElements);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, codeletName2D,
                               {{"out", outFlat.slices(regions)},
                                {"outGrad", outGradFlat.slices(regions)},
                                {"inGrad", inGradFlat.slices(regions)}});
      graph.setTileMapping(v, tile);
    }
  }
  di.addOutput(inGradient);
  return inGradient;
}

Tensor nonLinearityInputGradient(Graph &graph,
                                 NonLinearityType nonLinearityType, Tensor out,
                                 Tensor outGradient,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(out, outGradient, nonLinearityType));

  if (isSoftMax(nonLinearityType)) {
    return softmaxInputGradientImpl(graph, out, outGradient, prog, {di});
  } else if (nonLinearityType == NonLinearityType::HARD_SIGMOID) {
    return hardSigmoidGradImpl(graph, out, outGradient, prog, {di});
  }

  auto cs = graph.addComputeSet({di, "NonLinearityGrad"});
  auto t = nonLinearityInputGradient(graph, nonLinearityType, out, outGradient,
                                     cs, {di});
  prog.add(Execute(cs, {di}));
  di.addOutput(t);
  return t;
}

// This is called for all the different types of non linearities, in-place or
// not.
// If the 'out' parameter is not empty (not nullopt), the vertex used will be
// a not-in-place vertex, and '*out' needs to be set as the 'out'
// field of the vertex.
void nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
                  poplar::Tensor t, std::optional<poplar::Tensor> out,
                  ComputeSet &cs, const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t, nonLinearityType, cs));

  if (isSoftMax(nonLinearityType)) {
    throw poputil::poplibs_error("Compute set variant of softmax not "
                                 "implemented");
  }

  logging::popnn::info("nonLinearity type={}, t={}, name={}", nonLinearityType,
                       t.shape(), debugContext.getPathName());

  //  out == true     Indicates that we must use a non-in-place vertex
  //  unaryOp == true Indicates that the vertex is supported by an elementwise
  //                  unaryOp vertex

  bool isInPlaceVertex = !bool(out);
  std::optional<expr::UnaryOpType> unaryOp = isPopops(nonLinearityType);
  if (!isInPlaceVertex && !unaryOp &&
      nonLinearityType != NonLinearityType::SWISH) {
    throw poputil::poplibs_error("Non In Place variant of nonlinearity not "
                                 "implemented");
  }

  std::vector<Tensor *> ts;
  Tensor *tCheckForWriteability; // check the correct tensor for writeability
  if (isInPlaceVertex) {
    tCheckForWriteability = &t;
  } else {
    out = out->flatten();
    ts.push_back(&(*out));
    tCheckForWriteability = &(*out);
  }
  if (!tCheckForWriteability->isParallelWriteable())
    throw poputil::poplibs_error("Trying to update tensor that cannot be "
                                 "written in parallel");

  t = t.flatten();
  graph.reorderToSimplify(&t, ts, false);

  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  auto mapping = graph.getTileMapping(t);
  const auto numWorkers = target.getNumWorkerContexts();
  const auto numTiles = target.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = target.getVectorWidth(dType);

  auto codeletName2D =
      templateVertex(isInPlaceVertex ? "popnn::NonLinearity2DInPlace"
                                     : "popnn::NonLinearity2D",
                     dType, nonLinearityType);
  auto codeletName1D =
      templateVertex(isInPlaceVertex ? "popnn::NonLinearity1DInPlace"
                                     : "popnn::NonLinearity1D",
                     dType, nonLinearityType);
  auto dataName = "data";

  Tensor outFlat;
  if (!isInPlaceVertex) {
    outFlat = out->flatten();
  }
  if (unaryOp) {
    // Popops vertices: could be in-place or not-in-place, both the vertex name
    // and the data field name depends on this.
    std::string name2D, name1D;
    if (isInPlaceVertex) {
      dataName = "inOut";
      name2D = "popops::UnaryOp2DInPlace";
      name1D = "popops::UnaryOp1DInPlace";
    } else {
      dataName = "in";
      name2D = "popops::UnaryOp2D";
      name1D = "popops::UnaryOp1D";
    }
    codeletName2D = templateVertex(name2D, *unaryOp, dType);
    codeletName1D = templateVertex(name1D, *unaryOp, dType);
  }

  // Maximum elements vertices can handle per-region is based on input vector
  // type and the max count the `rpt` instruction can handle.
  const auto rptLimitMaxElements1D =
      target.getRptCountMax() * numWorkers * vectorWidth;
  const auto rptLimitMaxElements2D = target.getRptCountMax() * vectorWidth;
  std::size_t max1DElements, max2DElements;
  if (unaryOp && !isInPlaceVertex) {
    // These vertices don't have limited size fields; the limits are only due
    // to the RPT instruction count.
    max1DElements = rptLimitMaxElements1D;
    max2DElements = rptLimitMaxElements2D;
  } else {
    max1DElements =
        std::min<std::size_t>(graph.getMaxVertexFieldValue(codeletName1D, "n"),
                              rptLimitMaxElements1D);
    max2DElements =
        std::min<std::size_t>(graph.getMaxFieldDim(codeletName2D, dataName, 1),
                              rptLimitMaxElements2D);
  }
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto thisTileMap = mapping[tile];
    if (thisTileMap.empty())
      continue;
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(t, thisTileMap);
    // If mapping of outGrad tensor on this tile is only region(s) from a
    // single variable, gather all the inputs to the non-linearity into
    // a single edge
    if (tileContiguousRegions.size() == 1) {
      const auto tThisTile = concat(tFlat.slices(tileContiguousRegions));
      const auto numElements = tThisTile.numElements();
      if (numElements <= max1DElements) {
        auto v = graph.addVertex(cs, codeletName1D, {{dataName, tThisTile}});
        if (!isInPlaceVertex) {
          // The not-in-place vertices have the 'out' field.
          graph.connect(v["out"],
                        concat(outFlat.slices(tileContiguousRegions)));
        }
        if (!unaryOp || (unaryOp && isInPlaceVertex)) {
          // The popops in-place vertices have the 'n' field
          // The non popops vertices have the 'n' field
          graph.setInitialValue(v["n"], numElements);
        }
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
        splitRegionsBetweenWorkers(target, tileContiguousRegions, vectorWidth,
                                   minVectors * vectorWidth, max2DElements);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, codeletName2D,
                               {{dataName, tFlat.slices(regions)}});
      if (!isInPlaceVertex) {
        graph.connect(v["out"], outFlat.slices(regions));
      }
      graph.setTileMapping(v, tile);
    }
  }
}

void nonLinearityInPlace(Graph &graph, NonLinearityType nonLinearityType,
                         poplar::Tensor t, ComputeSet &cs,
                         const poplar::DebugContext &debugContext) {
  nonLinearity(graph, nonLinearityType, t, std::nullopt, cs, debugContext);
}

void nonLinearityInPlace(Graph &graph, NonLinearityType nonLinearityType,
                         Tensor t, Sequence &prog,
                         const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, nonLinearityType));

  const std::string fnPrefix = "Nonlinearity";
  const auto implDebugContext = DebugNameAndId(di, fnPrefix);

  if (isSoftMax(nonLinearityType)) {
    softmaxImpl(graph, t, isStableAlgorithm(nonLinearityType), true,
                isScaled(nonLinearityType), prog, implDebugContext);
  } else if (nonLinearityType == NonLinearityType::HARD_SIGMOID) {
    hardSigmoidImpl(graph, t, prog, implDebugContext);
  } else {
    ComputeSet cs = graph.addComputeSet(implDebugContext);
    nonLinearityInPlace(graph, nonLinearityType, t, cs, implDebugContext);
    prog.add(Execute(cs, {di}));
  }
}

Tensor nonLinearity(Graph &graph, NonLinearityType nonLinearityType, Tensor t,
                    Sequence &prog, const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, nonLinearityType));

  const std::string fnPrefix = "Nonlinearity";
  const auto implDebugContext = DebugNameAndId(di, fnPrefix);

  if (isSoftMax(nonLinearityType)) {
    return softmaxImpl(graph, t, isStableAlgorithm(nonLinearityType), false,
                       isScaled(nonLinearityType), prog, implDebugContext);
  } else if (nonLinearityType == NonLinearityType::HARD_SIGMOID) {
    auto copy = graph.clone(t);
    prog.add(Copy(t, copy));
    hardSigmoidImpl(graph, copy, prog, implDebugContext);

    return copy;
  }
  ComputeSet cs = graph.addComputeSet({di, fnPrefix});

  auto out = createOutputForElementWiseOp(graph, {t}, t.elementType(),
                                          {di, fnPrefix + "/out"});
  if (nonLinearityType == NonLinearityType::SWISH ||
      isPopops(nonLinearityType)) {
    nonLinearity(graph, nonLinearityType, t, out, cs, {di, fnPrefix});
  } else {
    nonLinearity(graph, nonLinearityType, out, std::nullopt, cs,
                 {di, fnPrefix});
    prog.add(Copy(t, out, false, {di}));
  }

  prog.add(Execute(cs, {di}));
  di.addOutput(out);
  return out;
}
// Functions with a reference to a float, which will return the scaling
// that is used by the nonLinearityType selected.

void nonLinearityInPlace(Graph &graph, NonLinearityType nonLinearityType,
                         Tensor t, float &nonLinearityScaling, Sequence &prog,
                         const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, nonLinearityType, nonLinearityScaling));

  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  nonLinearityInPlace(graph, nonLinearityType, t, prog, {di});
}

void nonLinearityInPlace(poplar::Graph &graph,
                         NonLinearityType nonLinearityType, poplar::Tensor t,
                         ComputeSet &cs, float &nonLinearityScaling,
                         const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, nonLinearityType, nonLinearityScaling, cs));

  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  nonLinearityInPlace(graph, nonLinearityType, t, cs, {di});
}

Tensor nonLinearity(Graph &graph, NonLinearityType nonLinearityType, Tensor t,
                    float &nonLinearityScaling, Sequence &prog,
                    const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, nonLinearityType, nonLinearityScaling));

  nonLinearityScaling = getNonLinearityScaling(nonLinearityType);
  auto output = nonLinearity(graph, nonLinearityType, t, prog, {di});
  di.addOutput(output);
  return output;
}

} // end namespace popnn
