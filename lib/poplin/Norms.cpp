// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/logging.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Convolution.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/icl/interval_map.hpp>
#include <cassert>
#include <cmath>
#include <set>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;
namespace logging = poplibs_support::logging;

namespace poplin {

static Tensor normReduce(Graph &graph, const Tensor &actsUngrouped,
                         const Tensor &scale, bool doSquare,
                         std::vector<ComputeSet> &css,
                         const Type &, // partialsType,
                         const Type &outputType,
                         const Tensor *outputToCloneFrom,
                         const std::string &debugPrefix) {
  std::string name = debugPrefix + "/ReduceResult";
  Tensor t;

  // The output tensor mapping may be specified or created
  if (outputToCloneFrom) {
    t = graph.clone(outputType, *outputToCloneFrom, name);
  } else {
    t = createBroadcastOperand(graph, actsUngrouped, outputType, 1, true, name);
  }

  if (actsUngrouped.rank() < 2)
    throw poplibs_error("NormReduce with rank " +
                        std::to_string(actsUngrouped.rank()) + " expected >=2");

  std::vector<std::size_t> reduceDims(actsUngrouped.rank() - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  popops::reduceWithOutput(
      graph, actsUngrouped, t, reduceDims,
      {doSquare ? popops::Operation::SQUARE_ADD : popops::Operation::ADD, false,
       scale},
      css, debugPrefix);
  return t;
}

static Tensor normReduce(Graph &graph, const Tensor &actsUngrouped, float scale,
                         bool doSquare, std::vector<ComputeSet> &css,
                         const Type &partialsType, const Type &outputType,
                         const Tensor *outputToCloneFrom,
                         const std::string &debugPrefix) {
  auto constantScale =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/constantScale");
  graph.setTileMapping(constantScale, 0);

  return normReduce(graph, actsUngrouped, constantScale, doSquare, css,
                    partialsType, outputType, outputToCloneFrom,
                    debugPrefix + "/ConstScale");
}

static Tensor computeInvStdDev(Graph &graph, const Tensor &mean,
                               const Tensor &power, float eps, float scaleVar,
                               Sequence &prog, const Type &invStdDevType,
                               bool stableAlgo, const std::string debugPrefix) {
  const auto meanType = mean.elementType();
  const auto powerType = power.elementType();
  auto iStdDev = graph.clone(invStdDevType, mean, debugPrefix + "/iStdDev");

  const auto meanFlat = mean.flatten();
  const auto powerFlat = power.flatten();
  const auto iStdDevFlat = iStdDev.flatten();

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/iStdDev");

  const auto mapping = graph.getTileMapping(iStdDev);
  const auto grainSize = target.getVectorWidth(invStdDevType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(iStdDevFlat, mapping[tile]);
    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poplin::InverseStdDeviation",
                                              meanType, powerType,
                                              invStdDevType, stableAlgo),
                               {{"mean", meanFlat.slices(regions)},
                                {"power", powerFlat.slices(regions)},
                                {"iStdDev", iStdDevFlat.slices(regions)}});
      graph.setInitialValue(v["eps"], eps);
      graph.setInitialValue(v["scaleVar"], scaleVar);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return iStdDev;
}

static Tensor broadcastChannelToMatch(const Tensor &ref, const Tensor &t) {
  return t.flatten().expand(std::vector<std::size_t>(ref.rank() - 2, 1));
}

std::pair<Tensor, Tensor>
normStatistics(Graph &graph, const Tensor &acts, float eps, Sequence &prog,
               bool unbiasedVarEstimate, bool stableAlgo,
               const Type &partialsType,
               const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  const auto fnPrefix = debugPrefix + "/Norm/statistics";
  logging::poplin::info(
      "normStatistics acts={}, eps={}, unbiasedVarEstimate={}, "
      "type={}, name={}",
      acts.shape(), eps, unbiasedVarEstimate, partialsType, fnPrefix);

  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(1);
  const float scaleVar =
      unbiasedVarEstimate ? static_cast<float>(numElements) / (numElements - 1)
                          : 1.0f;
  const auto powerOutputType = partialsType;
  const auto meanOutputType = acts.elementType();

  std::vector<ComputeSet> css;
  auto mean =
      normReduce(graph, acts, 1.0f / numElements, false, css, partialsType,
                 meanOutputType, nullptr, fnPrefix + "/mean");

  auto maybeZeroMeanActs = acts;
  if (stableAlgo) {
    for (const auto &cs : css) {
      prog.add(Execute(cs));
    }
    css.clear();
    logging::poplin::info("Stable statistics estimator used");
    using namespace popops::expr;
    maybeZeroMeanActs = popops::map(graph, _1 - Cast(_2, acts.elementType()),
                                    {acts, broadcastChannelToMatch(acts, mean)},
                                    prog, fnPrefix + "/removeMean");
  }
  // The actual output type for squared sum may be different as the dynamic
  // range is higher. The selection should be based on actual statistics
  // gathered from training experiments. For now keep it at reduced precision
  // to save memory
  auto power =
      normReduce(graph, maybeZeroMeanActs, 1.0f / numElements, true, css,
                 partialsType, powerOutputType, &mean, fnPrefix + "/power");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  auto iStdDev = computeInvStdDev(graph, mean, power, eps, scaleVar, prog,
                                  acts.elementType(), stableAlgo, debugPrefix);
  return std::make_pair(mean, iStdDev);
}

Tensor createNormGamma(Graph &graph, const Tensor &acts, const Type &type) {
  return createBroadcastOperand(graph, acts, type, 1, true, "gamma");
}

Tensor createNormGamma(Graph &graph, const Tensor &acts) {
  return createNormGamma(graph, acts, acts.elementType());
}

Tensor createNormBeta(Graph &graph, const Tensor &acts, const Type &type) {
  return createBroadcastOperand(graph, acts, type, 1, true, "beta");
}

Tensor createNormBeta(Graph &graph, const Tensor &acts) {
  return createNormBeta(graph, acts, acts.elementType());
}

std::pair<Tensor, Tensor> createNormParams(Graph &graph, const Tensor &acts) {
  auto gamma = createNormGamma(graph, acts);
  auto beta = createNormBeta(graph, acts);
  return std::make_pair(gamma, beta);
}

Tensor normWhiten(Graph &graph, const Tensor &acts, const Tensor &mean,
                  const Tensor &iStdDev, Sequence &prog,
                  const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  const auto fnPrefix = debugPrefix + "/Whiten";
  logging::poplin::info("normWhiten acts={}, mean={}, iStdDev={}, name={}",
                        acts.shape(), mean.shape(), iStdDev.shape(), fnPrefix);

  auto meanBroadcast = broadcastChannelToMatch(acts, mean);
  auto actsWhitened = sub(graph, acts, meanBroadcast, prog, fnPrefix + "/mean");
  auto iStdDevBroadcast = broadcastChannelToMatch(actsWhitened, iStdDev);
  mulInPlace(graph, actsWhitened, iStdDevBroadcast, prog,
             fnPrefix + "/istdDev");
  return actsWhitened;
}

Tensor normalise(Graph &graph, const Tensor &actsWhitened, const Tensor &gamma,
                 const Tensor &beta, Sequence &prog,
                 const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  const auto fnPrefix = debugPrefix + "/Norm/normalise";
  logging::poplin::info("normalise actsWhitened={}, gamma={}, beta={}, name={}",
                        actsWhitened.shape(), gamma.shape(), beta.shape(),
                        fnPrefix);

  auto gammaBroadcast = broadcastChannelToMatch(actsWhitened, gamma);
  auto actsNormalised =
      mul(graph, actsWhitened, gammaBroadcast, prog, fnPrefix + "/gamma");
  auto betaBroadcast = broadcastChannelToMatch(actsNormalised, beta);
  addInPlace(graph, actsNormalised, betaBroadcast, prog, fnPrefix + "/beta");
  return actsNormalised;
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, float scale, Sequence &prog,
                   const Type &partialsType, bool attemptRegroup,
                   const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/deltas";
  logging::poplin::info(
      "normParamGradients actsWhitened={}, gradsIn={}, scale={}, "
      "type={}, attemptRegroup={}, name={}",
      actsWhitened.shape(), gradsIn.shape(), scale, partialsType,
      attemptRegroup, fnPrefix);

  auto gradsInMaybeRegrouped =
      attemptRegroup ? popops::rearrange::regroupIfBeneficial(
                           graph, gradsIn, actsWhitened, prog, debugPrefix)
                     : gradsIn;
  const auto gradsInMultActs =
      mul(graph, actsWhitened, gradsInMaybeRegrouped, prog, fnPrefix);

  auto numChannels = gradsInMultActs.dim(1);
  const auto concatInputs = concat({gradsInMultActs, gradsInMaybeRegrouped}, 1);

  std::vector<ComputeSet> css;

  // For beta = Re{gradsIn} where Re{x} reduces the tensor x along the
  //                              second dimension to produce a vector
  //                              of length x.dim(1)
  // For gamma = Re{actsWhitened .* gradsIn}
  //                              .* is element-wise multiplication operator
  //                              Reduction along second dimension

  auto scaleTensor =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/scaleTensor");
  graph.setTileMapping(scaleTensor, 0);
  const auto concatDeltas =
      normReduce(graph, concatInputs, scaleTensor, false, css, partialsType,
                 gradsInMaybeRegrouped.elementType(), nullptr,
                 fnPrefix + "/JointGammaDelta");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  return std::make_pair(concatDeltas.slice(0, numChannels),
                        concatDeltas.slice(numChannels, 2 * numChannels));
}

std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, Sequence &prog,
                   const Type &partialsType,
                   const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  return normParamGradients(graph, actsWhitened, gradsIn, 1.0, prog,
                            partialsType, true, debugPrefix);
}

Tensor normGradients(Graph &graph, const Tensor &gradsIn, const Tensor &gamma,
                     Sequence &prog, const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  const auto fnPrefix = debugPrefix + "/NormGrad";
  logging::poplin::info("normGradients gradsIn={}, gamma={}, name={}",
                        gradsIn.shape(), gamma.shape(), fnPrefix);
  auto gammaBroadcast = broadcastChannelToMatch(gradsIn, gamma);
  return mul(graph, gradsIn, gammaBroadcast, prog, fnPrefix);
}

Tensor normStatisticsGradients(Graph &graph, const Tensor &actsWhitened,
                               const Tensor &gradsIn, const Tensor &invStdDev,
                               Sequence &prog,
                               const Type &partialsType, // currently unused
                               const poplar::DebugContext &debugContext) {
  const auto debugPrefix = debugContext.getPathName();
  const auto fnPrefix = debugPrefix + "/Norm/gradients";
  logging::poplin::info("normStatisticsGradients actsWhitened={}, gradsIn={}, "
                        "invStdDev={}, name={}",
                        actsWhitened.shape(), gradsIn.shape(),
                        invStdDev.shape(), fnPrefix);

  const auto actsShape = actsWhitened.shape();
  const auto numElements = actsWhitened.numElements() / actsWhitened.dim(1);
  const float rScale = 1.0f / numElements;

  auto gradsInMaybeRegrouped = popops::rearrange::regroupIfBeneficial(
      graph, gradsIn, actsWhitened, prog, debugPrefix);

  // split rScale = rScale1 * rScale2;
  // TODO: T12898 Research what the optimal split would be dependent on model
  // and field size.
  const auto scaleSplit = 3.0f / 4;
  const float rScale1 = std::pow(rScale, scaleSplit);
  const float rScale2 = rScale / rScale1;

  Tensor varDelta, meanDelta;
  // See Description of Re{} operator in normParamGradients
  // varDelta = Re{actsWhitened .* gradsIn} * -rScale
  //   Size of varDelta is the size of inverse standard deviation
  // meanDelta = Re{gradsIn} * -rScale
  std::tie(varDelta, meanDelta) =
      normParamGradients(graph, actsWhitened, gradsInMaybeRegrouped, -rScale1,
                         prog, partialsType, false, debugPrefix);

  auto gradient = graph.clone(actsWhitened, fnPrefix + "/gradsIn");
  prog.add(Copy(gradsInMaybeRegrouped, gradient));

  // gradOut = gradsIn - rScale * actsWhitened .* Br{varDelta}
  // where Br{x} broadcast x along all dimensions other than dim(1) of
  // actsWhitened
  // gradsOut = gradsIn - rScale * actsWhitened .* Br{varDelta} + Br{meanDelta}

  auto varDeltaBroadcast = broadcastChannelToMatch(actsWhitened, varDelta);
  auto varGrads =
      mul(graph, actsWhitened, varDeltaBroadcast, prog, fnPrefix + "/varGrads");
  mulInPlace(graph, meanDelta, rScale2, prog, fnPrefix + "/scaleMeanDelta");
  auto meanDeltaBroadcast = broadcastChannelToMatch(gradient, meanDelta);
  addInPlace(graph, gradient, meanDeltaBroadcast, prog,
             fnPrefix + "/meanGrads");
  // TODO: T12899 Once scaledAddTo is targeted efficiently in element-wise ops,
  // this should become a mapInPlace() expression.
  scaledAddTo(graph, gradient, varGrads, rScale2, prog, fnPrefix + "/addGrads",
              {{"scaleFloatToHalfTolerance", "1e-3"}});

  // Br{invStdDev} .* (gradsIn - rScale * actsWhitened .* Br{varDelta}
  //                   + Br{meanDelta})
  auto invStdDevBroadcast = broadcastChannelToMatch(gradient, invStdDev);
  mulInPlace(graph, gradient, invStdDevBroadcast, prog, fnPrefix);
  return gradient;
}

} // namespace poplin
