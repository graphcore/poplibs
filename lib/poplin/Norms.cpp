// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplin/Norms.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Convolution.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/DebugInfo.hpp"
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
                         const DebugNameAndId &dnai) {
  std::string layer = "ReduceResult";
  Tensor t;

  // The output tensor mapping may be specified or created
  if (outputToCloneFrom) {
    t = graph.clone(outputType, *outputToCloneFrom, {dnai, layer});
  } else {
    t = createBroadcastOperand(graph, actsUngrouped, outputType, 1, true,
                               {dnai, layer});
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
      css, {dnai});
  return t;
}

static Tensor normReduce(Graph &graph, const Tensor &actsUngrouped, float scale,
                         bool doSquare, std::vector<ComputeSet> &css,
                         const Type &partialsType, const Type &outputType,
                         const Tensor *outputToCloneFrom,
                         const DebugNameAndId &dnai) {
  auto constantScale =
      graph.addConstant(FLOAT, {}, scale, {dnai, "constantScale"});
  graph.setTileMapping(constantScale, 0);

  return normReduce(graph, actsUngrouped, constantScale, doSquare, css,
                    partialsType, outputType, outputToCloneFrom,
                    {dnai, "ConstScale"});
}

static Tensor computeInvStdDev(Graph &graph, const Tensor &mean,
                               const Tensor &power, float eps, float scaleVar,
                               Sequence &prog, const Type &invStdDevType,
                               bool stableAlgo, const DebugNameAndId &dnai) {
  const auto meanType = mean.elementType();
  const auto powerType = power.elementType();
  auto iStdDev = graph.clone(invStdDevType, mean, {dnai, "iStdDev"});

  const auto meanFlat = mean.flatten();
  const auto powerFlat = power.flatten();
  const auto iStdDevFlat = iStdDev.flatten();

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet({dnai, "iStdDev"});

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
  prog.add(Execute(cs, {dnai}));
  return iStdDev;
}

static Tensor broadcastChannelToMatch(const Tensor &ref, const Tensor &t) {
  return t.flatten().expand(std::vector<std::size_t>(ref.rank() - 2, 1));
}

static bool shouldExecuteCallback(unsigned replicaNormSize, unsigned normSize,
                                  DistributedNormReduceCallback callback) {
  // The callback should only be invoked when there is distributed reduction to
  // be performed.
  bool executeCallback =
      normSize > replicaNormSize && callback != nullptr && replicaNormSize > 0;
  if (executeCallback && normSize % replicaNormSize) {
    throw poplibs_error("Norm batch size must be an integer multiple of "
                        "replica batch size");
  }
  if (executeCallback) {
    logging::poplin::info("All-reduce callback called with group size of {}",
                          normSize / replicaNormSize);
  }
  return executeCallback;
}

static std::pair<Tensor, Tensor>
normStatisticsImpl(Graph &graph, const Tensor &acts, float eps, Sequence &prog,
                   bool unbiasedVarEstimate, bool stableAlgo,
                   const Type &partialsType,
                   DistributedNormReduceCallback reduceCallback,
                   unsigned normSize, const DebugNameAndId &dnai) {
  const std::string layer = "Norm/statistics";
  logging::poplin::info(
      "normStatistics acts={}, eps={}, unbiasedVarEstimate={}, type={}, "
      "normSize={} name={}",
      acts.shape(), eps, unbiasedVarEstimate, partialsType, normSize,
      dnai.getPathName() + "/" + layer);

  size_t numElements = acts.numElements();
  const auto replicaSize = acts.dim(0);
  bool executeCallback =
      shouldExecuteCallback(replicaSize, normSize, reduceCallback);

  // Ideally, we would like the scaling due to the distributed all-reduced to
  // be done by the all-reduce (see T36825).
  // But we know it doesn't do that for now and hence the scaling is folded in
  // here.
  if (executeCallback) {
    numElements *= normSize / replicaSize;
  }

  // Avoid a possible divide by zero FP exception.
  // Note that numElements will be 0 if acts.dim(1) is 0.
  if (acts.dim(1) > 0)
    numElements /= acts.dim(1);

  float scale = 1.0f;
  float scaleVar = 1.0f;
  if (numElements > 1) {
    scale /= numElements;
    if (unbiasedVarEstimate)
      scaleVar = static_cast<float>(numElements) / (numElements - 1);
  }

  const auto powerOutputType = partialsType;
  const auto meanOutputType = acts.elementType();

  std::vector<ComputeSet> css;
  auto mean = normReduce(graph, acts, scale, false, css, partialsType,
                         meanOutputType, nullptr, {dnai, layer + "/mean"});

  auto maybeZeroMeanActs = acts;
  if (stableAlgo) {
    for (const auto &cs : css) {
      prog.add(Execute(cs, {dnai}));
    }
    css.clear();
    logging::poplin::info("Stable statistics estimator used");
    if (executeCallback) {
      auto allReduceResult =
          reduceCallback(graph, {mean}, prog, normSize / replicaSize,
                         {dnai, layer + "/mean"}, {});
      mean = allReduceResult.at(0);
    }
    using namespace popops::expr;
    maybeZeroMeanActs = popops::map(graph, _1 - Cast(_2, acts.elementType()),
                                    {acts, broadcastChannelToMatch(acts, mean)},
                                    prog, {dnai, layer + "/removeMean"});
  }
  // The actual output type for squared sum may be different as the dynamic
  // range is higher. The selection should be based on actual statistics
  // gathered from training experiments. For now keep it at reduced precision
  // to save memory
  auto power =
      normReduce(graph, maybeZeroMeanActs, scale, true, css, partialsType,
                 powerOutputType, &mean, {dnai, layer + "/power"});

  for (const auto &cs : css) {
    prog.add(Execute(cs, {dnai}));
  }

  if (executeCallback) {
    std::vector<Tensor> inputsToCallback;
    std::string str;
    if (stableAlgo) {
      str = "/power";
    } else {
      inputsToCallback.push_back(mean);
      str = "/meanAndPower";
    }
    inputsToCallback.push_back(power);

    auto allReduceResult =
        reduceCallback(graph, inputsToCallback, prog, normSize / replicaSize,
                       {dnai, layer + str}, {});
    if (stableAlgo) {
      power = allReduceResult.at(0);
    } else {
      mean = allReduceResult.at(0);
      power = allReduceResult.at(1);
    }
  }

  auto iStdDev = computeInvStdDev(graph, mean, power, eps, scaleVar, prog,
                                  acts.elementType(), stableAlgo, {dnai});
  return std::make_pair(mean, iStdDev);
}

std::pair<Tensor, Tensor>
normStatistics(Graph &graph, const Tensor &acts, float eps, Sequence &prog,
               bool unbiasedVarEstimate, bool stableAlgo,
               const Type &partialsType,
               const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, eps, unbiasedVarEstimate, stableAlgo, partialsType));

  auto [mean, iStdDev] = normStatisticsImpl(
      graph, acts, eps, prog, unbiasedVarEstimate, stableAlgo, partialsType,
      nullptr, acts.dim(0), {di, "nonDistributed"});
  di.addOutputs(DI_ARGS(mean, iStdDev));
  return std::make_pair(mean, iStdDev);
}

std::pair<poplar::Tensor, poplar::Tensor> distributedNormStatistics(
    poplar::Graph &graph, const poplar::Tensor &acts, float eps,
    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
    DistributedNormReduceCallback callback, unsigned normSize, bool stableAlgo,
    const poplar::Type &partialsType,
    const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, eps, unbiasedVarEstimate, stableAlgo, partialsType));
  auto [mean, iStdDev] = normStatisticsImpl(
      graph, acts, eps, prog, unbiasedVarEstimate, stableAlgo, partialsType,
      callback, normSize, {di, "Distributed"});
  di.addOutputs(DI_ARGS(mean, iStdDev));
  return std::make_pair(mean, iStdDev);
}

Tensor createNormGamma(Graph &graph, const Tensor &acts, const Type &type,
                       const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts, type));
  auto output =
      createBroadcastOperand(graph, acts, type, 1, true, {di, "gamma"});
  di.addOutput(output);
  return output;
}

Tensor createNormGamma(Graph &graph, const Tensor &acts,
                       const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));
  auto output = createNormGamma(graph, acts, acts.elementType(), {di});
  di.addOutput(output);
  return output;
}

Tensor createNormBeta(Graph &graph, const Tensor &acts, const Type &type,
                      const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts, type));
  auto output =
      createBroadcastOperand(graph, acts, type, 1, true, {di, "beta"});
  di.addOutput(output);
  return output;
}

Tensor createNormBeta(Graph &graph, const Tensor &acts,
                      const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));
  auto output = createNormBeta(graph, acts, acts.elementType(), {di});
  di.addOutput(output);
  return output;
}

std::pair<Tensor, Tensor>
createNormParams(Graph &graph, const Tensor &acts,
                 const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));
  auto gamma = createNormGamma(graph, acts, {di});
  auto beta = createNormBeta(graph, acts, {di});
  di.addOutputs(DI_ARGS(gamma, beta));
  return std::make_pair(gamma, beta);
}

Tensor normWhiten(Graph &graph, const Tensor &acts, const Tensor &mean,
                  const Tensor &iStdDev, Sequence &prog,
                  const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts, mean, iStdDev));

  const std::string layer = "Whiten";
  logging::poplin::info("normWhiten acts={}, mean={}, iStdDev={}, name={}",
                        acts.shape(), mean.shape(), iStdDev.shape(),
                        debugContext.getPathName() + "/" + layer);

  auto meanBroadcast = broadcastChannelToMatch(acts, mean);
  auto actsWhitened =
      sub(graph, acts, meanBroadcast, prog, {di, layer + "/mean"});
  auto iStdDevBroadcast = broadcastChannelToMatch(actsWhitened, iStdDev);
  mulInPlace(graph, actsWhitened, iStdDevBroadcast, prog,
             {di, layer + "/istdDev"});
  di.addOutput(actsWhitened);
  return actsWhitened;
}

Tensor normalise(Graph &graph, const Tensor &actsWhitened, const Tensor &gamma,
                 const Tensor &beta, Sequence &prog,
                 const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(actsWhitened, gamma, beta));

  const std::string layer = "Norm/normalise";
  logging::poplin::info("normalise actsWhitened={}, gamma={}, beta={}, name={}",
                        actsWhitened.shape(), gamma.shape(), beta.shape(),
                        debugContext.getPathName() + "/" + layer);

  auto gammaBroadcast = broadcastChannelToMatch(actsWhitened, gamma);
  auto actsNormalised =
      mul(graph, actsWhitened, gammaBroadcast, prog, {di, layer + "/gamma"});
  auto betaBroadcast = broadcastChannelToMatch(actsNormalised, beta);
  addInPlace(graph, actsNormalised, betaBroadcast, prog, {di, layer + "/beta"});
  di.addOutput(actsNormalised);
  return actsNormalised;
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, float scale, Sequence &prog,
                   const Type &partialsType, bool attemptRegroup,
                   const DebugNameAndId &dnai) {
  const std::string layer = "Norm/deltas";
  logging::poplin::info(
      "normParamGradients actsWhitened={}, gradsIn={}, scale={}, "
      "type={}, attemptRegroup={}, name={}",
      actsWhitened.shape(), gradsIn.shape(), scale, partialsType,
      attemptRegroup, dnai.getPathName() + "/" + layer);

  auto gradsInMaybeRegrouped =
      attemptRegroup ? popops::rearrange::regroupIfBeneficial(
                           graph, gradsIn, actsWhitened, prog, {dnai})
                     : gradsIn;
  const auto gradsInMultActs =
      mul(graph, actsWhitened, gradsInMaybeRegrouped, prog, {dnai, layer});

  auto numChannels = gradsInMultActs.dim(1);
  const auto concatInputs = concat({gradsInMultActs, gradsInMaybeRegrouped}, 1);

  std::vector<ComputeSet> css;

  // For beta = Re{gradsIn} where Re{x} reduces the tensor x along the
  //                              second dimension to produce a vector
  //                              of length x.dim(1)
  // For gamma = Re{actsWhitened .* gradsIn}
  //                              .* is element-wise multiplication operator
  //                              Reduction along second dimension

  auto scaleTensor = graph.addConstant(FLOAT, {}, scale, {dnai, "scaleTensor"});
  graph.setTileMapping(scaleTensor, 0);
  const auto concatDeltas =
      normReduce(graph, concatInputs, scaleTensor, false, css, partialsType,
                 gradsInMaybeRegrouped.elementType(), nullptr,
                 {dnai, layer + "/JointGammaDelta"});

  for (const auto &cs : css) {
    prog.add(Execute(cs, {dnai}));
  }

  return std::make_pair(concatDeltas.slice(0, numChannels),
                        concatDeltas.slice(numChannels, 2 * numChannels));
}

std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, Sequence &prog,
                   const Type &partialsType,
                   const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(actsWhitened, gradsIn, partialsType));

  auto outputs = normParamGradients(graph, actsWhitened, gradsIn, 1.0, prog,
                                    partialsType, true, {di});
  di.addOutputs({{"gammaGrad", toProfileValue(outputs.first)},
                 {"betaGrad", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor normGradients(Graph &graph, const Tensor &gradsIn, const Tensor &gamma,
                     Sequence &prog, const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(gradsIn, gamma));

  const auto layer = "NormGrad";
  logging::poplin::info("normGradients gradsIn={}, gamma={}, name={}",
                        gradsIn.shape(), gamma.shape(),
                        debugContext.getPathName() + "/" + layer);
  auto gammaBroadcast = broadcastChannelToMatch(gradsIn, gamma);
  auto output = mul(graph, gradsIn, gammaBroadcast, prog, {di, layer});
  di.addOutput(output);
  return output;
}

Tensor normStatisticsGradientsImpl(Graph &graph, const Tensor &actsWhitened,
                                   const Tensor &gradsIn,
                                   const Tensor &invStdDev, Sequence &prog,
                                   const Type &partialsType, // currently unused
                                   DistributedNormReduceCallback reduceCallback,
                                   unsigned normSize,
                                   const DebugNameAndId &dnai) {
  logging::poplin::info("normStatisticsGradients actsWhitened={}, gradsIn={}, "
                        "invStdDev={}, name={}",
                        actsWhitened.shape(), gradsIn.shape(),
                        invStdDev.shape(), dnai.getPathName());

  const auto replicaNormSize = actsWhitened.dim(0);
  auto executeCallback =
      shouldExecuteCallback(replicaNormSize, normSize, reduceCallback);
  const auto actsShape = actsWhitened.shape();
  auto numElements = actsWhitened.numElements() / actsWhitened.dim(1);

  // Ideally, we would like the scaling due to the distributed all-reduced to
  // be done by the all-reduce. But we know it doesn't do that for now and hence
  // the scaling is folded in here.
  if (executeCallback) {
    numElements *= normSize / replicaNormSize;
  }
  const float rScale = 1.0f / numElements;

  auto gradsInMaybeRegrouped = popops::rearrange::regroupIfBeneficial(
      graph, gradsIn, actsWhitened, prog, {dnai});

  // split rScale = rScale1 * rScale2;
  // TODO: T12898 Research what the optimal split would be dependent on model
  // and field size.
  const auto scaleSplit = 3.0f / 4;
  float rScale1 = std::pow(rScale, scaleSplit);
  float rScale2 = rScale / rScale1;
  const auto dType = actsWhitened.elementType();

  // If type is half, ensure that rScale2 is exactly representable in device
  // HALF type so that the fastest codelet is picked up when rScale2 is used
  // in the scaledAddTo below.
  if (dType == HALF) {
    rScale2 = castToDeviceHalfValue(graph.getTarget(), rScale2);
    // re-evaluate to get better combined precision
    rScale1 = rScale / rScale2;
  }
  Tensor varDelta, meanDelta;
  // See Description of Re{} operator in normParamGradients
  // varDelta = Re{actsWhitened .* gradsIn} * -rScale
  //   Size of varDelta is the size of inverse standard deviation
  // meanDelta = Re{gradsIn} * -rScale
  std::tie(varDelta, meanDelta) =
      normParamGradients(graph, actsWhitened, gradsInMaybeRegrouped, -rScale1,
                         prog, partialsType, false, {dnai});

  if (executeCallback) {
    auto reducedGrads = reduceCallback(graph, {varDelta, meanDelta}, prog,
                                       normSize / replicaNormSize, {dnai}, {});
    varDelta = reducedGrads.at(0);
    meanDelta = reducedGrads.at(1);
  }

  auto gradient = graph.clone(actsWhitened, {dnai, "/gradsIn"});
  prog.add(Copy(gradsInMaybeRegrouped, gradient, false, {dnai}));

  // gradOut = gradsIn - rScale * actsWhitened .* Br{varDelta}
  // where Br{x} broadcast x along all dimensions other than dim(1) of
  // actsWhitened
  // gradsOut = gradsIn - rScale * actsWhitened .* Br{varDelta} + Br{meanDelta}

  auto varDeltaBroadcast = broadcastChannelToMatch(actsWhitened, varDelta);
  auto varGrads =
      mul(graph, actsWhitened, varDeltaBroadcast, prog, {dnai, "varGrads"});
  mulInPlace(graph, meanDelta, rScale2, prog, {dnai, "scaleMeanDelta"});
  auto meanDeltaBroadcast = broadcastChannelToMatch(gradient, meanDelta);
  addInPlace(graph, gradient, meanDeltaBroadcast, prog, {dnai, "meanGrads"});
  // TODO: T12899 Once scaledAddTo is targeted efficiently in element-wise ops,
  // this should become a mapInPlace() expression.
  scaledAddTo(graph, gradient, varGrads, rScale2, prog, {dnai, "addGrads"});

  // Br{invStdDev} .* (gradsIn - rScale * actsWhitened .* Br{varDelta}
  //                   + Br{meanDelta})
  auto invStdDevBroadcast = broadcastChannelToMatch(gradient, invStdDev);
  mulInPlace(graph, gradient, invStdDevBroadcast, prog, {dnai});
  return gradient;
}

Tensor normStatisticsGradients(Graph &graph, const Tensor &actsWhitened,
                               const Tensor &gradsIn, const Tensor &invStdDev,
                               Sequence &prog,
                               const Type &partialsType, // currently unused
                               const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, invStdDev, partialsType));
  const std::string layer = "NonDistributedNorm/gradients";

  auto gradient = normStatisticsGradientsImpl(
      graph, actsWhitened, gradsIn, invStdDev, prog, partialsType, nullptr,
      actsWhitened.dim(0), {di, layer});
  di.addOutput(gradient);
  return gradient;
}

Tensor distributedNormStatisticsGradients(
    Graph &graph, const Tensor &actsWhitened, const Tensor &gradsIn,
    const Tensor &invStdDev, Sequence &prog,
    DistributedNormReduceCallback normReduceCallback, unsigned normSize,
    const Type &partialsType, const poplar::DebugContext &debugContext) {
  POPLIN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, invStdDev, partialsType));
  const std::string layer = "DistributedNorm/gradients";
  auto gradient = normStatisticsGradientsImpl(
      graph, actsWhitened, gradsIn, invStdDev, prog, partialsType,
      normReduceCallback, normSize, {di, layer});
  di.addOutput(gradient);
  return gradient;
}

} // namespace poplin
