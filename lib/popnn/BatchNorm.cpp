// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popnn/BatchNorm.hpp"
#include "NormsInternal.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplin/Norms.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <functional>
#include <map>
#include <numeric>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace popnn {
namespace bn {

std::pair<Tensor, Tensor>
batchNormStatistics(Graph &graph, const Tensor acts, float eps, Sequence &prog,
                    bool unbiasedVarEstimate, bool stableAlgo,
                    const Type &partialsType_,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options) {

  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, eps, unbiasedVarEstimate,
                                         stableAlgo, partialsType_, options));

  checkTensorShape(acts);
  auto partialsType = partialsType_;
  checkNormTensorTypes(acts.elementType(), graph.getTarget(), partialsType);
  auto outputs =
      poplin::normStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                             stableAlgo, partialsType, {di});

  di.addOutputs({{"mean", toProfileValue(outputs.first)},
                 {"inverseStd", toProfileValue(outputs.second)}});
  return outputs;
}

std::pair<Tensor, Tensor> distributedBatchNormStatistics(
    Graph &graph, const Tensor acts, float eps, Sequence &prog,
    bool unbiasedVarEstimate, poplin::DistributedNormReduceCallback callback,
    unsigned normBatchSize, bool stableAlgo, const Type &partialsType_,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts, eps, unbiasedVarEstimate, stableAlgo,
                            partialsType_, normBatchSize, options));

  checkTensorShape(acts);
  auto partialsType = partialsType_;
  checkNormTensorTypes(acts.elementType(), graph.getTarget(), partialsType);
  auto outputs = poplin::distributedNormStatistics(
      graph, acts, eps, prog, unbiasedVarEstimate, callback, normBatchSize,
      stableAlgo, partialsType, {di});

  di.addOutputs({{"mean", toProfileValue(outputs.first)},
                 {"inverseStd", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor batchNormWhiten(Graph &graph, const Tensor &acts_, const Tensor &mean,
                       const Tensor &iStdDev, Sequence &prog,
                       const poplar::DebugContext &debugContext,
                       const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts_, mean, iStdDev, options));

  const auto rank = acts_.rank();
  auto acts = preProcessNormActs(acts_);
  auto whitenedActs =
      poplin::normWhiten(graph, acts, mean, iStdDev, prog, {di});
  auto output = postProcessNormActs(whitenedActs, rank);
  di.addOutput(output);
  return output;
}

std::pair<Tensor, Tensor>
batchNormalise(Graph &graph, const Tensor &acts, const Tensor &gamma,
               const Tensor &beta, const Tensor &mean, const Tensor &iStdDev,
               Sequence &prog, const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts, mean, beta, gamma, iStdDev, options));

  const auto rank = acts.rank();
  checkTensorShape(acts);
  auto preProcessActs = preProcessNormActs(acts);
  auto whitenedActs =
      batchNormWhiten(graph, preProcessActs, mean, iStdDev, prog, {di});
  auto outputActs =
      poplin::normalise(graph, whitenedActs, gamma, beta, prog, {di});
  auto outputs = std::make_pair(postProcessNormActs(outputActs, rank),
                                postProcessNormActs(whitenedActs, rank));
  di.addOutputs({{"normalizedActivations", toProfileValue(outputs.first)},
                 {"whitenedActivations", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor batchNormalise(Graph &graph, const Tensor &acts,
                      const Tensor &combinedMultiplicand, const Tensor &addend,
                      Sequence &prog, const poplar::DebugContext &debugContext,
                      const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts, combinedMultiplicand, addend, options));

  const auto rank = acts.rank();
  checkTensorShape(acts);
  auto preProcessedActs = preProcessNormActs(acts);
  auto actsNormalised = poplin::normalise(
      graph, preProcessedActs, combinedMultiplicand, addend, prog, {di});
  auto output = postProcessNormActs(actsNormalised, rank);
  di.addOutput(output);
  return output;
}

std::pair<Tensor, Tensor>
batchNormParamGradients(Graph &graph, const Tensor &actsWhitened,
                        const Tensor &gradsIn, Sequence &prog,
                        const Type &partialsType_,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, partialsType_, options));

  checkTensorShape(gradsIn);
  checkTensorShape(actsWhitened);
  auto partialsType = partialsType_;
  checkNormTensorTypes(actsWhitened.elementType(), graph.getTarget(),
                       partialsType);
  auto outputs = poplin::normParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, {di});
  di.addOutputs({{"meanGrad", toProfileValue(outputs.first)},
                 {"iStdDevGrad", toProfileValue(outputs.second)}});
  return outputs;
}

std::pair<Tensor, Tensor>
batchNormParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                        const Tensor &mean, const Tensor &iStdDev,
                        Sequence &prog, const Type &partialsType_,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, gradsIn, mean, iStdDev, partialsType_, options));

  checkTensorShape(gradsIn);
  checkTensorShape(acts);
  auto partialsType = partialsType_;
  checkNormTensorTypes(acts.elementType(), graph.getTarget(), partialsType);
  auto actsWhitened = batchNormWhiten(graph, acts, mean, iStdDev, prog, {di});
  auto outputs = batchNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, {di});
  di.addOutputs({{"meanGrad", toProfileValue(outputs.first)},
                 {"iStdDevGrad", toProfileValue(outputs.second)}});
  return outputs;
}

static Tensor batchNormGradientsImpl(
    Graph &graph, const Tensor &actsWhitened_, const Tensor &gradsIn_,
    const Tensor &iStdDev, const Tensor &gamma, Sequence &prog,
    const Type &partialsType_, poplin::DistributedNormReduceCallback callback,
    unsigned normSize, const poplar::DebugNameAndId &dnai) {
  const auto rank = actsWhitened_.rank();
  checkTensorShape(actsWhitened_);
  checkTensorShape(gradsIn_);
  auto partialsType = partialsType_;
  checkNormTensorTypes(actsWhitened_.elementType(), graph.getTarget(),
                       partialsType);

  auto actsWhitened = preProcessNormActs(actsWhitened_);
  auto gradsIn = preProcessNormActs(gradsIn_);
  auto gradsNorm = poplin::normGradients(graph, gradsIn, gamma, prog, {dnai});
  Tensor gradsOut;
  if (callback) {
    gradsOut = poplin::distributedNormStatisticsGradients(
        graph, actsWhitened, gradsNorm, iStdDev, prog, callback, normSize,
        partialsType, {dnai});
  } else {
    gradsOut = poplin::normStatisticsGradients(
        graph, actsWhitened, gradsNorm, iStdDev, prog, partialsType, {dnai});
  }
  auto output = postProcessNormActs(gradsOut, rank);
  return output;
}

Tensor batchNormGradients(Graph &graph, const Tensor &actsWhitened_,
                          const Tensor &gradsIn_, const Tensor &iStdDev,
                          const Tensor &gamma, Sequence &prog,
                          const Type &partialsType,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(actsWhitened_, gradsIn_, iStdDev, gamma, partialsType, options));
  auto gradsOut = batchNormGradientsImpl(
      graph, actsWhitened_, gradsIn_, iStdDev, gamma, prog, partialsType,
      nullptr, 1, {di, "NonDistributedBatchNormGrads"});
  di.addOutput(gradsOut);
  return gradsOut;
}

Tensor distributedBatchNormGradients(
    Graph &graph, const Tensor &actsWhitened_, const Tensor &gradsIn_,
    const Tensor &iStdDev, const Tensor &gamma, Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, const Type &partialsType,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened_, gradsIn_, iStdDev, gamma,
                            partialsType, normBatchSize, options));
  auto gradsOut = batchNormGradientsImpl(
      graph, actsWhitened_, gradsIn_, iStdDev, gamma, prog, partialsType,
      reduceCallback, normBatchSize, {di, "distributedBatchNormGrads"});
  di.addOutput(gradsOut);
  return gradsOut;
}

Tensor batchNormGradients(Graph &graph, const Tensor &acts_,
                          const Tensor &gradsIn_, const Tensor &mean,
                          const Tensor &iStdDev, const Tensor &gamma,
                          Sequence &prog, const Type &partialsType_,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts_, gradsIn_, mean, iStdDev, gamma, partialsType_, options));

  checkTensorShape(acts_);
  auto partialsType = partialsType_;
  checkNormTensorTypes(acts_.elementType(), graph.getTarget(), partialsType);

  auto actsWhitened = batchNormWhiten(graph, acts_, mean, iStdDev, prog, {di});
  auto output = batchNormGradientsImpl(graph, actsWhitened, gradsIn_, iStdDev,
                                       gamma, prog, partialsType, nullptr, 1,
                                       {di, "NonDistributedBatchNormGrads"});
  di.addOutput(output);
  return output;
}

Tensor distributedBatchNormGradients(
    Graph &graph, const Tensor &acts_, const Tensor &gradsIn_,
    const Tensor &mean, const Tensor &iStdDev, const Tensor &gamma,
    Sequence &prog, poplin::DistributedNormReduceCallback callback,
    unsigned normBatchSize, const Type &partialsType_,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts_, gradsIn_, mean, iStdDev, gamma,
                            normBatchSize, partialsType_, options));

  checkTensorShape(acts_);
  auto partialsType = partialsType_;
  checkNormTensorTypes(acts_.elementType(), graph.getTarget(), partialsType);

  auto actsWhitened = batchNormWhiten(graph, acts_, mean, iStdDev, prog, {di});
  auto output = batchNormGradientsImpl(
      graph, actsWhitened, gradsIn_, iStdDev, gamma, prog, partialsType,
      callback, normBatchSize, {di, "DistributedBatchNormGrads"});
  di.addOutput(output);
  return output;
}

void batchNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, float scale, Tensor &gamma,
                          Tensor &beta, Sequence &prog,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(gammaDelta, betaDelta, beta, gamma, scale, options));

  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, {di, "BN/paramUpdate"});
}

void batchNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, const Tensor &scale,
                          Tensor &gamma, Tensor &beta, Sequence &prog,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(gammaDelta, betaDelta, scale, gamma, options));

  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, {di, "BN/paramUpdate"});
}

} // namespace bn
} // namespace popnn
