// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popnn/BatchNorm.hpp"
#include "NormsInternal.hpp"
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
                    const Type &partialsType,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, eps, unbiasedVarEstimate,
                                         stableAlgo, partialsType, options));

  checkTensorShape(acts);
  auto outputs =
      poplin::normStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                             stableAlgo, partialsType, {di});

  di.addOutputs({{"mean", toProfileValue(outputs.first)},
                 {"inverseStd", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor batchNormWhiten(Graph &graph, const Tensor &acts_, const Tensor &mean,
                       const Tensor &iStdDev, Sequence &prog,
                       const poplar::DebugContext &debugContext,
                       const poplar::OptionFlags &options) {
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
                        const Type &partialsType,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, partialsType, options));

  checkTensorShape(gradsIn);
  checkTensorShape(actsWhitened);
  auto outputs = poplin::normParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, {di});
  di.addOutputs({{"meanGrad", toProfileValue(outputs.first)},
                 {"iStdDevGrad", toProfileValue(outputs.second)}});
  return outputs;
}

std::pair<Tensor, Tensor>
batchNormParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                        const Tensor &mean, const Tensor &iStdDev,
                        Sequence &prog, const Type &partialsType,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, gradsIn, mean, iStdDev, partialsType, options));

  checkTensorShape(gradsIn);
  checkTensorShape(acts);
  auto actsWhitened = batchNormWhiten(graph, acts, mean, iStdDev, prog, {di});
  auto outputs = batchNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, {di});
  di.addOutputs({{"meanGrad", toProfileValue(outputs.first)},
                 {"iStdDevGrad", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor batchNormGradients(Graph &graph, const Tensor &actsWhitened_,
                          const Tensor &gradsIn_, const Tensor &iStdDev,
                          const Tensor &gamma, Sequence &prog,
                          const Type &partialsType,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(actsWhitened_, gradsIn_, iStdDev, gamma, partialsType, options));

  const auto rank = actsWhitened_.rank();
  checkTensorShape(actsWhitened_);
  checkTensorShape(gradsIn_);
  auto actsWhitened = preProcessNormActs(actsWhitened_);
  auto gradsIn = preProcessNormActs(gradsIn_);
  auto gradsNorm = poplin::normGradients(graph, gradsIn, gamma, prog, {di});
  auto gradsOut = poplin::normStatisticsGradients(
      graph, actsWhitened, gradsNorm, iStdDev, prog, partialsType, {di});
  auto output = postProcessNormActs(gradsOut, rank);
  di.addOutput(output);
  return output;
}

Tensor batchNormGradients(Graph &graph, const Tensor &acts_,
                          const Tensor &gradsIn_, const Tensor &mean,
                          const Tensor &iStdDev, const Tensor &gamma,
                          Sequence &prog, const Type &partialsType,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts_, gradsIn_, mean, iStdDev, gamma, partialsType, options));

  checkTensorShape(acts_);
  auto actsWhitened = batchNormWhiten(graph, acts_, mean, iStdDev, prog, {di});
  auto output = batchNormGradients(graph, actsWhitened, gradsIn_, iStdDev,
                                   gamma, prog, partialsType, {di});
  di.addOutput(output);
  return output;
}

void batchNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, float scale, Tensor &gamma,
                          Tensor &beta, Sequence &prog,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
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
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(gammaDelta, betaDelta, scale, gamma, options));

  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, {di, "BN/paramUpdate"});
}

} // namespace bn
} // namespace popnn
