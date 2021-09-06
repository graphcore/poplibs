// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *  Instance normalization operations.
 *
 * Instance norm uses group norm with number of groups = number of channels.
 */

#ifndef popnn_InstanceNorm_hpp
#define popnn_InstanceNorm_hpp
#include "popnn/GroupNorm.hpp"
#include "poputil/DebugInfo.hpp"

/* **Instance normalisation options**
 *
 * As instance norm uses group norm, options are passed through - see the group
 * norm header documentation for the option list.
 */

namespace popnn {
namespace in {

/// Estimate mean and inverse of standard deviation of activations.
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormStatistics(poplar::Graph &graph, const poplar::Tensor acts,
                       float eps, poplar::program::Sequence &prog,
                       bool unbiasedVarEstimate, bool stableAlgo,
                       const poplar::Type &partialsType = poplar::FLOAT,
                       const poplar::DebugContext &debugContext = {},
                       const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, eps, unbiasedVarEstimate,
                                         stableAlgo, partialsType, options));

  auto outputs = popnn::gn::groupNormStatistics(
      graph, acts, eps, prog, acts.dim(1), unbiasedVarEstimate, stableAlgo,
      partialsType, {di}, options);

  di.addOutputs({{"mean", poputil::toProfileValue(outputs.first)},
                 {"iStd", poputil::toProfileValue(outputs.second)}});
  return outputs;
}

/// Whiten activations given mean and standard deviation.
inline poplar::Tensor
instanceNormWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                   const poplar::Tensor &mean, const poplar::Tensor &invStdDev,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, mean, invStdDev, options));

  auto output = popnn::gn::groupNormWhiten(graph, acts, mean, invStdDev, prog,
                                           {di}, options);
  di.addOutput(output);
  return output;
}

/// Instance normalise activations given mean, standard deviation and norm
/// parameters.
///
/// The result is two tensors
///   1. normalised activations
///   2. whitened activations
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
                  const poplar::Tensor &gamma, const poplar::Tensor &beta,
                  const poplar::Tensor &mean, const poplar::Tensor &invStdDev,
                  poplar::program::Sequence &prog,
                  const poplar::DebugContext &debugContext = {},
                  const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts, gamma, beta, mean, invStdDev, options));

  auto outputs = popnn::gn::groupNormalise(graph, acts, gamma, beta, mean,
                                           invStdDev, prog, {di}, options);

  di.addOutputs({{"normActs", poputil::toProfileValue(outputs.first)},
                 {"whitenedActs", poputil::toProfileValue(outputs.second)}});
  return outputs;
}

/// Compute gradients w.r.t parameters for parameter update.
inline std::pair<poplar::Tensor, poplar::Tensor> instanceNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &iStdDev, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}) {

  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, gradsIn, mean, iStdDev, partialsType, options));

  auto outputs = popnn::gn::groupNormParamGradients(
      graph, acts, gradsIn, mean, iStdDev, prog, partialsType, {di}, options);

  di.addOutputs({{"meanGrad", poputil::toProfileValue(outputs.first)},
                 {"iStdDevGrad", poputil::toProfileValue(outputs.second)}});
  return outputs;
}

/// Compute gradients w.r.t parameters for parameter update.
inline std::pair<poplar::Tensor, poplar::Tensor> instanceNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}) {

  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, partialsType, options));

  auto outputs = popnn::gn::groupNormParamGradients(
      graph, actsWhitened, gradsIn, prog, partialsType, {di}, options);

  di.addOutputs({{"meanGrad", poputil::toProfileValue(outputs.first)},
                 {"iStdDevGrad", poputil::toProfileValue(outputs.second)}});
  return outputs;
}

/// Compute gradients w.r.t input activations for the instance norm layer.
/// Gradients are propagated through the complete layer including
/// statistics computation.
inline poplar::Tensor
instanceNormGradients(poplar::Graph &graph, const poplar::Tensor &acts,
                      const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
                      const poplar::Tensor &invStdDev,
                      const poplar::Tensor &gamma,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {}) {

  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, gradsIn, mean, invStdDev, gamma, partialsType, options));

  auto output =
      popnn::gn::groupNormGradients(graph, acts, gradsIn, mean, invStdDev,
                                    gamma, prog, partialsType, {di}, options);

  di.addOutput(output);
  return output;
}

/// Compute gradients w.r.t input activations for the instance norm layer.
/// Gradients are propagated through the complete layer including
/// statistics computation.
inline poplar::Tensor instanceNormGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    const poplar::Tensor &gamma, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}) {

  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(actsWhitened, gradsIn, invStdDev, gamma, partialsType, options));

  auto output =
      popnn::gn::groupNormGradients(graph, actsWhitened, gradsIn, invStdDev,
                                    gamma, prog, partialsType, {di}, options);
  di.addOutput(output);
  return output;
}

/// Update parameters given gradients w.r.t. parameters.
inline void
instanceNormParamUpdate(poplar::Graph &graph, const poplar::Tensor &gammaDelta,
                        const poplar::Tensor &betaDelta, float scale,
                        poplar::Tensor &gamma, poplar::Tensor &beta,
                        poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(gammaDelta, betaDelta, scale, gamma, beta, options));

  return popnn::gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, scale,
                                         gamma, beta, prog, {di}, options);
}

inline void
instanceNormParamUpdate(poplar::Graph &graph, const poplar::Tensor &gammaDelta,
                        const poplar::Tensor &betaDelta,
                        const poplar::Tensor &scale, poplar::Tensor &gamma,
                        poplar::Tensor &beta, poplar::program::Sequence &prog,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(gammaDelta, betaDelta, scale, gamma, beta, options));
  return popnn::gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, scale,
                                         gamma, beta, prog, {di}, options);
}

/// In flop computation, the following applies:
///   - Acts per channel:
///     - for fc layers: the total number of batches.
///     - for conv layers: the field size per channel * batch size.
///
///   - Number of channels:
///     - for fc layers: the total number of activations in a batch.
///     - for conv layers: the total number of channels.

uint64_t getFwdFlops(uint64_t numChannels, uint64_t actsPerChannel,
                     bool computeEstimates);
uint64_t getBwdFlops(uint64_t numChannels, uint64_t actsPerChannel);
uint64_t getWuFlops(uint64_t numChannels, uint64_t actsPerChannel);

} // namespace in
} // namespace popnn
#endif // popnn_InstanceNorm_hpp
