// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *  Layer normalisation operations.
 *
 * Layer norm uses group norm with number of groups = 1.
 */

#ifndef popnn_LayerNorm_hpp
#define popnn_LayerNorm_hpp
#include "popnn/GroupNorm.hpp"
#include "poputil/DebugInfo.hpp"

/* **Layer normalisation options**
 *
 * As layer norm uses group norm, options are passed through - see the group
 * norm header documentation for the option list.
 */

namespace popnn {
namespace ln {

/// Estimate mean and inverse of standard deviation of activations.
inline std::pair<poplar::Tensor, poplar::Tensor>
layerNormStatistics(poplar::Graph &graph, const poplar::Tensor acts, float eps,
                    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
                    bool stableAlgo = false,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {}) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, eps, unbiasedVarEstimate,
                                         stableAlgo, partialsType, options));

  auto outputs = popnn::gn::groupNormStatistics(graph, acts, eps, prog, 1,
                                                unbiasedVarEstimate, stableAlgo,
                                                partialsType, {di}, options);

  di.addOutputs({{"mean", poputil::toProfileValue(outputs.first)},
                 {"iStdDev", poputil::toProfileValue(outputs.second)}});
  return outputs;
}

/// Whiten activations given mean and standard deviation.
inline poplar::Tensor
layerNormWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
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

/// Layer normalise activations given mean, standard deviation and norm
/// parameters.
///
/// The result is two tensors:
///   1. normalised activations
///   2. whitened activations
inline std::pair<poplar::Tensor, poplar::Tensor>
layerNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
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
inline std::pair<poplar::Tensor, poplar::Tensor> layerNormParamGradients(
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
inline std::pair<poplar::Tensor, poplar::Tensor> layerNormParamGradients(
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

/// Compute gradients w.r.t input activations for the layer norm layer.
/// Gradients are propagated through the complete layer including
/// statistics computation.
inline poplar::Tensor
layerNormGradients(poplar::Graph &graph, const poplar::Tensor &acts,
                   const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
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

/// Compute gradients w.r.t input activations for the layer norm layer.
/// Gradients are propagated through the complete layer including
/// statistics computation.
inline poplar::Tensor
layerNormGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
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

/// Update layer norm parameters given the gradients w.r.t. parameters.
inline void layerNormParamUpdate(poplar::Graph &graph,
                                 const poplar::Tensor &gammaDelta,
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

inline void layerNormParamUpdate(poplar::Graph &graph,
                                 const poplar::Tensor &gammaDelta,
                                 const poplar::Tensor &betaDelta,
                                 const poplar::Tensor &scale,
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
} // namespace ln
} // namespace popnn
#endif // popnn_LayerNorm_hpp
