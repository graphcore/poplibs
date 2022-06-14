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

namespace popnn {
namespace ln {

/** Estimate the mean and inverse of the standard deviation of the activations.
 *
 * \param graph          The graph that the normalisation operation is added to.
 * \param acts           The activations for which the mean and variance are
 *                       estimated.
 * \param eps            The epsilon value added to the variance to avoid
 *                       division by zero.
 * \param prog           The program sequence to add the operation to.
 * \param unbiasedVarEstimate If true, an unbiased variance estimate will be
 *                            computed.
 * \param stableAlgo          If true, computes the mean first then subtracts
 *                            the activations from it before computing the
 *                            variance. The implementation with this flag set to
 *                            true is slower than when set to false.
 * \param partialsType  Poplar type used for partial results.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Layer normalisation options. See groupNormalise().
 *
 * \returns                   A vector pair with mean and inverse standard
 *                            deviation.
 */
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

/** Whiten the activations using the mean and standard deviation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The input activations that will be whitened.
 * \param mean          The previously calculated mean to subtract from the
 *                      activations.
 *                      Typically calculated using layerNormStatistics().
 * \param invStdDev     The previously calculated inverse standard deviation
 *                      to multiply the activations by.
 *                      Typically calculated using layerNormStatistics().
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Layer normalisation options. See groupNormalise().
 *
 * \returns             A new tensor with the whitened activations.
 */
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

/** Layer normalise the activations using the mean, standard deviation and batch
 *  norm parameters.
 *
 * As layer normalise uses group normalise, options are passed through.
 * See the groupNormalise() documentation for details of the options.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The input activations to whiten and normalise,
 *                      with shape `[B][C][..F..]` \n
 *                      where:
 *                           - `B` is the batch size
 *                           - `C` is the number of channels
 *                           - `..F..` are the dimensions of an N-dimensional
 *                             field.
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param beta          The beta weights to add when normalising the whitened
 *                      activations.
 * \param mean          The mean to subtract when whitening the activations.
 * \param invStdDev     The inverse standard deviation to multiply by when
 *                      whitening the activations.
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Layer normalisation options. See groupNormalise().
 *
 * \returns Two tensors containing:
 *          * normalised activations
 *          * whitened activations
 */
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

/** Compute gradients with respect to the parameters for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using layerNormStatistics().
 * \param iStdDev       The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using layerNormStatistics().
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  The Poplar type to be used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Layer normalisation options. See groupNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
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

/** Compute gradients with respect to parameters for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass whitened activation inputs to this
 *                      layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  The Poplar type to be used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Layer normalisation options. See groupNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
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

/** Compute gradients with respect to the input activations for the layer norm
 * layer. Gradients are propagated through the complete layer including
 * statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using layerNormStatistics().
 * \param invStdDev     The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using layerNormStatistics().
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  The Poplar type to be used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Layer normalisation options. See groupNormalise().
 *
 * \returns A tensor containing the gradients with respect to the input
 *          activations for this layer.
 */
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

/** Compute gradients with respect to the input activations for the layer norm
 *  layer. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass whitened activation inputs to this
 *                      layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param invStdDev     The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using layerNormStatistics().
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  The Poplar type to be used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Layer normalisation options. See groupNormalise().
 *
 * \returns A tensor containing the gradients with respect to the input
 *          activations for this layer.
 */
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

/** Update layer norm parameters from the gradients with respect to the
 *  parameters. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * The \p gamma and \p beta parameters are updated as follows:
 * - \p gamma += \p gammaDelta * \p scale
 * - \p beta  += \p betaDelta * \p scale
 *
 * \p scale is a float and therefore constant.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param gammaDelta    Value used to update \p gamma.
 * \param betaDelta     Value used to update \p beta.
 * \param scale         Scale factor for \p gammaDelta and \p betaDelta.
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      activations.
 * \param beta          The beta weights to add when normalising the
 *                      activations.
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Layer normalisation options. See groupNormalise().
 */
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

/** Update the layer norm parameters from the gradients with respect to the
 *  parameters. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * The `gamma` and `beta` parameters are updated as follows:
 * - `gamma += gammaDelta * scale`
 * - `beta  += betaDelta * scale`
 *
 * \p scale is a tensor and therefore variable.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param gammaDelta    Value used to update \p gamma.
 * \param betaDelta     Value used to update \p beta.
 * \param scale         Scale factor for \p gammaDelta and \p betaDelta.
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      activations.
 * \param beta          The beta weights to add when normalising the
 *                      activations.
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Layer normalisation options. See groupNormalise().
 */
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
