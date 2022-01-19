// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *  Group normalization operations.
 */

#ifndef popnn_GroupNorm_hpp
#define popnn_GroupNorm_hpp
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include <poplar/OptionFlags.hpp>
#include <utility>

namespace popnn {
namespace gn {

/** Estimate mean and inverse of standard deviation of activations.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The activations for which the mean and variance are
 *                      estimated.
 * \param eps               The epsilon value added to the variance to avoid
 *                          division by zero.
 * \param prog              The program sequence to add the operation to.
 * \param numGroups     The number of groups to split the channel dimension
 *                      into when calculating group norm statistics. The
 *                      \c groupNormStridedChannelGrouping option defines
 *                      how the split is made.
 * \param unbiasedVarEstimate If true, an unbiased variance estimate will be
 *                            computed.
 * \param stableAlgo          If true, computes the mean first then subtracts
 *                            the activations from it before computing the
 *                            variance. The implementation with this flag set to
 *                            true is slower than when set to false.
 *  \param partialsType       Poplar type used for intermediate values.
 *                            If the type specified is smaller than the input/
 *                            output type then \p partialsType is ignored and
 *                            the input/output type is used instead.
 * \param debugContext        Optional debug information.
 * \param options             Group normalisation options. See groupNormalise().
 *
 * \returns                   A vector pair with mean and inverse standard
 *                            deviation.
 */
std::pair<poplar::Tensor, poplar::Tensor>
groupNormStatistics(poplar::Graph &graph, const poplar::Tensor acts, float eps,
                    poplar::program::Sequence &prog, unsigned numGroups,
                    bool unbiasedVarEstimate, bool stableAlgo = false,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

/** Whiten activations given the mean and standard deviation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The input activations that will be whitened.
 * \param mean          The previously calculated mean to subtract from the
 *                      activations.
 *                      Typically calculated using groupNormStatistics().
 * \param invStdDev     The previously calculated inverse standard deviation
 *                      to multiply the activations by.
 *                      Typically calculated using groupNormStatistics().
 * \param prog          The program sequence to add the operation to.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options. See groupNormalise().
 *
 * \returns             A new tensor with the whitened activations.
 */
poplar::Tensor groupNormWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::Tensor &mean,
                               const poplar::Tensor &invStdDev,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {},
                               const poplar::OptionFlags &options = {});

/** Group normalise activations given the mean, standard deviation and group
 *  norm parameters.
 *
 * **Group normalisation options**
 *
 *    * `groupNormStridedChannelGrouping` (true, false) [=true]
 *
 *      Select groups of channels for group normalisation with a stride between
 *      channels.  This makes the implementation more efficient but is
 *      unconventional.  Among other things this will mean that using
 *      pre-trained weights would not be possible if not produced with this
 *      unconventional implementation.
 *
 *      If we have `numGroups` groups then the channels in the group
 *      `groups[groupIdx]` are given by:
 *
 *      * Strided channel grouping: channelInGroupIdx * numGroups + groupIdx
 *      * Otherwise: channelInGroupIdx + channelsPerGroup * groupIdx
 *
 *      In the case of instanceNormalise() and layerNormalise() (which use group
 *      norm in their implementation) this option will have no effect.
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
 * \param prog          The program sequence to add the operation to.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options.
 *
 * \returns Two tensors containing:
 *          * normalised activations
 *          * whitened activations
 */
std::pair<poplar::Tensor, poplar::Tensor>
groupNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
               const poplar::Tensor &gamma, const poplar::Tensor &beta,
               const poplar::Tensor &mean, const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               const poplar::OptionFlags &options = {});

/** Compute gradients with respect to parameters for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using groupNormStatistics().
 * \param iStdDev       The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using groupNormStatistics().
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  Poplar type used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options. See groupNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
std::pair<poplar::Tensor, poplar::Tensor> groupNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &iStdDev, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Compute gradients with respect to parameters for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass whitened activation inputs to this
 *                      layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  Poplar type used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options. See groupNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
std::pair<poplar::Tensor, poplar::Tensor> groupNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Compute gradients with respect to input activations for the group norm
 * layer. Gradients are propagated through the complete layer including
 * statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using groupNormStatistics().
 * \param invStdDev     The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using groupNormStatistics().
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  Poplar type used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options. See groupNormalise().
 *
 * \returns A tensor containing the gradients with respect to the input
 *          activations for this layer.
 */
poplar::Tensor
groupNormGradients(poplar::Graph &graph, const poplar::Tensor &acts,
                   const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/** Compute gradients with respect to input activations for the group norm
 *  layer. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param invStdDev     The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using groupNormStatistics().
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  Poplar type used for intermediate values.
 *                      If the type specified is smaller than the input/output
 *                      type then \p partialsType is ignored and the
 *                      input/output type is used instead.
 * \param debugContext  Optional debug information.
 * \param options       Group normalisation options. See groupNormalise().
 *
 * \returns A tensor containing the gradients with respect to the input
 *          activations for this layer.
 */
poplar::Tensor
groupNormGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/** Update parameters for the group norm layer.
 *  Gradients are propagated through the complete layer including
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
 * \param options            Group normalisation options. See groupNormalise().
 */
void groupNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta, float scale,
                          poplar::Tensor &gamma, poplar::Tensor &beta,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {});

/** Update parameters for the group norm layer.
 *  Gradients are propagated through the complete layer including
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
 * \param options            Group normalisation options. See groupNormalise().
 */
void groupNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta,
                          const poplar::Tensor &scale, poplar::Tensor &gamma,
                          poplar::Tensor &beta, poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {});
} // namespace gn
} // namespace popnn
#endif // popnn_GroupNorm_hpp
