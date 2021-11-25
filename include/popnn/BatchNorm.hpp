// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *  Batch normalization operations.
 */

#ifndef popnn_BatchNorm_hpp
#define popnn_BatchNorm_hpp
#include "poplar/DebugContext.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplin/Norms.hpp"
#include <utility>

namespace popnn {
namespace bn {

/** Estimate mean and inverse of standard deviation of batched activations.
 *
 * \param graph          The graph that the normalisation operation is added to.
 * \param acts           The activations for which the mean and variance are
 *                       estimated.
 * \param eps            The epsilon value added to the variance to avoid
 *                       division by zero.
 * \param prog           The program sequence to add the operation to.
 * \param unbiasedVarEstimate If true, an unbiased variance estimate will be
 *                            computed.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. See batchNormalise().
 *
 * \returns             A vector pair with mean and inverse standard deviations.
 */
std::pair<poplar::Tensor, poplar::Tensor>
batchNormStatistics(poplar::Graph &graph, const poplar::Tensor acts, float eps,
                    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
                    bool stableAlgo = false,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

/** Compute the batch normalisation statistics for a part of the activations
 *  tensor. \p normBatchSize batch elements are distributed over
 *  multiple replicas. Each replica gets equal-sized batches (`B`).
 *  A callback does the required reduction over multiple replicas.
 *  The activations tensor is of shape `[B][C][..F..]`. The mean and inverse
 *  standard deviation are computed over dimensions `{[B] [..F..]}` and vectors
 *  of length `C` are returned as estimates.
 *
 *  \param replicatedGraph
 *                        The replicated graph in which the computation is
 *                        performed.
 *  \param acts           The activation tensor
 *                        with shape `[B][C][..F..]` \n
 *                        where:
 *                            - `B` is the batch size
 *                            - `C` is the number of channels
 *                            - `..F..` are the dimensions of an N-dimensional
 *                              field.
 *  \param eps            The epsilon value added to the variance to avoid
 *                        division by zero.
 *  \param prog           A program sequence that the code to
 *                        perform the normalisation will be appended to.
 *  \param unbiasedVarEstimate
 *                        If true an unbiased variance estimate will be
 *                        computed.
 *  \param stableAlgo     If true, computes the mean first then subtracts
 *                        the activations from it before computing the variance.
 *                        The implementation with this flag set to true is
 *                        slower than when set to false.
 *  \param partialsType   Poplar type used for partials.
 *  \param allReduceCallback
 *                        Callback to perform all-reduce over \p normBatchSize
 *                        batch elements.
 *  \param normBatchSize  Number of batch elements over which statistics
 *                        are estimated.
 *  \param debugContext   Optional debug information.
 *
 *  \returns             A vector pair with mean and inverse standard deviation.
 */
std::pair<poplar::Tensor, poplar::Tensor> distributedBatchNormStatistics(
    poplar::Graph &replicatedGraph, const poplar::Tensor acts, float eps,
    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, bool stableAlgo = false,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Whiten activations given the mean and standard deviation.
 *
 * \param graph          The graph that the normalisation operation is added to.
 * \param acts           The input activations that will be whitened.
 * \param mean           The previously calculated mean to subtract from the
 *                       activations.
 *                       Typically calculated using batchNormStatistics().
 * \param invStdDev      The previously calculated inverse standard deviation
 *                       to multiply the activations by.
 *                       Typically calculated using batchNormStatistics().
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. See batchNormalise().
 *
 * \returns               A new tensor with the whitened activations.
 */
poplar::Tensor batchNormWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::Tensor &mean,
                               const poplar::Tensor &invStdDev,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {},
                               const poplar::OptionFlags &options = {});

/** Batch normalise the activations using the given mean, standard deviation
 * and batch norm parameters.
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
 * \param options            Batch normalisation options. Presently, there are
 *                           no options that affect the operation of batch norm.
 *
 * \returns Two tensors containing:
 *          * normalised activations
 *          * whitened activations
 */
std::pair<poplar::Tensor, poplar::Tensor>
batchNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
               const poplar::Tensor &gamma, const poplar::Tensor &beta,
               const poplar::Tensor &mean, const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               const poplar::OptionFlags &options = {});

/** Computes the batch normalisation from a combined multiplicand and addend.
 *
 * \param graph          The graph that the normalisation operation is added to.
 * \param acts           The input activations that will be normalised using
 *                       the combined multiplicand and addend.
 * \param combinedMultiplicand = gamma * invStdDev
 * \param addend = beta - (gamma * mean * invStdDev)
 * \param prog               The program sequence to add the operation to.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. Presently, there are
 *                           no options that affect the operation of batch norm.
 *
 * \returns A new tensor with the normalised activations.
 */
poplar::Tensor batchNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
                              const poplar::Tensor &combinedMultiplicand,
                              const poplar::Tensor &addend,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext = {},
                              const poplar::OptionFlags &options = {});

/** Compute gradients with respect to parameters required for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using batchNormStatistics().
 * \param iStdDev       The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using batchNormStatistics().
 * \param prog               The program sequence to add the operation to.
 * \param partialsType       The Poplar type to be used for intermediate values.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. See batchNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
std::pair<poplar::Tensor, poplar::Tensor> batchNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &iStdDev, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Compute gradients with respect to parameters required for parameter update.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass whitened activation inputs to this
 *                      layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param prog          The program sequence to add the operation to.
 * \param partialsType  The Poplar type to be used for intermediate values.
 * \param debugContext  Optional debug information.
 * \param options       Batch normalisation options. See batchNormalise().
 *
 * \returns A pair of tensors, \c gammaDelta and \c betaDelta which are the
 * gradients with respect to \c gamma and \c beta.
 */
std::pair<poplar::Tensor, poplar::Tensor> batchNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Compute gradients with respect to input activations for the batch norm
 *  layer. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param acts          The forward-pass activation inputs to this layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param mean          The mean of the \p acts tensor, typically calculated
 *                      using batchNormStatistics().
 * \param invStdDev     The inverse standard deviation of the \p acts tensor,
 *                      typically calculated using batchNormStatistics().
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog               The program sequence to add the operation to.
 * \param partialsType       The Poplar type to be used for intermediate values.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. See batchNormalise().
 *
 * \returns     A tensor containing the gradients with respect to the input
 *              activations for this layer.
 */
poplar::Tensor
batchNormGradients(poplar::Graph &graph, const poplar::Tensor &acts,
                   const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/** Compute gradients with respect to input activations for the batch norm
 *  layer. Gradients are propagated through the complete layer including
 *  statistics computation.
 *
 * \param graph         The graph that the normalisation operation is added to.
 * \param actsWhitened  The forward-pass whitened activation inputs to this
 *                      layer.
 * \param gradsIn       The gradient with respect to the output of this layer.
 * \param invStdDev     The inverse standard deviation to multiply by when
 *                      whitening the activations.
 * \param gamma         The gamma weights to multiply by when normalising the
 *                      whitened activations.
 * \param prog               The program sequence to add the operation to.
 * \param partialsType       The Poplar type to be used for intermediate values.
 * \param debugContext       Optional debug information.
 * \param options            Batch normalisation options. See batchNormalise().
 *
 * \returns     A tensor containing the gradients with respect to the input
 *              activations for this layer.
 */
poplar::Tensor
batchNormGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/// Propagate the gradients through the batch norm layer where equal-sized
/// batch elements are distributed over replicas to effectively compute the
/// batch norm over \p normBatchSize elements.
/// Each replica gets the same number of batches (\a N) with
/// \p normBatchSize = \a N * \a number-of-devices
///
/// A callback does the required reduction over the replicas the norm is spread
/// over.
///
/// The input to the layer is the output gradients from the normalisation layer.
/// The whitened activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension is the same as
/// \p invStdDev.
///
/// \param replicatedGraph
///                     The replicated graph to which the normalisation
///                     operation is added.
/// \param actsWhitened The forward-pass whitened activation inputs to this
///                     layer.
/// \param gradsIn      The gradient with respect to the output of this layer.
/// \param invStdDev    The inverse standard deviation of the \p acts tensor,
///                     typically calculated using batchNormStatistics().
/// \param gamma        The gamma weights to multiply by when normalising the
///                     whitened activations.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param reduceCallback
///                     A callback to perform all-reduce of the statistics
///                     gradients.
/// \param normBatchSize
///                     The batch size over which the norm is done.
/// \param debugContext Optional debug information.
///
/// \returns A tensor containing the gradients with respect to the input
///          activations for this layer.
///
poplar::Tensor distributedBatchNormGradients(
    poplar::Graph &replicatedGraph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    const poplar::Tensor &gamma, poplar::program::Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/// Propagate the gradients through the batch norm layer where equal-sized
/// batch elements are distributed over replicas to effectively compute the
/// batch norm over \p normBatchSize elements.
/// Each replica gets the same number of batches (\a N) with
/// \p normBatchSize = \a N * \a number-of-devices
///
/// A callback does the required reduction over the replicas the norm is spread
/// on.
///
/// The input to the layer is the output gradients from the normalisation layer.
/// The activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension has the same elements as
/// \p invStdDev. The activations are whitened within the function by
/// applying the \p mean and \p invStdDev.
///
/// \param replicatedGraph
///                     The replicated graph to which the normalisation
///                     operation is added.
/// \param acts         The forward-pass activation inputs to this layer.
/// \param gradsIn      The gradient with respect to the output of this layer.
/// \param mean         The mean of the \p acts tensor, typically calculated
///                     using batchNormStatistics().
/// \param invStdDev    The inverse standard deviation of the \p acts tensor,
///                     typically calculated using batchNormStatistics().
/// \param gamma        The gamma weights to multiply by when normalising the
///                     whitened activations.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param reduceCallback
///                     A callback to perform all-reduce of the statistics
///                     gradients.
/// \param normBatchSize
///                     The batch size over which the norm is done.
/// \param debugContext Optional debug information.
///
/// \returns A tensor containing the gradients with respect to the input
///          activations for this layer.
///
poplar::Tensor distributedBatchNormGradients(
    poplar::Graph &replicatedGraph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
    poplar::program::Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Update the parameters for the batch norm layer.
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
 * \param prog          The program sequence to add the operation to.
 * \param debugContext  Optional debug information.
 * \param options       Batch normalisation options. See batchNormalise().
 */
void batchNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta, float scale,
                          poplar::Tensor &gamma, poplar::Tensor &beta,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {});

/** Update parameters for the batch norm layer.
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
 * \param prog          The program sequence to add the operation to.
 * \param debugContext  Optional debug information.
 * \param options       Batch normalisation options. See batchNormalise().
 */
void batchNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta,
                          const poplar::Tensor &scale, poplar::Tensor &gamma,
                          poplar::Tensor &beta, poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {});
} // namespace bn
} // namespace popnn
#endif // popnn_BatchNorm_hpp
