// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popnn_BatchNorm_hpp
#define popnn_BatchNorm_hpp
#include "poplar/DebugContext.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplin/Norms.hpp"
#include <utility>

/* **Batch normalisation options**
 *
 * Presently there are no options that affect the operation of batch norm.
 * Options are included in the header prototypes for consistency with other
 * normalisation functions.
 */

namespace popnn {
namespace bn {

/// Estimate mean and inverse of standard deviation of batched activations.
std::pair<poplar::Tensor, poplar::Tensor>
batchNormStatistics(poplar::Graph &graph, const poplar::Tensor acts, float eps,
                    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
                    bool stableAlgo = false,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

/// Compute the batch normalisation statistics for a part of the activations
/// tensor where the `normBatchSize` batch elements are distributed over
/// multiple replicas. Each replica gets equal sized batches (`N`).
/// A callback does the required reduction over multiple replicas.
/// The activations tensor is of shape `[N][C][..F..]`. The mean and inverse
/// standard deviation is computed over dimensions `{[N] [..F..]}` and vectors
/// of length `C` are returned as estimates.
///
/// \param replicatedGraph
///                       The replicated graph in which the computation is
///                       performed.
/// \param acts           The activation with shape `[N][C][..F..]`
///                       where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field.
/// \param eps            The epsilon added to the variance to avoid divide by
///                       zero.
/// \param prog           A program sequence that the code to
///                       perform the normalisation will be appended to.
/// \param unbiasedVarEstimate
///                       Compute unbiased variance estimate.
/// \param stableAlgo     If true, computes the mean first and subtracts
///                       the activations by it before computing the variance.
///                       The implementation with this flag set to true is
//                        slower than when set to false.
/// \param partialsType   Poplar type used for partials.
/// \param allReduceCallback
///                       Callback to perform all-reduce over 'normBatchSize'
///                       batch elements.
/// \param normBatchSize  Number of batch elements over which statistics
///                       are estimated.
/// \param debugContext   Optional debug information.
///
/// \returns             A vector pair with mean and inverse standard deviation.
std::pair<poplar::Tensor, poplar::Tensor> distributedBatchNormStatistics(
    poplar::Graph &replicatedGraph, const poplar::Tensor acts, float eps,
    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, bool stableAlgo = false,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/// Whiten activations given mean and standard deviation.
poplar::Tensor batchNormWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::Tensor &mean,
                               const poplar::Tensor &invStdDev,
                               poplar::program::Sequence &prog,
                               const poplar::DebugContext &debugContext = {},
                               const poplar::OptionFlags &options = {});

/// Batch normalise activations given mean, standard deviation and batch norm
/// parameters.
/// The result is two tensors
///   1. normalised activations
///   2. whitened activations
std::pair<poplar::Tensor, poplar::Tensor>
batchNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
               const poplar::Tensor &gamma, const poplar::Tensor &beta,
               const poplar::Tensor &mean, const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               const poplar::OptionFlags &options = {});

/// Computes the output of batch normalisation given:
///   - combinedMultiplicand = gamma / stdDev
///   - addend = beta - gamma * mean / stdDev
poplar::Tensor batchNormalise(poplar::Graph &graph, const poplar::Tensor &acts,
                              const poplar::Tensor &combinedMultiplicand,
                              const poplar::Tensor &addend,
                              poplar::program::Sequence &prog,
                              const poplar::DebugContext &debugContext = {},
                              const poplar::OptionFlags &options = {});

/// Compute gradients w.r.t parameters required for parameter update.
std::pair<poplar::Tensor, poplar::Tensor> batchNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &iStdDev, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/// Compute gradients w.r.t parameters required for parameter update.
std::pair<poplar::Tensor, poplar::Tensor> batchNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/// Compute gradients w.r.t input activations for the batch norm layer.
/// i.e. gradients are propagated through the complete layer including
/// statistics computation.
poplar::Tensor
batchNormGradients(poplar::Graph &graph, const poplar::Tensor &acts,
                   const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/// Compute gradients w.r.t input activations for the batch norm layer.
/// i.e. gradients are propagated through the complete layer including
/// statistics computation.

/// Compute gradients w.r.t input activations for the batch norm layer.
/// i.e. gradients are propagated through the complete layer including
/// statistics computation.
poplar::Tensor
batchNormGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {});

/// Propagate the gradients through the batch norm layer where equal sized
/// batch elements are distributed over replicas to effectively compute the
/// batch norm over `normBatchSize` elements.
/// Each replica gets the same number of batches (`N') with
/// `normBatchSize` = `N` * number of devices
/// A callback does the required reduction over the replicas the norm is spread
/// over.
///
/// The input to the layer is the output gradients from the normalisation layer.
/// The whitened activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension is the same as
/// \p invStdDev.
/// \param replicatedGraph
///                     The replicated graph to which the normalisaton operation
///                     is added.
/// \param actsWhitened Forward whitened activations.
/// \param gradsIn      Input gradients to the normalisation layer.
/// \param invStdDev    Inverse standard deviation from norm statistics.
/// \param gamma        Parameter gamma.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param reduceCallback
///                     A call back to perform all reduce of the statistics
///                     gradients.
/// \param normBatchSize
///                     The batch size over which the norm is done.
/// \param debugContext Optional debug information.
poplar::Tensor distributedBatchNormGradients(
    poplar::Graph &replicatedGraph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    const poplar::Tensor &gamma, poplar::program::Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/// Propagate the gradients through the batch norm layer where equal sized
/// batch elements are distributed over replicas to effectively compute the
/// batch norm over `normBatchSize` elements.
/// Each replica gets the same number of batches (`N') with
/// `normBatchSize` = `N` * number of replicas
/// A callback does the required reduction over the replicas the norm is spread
/// on.
///
/// The input to the layer is the output gradients from the normalisation layer.
/// The activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension has the same elements as
/// \p invStdDev. The activations are whitened within the function by
/// applying the \p mean and \p invStdDev.
/// \param replicatedGraph
///                     The replicated graph to which the normalisaton operation
///                     is added.
/// \param acts         Inputs to the batch norm layer.
/// \param gradsIn      Input gradients to the normalisation layer.
/// \param mean         Estimated mean.
/// \param invStdDev    Inverse standard deviation from norm statistics.
/// \param gamma        Parameter gamma.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param reduceCallback
///                     A call back to perform all reduce of the statistics
///                     gradients.
/// \param normBatchSize
///                     The batch size over which the norm is done.
/// \param debugContext Optional debug information.
poplar::Tensor distributedBatchNormGradients(
    poplar::Graph &replicatedGraph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &invStdDev, const poplar::Tensor &gamma,
    poplar::program::Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback,
    unsigned normBatchSize, const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

void batchNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta, float scale,
                          poplar::Tensor &gamma, poplar::Tensor &beta,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {});

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
