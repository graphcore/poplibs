// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions to support normalising values in a tensor.
 *
 */

#ifndef poplin_Norms_hpp
#define poplin_Norms_hpp
#include <functional>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <tuple>

namespace poplin {

// Note that the functionality here should move to popnn once operations on
// broadcasting some of the dimensions is added. Currently these operations are
// optimised for tensors produced by convolutions/matrix multiplications.
// (see T6054)

/// Create and map the per-channel multiplicative gamma parameter tensor used
/// for normalisation in convolution layers.
/// \param graph           The graph with the activations and gamma tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field.
/// \param type            The type of the output tensor.
/// \param debugContext   Optional debug information.
/// \returns               Gamma vector of dimension `C`.
poplar::Tensor createNormGamma(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::Type &type,
                               const poplar::DebugContext &debugContext = {});

/// Create and map the per-channel multiplicative gamma parameter tensor used
/// for normalisation in convolution layers.
/// \param graph           The graph with the activations and gamma tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field.
/// \param debugContext   Optional debug information.
/// \returns               Gamma vector of dimension `C`.
poplar::Tensor createNormGamma(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::DebugContext &debugContext = {});

/// Create and map the per-channel additive beta parameter tensor used for
/// normalisation in convolution layers.
/// \param graph           The graph with the activations and beta tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field
/// \param type            The type of the output tensor.
/// \param debugContext   Optional debug information.
/// \returns               Beta vector of dimension `C`.
poplar::Tensor createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts,
                              const poplar::Type &type,
                              const poplar::DebugContext &debugContext = {});

/// Create and map the per-channel additive beta parameter tensor used for
/// normalisation in convolution layers.
/// \param graph           The graph with the activations and beta tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field
/// \param debugContext   Optional debug information.
/// \returns               Beta vector of dimension `C`.
poplar::Tensor createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts,
                              const poplar::DebugContext &debugContext = {});

/// Creates a tensor pair of normalisation parameters (gamma, beta).
/// \param graph           The graph with the activations and beta/gamma
///                        tensors.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where:
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field
/// \param debugContext   Optional debug information.
/// \returns               A pair of vectors of dimension `C`.
std::pair<poplar::Tensor, poplar::Tensor>
createNormParams(poplar::Graph &graph, const poplar::Tensor &acts,
                 const poplar::DebugContext &debugContext = {});

/// Compute the normalisation statistics from the activations tensor. The
/// activations tensor is of shape `[N][C][..F..]`. The mean and inverse
/// standard
/// deviation is computed over dimensions `{[N] [..F..]}` and vectors of
/// length `C` are returned as estimates.
///
/// The input activations tensor must be rearranged such that statistics are
/// computed for `C` channels.
/// \param graph          The graph in which the computation is performed.
/// \param actsUngrouped  The activation with shape `[N][C][..F..]`
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
/// \param debugContext   Optional debug information.
///
/// \returns             A vector pair with mean and inverse standard deviation.
std::pair<poplar::Tensor, poplar::Tensor>
normStatistics(poplar::Graph &graph, const poplar::Tensor &actsUngrouped,
               float eps, poplar::program::Sequence &prog,
               bool unbiasedVarEstimate, bool stableAlgo = false,
               const poplar::Type &partialsType = poplar::FLOAT,
               const poplar::DebugContext &debugContext = {});

/// Callback to reduce statistics and gradients. The reduce operation is
/// reduce-add.
/// \param graph     The replicated graph in which the computation is performed.
/// \param inputsToReduce
///                  A vector of independent tensors to reduce
/// \param prog      A program sequence that the code to perform the
///                  normalisation will be appended to.
/// \param groupSize The number of replicas that need to be reduced. This may
///                  be less than the total number of replicas in the top level
///                  graph. A group is formed by adjacent replicas such that
///                  the top level graph contains an integral number of
///                  `groupSize` replicas.
/// \param debugContext
///                  Optional debug information.
/// \param options   The structure describing options on how the reduction
///                   should be implemented.
/// \return  A vector of reduced tensors in the same order as supplied in
///          `inputsToReduce`
using DistributedNormReduceCallback = std::function<std::vector<poplar::Tensor>(
    poplar::Graph &replicatedGraph,
    const std::vector<poplar::Tensor> &inputsToReduce,
    poplar::program::Sequence &prog, unsigned groupSize,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options)>;

/// Compute the normalisation statistics for a part of the activations tensor
/// which is distributed over multiple replicas. Each replica gets equal sized
/// batches (`N`) with normalisation done over `normSize` batches.
/// A callback does the required mean reduction over multiple replicas.
/// The activations tensor is of shape `[N][C][..F..]`. The mean and inverse
/// standard deviation is computed over dimensions `{[N] [..F..]}` and vectors
/// of length `C` are returned as estimates.
///
/// The input activations tensor must be rearranged such that statistics are
/// computed for `C` channels.
/// \param replicatedGraph
///                       The replicated graph in which the computation is
///                       performed.
/// \param actsUngrouped  The activation with shape `[N][C][..F..]`
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
///                       Callback to perform all-reduce over 'normSize'
///                       batch elements.
/// \param normSize       Number of batch elements over which statistics
///                       are estimated.
/// \param debugContext   Optional debug information.
///
/// \returns             A vector pair with mean and inverse standard deviation.
///
std::pair<poplar::Tensor, poplar::Tensor> distributedNormStatistics(
    poplar::Graph &replicatedGraph, const poplar::Tensor &actsUngrouped,
    float eps, poplar::program::Sequence &prog, bool unbiasedVarEstimate,
    DistributedNormReduceCallback allReduceCallback, unsigned normSize,
    bool stableAlgo = false, const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {});

/// Compute the whitened activations using the supplied mean and inverse
/// standard deviation.
///
/// The input activations undergo a prior rearrangement such that `C`
/// is the size of the statistics \p mean and \p iStdDev tensors.
/// \param graph          The graph which the computation is in.
/// \param acts           The activations tensor of shape [N][C][..F..].
/// \param mean           Mean of the activations with dimension C.
/// \param iStdDev        Inverse standard deviation with dimension C.
/// \param prog           A program sequence that the code to
///                       perform the normalisation will be appended to.
/// \param debugContext   Optional debug information.
///
/// \returns              A new tensor with the whitened activations.
///
poplar::Tensor normWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                          const poplar::Tensor &mean,
                          const poplar::Tensor &iStdDev,
                          poplar::program::Sequence &prog,
                          const poplar::DebugContext &debugContext = {});

/// Computes the normalised output from whitened activations.
/// \param graph        The graph to which the normalisation operation is added.
/// \param actsWhitened  Whitened activations.
/// \param gamma         Per-channel multiplicative normalisation parameter.
/// \param beta          Per-channel additive normalisation parameter.
/// \param prog          A program sequence that the code to
///                      perform the normalisation will be appended to.
/// \param debugContext   Optional debug information.
///
/// \returns             A tensor containing the normalised activations.
///
poplar::Tensor normalise(poplar::Graph &graph,
                         const poplar::Tensor &actsWhitened,
                         const poplar::Tensor &gamma,
                         const poplar::Tensor &beta,
                         poplar::program::Sequence &prog,
                         const poplar::DebugContext &debugContext = {});

/// Compute gradients with respect to parameters required for parameter update.
///
/// \param graph        The graph to which the normalisation operation is added.
/// \param actsWhitened  Whitened activations.
/// \param gradsIn       The gradient with respect to the output of this layer.
/// \param prog          A program sequence that the code to
///                      perform the normalisation will be appended to.
/// \param partialsType  The intermediate type kept in the computation.
/// \param debugContext   Optional debug information.
///
/// \returns            A pair of tensors, \c gammaDelta and \c betaDelta which
///                     are the gradients with respect to \c gamma and \c beta.
///
std::pair<poplar::Tensor, poplar::Tensor>
normParamGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const poplar::DebugContext &debugContext = {});

/// Propagate the gradients through the normalisation layer.
/// \param graph        The graph to which the normalisation operation is added.
/// \param gradsIn       The gradient with respect to the output of this layer.
/// \param gamma         Multiplicative parameter used in the normalisation.
/// \param prog          A program sequence that the code to
///                      perform the normalisation will be appended to.
/// \param debugContext   Optional debug information.
///
/// \returns            The gradient with respect to the input of this layer.
///
poplar::Tensor normGradients(poplar::Graph &graph,
                             const poplar::Tensor &gradsIn,
                             const poplar::Tensor &gamma,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext = {});

/// Propagate the gradients through the norm statistics layer. The input to the
/// layer is the output gradients from the normalisation layer. The whitened
/// activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension has the same elements as
/// \p invStdDev.
///
/// \param graph        The graph to which the normalisation operation is added.
/// \param actsWhitened Forward whitened activations.
/// \param gradsIn      The gradient with respect to the output of this layer.
/// \param invStdDev    Inverse standard deviation from norm statistics.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param debugContext   Optional debug information.
///
/// \returns            The gradient with respect to the input of this layer.
///
poplar::Tensor normStatisticsGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {});

/// Propagate the gradients through the norm statistics layer where equal sized
/// batch elements are distributed over replicas.
/// Each replica gets the same number of batches and norm gradients are computed
/// over `normSize` batch elements. Each replica is given
/// `N` batch elements. A callback does the required reduction over multiple
/// replicas.
///
/// The input to the layer is the output gradients from the normalisation layer.
/// The whitened activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension has the same elements as
/// \p invStdDev.
/// \param replicatedGraph
///                    The replicated graph to which the normalisation operation
///                     is added.
/// \param actsWhitened Forward whitened activations.
/// \param gradsIn      The gradient with respect to the output of this layer.
/// \param invStdDev    Inverse standard deviation from norm statistics.
/// \param prog         A program sequence that the code to
///                     perform the normalisation will be appended to.
/// \param reduceCallback
///                     A call back to perform all reduce of the statistics
///                     gradients across the replicas.
/// \param normSize     The batch size over which the norm is done.
/// \param debugContext Optional debug information.
///
/// \returns            The gradient with respect to the input of this layer.
///
poplar::Tensor distributedNormStatisticsGradients(
    poplar::Graph &replocatedGraph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    poplar::program::Sequence &prog,
    poplin::DistributedNormReduceCallback reduceCallback, unsigned normSize,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {});

} // namespace poplin

#endif // poplin_Norms_hpp
