// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_Norms_hpp
#define poplin_Norms_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <tuple>

namespace poplin {

// Note that the functionality here should move to popnn once operations on
// broadcasting some of the dimensions is added. Currently these operations are
// optimised for tensors produced by convolutions/matrix multiplications.
// (see T6054)

/// Create and map the per channel multiplicative gamma parameter tensor used
/// for normalisation in convolution layers.
/// \param graph           The graph with the activations and gamma tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where\n
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field.
/// \returns               Gamma vector of dimension `C`.
poplar::Tensor createNormGamma(poplar::Graph &graph,
                               const poplar::Tensor &acts);

/// Create and map the per channel additive beta parameter tensor used for
/// normalisation in convolution layers.
/// \param graph           The graph with the activations and beta tensor.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where\n
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field
/// \returns               Beta vector of dimension `C`.
poplar::Tensor createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts);

/// Creates a tensor pair of normalisation parameters (gamma, beta).
/// \param graph           The graph with the activations and beta/gamma
///                        tensors.
/// \param acts            The activations tensor has shape `[N][C][..F..]`
///                        where\n
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field
/// \returns               A pair of vectors of dimension `C`.
std::pair<poplar::Tensor, poplar::Tensor>
createNormParams(poplar::Graph &graph, const poplar::Tensor &acts);

/// Compute the normalisation statistics from the activations tensor. The
/// activations tensor is of shape `[N][C][..F..]`. The mean and inverse
/// standard
/// deviation is computed over dimensions `{[N] [..F..]|` and vectors of
/// length `C` are returned as estimates.
///
/// The input activations tensor must be rearranged such that statistics are
/// computed for `C` channels.
/// \param graph          The graph in which the computation is performed.
/// \param actsUngrouped  The activation with shape `[N][C][..F..]`
///                       where\n
///                           - `N` is the batch size
///                           - `C` is the number of channels
///                           - `..F..` is dimensions of a N-dimensional field.
/// \param eps            The epsilon added to the variance to avoid divide by
///                       zero.
/// \param prog           A reference to the a program sequence which will be
///                       appended with the code to perform the normalisation.
/// \param unbiasedVarEstimate
///                       Compute unbiased variance estimate.
/// \param stableAlgo     If true, computes the mean first and subtracts
///                       the activations by it before computing the variance.
///                       The implementation with this flag set to true is
//                        slower than when set to false.
/// \param partialsType   Poplar type used for partials.
/// \param debugPrefix    A debug prefix added to compute set and tensor names.
///
/// \returns             A vector pair with mean and inverse standard deviation.
std::pair<poplar::Tensor, poplar::Tensor>
normStatistics(poplar::Graph &graph, const poplar::Tensor &actsUngrouped,
               float eps, poplar::program::Sequence &prog,
               bool unbiasedVarEstimate, bool stableAlgo = false,
               const poplar::Type &partialsType = poplar::FLOAT,
               const std::string &debugPrefix = "");

/// Compute the whitened activations using the supplied mean and inverse
/// standard deviation.
///
/// The input activations undergo a prior rearrangement such that `C`
/// is the size of the statistics mean and iStdDev.
/// \param graph          The graph which the computation is in.
/// \param acts           The activations tensor of shape [N][C][..F..].
/// \param mean           Mean of the activations with dimension C.
/// \param iStdDev        Inverse standard deviation with dimension C.
/// \param prog           A reference to the a program sequence which will be
///                       appended with the code to perform the normalisation.
/// \param debugPrefix    A debug prefix added to compute set and tensor names.
///
/// \returns              Whitened activations.
poplar::Tensor normWhiten(poplar::Graph &graph, const poplar::Tensor &acts,
                          const poplar::Tensor &mean,
                          const poplar::Tensor &iStdDev,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix);

/// Computes the normalised output given whitened activations.
/// \param graph         The graph to which the normalisaton operation is added.
/// \param actsWhitened  Whitened activations.
/// \param gamma         Per channel multiplicative normalisation parameter.
/// \param beta          Per channel additive normalisation parameter.
/// \param prog          A reference to the a program sequence which will be
///                      appended with the code to perform the normalisation.
/// \param debugPrefix   A debug prefix added to compute set and tensor names.
poplar::Tensor
normalise(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
          const poplar::Tensor &gamma, const poplar::Tensor &beta,
          poplar::program::Sequence &prog, const std::string &debugPrefix = "");

/// Compute gradients with respect to parameters required for parameter update.
/// \param graph         The graph to which the normalisaton operation is added.
/// \param actsWhitened  Whitened activations.
/// \param gradsIn       Input gradients to the normalisation layer.
/// \param prog          A reference to the a program sequence which will be
///                      appended with the code to perform the normalisation.
/// \param partialsType  The intermediate type kept in the computation.
/// \param debugPrefix   A debug prefix added to compute set and tensor names.
std::pair<poplar::Tensor, poplar::Tensor>
normParamGradients(poplar::Graph &graph, const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const std::string &debugPrefix = "");

/// Propagate the gradients through the normalisation layer.
/// \param graph         The graph to which the normalisaton operation is added.
/// \param gradsIn       Input gradients to the normalisation layer.
/// \param gamma         Multiplicative parameter used in the normalisation.
/// \param prog          A reference to the a program sequence which will be
///                      appended with the code to perform the normalisation.
/// \param debugPrefix   A debug prefix added to compute set and tensor names.
poplar::Tensor normGradients(poplar::Graph &graph,
                             const poplar::Tensor &gradsIn,
                             const poplar::Tensor &gamma,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "");

/// Propagate the gradients through the norm statistics layer. The input to the
/// layer is the output gradients from the normalisation layer. The whitened
/// activations and the input gradients must have undergone a prior
/// rearrangement such that the channel dimension has the same elements as
/// \p invStdDev.
/// \param graph        The graph to which the normalisaton operation is added.
/// \param actsWhitened Forward whitened activations.
/// \param gradsIn      Input gradients to the normalisation layer.
/// \param invStdDev    Inverse standard deviation from norm statistics.
/// \param prog         A reference to the a program sequence which will be
///                     appended with the code to perform the normalisation.
/// \param debugPrefix  A debug prefix added to compute set and tensor names.
poplar::Tensor normStatisticsGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, const poplar::Tensor &invStdDev,
    poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const std::string &debugPrefix = "");

} // namespace poplin

#endif // poplin_Norms_hpp
