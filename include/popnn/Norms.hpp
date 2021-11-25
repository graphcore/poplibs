// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *  Normalisation operations.
 */

#ifndef popnn_Norms_hpp
#define popnn_Norms_hpp
#include "poplar/DebugContext.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"

namespace popnn {

/// Calculate the floating point operations required for the forward pass of a
/// norm layer. For inference with batchNormalise(), \p computeStats should be
/// set to false if the batch statistics are not computed. This is because
/// averaged batch statistics may be combined with the norm parameters.
///
/// \param statisticsSize   The size of the statistics vector.
/// \param numActsElements  The number of elements in the activation inputs.
/// \param computeStats     Set to false for inference with batch norm.
/// \return                 Number of floating point operations.
std::uint64_t getNormFwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements,
                              bool computeStats = true);

/// Calculate the floating point operations required for computation of the
/// gradient with respect to the activations for a norm layer.
///
/// \param statisticsSize   The size of the statistics vector.
/// \param numActsElements  The number of elements in the activation inputs.
/// \return                 Number of floating point operations.
std::uint64_t getNormBwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements);

/// Calculate the floating point operations required for parameter update for a
/// norm layer.
///
/// \param paramsSize       The size of the parameter vector.
/// \param numActsElements  The number of elements in the activation inputs.
/// \return                 Number of floating point operations.
std::uint64_t getNormWuFlops(std::size_t paramsSize,
                             std::size_t numActsElements);

/// \param graph         The graph that the normalisation operation is added to.
/// \param acts             Activations that are inputs to the norm.
/// \param debugContext     Optional debug information.
/// \return                 The gamma values for the activations.
poplar::Tensor createNormGamma(poplar::Graph &graph, const poplar::Tensor &acts,
                               const poplar::DebugContext &debugContext = {});

/// \param graph         The graph that the normalisation operation is added to.
/// \param acts             Activations that are inputs to the norm.
/// \param debugContext     Optional debug information.
/// \return                 The beta values for the activations.
poplar::Tensor createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts,
                              const poplar::DebugContext &debugContext = {});

/// \param graph         The graph that the normalisation operation is added to.
/// \param acts             Activations that are inputs to the norm.
/// \param debugContext     Optional debug information.
/// \return             A pair of tensors containing the gamma and beta values
///                     for the activations.
std::pair<poplar::Tensor, poplar::Tensor>
createNormParams(poplar::Graph &graph, const poplar::Tensor acts,
                 const poplar::DebugContext &debugContext = {});

} // namespace popnn
#endif // popnn_Norms_hpp
