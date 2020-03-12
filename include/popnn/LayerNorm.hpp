// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popnn_LayerNorm_hpp
#define popnn_LayerNorm_hpp
#include "popnn/GroupNorm.hpp"

// Layer norm uses group norm with number of groups = 1

namespace popnn {
namespace ln {

/// Estimate mean and inverse of standard deviation of activations.
inline std::pair<poplar::Tensor, poplar::Tensor>
layerNormStatistics(poplar::Graph &graph, const poplar::Tensor acts, float eps,
                    poplar::program::Sequence &prog, bool unbiasedVarEstimate,
                    bool stableAlgo = false,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    const std::string &debugPrefix = "") {
  return popnn::gn::groupNormStatistics(graph, acts, eps, prog, 1,
                                        unbiasedVarEstimate, stableAlgo,
                                        partialsType, debugPrefix);
}

/// Whiten activations given mean and standard deviation.
inline poplar::Tensor layerNormWhiten(poplar::Graph &graph,
                                      const poplar::Tensor &acts,
                                      const poplar::Tensor &mean,
                                      const poplar::Tensor &invStdDev,
                                      poplar::program::Sequence &prog,
                                      const std::string &debugPrefix = "") {
  return popnn::gn::groupNormWhiten(graph, acts, mean, invStdDev, prog,
                                    debugPrefix);
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
               const std::string &debugPrefix = "") {
  return popnn::gn::groupNormalise(graph, acts, gamma, beta, mean, invStdDev,
                                   prog, debugPrefix);
}

/// Compute gradients w.r.t parameters for parameter update.
inline std::pair<poplar::Tensor, poplar::Tensor> layerNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &acts,
    const poplar::Tensor &gradsIn, const poplar::Tensor &mean,
    const poplar::Tensor &iStdDev, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                            prog, partialsType, debugPrefix);
}

/// Compute gradients w.r.t parameters for parameter update.
inline std::pair<poplar::Tensor, poplar::Tensor> layerNormParamGradients(
    poplar::Graph &graph, const poplar::Tensor &actsWhitened,
    const poplar::Tensor &gradsIn, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, debugPrefix);
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
                   const std::string &debugPrefix = "") {
  return popnn::gn::groupNormGradients(graph, acts, gradsIn, mean, invStdDev,
                                       gamma, prog, partialsType, debugPrefix);
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
                   const std::string &debugPrefix = "") {
  return popnn::gn::groupNormGradients(graph, actsWhitened, gradsIn, invStdDev,
                                       gamma, prog, partialsType, debugPrefix);
}

/// Update layer norm parameters given the gradients w.r.t. parameters.
inline void layerNormParamUpdate(poplar::Graph &graph,
                                 const poplar::Tensor &gammaDelta,
                                 const poplar::Tensor &betaDelta, float scale,
                                 poplar::Tensor &gamma, poplar::Tensor &beta,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, scale,
                                         gamma, beta, prog, debugPrefix);
}

inline void layerNormParamUpdate(poplar::Graph &graph,
                                 const poplar::Tensor &gammaDelta,
                                 const poplar::Tensor &betaDelta,
                                 const poplar::Tensor &scale,
                                 poplar::Tensor &gamma, poplar::Tensor &beta,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamUpdate(graph, gammaDelta, betaDelta, scale,
                                         gamma, beta, prog, debugPrefix);
}
} // namespace ln
} // namespace popnn
#endif // popnn_LayerNorm_hpp
