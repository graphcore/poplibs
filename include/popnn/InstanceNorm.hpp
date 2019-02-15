// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popnn_InstanceNorm_hpp
#define popnn_InstanceNorm_hpp
#include "popnn/GroupNorm.hpp"

// Instance norm uses group norm with number of groups = number of channels

namespace popnn {
namespace in {

// Estimate mean and inverse of standard deviation of activtaions
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormStatistics(poplar::Graph &graph, const poplar::Tensor acts,
                       float eps,
                       poplar::program::Sequence &prog,
                       bool unbiasedVarEstimate,
                       const poplar::Type &partialsType= poplar::FLOAT,
                       const std::string &debugPrefix = "") {
  return popnn::gn::groupNormStatistics(graph, acts, eps, prog, acts.dim(1),
                                        unbiasedVarEstimate,
                                        partialsType, debugPrefix);
}

// Whiten activations given mean and standard deviation
inline poplar::Tensor
instanceNormWhiten(poplar::Graph &graph,
                   const poplar::Tensor &acts,
                   const poplar::Tensor &mean,
                   const poplar::Tensor &invStdDev,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "") {
  return popnn::gn::groupNormWhiten(graph, acts, mean, invStdDev, prog,
                                    debugPrefix);
}

// Instance normalise activations given mean, standard deviation and norm
// parameters.
// The result is two tensors
// 1. normalised activations
// 2. whitened activations
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormalise(poplar::Graph &graph,
                  const poplar::Tensor &acts,
                  const poplar::Tensor &gamma,
                  const poplar::Tensor &beta,
                  const poplar::Tensor &mean,
                  const poplar::Tensor &invStdDev,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "") {
  return popnn::gn::groupNormalise(graph, acts, gamma, beta, mean, invStdDev,
                                   prog, debugPrefix);
}

// Compute gradients w.r.t parameters for parameter update
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormParamGradients(poplar::Graph &graph,
                           const poplar::Tensor &acts,
                           const poplar::Tensor &gradsIn,
                           const poplar::Tensor &mean,
                           const poplar::Tensor &iStdDev,
                           poplar::program::Sequence &prog,
                           const poplar::Type &partialsType = poplar::FLOAT,
                           const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamGradients(graph, acts, gradsIn, mean, iStdDev,
                                            prog, partialsType, debugPrefix);
}

// Compute gradients w.r.t parameters for parameter update
inline std::pair<poplar::Tensor, poplar::Tensor>
instanceNormParamGradients(poplar::Graph &graph,
                           const poplar::Tensor &actsWhitened,
                           const poplar::Tensor &gradsIn,
                           poplar::program::Sequence &prog,
                           const poplar::Type &partialsType = poplar::FLOAT,
                           const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, debugPrefix);
}

// Compute gradients w.r.t input activations for the instance norm layer.
// i.e. gradients are propagated through the complete layer including
// statistics computation.
inline poplar::Tensor
instanceNormGradients(poplar::Graph &graph,
                      const poplar::Tensor &acts,
                      const poplar::Tensor &gradsIn,
                      const poplar::Tensor &mean,
                      const poplar::Tensor &invStdDev,
                      const poplar::Tensor &gamma,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const std::string &debugPrefix = "") {
  return popnn::gn::groupNormGradients(graph, acts, gradsIn, mean, invStdDev,
                                       gamma, prog, partialsType, debugPrefix);
}

// Compute gradients w.r.t input activations for the instance norm layer.
// i.e. gradients are propagated through the complete layer including
// statistics computation.
inline poplar::Tensor
instanceNormGradients(poplar::Graph &graph,
                      const poplar::Tensor &actsWhitened,
                      const poplar::Tensor &gradsIn,
                      const poplar::Tensor &invStdDev,
                      const poplar::Tensor &gamma,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const std::string &debugPrefix = "") {
  return popnn::gn::groupNormGradients(graph, actsWhitened, gradsIn, invStdDev,
                                       gamma, prog, partialsType, debugPrefix);
}

// update parameters given gradients w.r.t. parameters
inline void
instanceNormParamUpdate(poplar::Graph &graph,
                        const poplar::Tensor &gammaDelta,
                        const poplar::Tensor &betaDelta,
                        float learningRate,
                        poplar::Tensor &gamma,
                        poplar::Tensor &beta,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "") {
  return popnn::gn::groupNormParamUpdate(graph, gammaDelta, betaDelta,
                                         learningRate, gamma, beta, prog,
                                         debugPrefix);
}

// In flop computation, following applies
// Acts per channel:
// - for fc layers is the total number of batches.
// - for conv layers it is the field size per channel * batch size
//
// Number of channels:
// - for fc layers is the total number of activations in a batch
// - for conv layers is the total number of channels

uint64_t getFwdFlops(uint64_t numChannels, uint64_t actsPerChannel,
                     bool computeEstimates);
uint64_t getBwdFlops(uint64_t numChannels, uint64_t actsPerChannel);
uint64_t getWuFlops(uint64_t numChannels, uint64_t actsPerChannel);

} // namespace in
} // namespace popnn
#endif // popnn_InstanceNorm_hpp
