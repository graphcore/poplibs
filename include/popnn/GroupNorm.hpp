// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_GroupNorm_hpp
#define popnn_GroupNorm_hpp
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include <utility>

// The group normalisation implementation groups channels such that channels
// belonging to a group are strided.

namespace popnn {
namespace gn {

// Estimate mean and inverse of standard deviation of activations
std::pair<poplar::Tensor, poplar::Tensor>
groupNormStatistics(poplar::Graph &graph, const poplar::Tensor acts,
                    float eps,
                    poplar::program::Sequence &prog,
                    unsigned numGroups,
                    bool unbiasedVarEstimate,
                    const poplar::Type &partialsType= poplar::FLOAT,
                    const std::string &debugPrefix = "");

// Whiten activations given mean and standard deviation
poplar::Tensor
groupNormWhiten(poplar::Graph &graph,
                const poplar::Tensor &acts,
                const poplar::Tensor &mean,
                const poplar::Tensor &invStdDev,
                poplar::program::Sequence &prog,
                const std::string &debugPrefix = "");

// Group normalise activations given mean, standard deviation and batch norm
// parameters. The outputs produced are
// 1) group normalised activations (whitened, scaled by gamma, offset by beta)
// 2) whitened activations
std::pair<poplar::Tensor, poplar::Tensor>
groupNormalise(poplar::Graph &graph,
               const poplar::Tensor &acts,
               const poplar::Tensor &gamma,
               const poplar::Tensor &beta,
               const poplar::Tensor &mean,
               const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

// Compute gradients w.r.t parameters for parameter update
std::pair<poplar::Tensor, poplar::Tensor>
groupNormParamGradients(poplar::Graph &graph,
                        const poplar::Tensor &actsWhitened,
                        const poplar::Tensor &gradsIn,
                        poplar::program::Sequence &prog,
                        const poplar::Type &partialsType = poplar::FLOAT,
                        const std::string &debugPrefix = "");

// Compute gradients w.r.t input activations for the group norm layer.
// i.e. gradients are propagated through the complete layer including
// statistics computation.
poplar::Tensor
groupNormGradients(poplar::Graph &graph,
                   const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &invStdDev,
                   const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const std::string &debugPrefix = "");

void groupNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta,
                          float learningRate,
                          poplar::Tensor &gamma,
                          poplar::Tensor &beta,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");
} // namespace gn
} // namespace popnn
#endif // popnn_GroupNorm_hpp
