// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_Norms_hpp
#define popnn_Norms_hpp
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"

namespace popnn {

// Flops for forward pass of a norm layer with a given size of statistics vector
// and the total elements in the activations input to the layer
// For Batch Norm, computeStats should be set to false for inference if batch
// statistics are  not computed as averaged batch statistics may be combined
// with norm parameters.
std::uint64_t getNormFwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements,
                              bool computeStats = true);

// Flops for computation of gradient w.r.t activations for a norm layer with a
// given size of statistics vector and the total elements in the activations
// input to the layer
std::uint64_t getNormBwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements);

// Flops for parameter update for a norm layer with a given parameter vector
// size and the total elements in the activations input to the layer
std::uint64_t getNormWuFlops(std::size_t paramsSize,
                             std::size_t numActsElements);
poplar::Tensor createNormGamma(poplar::Graph &graph,
                               const poplar::Tensor &acts);

poplar::Tensor createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts);

std::pair<poplar::Tensor, poplar::Tensor>
createNormParams(poplar::Graph &graph, const poplar::Tensor acts);

} // namespace popnn
#endif // popnn_Norms_hpp
