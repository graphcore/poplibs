// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_Norms_hpp
#define popnn_Norms_hpp
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"

namespace popnn {

std::size_t normNumChannels(poplar::Tensor acts);

std::size_t normNumActsPerChannel(poplar::Tensor acts);

poplar::Tensor
createNormGamma(poplar::Graph &graph, const poplar::Tensor &acts);

poplar::Tensor
createNormBeta(poplar::Graph &graph, const poplar::Tensor &acts);

std::pair<poplar::Tensor, poplar::Tensor>
createNormParams(poplar::Graph &graph, const poplar::Tensor acts);

} // namespace popnn
#endif // popnn_Norms_hpp
