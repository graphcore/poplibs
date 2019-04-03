// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popops_ChannelOps_hpp
#define popops_ChannelOps_hpp

#include <boost/variant.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Program.hpp>

#include <string>

namespace popops {

// Vertex creation within popops for channelMul and AddToChannel
void broadcastAddVectorInnermostInPlace(poplar::Graph &graph,
                                        const poplar::Tensor &acts,
                                        const poplar::Tensor &addendByGroup,
                                        const float scale,
                                        poplar::ComputeSet &cs);

void broadcastMulVectorInnermost(poplar::Graph &graph,
                                const poplar::Tensor &acts,
                                const poplar::Tensor &actsOut,
                                const poplar::Tensor &scaleByGroup,
                                poplar::ComputeSet &cs,
                                const std::string &debugPrefix);

}

#endif // popops_ChannelOps_hpp
