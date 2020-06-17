// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/ConvParams.hpp>
#include <string>

namespace poplin {

void createConvPartialSlicVertex(
    poplar::Graph &graph, unsigned slicWindowWidth, unsigned convGroupsPerGroup,
    unsigned chansPerGroup, unsigned convUnitsRequired, unsigned tile,
    ConvParams params, std::vector<poplar::program::Copy> &transformPre,
    poplar::Tensor &copyWritten, poplar::ComputeSet fwdCS,
    poplar::program::Sequence &postConvProg, poplar::Tensor in,
    poplar::Tensor weights, poplar::Tensor out, const std::string &debugPrefix);

} // namespace poplin
