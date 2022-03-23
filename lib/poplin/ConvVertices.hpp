// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "ConvPlan.hpp"
#include "ConvProgramTree.hpp"
#include <map>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplin/ConvParams.hpp>
#include <string>
#include <utility>

namespace poplin {

void calcPartialConvOutput(poplar::Graph &graph, const Plan &plan,
                           unsigned tile, ConvParams params,
                           std::vector<poplar::program::Copy> &transformPre,
                           std::vector<poplar::Tensor> &copyWritten,
                           ConvProgramTree::ComputeSetsGroup &convolveCS,
                           poplar::Tensor in, poplar::Tensor weights,
                           poplar::Tensor out, bool use128BitConvUnitLoad,
                           bool disableSRForAMPVertices,
                           const poplar::DebugNameAndId &dnai);

void createConvPartialSlicVertex(
    poplar::Graph &graph, const Plan &plan, unsigned tile, ConvParams params,
    std::vector<poplar::program::Copy> &transformPre,
    std::vector<poplar::Tensor> &copyWritten, poplar::ComputeSet fwdCS,
    std::map<poplar::Type, std::pair<std::vector<poplar::Tensor>,
                                     std::vector<poplar::Tensor>>> &postProg,
    poplar::Tensor in, poplar::Tensor weights, poplar::Tensor out,
    bool disableSRForAMPVertices, const poplar::DebugNameAndId &dnai);

} // namespace poplin
