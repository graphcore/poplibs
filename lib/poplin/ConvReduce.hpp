// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef ConvReduce_hpp
#define ConvReduce_hpp

#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

namespace poplin {

class ConvOptions;

poplar::Tensor
multiStageGroupedReduce(poplar::Graph &graph, const poplar::Tensor &partials,
                        const poplar::Type &resultType,
                        std::vector<poplar::ComputeSet> &computeSets,
                        const ConvOptions &options, unsigned startTile,
                        bool ascendingMapping,
                        const poplar::DebugNameAndId &dnai);

poplar::Tensor createMultiStageGroupedReduceOutput(
    poplar::Graph &graph, const poplar::Tensor &partials,
    const poplar::Type &resultType, const ConvOptions &options,
    unsigned startTile, bool ascendingMapping,
    const poplar::DebugNameAndId &dnai);

void multiStageGroupedReduceWithOutput(
    poplar::Graph &graph, const poplar::Tensor &output,
    const poplar::Tensor &partials, const poplar::Type &resultType,
    std::vector<poplar::ComputeSet> &computeSets, const ConvOptions &options,
    unsigned startTile, bool ascendingMapping,
    const poplar::DebugNameAndId &dnai);

} // namespace poplin

#endif // ConvReduce_hpp
