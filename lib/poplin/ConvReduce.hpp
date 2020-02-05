// Copyright (c) Graphcore Ltd, All rights reserved.
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
multiStageGroupedReduce(poplar::Graph &graph, poplar::Tensor partials,
                        const poplar::Type &resultType,
                        std::vector<poplar::ComputeSet> &computeSets,
                        const ConvOptions &options,
                        const std::string &debugPrefix);
} // namespace poplin

#endif // ConvReduce_hpp
