// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popops_ScalarMultiply_hpp
#define popops_ScalarMultiply_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/DebugInfo.hpp>

namespace popops {

void scalarMultiplyInplace(poplar::Graph &graph, const poplar::Tensor &a,
                           const poplar::Tensor &b,
                           poplar::program::Sequence &prog,
                           poputil::PoplibsOpDebugInfo &di,
                           const poplar::OptionFlags &options = {});

poplar::Tensor scalarMultiply(poplar::Graph &graph, const poplar::Tensor &a,
                              const poplar::Tensor &b,
                              poplar::program::Sequence &prog,
                              poputil::PoplibsOpDebugInfo &di,
                              const poplar::OptionFlags &options = {});

bool inputsMatchMixedPrecisionScalarMultiplyPattern(
    const poplar::Tensor &a, const poplar::Tensor &b,
    bool orderInvariant = false);

} // namespace popops

#endif // popops_ScalarMultiply_hpp
