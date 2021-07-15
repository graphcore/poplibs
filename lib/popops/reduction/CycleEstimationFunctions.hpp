// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef CycleEstimationFunctions_hpp
#define CycleEstimationFunctions_hpp

#include <cstdint>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <popops/Reduce.hpp>

namespace popops {
enum class ReductionSpecialisation;

/// Get the cycle estimate for a strided reduce vertex with the
/// given parameters. This is exposed for modelling.
std::uint64_t getCyclesEstimateForStridedReduce(
    const std::size_t partialsSize, const std::size_t numPartials,
    const std::size_t numOutputs, const unsigned stride,
    const unsigned numOuterStrides, const unsigned dataPathWidth,
    const unsigned vectorWidth, const poplar::Type &partialsType,
    const poplar::Type &outType, const popops::Operation operation,
    bool update);

/// Get the cycle estimate for a reduction. This obtains field sizes from the
/// vertex and calls through to getCyclesEstimateForReduce(). See that
/// function for details.
poplar::VertexPerfEstimate getCycleEstimateForReduceVertex(
    const poplar::VertexIntrospector &vertex, const poplar::Target &target,
    const poplar::Type &partialsType, const poplar::Type &outType,
    popops::Operation operation, bool isUpdate,
    popops::ReductionSpecialisation specialisation);

} // namespace popops

#endif // CycleEstimationFunctions_hpp
