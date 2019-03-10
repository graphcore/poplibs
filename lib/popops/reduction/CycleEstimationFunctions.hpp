#ifndef CycleEstimationFunctions_hpp
#define CycleEstimationFunctions_hpp

#include <cstdint>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <popops/Reduce.hpp>

namespace popops {

/// Get the cycle estimate for reduction. This is broken out from
/// getCyclesEstimateForReduceVertex() so that it can be used from
/// visualisations and planning where VertexIntrospector is not available.
///
/// /param partialsSizes  The size of each partial fed into the codelet. These
///                       must always be a multiple of the size of the
///                       corresponding output.
/// /param outSizes       The size of each output region.
/// /param numPartials    The number of partials for each output. This must be
///                       the same length as outSizes, and its values must
///                       sum to the length of partialsSizes.
/// /param vectorWidth    The vector width for the tile. This may be doubled
///                       internally for double-data-width operations such
///                       as add. It should be passed here in undoubled form.
/// /param partialsType   The type of the variables connected to the partials
///                       field of the codelet.
/// /param outType        The type of the variables connected to the output
///                       field of the codelet.
/// /param operation      The basic operation to perform. Add is faster than
///                       the others.
/// /param isUpdate       True if the operation is A += reduce(B) rather than
///                       A = reduce(B).
/// /param outputRegionSizeisOne True if all regions connected to output edge
///                       are of size 1.
///
/// /returns  The estimated number of thread cycles used by the vertex.
///
std::uint64_t
getCyclesEstimateForReduce(const std::vector<std::size_t> &partialsSizes,
                           const std::vector<std::size_t> &outSizes,
                           const std::vector<unsigned> &numPartials,
                           unsigned vectorWidth,
                           const poplar::Type &partialsType,
                           const poplar::Type &outType,
                           popops::Operation operation,
                           bool isUpdate,
                           bool outputRegionsSizeIsOne);

/// Get the cycle estimate for a reduction. This obtains field sizes from the
/// vertex and calls through to getCyclesEstimateForReduce(). See that
/// function for details.
std::uint64_t
getCycleEstimateForReduceVertex(const poplar::VertexIntrospector &vertex,
                                const poplar::Target &target,
                                const poplar::Type &partialsType,
                                const poplar::Type &outType,
                                popops::Operation operation,
                                bool isUpdate, bool outputRegionSizeIsOne);

}

#endif // CycleEstimationFunctions_hpp
