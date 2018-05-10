#ifndef ReductionStages_hpp
#define ReductionStages_hpp

#include <string>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include "IntermediatePartials.hpp"
#include "ReductionDebug.hpp"

namespace popops {

/// Take an input tensor, and reduce it to the output data with no exchange.
/// All the input elements for each output element must be on the same
/// tile or an exception is thrown.
///
/// Note that the output may not be mapped to the same tiles as the partials
/// so there may be a post-exchange step but that is fine -
/// if this method is possible it will definitely be the most efficient because
/// it is always quicker to reduce a value than to send it somewhere else.
///
/// \param graph    The graph
/// \param A        The 2D input tensor
/// \param mapping  The result of graph.getTileMapping(A)
/// \param out      The output tensor. Doesn't not have to have its tile mapping
///                 set yet.
/// \param params   The reduce operation to do, including scale & update.
/// \param prog     Sequence to append to.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns The output tensor.
poplar::Tensor inputToOutputNoExchange(
    poplar::Graph &graph,
    const poplar::Tensor &A,
    const poplar::Graph::TileToTensorMapping &mapping,
    const poplar::Tensor &out,
    ReduceParams params,
    poplar::program::Sequence &prog,
    const std::string &debugPrefix,
    ReductionDebug *debug);

/// Take an input tensor and reduce it as much as possible on each tile without
/// doing any exchange.
///
/// \param graph    The graph
/// \param A        The 2D input tensor
/// \param mapping  The result of graph.getTileMapping(A)
/// \param params   The reduce operation to do, including scale & update.
/// \param prog     Sequence to append to.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns A structure containing the intermediate partials.
IntermediatePartials inputToIntermediateNoExchange(poplar::Graph &graph,
    const poplar::Tensor &A,
    const poplar::Graph::TileToTensorMapping &mapping,
    ReduceParams params,
    poplar::program::Sequence &prog,
    const std::string &debugPrefix,
    ReductionDebug *debug);

/// Reduce an intermediate result to another intermediate result by the given
/// ratio. This is the most difficult of the stages.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the prevoius stage.
/// \param op       The reduction operation, not including scale or update.
/// \param prog     Sequence to append to.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns The intermediate partials produced by this reduction stage.
IntermediatePartials intermediateToIntermediate(poplar::Graph &graph,
    const IntermediatePartials &ipIn,
    ReduceParams params,
    poplar::program::Sequence &prog,
    const std::string &debugPrefix,
    ReductionDebug *debug);

/// Reduce an intermediate reduction to a final output tensor. The reduction
/// may or may not be done at the location of the output tensor. If the output
/// tensor does not have a tile mapping set then it is mapped linearly.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the prevoius stage.
/// \param output   The output tensor, may not be mapped.
/// \param params   The reduction operation, scale and update are applied.
/// \param prog     Sequence to append to.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns The output tensor.
poplar::Tensor intermediateToOutput(poplar::Graph &graph,
    const IntermediatePartials &ipIn,
    const poplar::Tensor &output,
    ReduceParams params,
    poplar::program::Sequence &prog,
    const std::string &debugPrefix,
    ReductionDebug *debug);

}

#endif // ReductionStages_hpp
