#ifndef ReductionStages_hpp
#define ReductionStages_hpp

#include <string>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include "ComputeSetList.hpp"
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
/// \param in       The 2D input tensor
/// \param mapping  The result of graph.getTileMapping(in)
/// \param out      The output tensor. Doesn't have to have its tile mapping
///                 set yet.
/// \param params   The reduce operation to do, including scale & update.
/// \param inVertexType   The accumulation type of the reduction - this may
///                       be different to the type of the 'out' tensor.
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A vector into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
void inputToOutputNoExchange(
    poplar::Graph &graph, const poplar::Tensor &in,
    const poplar::Graph::TileToTensorMapping &mapping,
    const poplar::Tensor &out, poplar::Type inVertexType, ReduceParams params,
    ComputeSetList &css, std::vector<poplar::Tensor> &reductionResultTensors,
    const std::string &debugPrefix, ReductionDebug *debug);

/// Take an input tensor and reduce it as much as possible on each tile without
/// doing any exchange.
///
/// \param graph    The graph
/// \param in       The 2D input tensor
/// \param mapping  The result of graph.getTileMapping(in)
/// \param op       The reduce operation to do. This never does scale or update.
/// \param inVertexType   The accumulation type of the reduction - this may
///                       be different to `outType`.
/// \param outType  The required output type
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A vector into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns A structure containing the intermediate partials.
IntermediatePartials inputToIntermediateNoExchange(
    poplar::Graph &graph, const poplar::Tensor &in,
    const poplar::Graph::TileToTensorMapping &mapping, Operation op,
    const poplar::Type &inVertexType, const poplar::Type &outType,
    ComputeSetList &css, std::vector<poplar::Tensor> &reductionResultTensors,
    const std::string &debugPrefix, ReductionDebug *debug);

/// Reduce an intermediate result to another intermediate result by the given
/// ratio. This is the most difficult of the stages.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the prevoius stage.
/// \param op       The reduce operation to do. This never does scale or update.
/// \param inVertexType   The accumulation type of the reduction - this may
///                       be different to `outType`.
/// \param outType  The required output type
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A vector into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
/// \returns The intermediate partials produced by this reduction stage.
IntermediatePartials intermediateToIntermediate(
    poplar::Graph &graph, const IntermediatePartials &ipIn, Operation op,
    const poplar::Type &inVertexType, const poplar::Type &outType,
    ComputeSetList &css, std::vector<poplar::Tensor> &reductionResultTensors,
    const std::string &debugPrefix, ReductionDebug *debug);

/// Reduce an intermediate reduction to a final output tensor. The reduction
/// may or may not be done at the location of the output tensor. If the output
/// tensor does not have a tile mapping set then it is mapped linearly.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the prevoius stage.
/// \param output   The output tensor, may not be mapped.
/// \param params   The reduction operation, scale and update are applied.
/// \param inVertexType   The accumulation type of the reduction - this may
///                       be different to the type of the 'out' tensor.
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A vector into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
/// \param debug    Optional pointer (can be null) to be filled with debug info.
///
void intermediateToOutput(poplar::Graph &graph,
                          const IntermediatePartials &ipIn,
                          const poplar::Tensor &output, ReduceParams params,
                          poplar::Type inVertexType, ComputeSetList &css,
                          std::vector<poplar::Tensor> &reductionResultTensors,
                          const std::string &debugPrefix,
                          ReductionDebug *debug);

// Initially each reduction is referenced as a series of patterns which
// describe the part of a contiguous region/regions of data that are
// required by a given reduction.  A pattern takes the form:
//
// length : number of contiguous elements of any single column in the pattern
// start: index of the 1st element that is required relative to the region start
// stride: number of elements within the whole pattern before it repeats
// repetitions: number of times the pattern repeats
// regionIdx: The index into the contiguous regions list that the column data is
//            found in. (Ie which of the contiguous regions on tile is it in)
struct PartialsPattern {
  unsigned length;
  unsigned regionOffset;
  unsigned stride;
  unsigned repetitions;
  unsigned regionIdx;
};
// The PartialsDescription structure is used in 2 ways.
// Initially a single column is identified and recorded in 'columns'.  All
// elements of that column that are found on tile are recorded in 'patterns'.
// Where the regularity of the layout of the column elements is broken,
// (either by a change in 'stride' or similar or a new region) a fresh pattern
// is started.
//
// Later, PartialsDescriptions can be grouped. Those with compatible patterns
// can be combined. The patterns listed are unchanged and describe the layout
// of the 1st column in the 'columns' vector.  Further columns are added to the
// columns vector, which have the same layout in memory but a 'start' parameter
// which increments with position in the 'columns' vector.
struct PartialsDescription {
  // List of the columns which these patterns describe
  std::vector<unsigned> columns;
  // The partials described in the form of patterns
  std::vector<PartialsPattern> patterns;
};

// Internal access to functions for test purposes
// Given a set of contiguous regions for a tensor with shape {rows, columns},
// fills partialsDescription with patterns describing each column individually.
void gatherReductionPatterns(
    std::vector<PartialsDescription> &partialsDescription,
    const std::vector<std::vector<poplar::Interval>> &regions,
    unsigned columns);

// Given a set of PartialsDescriptions, group together those that sit next to
// each other in memory and which can be described with the same
// PartialsPattern.
std::vector<PartialsDescription>
groupPartials(std::vector<PartialsDescription> &partialsDescription,
              unsigned columns);

} // namespace popops

#endif // ReductionStages_hpp
