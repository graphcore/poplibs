// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef ReductionStages_hpp
#define ReductionStages_hpp

#include "ComputeSetList.hpp"
#include "IntermediatePartials.hpp"
#include "ReductionIntrospection.hpp"
#include "popops/Reduce.hpp"

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include <boost/optional.hpp>

#include <string>

namespace popops {
// Storage of reduction result tensors, containing partials and output.  It is
// possible that partials can be a mixture of types, where float and half can
// be used in the same reduction, but there can be only 2.  We store the tensors
// according to the type found for convenience.
struct ResultTensors {
  std::vector<poplar::Tensor> typeA;
  std::vector<poplar::Tensor> typeB;
};

/// Take an input tensor, and reduce it to the output data with no exchange.
/// All the input elements for each output element must be on the same
/// tile or an exception is thrown.
///
/// Note that the output may not be mapped to the same tiles as the partials
/// so there may be a post-exchange step but that is fine -
/// if this method is possible it will definitely be the most efficient because
/// it is always quicker to reduce a value than to send it somewhere else.
///
/// \param graph                   The graph
/// \param in                      The 2D input tensor
/// \param contiguousRegionsByTile A vector of intervals of the input tensor
///                                that lie in each region found on each tile
/// \param groupedPartials The result of analysis of the contiguousRegions
///                        which identifies the partials on each tile.
/// \param out             Optional output tensor. Doesn't have to have its tile
///                        mapping set yet.
///                        If a tensor is not passed in one will be created
///                        using the outputShape and outputType provided.
/// \param originalOut     The original view of the output Tensor, which can
///                        be recorded for a later writeUndef
/// \param outputShape     The shape of the output Tensor to be created
/// \param inVertexType    The accumulation type of the reduction - this may
///                        be different to the type of the 'out' tensor.
/// \param outputType      The type of the output Tensor to be created
/// \param params   The reduce operation to do, including scale & update.
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A struct into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
///
void inputToOutputNoExchange(poplar::Graph &graph, const poplar::Tensor &in,
                             const RegionsByTile &contiguousRegionsByTile,
                             const TilePartialsDescription &groupedPartials,
                             boost::optional<poplar::Tensor> &out,
                             boost::optional<poplar::Tensor> &originalOutput,
                             const std::vector<std::size_t> outputShape,
                             poplar::Type inVertexType, poplar::Type outputType,
                             ReduceParams params, ComputeSetList &css,
                             ResultTensors &reductionResultTensors,
                             const std::string &debugPrefix);

/// Take an input tensor and reduce it as much as possible on each tile without
/// doing any exchange.
///
/// \param graph                   The graph
/// \param in                      The 2D input tensor
/// \param contiguousRegionsByTile A vector of intervals of the input tensor
///                                that lie in each region found on each tile
/// \param groupedPartials The result of analysis of the contiguousRegions
///                        which identifies the partials on each tile.
/// \param op              The reduce operation to do. This never does scale or
///                        update.
/// \param inVertexType    The accumulation type of the reduction - this may
///                        be different to `outType`.
/// \param outType  The required output type
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A struct into which this function will push
///                                 any tensor that is written to with a
///                                 reduction result.
/// \param debugPrefix
///
/// \returns A structure containing the intermediate partials.
IntermediatePartials
inputToIntermediateNoExchange(poplar::Graph &graph, const poplar::Tensor &in,
                              const RegionsByTile &contiguousRegionsByTile,
                              const TilePartialsDescription &groupedPartials,
                              Operation op, const poplar::Type &inVertexType,
                              const poplar::Type &outType, ComputeSetList &css,
                              ResultTensors &reductionResultTensors,
                              const std::string &debugPrefix);

/// Reduce an intermediate result to another intermediate result by the given
/// ratio. This is the most difficult of the stages.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the previous stage.
/// \param op       The reduce operation to do. This never does scale or update.
/// \param outType  The required output type
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A struct into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param startTile The tile to begin linearly laying out the intermediate
///                  reduction stages.
/// \param debugPrefix
///
/// \returns The intermediate partials produced by this reduction stage.
IntermediatePartials
intermediateToIntermediate(poplar::Graph &graph,
                           const IntermediatePartials &ipIn, Operation op,
                           const poplar::Type &outType, ComputeSetList &css,
                           ResultTensors &reductionResultTensors,
                           unsigned startTile, const std::string &debugPrefix);

/// Reduce an intermediate reduction to a final output tensor. The reduction
/// may or may not be done at the location of the output tensor. If the output
/// tensor does not have a tile mapping set then it is mapped linearly.
///
/// \param graph    The graph
/// \param ipIn     The intermediate partials from the prevoius stage.
/// \param output   Optional output tensor. Doesn't have to have its tile
///                 mapping set yet.  If a tensor is not passed in one will be
///                 created using the outputShape and outputType provided.
/// \param originalOut    The original view of the output Tensor, which can
///                       be recorder for a later writeUndef
/// \param outputShape    The shape of the output Tensor to be created
/// \param outputType     The type of the output Tensor to be created
/// \param params   The reduction operation, scale and update are applied.
/// \param inVertexType   The accumulation type of the reduction - this may
///                       be different to the type of the 'out' tensor.
/// \param css      Vertices are added to these compute sets - they must be
///                 added as a Sequence of Executes afterwards.
/// \param reductionResultTensors   A struct into which this function will push
///                       any tensor that is written to with a reduction result.
/// \param debugPrefix
///
void intermediateToOutput(poplar::Graph &graph,
                          const IntermediatePartials &ipIn,
                          boost::optional<poplar::Tensor> &output,
                          boost::optional<poplar::Tensor> &originalOutput,
                          const std::vector<std::size_t> outputShape,
                          poplar::Type outputType, ReduceParams params,
                          poplar::Type inVertexType, ComputeSetList &css,
                          ResultTensors &reductionResultTensors,
                          const poplar::Tensor &in,
                          const std::string &debugPrefix);

unsigned findGrainSizeForOp(poplar::Graph &graph, poplar::Type partialType,
                            popops::Operation &operation);

} // namespace popops

#endif // ReductionStages_hpp
