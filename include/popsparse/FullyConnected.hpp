// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef popsparse_FullyConnected_hpp
#define popsparse_FullyConnected_hpp

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <popsparse/FullyConnectedParams.hpp>
#include <popsparse/PlanningCache.hpp>
#include <popsparse/SparseTensor.hpp>

namespace popsparse {
/// Support for dynamic sparse matrices.
namespace dynamic {

/**
 * Create a sparse tensor that is used as the weights W for a fully connected
 * layer.
 *
 * The following options are available:
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 [=0.6]
 *
 *      The maximum proportion of available memory on each tile that this
 *      layer should consume temporarily during the course of the operation.
 *
 *    * `metaInfoBucketOversizeProportion` Decimal between 0 and 1 [=0.3]
 *
 *      This specifies additional elements to allocate in each bucket of
 *      meta-information as a proportion of the required size for a perfectly
 *      uniformly distributed sparsity pattern.
 *
 *    * `doGradAPass` (true, false) [=false]
 *
 *      `doGradWPass` (true, false) [=false]
 *
 *      Indicate which passes are present for the operation of the layer as a
 *      whole. It is assumed that the forward pass is always present.
 *
 *    * `partialsType` poplar::Type [=poplar::FLOAT]
 *
 *      The type to use for partial results.
 *
 *    * `sharedBuckets` (true, false) [=true]
 *
 *      If set, forces the same buckets to be used for all three passes.
 *
 *
 * \param graph The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params Parameters for the fully connected layer.
 * \param debugContext Optional debug information.
 * \param options Implementation options for the fully connected layer.
 * \param cache Optional pointer to planning cache to use.
 *
 * \returns A tensor with sparse representation of weights for the fully
 *          connected layer.
 */
SparseTensor
createFullyConnectedWeights(poplar::Graph &graph, const poplar::Type &inputType,
                            const FullyConnectedParams &params,
                            const poplar::DebugContext &debugContext = {},
                            const poplar::OptionFlags &options = {},
                            PlanningCache *cache = nullptr);

/**
 * Create a dense tensor that is used as the input activations for a fully
 * connected layer. This returned tensor is of shape
 * [batchSize, inputChannelsPerGroup].
 *
 * \param graph The Poplar graph.
 * \param inputType The type for inputs to the operation.
 * \param params Parameters for the fully connected layer.
 * \param debugContext    Optional debug information.
 * \param options Implementation options for the fully connected layer.
 *        See createFullyConnectedWeights() for details.
 * \param cache Optional pointer to planning cache to use.
 */
poplar::Tensor
createFullyConnectedInput(poplar::Graph &graph, const poplar::Type &inputType,
                          const FullyConnectedParams &params,
                          const poplar::DebugContext &debugContext = {},
                          const poplar::OptionFlags &options = {},
                          PlanningCache *cache = nullptr);

/** Run a fully connected forward (or inference) pass.
 *
 *  The sparse-weights tensor is made up of meta information for the sparsity
 *  and the non-zero values. Does the Fwd operation described in the Note
 *  above but with input and output transposed.
 *
 *  The meta information for the sparse weights tensor must be created for the
 *  forward (or inference) pass and should be created by use of the
 *  createFullyConnectedWeights() function.
 *
 *  \param graph           The Poplar graph.
 *  \param weights         Sparsity information of the weights tensor.
 *  \param activations     The dense activations have shape
 *                         [batchSize][inputChannelsPerGroup * numGroups]
 *  \param fcParams        Fully connected layer parameters.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         forward operation.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         operation should be implemented.
 *                         See createFullyConnectedWeights() for details.
 *  \param cache           Optional pointer to planning cache to use.
 *  \returns               The tensor holding the result.
 *                         This tensor will be created, added to the graph and
 *                         mapped to tiles. The result tensor is of shape
 *                         [batchSize][outputChannelsPerGroup * numGroups]
 */
poplar::Tensor fullyConnectedFwd(poplar::Graph &graph,
                                 const SparseTensor &weights,
                                 const poplar::Tensor &activations,
                                 const FullyConnectedParams &fcParams,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext = {},
                                 const poplar::OptionFlags &options = {},
                                 PlanningCache *cache = nullptr);

/** Run a fully connected GradA pass.
 *
 *  The sparse-weights tensor is made up of meta information
 *  for the sparsity and the non-zero values. Does the GradA
 *  computation as described in the Note above but with input and output
 *  transposed.
 *
 *  The meta information for the sparse-weights tensor must be created for the
 *  GradA pass and should be created by use of
 *  createFullyConnectedWeights() function.
 *
 *  \param graph           The Poplar graph.
 *  \param weights         Sparsity information of the weights tensor.
 *  \param gradients       The dense loss gradients with respect to output
 *                         activations and are of shape
 *                         [batchSize][outputChannelsPerGroup] .
 *  \param fcParams        Fully connected layer parameters.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         GradA operation.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         operation should be implemented.
 *                         See createFullyConnectedWeights() for details.
 *  \param cache           Optional pointer to planning cache to use.
 *
 *  \returns               The tensor holding the result.
 *                         This tensor will be created, added to the graph and
 *                         mapped to tiles. The tensor is of shape
 *                         [batchSize][inputChannelsPerGroup * numGroups]
 */
poplar::Tensor fullyConnectedGradA(
    poplar::Graph &graph, const SparseTensor &weights,
    const poplar::Tensor &gradients, const FullyConnectedParams &fcParams,
    poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/** Run a fully connected GradW pass to compute sparse gradients. The layout
 *  of the returned tensor is exactly as that of the representation of the
 *  weights NZ values so that any elementwise operation may be done between the
 *  two.
 *
 *  The actual implementation differs from that in the Note above as the
 *  transpose of the gradients and activations are supplied as parameters to
 *  this function.
 *
 *  \param graph           The Poplar graph.
 *  \param weightMetaInfo  Meta information for sparse weights. See
 *                         SparseTensor representation.
 *  \param gradA           Dense gradients wrt output activations of shape
 *                         [batchSize][outputChannelsPerGroup * numGroups]
 *  \param activations     Input activations of shape
 *                         [batchSize][inputChannelsPerGroup * numGroups]
 *  \param fcParams        Fully connected layer parameters.
 *  \param prog            A reference to a program sequence which will
 *                         be appended with the code to perform the
 *                         GradW operation.
 *  \param debugContext    Optional debug information.
 *  \param options         The structure describing options on how the
 *                         operation should be implemented.
 *                         See createFullyConnectedWeights() for details.
 *  \param cache           Optional pointer to planning cache to use.
 *  \returns               The tensor holding the result.
 *                         This tensor will be created, added to the graph and
 *                         mapped to tiles.
 */
poplar::Tensor fullyConnectedSparseGradW(
    poplar::Graph &graph, const poplar::Tensor sparsityMetaInfo,
    const poplar::Tensor &gradA, const poplar::Tensor &activations,
    const FullyConnectedParams &fcParams, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/** Report the serial splitting of a dense gradW output given the memory
 *  proportion limit given in options. A dense gradW output is of shape
 *  [numGroups][inputSize][outputSize]
 *
 *  \param graph           The Poplar graph.
 *  \param inputType       The type of input.
 *  \param params          Fully connected params.
 *  \param options         The structure describing options on how the
 *                         operation should be implemented.
 *                         See createFullyConnectedWeights() for details.
 *  \param cache           Optional pointer to planning cache to use.
 *  \returns               Serial splits for each of the output dimensions
 *                         [numGroups][inputSize][outputSize].
 */
std::tuple<unsigned, unsigned, unsigned> fullyConnectedDenseGradWSerialSplits(
    const poplar::Graph &graph, const poplar::Type &inputType,
    const FullyConnectedParams &fcParams,
    const poplar::OptionFlags &options_ = {}, PlanningCache *cache = nullptr);

} // namespace dynamic
} // namespace popsparse

#endif // popsparse_FullyConnected_hpp
