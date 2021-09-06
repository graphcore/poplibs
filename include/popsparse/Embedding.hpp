// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *  Functions for slicing and mapping sparse tensors.
 */

#ifndef popsparse_Embedding_hpp
#define popsparse_Embedding_hpp

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>

#include <popsparse/FullyConnected.hpp>
#include <popsparse/FullyConnectedParams.hpp>
#include <popsparse/SparseTensor.hpp>

namespace popsparse {
namespace dynamic {

/** Create and map a tensor to contain indices for slicing/updating
 *  a tensor efficiently.
 *
 * \param graph       The Poplar graph.
 * \param params      Parameters for the fully connected layer which defines
 *                    the embedding operation. Used to decide on layout for
 *                    the indices.
 * \param options     Implementation options for the fully connected layer.
 * \param numIndices  The number of indices this tensor should contain
 * \param debugContext   Optional debug information.
 *
 * \returns A 1D tensor of shape [\p numIndices]. Element type
 *          is always UNSIGNED_INT.
 */
poplar::Tensor
createIndicesTensor(poplar::Graph &graph, const FullyConnectedParams &params,
                    std::size_t numIndices,
                    const poplar::OptionFlags &options = {},
                    const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to be updated from efficiently.
 *
 *  Memory layout is based on the planned split of the sparse tensor.
 *
 * \param graph          The Poplar graph.
 * \param dataType       The data type of the returned tensor.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being updated
 * \param numIndices     The number of slices this tensor should contain.
 * \param debugContext   Optional debug information.
 * \param options        Implementation options for the fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 *
 *  \returns             A 2D tensor with shape [numIndices, \p
 * params.getInputChannels()] with layout optimised for slicing into/updating
 * from.
 */
poplar::Tensor createSliceTensor(poplar::Graph &graph,
                                 const poplar::Type &dataType,
                                 const FullyConnectedParams &params,
                                 std::size_t numIndices,
                                 const poplar::DebugContext &debugContext = {},
                                 const poplar::OptionFlags &options = {},
                                 PlanningCache *cache = nullptr);

/** Take multiple slices from a base tensor.
 *
 * The returned tensor will have dimensions [offsets, k (from params)]
 *
 * \param graph          The Poplar graph.
 * \param t              The sparse tensor being sliced.
 * \param indices        The indices of rows of \p t to be sliced.
 * \param prog           The program to be extended.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being sliced.
 * \param debugContext   Optional debug information.
 * \param options        Implementation options for the fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 */
poplar::Tensor embeddingSlice(poplar::Graph &graph, const SparseTensor &t,
                              const poplar::Tensor &indices,
                              poplar::program::Sequence &prog,
                              const FullyConnectedParams &params,
                              const poplar::DebugContext &debugContext = {},
                              const poplar::OptionFlags &options = {},
                              PlanningCache *cache = nullptr);

/** Update a sparse tensor with a set of slices at the given row indices.
 *
 * \param graph          The Poplar graph.
 * \param t              The sparse tensor being updated.
 * \param slices         The slices to accumulate.
 * \param indices        The indices of rows of \p t to accumulate each slice in
 *                       \p slices into.
 * \param scale          The scaling to apply to the update.
 * \param prog           The program to be extended.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being updated
 * \param debugContext   Optional debug information.
 * \param options        Implementation options for the fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 */
void embeddingUpdateAdd(
    poplar::Graph &graph, const SparseTensor &t, const poplar::Tensor &slices,
    const poplar::Tensor &indices, const poplar::Tensor &scale,
    poplar::program::Sequence &prog, const FullyConnectedParams &params,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

} // end namespace dynamic

} // namespace popsparse

#endif // popsparse_Embedding_hpp
