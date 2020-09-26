// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

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
 * \param dims        The dimensions of a tensor to be sliced/updated that will
 *                    be sliced/updated using these indices.
 * \param numIndices  The number of indices this tensor should contain
 * \param debugPrefix The prefix prepended to debugging info.
 *
 * \returns A tensor of shape [numIndices]. Element type
 *          is always UNSIGNED_INT.
 */
poplar::Tensor createIndicesTensor(poplar::Graph &graph,
                                   const std::vector<std::size_t> &dims,
                                   std::size_t numIndices,
                                   const std::string &debugPrefix = "");

/** Create and map a tensor to be updated from efficiently.
 *
 *  Memory layout is based on the planned split of the sparse tensor t
 *
 * \param graph          The Poplar graph.
 * \param t              The sparse tensor to be updated.
 * \param numIndices     The number of slices this tensor should contain.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being updated
 * \param debugPrefix    A string prepended to debugging info.
 * \param options        Implementation options for the  fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 *
 *  \returns             A tensor with shape [numIndices] mapped
 *                       appropriately to be sliced into/updated from.
 */
poplar::Tensor createSliceTensor(poplar::Graph &graph, const SparseTensor &t,
                                 std::size_t numIndices,
                                 const FullyConnectedParams &params,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {},
                                 PlanningCache *cache = nullptr);

/** Take multiple slices from a base tensor.
 *
 * The returned tensor will have dimensions [offsets, k (from params)]
 *
 * \param graph          The Poplar graph.
 * \param t              The sparse tensor being sliced.
 * \param offsets        The offsets within \p t to be sliced.
 * \param prog           The program to be extended.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being updated
 * \param debugPrefix    The prefix prepended to debugging info.
 * \param options        Implementation options for the fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 */
poplar::Tensor embeddingSlice(poplar::Graph &graph, const SparseTensor &t,
                              const poplar::Tensor &offsets,
                              poplar::program::Sequence &prog,
                              const FullyConnectedParams &params,
                              const std::string &debugPrefix = "",
                              const poplar::OptionFlags &options = {},
                              PlanningCache *cache = nullptr);

/** Accumulate multiple slices in a tensor
 *
 * \param graph          The Poplar graph.
 * \param t              The sparse tensor being updated.
 * \param slices         The slices to accumulate.
 * \param offsets        The offsets within \p t to be accumulated.
 * \param scale          The scaling to apply to the update.
 * \param prog           The program to be extended.
 * \param params         Parameters for the fully connected layer which will
 *                       provide the planned memory layout for the sparse tensor
 *                       being updated
 * \param debugPrefix    The prefix prepended to debugging info.
 * \param options        Implementation options for the fully connected layer.
 * \param cache          Optional pointer to planning cache to use.
 */
void embeddingUpdateAdd(
    poplar::Graph &graph, const SparseTensor &t, const poplar::Tensor &slices,
    const poplar::Tensor &offsets, const poplar::Tensor &scale,
    poplar::program::Sequence &prog, const FullyConnectedParams &params,
    const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

} // end namespace dynamic

} // namespace popsparse

#endif // popsparse_Embedding_hpp
