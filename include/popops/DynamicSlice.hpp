// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for dynamic slices.
 *
 */

#ifndef popops_DynamicSlice_hpp
#define popops_DynamicSlice_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poputil/DebugInfo.hpp>
#include <string>
#include <vector>

namespace poplar {
class Tensor;
}

namespace popops {

class SlicePlanInternal;
/** An object representing a plan that describes how to implement
 *  a slice or update. This can be used as a parameter to a function
 *  that will slice or update a tensor.
 */
class SlicePlan {
public:
  SlicePlan();
  ~SlicePlan();
  SlicePlan(const SlicePlan &other);
  SlicePlan(SlicePlan &&other);
  SlicePlan &operator=(const SlicePlan &other);
  SlicePlan &operator=(SlicePlan &&other);

  friend std::ostream &operator<<(std::ostream &o, const SlicePlan &p);
  friend bool operator<(const SlicePlan &a, const SlicePlan &b) noexcept;
  friend bool operator==(const SlicePlan &a, const SlicePlan &b) noexcept;
  friend poplar::ProfileValue poputil::toProfileValue<>(const SlicePlan &p);

  // Implementation
  SlicePlan(std::unique_ptr<SlicePlanInternal> internal);
  SlicePlanInternal &getImpl() const { return *internal; }

private:
  std::unique_ptr<SlicePlanInternal> internal;
};
bool operator<(const SlicePlan &a, const SlicePlan &b) noexcept;
bool operator==(const SlicePlan &a, const SlicePlan &b) noexcept;
bool operator!=(const SlicePlan &a, const SlicePlan &b) noexcept;

/** Create and map a tensor to be sliced/updated efficiently.
 *
 *  The returned tensor will be spread over as many tiles as possible while
 *  respecting the minimum number of elements per tile (\p minGrainSize) and
 *  still being in a form that can be sliced/updated efficiently.
 *
 *  \param graph        The Poplar graph.
 *  \param type         The type of the elements.
 *  \param shape        The shape of the tensor to be slice/updated.
 *  \param dims         The dimensions of the tensor that will be slice/updated.
 *  \param sizes        The size of the slice in each of the dimensions.
 *  \param minGrainSize The minimum elements per slice mapped to each tile
 *  \param debugContext Optional debug information.
 *  \returns            A tensor shape \p shape that is suitably mapped
 *
 */
poplar::Tensor createSliceableTensor(
    poplar::Graph &graph, const poplar::Type &type,
    const std::vector<size_t> &shape, const std::vector<size_t> &dims,
    const std::vector<size_t> &sizes, std::size_t minGrainSize = 0,
    const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to be sliced/updated efficiently.
 *
 *  The returned tensor will be laid out according to the plan.
 *
 *  \param graph        The Poplar graph.
 *  \param type         The type of the elements.
 *  \param shape        The shape of the tensor to be slice/updated.
 *  \param dims         The dimensions of the tensor that will be slice/updated.
 *  \param sizes        The size of the slice in each of the dimensions.
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 *  \returns            A tensor shape \p shape that is suitably mapped.
 **/
poplar::Tensor
createSliceableTensor(poplar::Graph &graph, const poplar::Type &type,
                      const std::vector<size_t> &shape,
                      const std::vector<size_t> &dims,
                      const std::vector<size_t> &sizes, const SlicePlan &plan,
                      const poplar::OptionFlags &options,
                      const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to be sliced into or updated from efficiently.
 *
 *  Introspection on the tensor \p t is used to lay out
 *  the created tensor such that it can be used to efficiently update \p t.
 *
 *  \param graph        The Poplar graph.
 *  \param t            The tensor to be updated.
 *  \param dims         The dimensions of the tensor that will be
 *                      sliced/updated.
 *  \param sizes        The number of elements of each dimension in \p dims
 *                      that will be sliced/updated.
 *  \param numIndices   The number of slices this tensor should contain.
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 *
 *  \returns            A tensor with shape [numIndices, shape...] mapped
 *                      appropriately to be sliced into/updated from.
 */
poplar::Tensor createSliceTensor(poplar::Graph &graph, const poplar::Tensor &t,
                                 const std::vector<size_t> &dims,
                                 const std::vector<size_t> &sizes,
                                 std::size_t numIndices,
                                 const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to be sliced into or updated from efficiently.
 *
 *  The returned tensor is laid out according to the plan for the
 *  slice/update operation.
 *
 *  \param graph        The Poplar graph.
 *  \param type         The type of the elements.
 *  \param shape        The shape of the tensor to be slice/updated.
 *  \param dims         The dimensions of the tensor that will be
 *                      sliced/updated.
 *  \param sizes        The number of elements of each dimension in \p dims
 *                      that will be sliced/updated.
 *  \param numIndices   The number of slices this tensor should contain.
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 *
 *  \returns            A tensor with shape [numIndices, shape...] mapped
 *                      appropriately to be sliced into/updated from.
 **/
poplar::Tensor createSliceTensor(poplar::Graph &graph, const poplar::Type &type,
                                 const std::vector<std::size_t> &shape,
                                 const std::vector<std::size_t> &dims,
                                 const std::vector<std::size_t> &sizes,
                                 std::size_t numIndices, const SlicePlan &plan,
                                 const poplar::OptionFlags &options,
                                 const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to contain indices for slicing or updating
 *  a tensor efficiently.
 *
 * \param graph       The Poplar graph.
 * \param dims        The dimensions of a tensor to be sliced/updated that will
 *                    be sliced/updated using these indices.
 * \param numIndices  The number of indices this tensor should contain
 * \param plan        Plan describing how the slicing/updating operation will
 *                    be implemented.
 * \param options     Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 *
 * \returns A tensor of shape [numIndices, dims.size()] mapped appropriately
 *          to be used as the indices for a slice/update operation. Element type
 *          is always UNSIGNED_INT.
 */
poplar::Tensor
createIndicesTensor(poplar::Graph &graph, const std::vector<std::size_t> &dims,
                    std::size_t numIndices, const SlicePlan &plan,
                    const poplar::OptionFlags &options,
                    const poplar::DebugContext &debugContext = {});

/** Create and map a tensor to be sliced/updated.
 *
 * The tensor is mapped in a way that can be efficiently sliced and updated
 * to/from the given slice tensor. It will be distributed across as many
 * tiles as the given slice and with the same contiguous regions on each tile.
 * The tensor's shape and mapping are derived from the reference slice tensor.
 *
 * \param graph       The Poplar graph.
 * \param s           The reference slice.
 * \param dims        The dimensions of the returned tensor that will be
 *                    sliced.
 * \param numSlices   The number of independent slices in each sliced
 *                    dimension.
 *  \param debugContext Optional debug information.
 *
 * \returns           A tensor to be sliced/updated.
 */
poplar::Tensor
createSliceableTensorFromSlice(poplar::Graph &graph, const poplar::Tensor &s,
                               const std::vector<std::size_t> &dims,
                               const std::vector<std::size_t> &numSlices,
                               const poplar::DebugContext &debugContext = {});

/** Slice a tensor based on offsets specified by a tensor.
 *
 *  \p dims gives the dimensions to slice, \p sizes defines the size of the
 *  slice in those dimensions and \p offset gives the base offsets on each
 *  execution.
 *
 *  \p offset[0], \p dims and \p sizes must have the same size. \p offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor.
 *  \param offset      A tensor of offsets at which the output is extracted.
 *  \param dims        The dimensions of \p t to slice.
 *  \param sizes       The size of the slice in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended
 *  \param debugContext Optional debug information.
 *  \returns           The specified subtensor
 **/
poplar::Tensor dynamicSlice(poplar::Graph &graph, const poplar::Tensor &t,
                            const poplar::Tensor &offset,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &sizes,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext = {});

/** Slice a tensor based on offsets specified by a tensor.
 *
 *  \p dims gives the dimensions to slice, \p sizes defines the size of the
 *  slice in those dimensions and \p offset gives the base offsets on each
 *  execution.
 *
 *  \p offset[0], \p dims and \p sizes must have the same size. \p offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The Poplar graph.
 *  \param output      The output tensor, This should ideally be created with
 *                     `createSliceTensor` to maximise efficiency,
 *  \param t           The source tensor.
 *  \param offset      A tensor of offsets at which the output is extracted.
 *  \param dims        The dimensions of \p t to slice.
 *  \param sizes       The size of the slice in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended \param debugContext Optional
 *                     debug information.
 **/
void dynamicSliceWithOutput(poplar::Graph &graph, const poplar::Tensor &output,
                            const poplar::Tensor &t,
                            const poplar::Tensor &offset,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &sizes,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext = {});

/** Get the tile mapping for a slice of a tensor.
 *
 *  \p dims gives the dimensions to slice, \p sizes defines the size of the
 *  slice in those dimensions.
 *  \param graph       The Poplar graph.
 *  \param t           The source tensor.
 *  \param dims        The dimensions of \p t to slice.
 *  \param sizes       The size of the slice in each of the dimensions in
 *                     \p dims.
 */
poplar::Graph::TileToTensorMapping
getSliceMapping(poplar::Graph &graph, const poplar::Tensor &t,
                const std::vector<std::size_t> &dims,
                const std::vector<std::size_t> &sizes);

/** Update a subtensor at offsets read from a tensor.
 *
 *  \p dims gives the dimensions that are partially updated, by \p sizes
 *  elements,
 *  at offsets \p offset. Unspecified dimensions are copied in full with zero
 *  offset.
 *
 *  \p offset[0], \p dims and \p sizes must have the same size. \p offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The Poplar graph.
 *  \param t           The tensor to update.
 *  \param s           The updates.
 *  \param offset      The offset within \p t to be updated.
 *  \param dims        The dimensions to be dynamically updated.
 *  \param sizes       The size of the update in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended.
 *  \param debugContext Optional debug information.
 **/
void dynamicUpdate(poplar::Graph &graph, const poplar::Tensor &t,
                   const poplar::Tensor &s, const poplar::Tensor &offset,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext = {});

/** Take multiple slices from a base tensor.
 *
 * The returned tensor will have a rank one greater than \p t. Its outer
 * dimension
 * will be \p offsets.dim(0). Note that \p dims refers to the dimensions
 * of \p t.
 * \p t can be created using \p createSliceableTensor() to ensure efficient
 * mapping.
 *  \param graph       The Poplar graph.
 *  \param t           The tensor being sliced.
 *  \param offsets     The offsets within \p t to be sliced.
 *  \param dims        The dimensions of \p t to be sliced.
 *  \param sizes       The size of the update in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended.
 *  \param plan        Plan describing how the operation will
 *                     be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 */
poplar::Tensor multiSlice(poplar::Graph &graph, const poplar::Tensor &t,
                          const poplar::Tensor &offsets,
                          const std::vector<std::size_t> &dims,
                          const std::vector<std::size_t> &sizes,
                          poplar::program::Sequence &prog,
                          const SlicePlan &plan,
                          const poplar::OptionFlags &options,
                          const poplar::DebugContext &debugContext = {});

/** Update multiple slices in a tensor.
 *
 *  \param graph       The Poplar graph.
 *  \param t           The tensor being updated.
 *  \param s           The slices to insert.
 *  \param offsets     The offsets within \p t to be updated.
 *  \param dims        The dimensions of \p t to be updated.
 *  \param sizes       The size of the update in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended.
 *  \param plan        Plan describing how the operation will be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 */
void multiUpdate(poplar::Graph &graph, const poplar::Tensor &t,
                 const poplar::Tensor &s, const poplar::Tensor &offsets,
                 const std::vector<std::size_t> &dims,
                 const std::vector<std::size_t> &sizes,
                 poplar::program::Sequence &prog, const SlicePlan &plan,
                 const poplar::OptionFlags &options,
                 const poplar::DebugContext &debugContext = {});

/** Accumulate multiple slices in a tensor
 * for i offsets:
 *   t[offsets[i]] += scale * s[i]
 * \p t, \p s must be of the same type
 *
 *  \param graph       The Poplar graph.
 *  \param t           The tensor being updated (must be rank 2).
 *  \param s           The slices to accumulate.
 *  \param offsets     The offsets within \p t to be accumulated.
 *  \param scale       The scaling to apply to the update. The type of the
 *                     tensor should be the same as that of \p t and \p s except
 *                     for the case when \p t and \p s are of type HALF. In
 *                     which case \p scale can be of type FLOAT or HALF.
 *  \param dims        The dimensions of \p t to be accumulated
 *                     (must be rank 1).
 *  \param sizes       The size of the accumulate in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended.
 *  \param plan        Plan describing how the operation will be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 */
void multiUpdateAdd(poplar::Graph &graph, const poplar::Tensor &t,
                    const poplar::Tensor &s, const poplar::Tensor &offsets,
                    const poplar::Tensor &scale,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog, const SlicePlan &plan,
                    const poplar::OptionFlags &options,
                    const poplar::DebugContext &debugContext = {});

/** Find maximum over multiple slices in a tensor
 * for i offsets:
 *   t[offsets[i]] = max(t[offsets[i]], s[i])
 * \p t, \p s must have the same element type
 *  offsets[i] >= t.dim(0) are ignored.
 *
 *  \param graph       The Poplar graph.
 *  \param t           The tensor being updated (must be rank 2).
 *  \param s           The slices to find maximum over.
 *  \param offsets     The offsets within \p t to find maximum over.
 *  \param dims        The dimensions of \p t to find maximum over
 *                     (must be rank 1).
 *  \param sizes       The size of the update in each of the dimensions in
 *                     \p dims.
 *  \param prog        The program to be extended.
 *  \param plan        Plan describing how the operation will be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugContext Optional debug information.
 */
void multiUpdateMax(poplar::Graph &graph, const poplar::Tensor &t,
                    const poplar::Tensor &s, const poplar::Tensor &offsets,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog, const SlicePlan &plan,
                    const poplar::OptionFlags &options,
                    const poplar::DebugContext &debugContext = {});

namespace embedding {

/** Create a plan for implementing a set of operations on an
 *  embedding matrix.
 *
 *  ** Embedding plan options **
 *
 *     * `usedForSlice` (true, false) [=true]
 *
 *       If true, you intend to use this embedding plan for both a multiSlice
 *       operation. An error is thrown if set to false and `usedForUpdate` is
 *       set to false.
 *
 *     * `usedForUpdate` (true, false) [=true]
 *
 *       If true, you intend to use this embedding plan for both a multiUpdate
 *       operation. An error is thrown if set to false and `usedForSlice` is
 *       set to false.
 *
 *     * `operationForUpdate` ("none", "add", "max") [="add"]
 *
 *       Only applicable when `usedForUpdate` = true. Is the type of operation
 *       used in multi-update.
 *         Set to "none" for multiUpdate
 *                "add" for multiUpdateAdd
 *                "max" for multiUpdateMax
 *
 *     * `availableMemoryProportion` Positive decimal
 *
 *       If set, gives the proportion of tile memory made available for
 * temporary variables (variables that become live and die during the operation)
 *       for this operation. If not set, the operation has the freedom to
 *       use unlimited temporary memory.
 *
 *     * `indicesDistribution` (uniform, onePoint) [=uniform]
 *
 *       A description of the statistical distribution of the indices that will
 *       be sliced/updated over the input size (\p numEntries) of the
 *       operation. This is used to when estimating the runtime of the
 *       multiSlice and multiUpdate* operation.
 *
 *       * `uniform`   Indices are assumed to be uniformly distributed over
 *                     the input size of the embedding.
 *       * `onePoint`  Indices are assumed to all be equal.
 *
 *     * `planMinimisationTarget` (memory, cycles) [=memory]
 *
 *       Select what should be minimised when planning this operation.
 *
 *       * `memory`    Minimise a weighted combination of estimated maximum
 *                     tile memory needed for code, for input/indices/output
 *                     operands, and temporary variables for the operation.
 *       * `cycles`    Minimise estimated total cycles for the operation.
 *
 *     * `indicesAreSorted` (true, false) [=false]
 *
 *       Plan assuming indices used in MultiUpdate/MultiUpdateOp are sorted in
 *       increasing order. The same option must then be used along with the plan
 *       when calling MultiUpdate with and without an operation.
 *
 * \param graph       The graph the operation will be added to.
 * \param dataType    The data type of the entries in the embedding matrix
 *                    and the resulting lookups from the matrix.
 * \param numEntries  Input size of embedding matrix.
 * \param outputSize  Output size of embedding matrix lookup.
 * \param numLookups  Vector of numbers of indices which will be looked
 *                    up in the embedding matrix.
 * \param options     Set of option flags controlling how the operation
 *                    will be implemented.
 *
 * \returns A plan which describes how the embedding matrix lookup/update
 *          operations should be implemented.
 */
SlicePlan plan(const poplar::Graph &graph, const poplar::Type &dataType,
               const std::size_t numEntries, const std::size_t outputSize,
               const std::vector<std::size_t> &numLookups,
               const poplar::OptionFlags &options);

} // end namespace embedding

} // end namespace popops
#endif // popops_DynamicSlice_hpp
