// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_DynamicSlice_hpp
#define popops_DynamicSlice_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace poplar {
class Tensor;
}

namespace popops {

/// A object representing a plan describing how to implement a
/// particular slice/update to slice/update APIs.
class SlicePlanInternal;
class SlicePlan {
public:
  SlicePlan();
  ~SlicePlan();
  SlicePlan(const SlicePlan &other);
  SlicePlan(SlicePlan &&other);
  SlicePlan &operator=(const SlicePlan &other);
  SlicePlan &operator=(SlicePlan &&other);

  friend std::ostream &operator<<(std::ostream &o, const SlicePlan &p);

  // Implementation
  SlicePlan(std::unique_ptr<SlicePlanInternal> internal);
  SlicePlanInternal &getImpl() const { return *internal; }
private:
  std::unique_ptr<SlicePlanInternal> internal;
};

/** Create and map a tensor to be sliced/updated efficiently.
 *
 *  The returned tensor will be laid out according to the plan.
 *
 *  \param graph        The poplar graph
 *  \param type         The type of the elements
 *  \param shape        The shape of the tensor to be slice/updated
 *  \param dims         The dimensions of the tensor that will be slice/updated
 *  \param sizes        The size of the slice in each of \a dims
 *  \param minGrainSize The minimum elements per slice mapped to each tile
 *  \param debugPrefix  A string prepended to debugging info.
 *  \returns            A tensor shape \a shape that is suitably mapped
 *
 */
poplar::Tensor
createSliceableTensor(poplar::Graph &graph,
                      const poplar::Type &type,
                      const std::vector<size_t> &shape,
                      const std::vector<size_t> &dims,
                      const std::vector<size_t> &sizes,
                      std::size_t minGrainSize = 0,
                      const std::string &debugPrefix = "");

/** Create and map a tensor to be sliced/updated efficiently.
 *
 *  The returned tensor will be spread over as many tiles
 *  as possible while respecting this minimum no. of elements per-tile and
 *  still being in a form to be sliced/updated efficiently.
 *
 *  \param graph        The poplar graph
 *  \param type         The type of the elements
 *  \param shape        The shape of the tensor to be slice/updated
 *  \param dims         The dimensions of the tensor that will be slice/updated
 *  \param sizes        The size of the slice in each of \a dims
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugPrefix  A string prepended to debugging info.
 *  \returns            A tensor shape \a shape that is suitably mapped
 **/
poplar::Tensor
createSliceableTensor(poplar::Graph &graph,
                      const poplar::Type &type,
                      const std::vector<size_t> &shape,
                      const std::vector<size_t> &dims,
                      const std::vector<size_t> &sizes,
                      const SlicePlan &plan,
                      const poplar::OptionFlags &options,
                      const std::string &debugPrefix = "");

/** Create and map a tensor to be sliced into/updated from efficiently.
 *
 *  Introspection on the tensor to update is used to lay out
 *  the returned tensor such that it can be used to update that tensor
 *  efficiently.
 *
 *  \param graph        The poplar graph.
 *  \param t            The tensor to be updated.
 *  \param dims         The dimensions of the tensor that will be
 *                      sliced/updated.
 *  \param sizes        The number of elements of each dimension in `dims` that
 *                      will be sliced/updated.
 *  \param numIndices   The number of slices this tensor should contain.
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugPrefix  A string prepended to debugging info.
 *
 *  \returns            A tensor with shape [numIndices, shape...] mapped
 *                      appropriately to be sliced into/updated from.
 */
poplar::Tensor
createSliceTensor(poplar::Graph &graph,
                  const poplar::Tensor &t,
                  const std::vector<size_t> &dims,
                  const std::vector<size_t> &sizes,
                  std::size_t numIndices,
                  const std::string &debugPrefix = "");

/** Create and map a tensor to be sliced into/updated from efficiently.
 *
 *  The returned tensor is laid out according to the plan for the
 *  slice/update operation.
 *
 *  \param graph        The poplar graph.
 *  \param type         The type of the elements.
 *  \param shape        The shape of the tensor to be slice/updated.
 *  \param dims         The dimensions of the tensor that will be
 *                      sliced/updated.
 *  \param sizes        The number of elements of each dimension in `dims` that
 *                      will be sliced/updated.
 *  \param numIndices   The number of slices this tensor should contain.
 *  \param plan         Plan describing how the slicing/updating operation will
 *                      be implemented.
 *  \param options      Flags controlling how the operation will be implemented.
 *  \param debugPrefix  A string prepended to debugging info.
 *
 *  \returns            A tensor with shape [numIndices, shape...] mapped
 *                      appropriately to be sliced into/updated from.
 **/
poplar::Tensor
createSliceTensor(poplar::Graph &graph,
                  const poplar::Type &type,
                  const std::vector<std::size_t> &shape,
                  const std::vector<std::size_t> &dims,
                  const std::vector<std::size_t> &sizes,
                  std::size_t numIndices,
                  const SlicePlan &plan,
                  const poplar::OptionFlags &options,
                  const std::string &debugPrefix = "");

/** Create and map a tensor to contain indices for slicing/updating
 *  a tensor efficiently.
 *
 * \param graph       The poplar graph.
 * \param dims        The dimensions of a tensor to be sliced/updated that will
 *                    be sliced/updated using these indices.
 * \param numIndices  The number of indices this tensor should contain
 * \param plan        Plan describing how the slicing/updating operation will
 *                    be implemented.
 * \param options     Flags controlling how the operation will be implemented.
 * \param debugPrefix The prefix prepended to debugging info.
 *
 * \returns A tensor of shape [numIndices, dims.size()] mapped appropriately
 *          to be used as the indices for a slice/update operation. Element type
 *          is always UNSIGNED_INT.
 */
poplar::Tensor
createIndicesTensor(poplar::Graph &graph,
                    const std::vector<std::size_t> &dims,
                    std::size_t numIndices,
                    const SlicePlan &plan,
                    const poplar::OptionFlags &options,
                    const std::string &debugPrefix = "");

/* Create and map a tensor to be sliced/updated
 *
 * The tensor is mapped in a way that can be efficiently sliced and updated
 * to/from the given slice tensor. It will be distributed across as many
 * tiles as the given slice and with the same contiguous regions on each tile.
 * The tensor's shape and mapping are derived from the reference slice tensor.
 *
 * \param graph       The poplar graph
 * \param s           The reference slice
 * \param dims        The dimensions of the returned tensor that will be sliced
 * \param numSlices   The number of independent slices in each sliced dimension
 * \param debugPrefix The prefix prepended to debugging info.
 */
poplar::Tensor
createSliceableTensorFromSlice(poplar::Graph &graph,
                               const poplar::Tensor &s,
                               const std::vector<std::size_t> &dims,
                               const std::vector<std::size_t> &numSlices,
                               const std::string &debugPrefix = "");

/** Slice a tensor based on offsets specified by a tensor.
 *  \a dims gives the dimensions to slice, \a sizes defines the size of the
 *  slice in those dimensions and \a offset gives the base offsets on each
 *  execution.
 *  \a offset[0], \a dims and \a sizes must have the same size. \a offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The poplar graph
 *  \param t           The source tensor
 *  \param offset      A tensor of offsets at which the output is extracted
 *  \param dims        The dimensions of \a t to slice
 *  \param sizes       The size of the slice in each of \a dims
 *  \param prog        The program to be extended
 *  \param debugPrefix The prefix prepended to debugging info
 *  \returns           The specified subtensor
 **/
poplar::Tensor dynamicSlice(poplar::Graph &graph,
                            const poplar::Tensor &t,
                            const poplar::Tensor &offset,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &sizes,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");

/** Get the tile mapping for a slice of a tensor
 *  \a dims gives the dimensions to slice, \a sizes defines the size of the
 *  slice in those dimensions
 *  \param graph       The poplar graph
 *  \param t           The source tensor
 *  \param dims        The dimensions of \a t to slice
 *  \param sizes       The size of the slice in each of \a dims
 */
poplar::Graph::TileToTensorMapping
getSliceMapping(poplar::Graph &graph,
                const poplar::Tensor &t,
                const std::vector<std::size_t> &dims,
                const std::vector<std::size_t> &sizes);

/** Update a subtensor at offsets read from a tensor
 *  \a dims gives the dimensions that are partialy updated, by \a sizes elements
 *  at offsets \a offset. Unspecified dimensions are copied in full with zero
 *  offset.
 *  \a offset[0], \a dims and \a sizes must have the same size. \a offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The poplar graph
 *  \param t           The tensor to update
 *  \param s           The updates
 *  \param offset      The offset within \a t to be updated
 *  \param dims        The dimensions to be dynamically updated
 *  \param sizes       The size of the update in each of \a dims
 *  \param prog        The program to be extended
 *  \param debugPrefix The prefix prepended to debugging info
 **/
void dynamicUpdate(poplar::Graph &graph,
                   const poplar::Tensor &t,
                   const poplar::Tensor &s,
                   const poplar::Tensor &offset,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

/** Take multiple slices from a base tensor
 * The returned tensor will have a rank 1 greater than t. Its outer dimension
 * will be offsets.dim(0). Note that \a dims refers to the dimensions of \a t.
 * \a t can be created using createSliceableTensor() to ensure efficient
 * mapping
 *  \param graph       The poplar graph
 *  \param t           The tensor being sliced
 *  \param offsets     The offsets within \a t to be sliced
 *  \param dims        The dimensions of \a t to be sliced
 *  \param sizes       The size of the update in each of \a dims
 *  \param prog        The program to be extended
 *  \param plan        Plan describing how the operation will
 *                     be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugPrefix The prefix prepended to debugging info
 */
poplar::Tensor multiSlice(poplar::Graph &graph,
                          const poplar::Tensor &t,
                          const poplar::Tensor &offsets,
                          const std::vector<std::size_t> &dims,
                          const std::vector<std::size_t> &sizes,
                          poplar::program::Sequence &prog,
                          const SlicePlan &plan,
                          const poplar::OptionFlags &options,
                          const std::string &debugPrefix = "");

/** Update multiple slices in a tensor
 *
 *  \param graph       The poplar graph
 *  \param t           The tensor being updated
 *  \param s           The slices to insert
 *  \param offsets     The offsets within \a t to be updated
 *  \param dims        The dimensions of \a t to be updated
 *  \param sizes       The size of the update in each of \a dims
 *  \param prog        The program to be extended
 *  \param plan        Plan describing how the operation will be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugPrefix The prefix prepended to debugging info
 */
void multiUpdate(poplar::Graph &graph,
                 const poplar::Tensor &t,
                 const poplar::Tensor &s,
                 const poplar::Tensor &offsets,
                 const std::vector<std::size_t> &dims,
                 const std::vector<std::size_t> &sizes,
                 poplar::program::Sequence &prog,
                 const SlicePlan &plan,
                 const poplar::OptionFlags &options,
                 const std::string &debugPrefix = "");

/** Accumulate multiple slices in a tensor
 * for i offsets:
 *   t[offsets[i]] += scale * s[i]
 * t, s and scale must have the same element type
 *
 *  \param graph       The poplar graph
 *  \param t           The tensor being updated (must be rank 2)
 *  \param s           The slices to accumulate
 *  \param offsets     The offsets within \a t to be accumulated
 *  \param scale       The scaling to apply to the update
 *  \param dims        The dimensions of \a t to be acumulated (must be rank 1)
 *  \param sizes       The size of the accumulate in each of \a dims
 *  \param prog        The program to be extended
 *  \param plan        Plan describing how the operation will be implemented.
 *  \param options     Flags controlling how the operation will be implemented.
 *  \param debugPrefix The prefix prepended to debugging info
 */
void multiUpdateAdd(poplar::Graph &graph,
                    const poplar::Tensor &t,
                    const poplar::Tensor &s,
                    const poplar::Tensor &offsets,
                    const poplar::Tensor &scale,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog,
                    const SlicePlan &plan,
                    const poplar::OptionFlags &options,
                    const std::string &debugPrefix = "");

namespace embedding {

/** Create a plan for implementing a set of operations on an
 *  embedding matrix.
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
SlicePlan plan(const poplar::Graph &graph,
               const poplar::Type &dataType,
               const std::size_t numEntries,
               const std::size_t outputSize,
               const std::vector<std::size_t> &numLookups,
               const poplar::OptionFlags &options);

} // end namespace embedding

} // end namespace popops
#endif // popops_DynamicSlice_hpp
