// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for gather operations.
 *
 */

#ifndef popops_Gather_hpp
#define popops_Gather_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {

/// Defines the parameters to a gather operation.
struct GatherParams {
  /// Suggested maximum number of elements to place on a tile.
  /// This can be used to balance the gather across the IPUs.
  std::size_t maxElementsPerTile = 65535;

  GatherParams() = default;
  GatherParams(std::size_t maxElementsPerTile_)
      : maxElementsPerTile(maxElementsPerTile_) {}
};

/**
 *  Create the input of the gather with only a single gather axis. This is
 *  designed to spread the gather, and each dynamic slice within the gather,
 *  across the tiles evenly.
 *
 *  \param graph        The Poplar graph.
 *  \param type         The data type of the required tensor.
 *  \param operandShape The desired shape of the input.
 *  \param axis         The axis that will be gathered on.
 *  \param params       The same parameters as used by the gather().
 *  \param debugContext Optional debug information.
 *
 *  \returns A tensor with the desired shape.
 */
poplar::Tensor createGatherInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::vector<std::size_t> &operandShape,
                                 unsigned axis, GatherParams params = {},
                                 const poplar::DebugContext &debugContext = {});

/**
 *  The gather operation stitches together several slices (each slice at a
 *  potentially different runtime offset) of an input tensor. To achieve the
 *  best performance, the input tensor should be created with
 *  createGatherInput().
 *
 *  \param graph       The Poplar graph.
 *  \param input       The tensor we are gathering from of rank x.
 *  \param indices     Tensor containing the indices of the slices we gather of
 *                     rank y.
 *  \param axis        The axis to gather on. The axis must be less than x.
 *  \param prog        The program sequence to add this operation to.
 *  \param params      Parameters for the form of the gather.
 *  \param debugContext Optional debug information.
 *
 *  \note The indices are treated as offsets along the chosen axis. At this
 *        offset a slice of depth 1 in the axis dimension is taken.
 *
 *  \returns The gathered slices from the input with rank y + (x - 1).
 */
poplar::Tensor gather(poplar::Graph &graph, const poplar::Tensor &input,
                      const poplar::Tensor &indices, unsigned axis,
                      poplar::program::Sequence &prog, GatherParams params,
                      const poplar::DebugContext &debugContext = {});

/**
 *  Create the input of the gather given a start index map. This is
 *  designed to spread the gather, and each dynamic slice within the gather,
 *  across the tiles evenly.
 *
 *  \param graph         The Poplar graph.
 *  \param type          The data type of the required tensor.
 *  \param inputShape    The desired shape of the input.
 *  \param sliceSizes    \p sliceSizes[i] is the bounds for
 *                        the slice on dimension \c i.
 *  \param startIndexMap A map that describes how to map indices in
 *                       \p indices in gather() to legal indices into the input.
 *  \param debugContext  Optional debug information.
 *
 *  \returns A tensor with the desired shape.
 */
poplar::Tensor createGatherInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::vector<std::size_t> &inputShape,
                                 const std::vector<std::size_t> &sliceSizes,
                                 std::vector<unsigned> startIndexMap,
                                 const poplar::DebugContext &debugContext = {});

/**
 *  The gather operation stitches together several slices (each slice at a
 *  potentially different runtime offset) of an input tensor. To achieve the
 *  best performance, the input tensor should be created with
 *  createGatherInput().
 *
 *  \param graph              The Poplar graph.
 *  \param input              The tensor we are gathering from.
 *  \param indices            Tensor containing the starting indices of the
 *                            slices we gather.
 *  \param indexVectorDim     The dimension in \p indices that "contains" the
 *                            starting indices.
 *  \param offsetDims         The set of dimensions in the output shape that
 *                            offset into a tensor sliced from \p input.
 *  \param sliceSizes         \p sliceSizes[i] is the bounds for the slice on
 *                            dimension \c i.
 *  \param collapsedSliceDims The set of dimensions in each slice that are
 *                            collapsed away. These dimensions must have size 1.
 *  \param startIndexMap      A map that describes how to map indices in
 *                            \p indices to legal indices into \p input.
 *  \param prog               The program sequence to add this operation to.
 *  \param debugContext       Optional debug information.
 *
 *  \note When `indexVectorDim == indices.rank()`, the indices are interpreted
 *        as scalar values.
 *
 *  \note This is a near direct port of
 *  https://www.tensorflow.org/xla/operation_semantics#gather from
 *  tensorflow/compiler/xla/service/gather_expander.cc
 *
 *  \returns The gathered slices from the input.
 *
 *  Example use where we want to take 2 elements from a given tensor:
 *
 *      // The runtime defined input tensor
 *      input = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; // shape = {3, 3}
 *
 *      // The runtime defined indices tensor containing the coords we want to
 *      // extract
 *      indices = {{1, 1}, {2, 1}}; // shape = {2, 2}
 *
 *      // We want to extract elems at [1, 1] and [2, 1] from the input
 *      // To achieve this we need to define the other parameters correctly
 *
 *      // We want to treat the rows of indices as coords into the input tensor
 *      indexVectorDim = 1;
 *
 *      // None of the output dims will correspond to any of the input dims
 *      offsetDims = {};
 *
 *      // We will be taking 1x1 slices to pick single elements
 *      sliceSizes = {1, 1};
 *
 *      // We will collapse both dims of the input slices
 *      collapsedSliceDims = {0, 1};
 *
 *      // An identity mapping between the indices coords and the input dims
 *      startIndexMap = {0, 1};
 *
 *      // Perform the desired gather
 *      result = gather(input,
 *                      indices,
 *                      indexVectorDim,
 *                      offsetDims,
 *                      sliceSizes
 *                      collapsedSliceDims,
 *                      startIndexMap) = {5, 8}; // shape = {2}
 *
 */
poplar::Tensor gather(poplar::Graph &graph, const poplar::Tensor &input,
                      const poplar::Tensor &indices, std::size_t indexVectorDim,
                      const std::vector<std::size_t> &offsetDims,
                      const std::vector<std::size_t> &sliceSizes,
                      const std::vector<std::size_t> &collapsedSliceDims,
                      const std::vector<unsigned> &startIndexMap,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_DynamicSlice_hpp
