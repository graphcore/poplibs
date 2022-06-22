// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef popops_GatherInternal_hpp
#define popops_GatherInternal_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {
namespace internal {

/**
 * Internal gather implementation
 *
 * This allows the user to extract multiple slices from the input tensor. The
 * indices must be a 2D tensor where the first dim is the number of slices to
 * take and the second dimension contains the slice coordinates.
 *
 * These k-d coordinates correspond to the outermost k dimensions of the input
 * tensor. The remaining dimensions are assumed to be taken as whole slices.
 *
 * The output shape must be
 *  [indices.dim(0), sliceSizes[0...k], input.dim(k...n)]
 *
 * Example where we would like to take the first and last row from a matrix:
 *  // Setup tensors
 *  input := {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; // shape = [3, 3]
 *  indices := {{0}, {2}};                      // shape = [2, 1]
 *  output := {{0, 0, 0}, {0, 0, 0}};           // shape = [2, 3]
 *
 *  // Since we want to take a single row, we set our slice size to 1.
 *  // We will also be taking the whole row, so we exclude the size and
 *  // implicitly take the whole row.
 *  sliceSizes := {1};
 *
 *  // Call gather
 *  output := gather(g, input, indices, sliceSizes, progs);
 *
 *  // We now expect output to be filled with the selected rows
 *  output == {{1, 2, 3}, {7, 8, 9}}; // shape = [2, 3]
 *
 *  \param graph              The poplar graph
 *  \param input              The input tensor
 *  \param indices            The 2D indices tensor
 *  \param sliceSizes         The size of each sliced dimension
 *  \param prog               The program sequence to add this operation to
 *  \param dnai               The debug reference
 *  \param optionFlags        Option flags
 *
 *  \returns The gathered slices tensor
 */
poplar::Tensor gather(poplar::Graph &graph, const poplar::Tensor &input,
                      const poplar::Tensor &indices,
                      const std::vector<std::size_t> &sliceSizes,
                      poplar::program::Sequence &prog,
                      const poplar::DebugNameAndId &dnai,
                      const poplar::OptionFlags &optionFlags);

poplar::Tensor
createGatherInputTensor(poplar::Graph &graph, poplar::Type type,
                        const std::vector<std::size_t> &inputShape,
                        const std::vector<std::size_t> &sliceSizes,
                        const poplar::DebugNameAndId &dnai);

} // namespace internal
} // namespace popops
#endif
