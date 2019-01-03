// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef popops_Gather_hpp
#define popops_Gather_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {

/**
 *  The gather operation stitches together several slices (each slice at a
 *  potentially different runtime offset) of an input array
 *
 *  \param graph              The poplar graph
 *  \param operand            The array we are gathering from
 *  \param indices            Array containing the starting indices of the
 *                            slices we gather
 *  \param indexVectorDim     The dimension in `indices` that "contains" the
 *                            starting indices
 *  \param offsetDims         The set of dimensions in the output shape that
 *                            offset into a array sliced from operand.
 *  \param sliceSizes         `slice_sizes[i]` is the bounds for the slice on
 *                            dimension `i`
 *  \param collapsedSliceDims The set of dimensions in each slice that are
 *                            collapsed away. These dimensions must have size 1
 *  \param startIndexMap      A map that describes how to map indices in
 *                            `startIndices` to legal indices into operand
 *  \param prog               The program sequence to add this operation to
 *  \param debugPrefix        A debug name for the operation
 *
 *  \returns The gathered slices from the operand
 */
poplar::Tensor gather(poplar::Graph &graph, const poplar::Tensor &operand,
                      const poplar::Tensor &indices, std::size_t indexVectorDim,
                      std::vector<std::size_t> offsetDims,
                      std::vector<std::size_t> sliceSizes,
                      std::vector<std::size_t> collapsedSliceDims,
                      std::vector<unsigned> startIndexMap,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_DynamicSlice_hpp
