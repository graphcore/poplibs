// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for dynamic slices.
 *
 */

#ifndef popops_SequenceSlice_hpp
#define popops_SequenceSlice_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poputil/DebugInfo.hpp>
#include <string>
#include <vector>

namespace poplar {
class Tensor;
}

namespace popops {

/** Slice a 2d tensor based on offsets specified by a tensor.
 *
 * Typically this is used to copy subsequences of one tensor to another.
 *  The outermost dimension is sliced;
 *    tOut[tOutOffset:tOutOffset+tN][...] = tIn[tInOffset:tInOffset+tN][...]
 *  for each entry in tN/tInOffset/tOutOffset; entries after the first tN==0 may
 *  be ignored.
 *  Unreferenced elements of tOut are zeroed if zeroUnused is set. The same
 *  output element should not be written by multiple inputs.
 *
 *  \p tIn and \p tOut must have rank >=2. The outer dimension is sliced; the
 *  product of the inner dimensions must match.
 *  \p tInOffset, \p tOutOffset and \p tN must be 1d and the same size.
 *  \param graph       The Poplar graph.
 *  \param tIn         The source tensor.
 *  \param tOut        The destination tensor.
 *  \param tN          The number of elements to copy.
 *  \param tInOffset   First element read from \p tIn.
 *  \param tOutOffset  First element written in \p tOut.
 *  \param zeroUnused	 Whether to zero unreferenced \p tOut elements.
 *  \param prog        The program to be extended.
 *  \param debugContext Optional debug information.
 **/
void sequenceSlice(poplar::Graph &graph, const poplar::Tensor &tIn,
                   const poplar::Tensor &tOut, const poplar::Tensor &tN,
                   const poplar::Tensor &tInOffset,
                   const poplar::Tensor &tOutOffset, bool zeroUnused,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext = {});

} // end namespace popops
#endif // popops_DynamicSlice_hpp
