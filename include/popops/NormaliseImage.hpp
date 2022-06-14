// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions for padding and normalising image tensors.
 *
 */

#ifndef popops_NormaliseImage_hpp
#define popops_NormaliseImage_hpp
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popops {

/// Add a tensor for a 3-channel image suitable for padding to 4 channels.
///
/// \param graph The graph to which the tensor will be added.
/// \param type  The type of the elements. Must be \c UNSIGNED_CHAR, \c HALF
///  or \c FLOAT.
/// \param shape Required tensor shape. Must have an inner
///              dimension of three.
/// \param debugContext Debugging context.
///
poplar::Tensor
createNormaliseImageInput(poplar::Graph &graph, const poplar::Type &type,
                          const poplar::ArrayRef<std::size_t> shape,
                          const poplar::DebugContext &debugContext = {});

/// Pad a tensor to have 4 channel dimensions.
///
/// Each channel is normalised via:
///
///      tIn[c] * inScale - offset[c]) * scale[c]
///
/// \a tIn must be mapped with a single region of complete pixels on each tile.
/// \ref createNormaliseImageInput() creates a variable that is suitably mapped.
///
/// \c UINT8 inputs are cast to \c HALF. Otherwise the output tensor follows the
/// input type.
///
/// \param graph      The graph containing the tensor.
/// \param seq        The sequence to which the normalisation programs will be
///                   added.
/// \param tIn        Input image. It must have an inner dimension of 3.
///                   and be \c UNSIGNED_CHAR, \c HALF or \c FLOAT.
/// \param inScale    Input scaling.
/// \param offsets    Offset for each channel. Must be shape {3} and must match
///                   the output type.
/// \param scales     Scaling factor for each channel. Must be shape {3} and
///                   must match the output type.
/// \param debugContext Debugging context.
///
poplar::Tensor normaliseImage(poplar::Graph &graph,
                              poplar::program::Sequence &seq,
                              poplar::Tensor tIn, float inScale,
                              poplar::Tensor offsets, poplar::Tensor scales,
                              const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_NormaliseImage_hpp
