// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Test for NaN values in a tensor.
 *
 */

#ifndef popops_NaN_hpp
#define popops_NaN_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace popops {
/** Test for NaN values in a tensor.
 * Takes a tensor of any shape and type float or half and returns a new
 *  scalar bool tensor whose only element is true if any of the elements of the
 *  \p src tensor contained a NaN.
 *
 *  \param graph        The graph to add the tensor and any vertices to.
 *  \param src          The input tensor, the type must be floating point.
 *  \param prog         Sequence to add programs to, which perform the check.
 *  \param debugContext Optional debug information.
 */
poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {});

/** Test for NaN or Inf values in a tensor.
 * Takes a tensor of any shape and type float or half and returns a new
 *  scalar bool tensor whose only element is true if any of the elements of the
 *  \p src tensor contained a NaN or an Inf.
 *
 *  \param graph        The graph to add the tensor and any vertices to.
 *  \param src          The input tensor, the type must be floating point.
 *  \param prog         Sequence to add programs to, which perform the check.
 *  \param debugContext Optional debug information.
 */
poplar::Tensor hasNaNOrInf(poplar::Graph &graph, const poplar::Tensor &src,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext = {});

} // namespace popops

#endif // popops_NaN_hpp
