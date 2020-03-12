// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popops_NaN_hpp
#define popops_NaN_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace popops {
/** Takes a tensor of any shape and type float or half and returns a new
 *  scalar bool tensor whose only element is true if any of the elements of the
 *  src tensor contained a NaN.
 *
 *  \param graph        The graph to add the tensor and any vertices to.
 *  \param src          The input tensor, the type must be floating point.
 *  \param prog         Sequence to add programs to to perform the check.
 *  \param debugPrefix  Optional debug prefix for programs/variables.
 */
poplar::Tensor hasNaN(poplar::Graph &graph, const poplar::Tensor &src,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_NaN_hpp
