// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Cast_hpp
#define popops_Cast_hpp

#include <poplar/Graph.hpp>
#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <vector>

namespace popops {

/// Cast elements of the specified \p src tensor to \p dstType, returning the
/// result as a new tensor.
poplar::Tensor cast(poplar::Graph &graph, const poplar::Tensor &src,
                    const poplar::Type &dstType,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "");

/// Create a program to copy tensor casting between types (for
/// example, half->float).
poplar::program::Program cast(poplar::Graph &graph, poplar::Tensor src,
                              poplar::Tensor dst,
                              const std::string &debugPrefix = "");

/// Create vertices to copy element wise from the \p src tensor to the \p dst
/// tensor casting between types (for example, half->float).
/// The vertices are added to the specified compute set.
void cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
          poplar::ComputeSet cs);

/// Create vertices to cast elements of the specified \p src tensor to \p
/// dstType, returning the result as a new tensor. The vertices are added to the
/// specified compute set.
poplar::Tensor cast(poplar::Graph &graph, poplar::Tensor src,
                    const poplar::Type &dstType, poplar::ComputeSet cs,
                    const std::string &debugPrefix = "");

/// Helper function which checks the relative error in the tensor \p input
/// when casting it to type \p outputType.  The result is a bool tensor
/// which is set to true if the error is < \p tolerance.
poplar::Tensor checkAccuracyWhenCast(poplar::Graph &graph,
                                     const poplar::Tensor &input,
                                     poplar::Type outputType, double tolerance,
                                     poplar::program::Sequence &prog,
                                     const std::string &debugPrefix = "");

} // namespace popops

#endif // popops_Cast_hpp
