// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Cast_hpp
#define popops_Cast_hpp

#include <poplar/Interval.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <vector>

namespace popops {

/// Cast elements of the specified \a src tensor to \a dstType, returning the
/// result as a new tensor.
poplar::Tensor
cast(poplar::Graph &graph, const poplar::Tensor &src,
     const poplar::Type &dstType, poplar::program::Sequence &prog,
     const std::string &debugPrefix = "");

/// Create a program to copy tensor casting between types (e.g. half->float).
poplar::program::Program
cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
     const std::string &debugPrefix = "");

/// Create vertices to copy element wise from the src tensor to the dst tensor
/// casting between types (e.g. half->float). The vertices are added to the
/// specified compute set.
void
cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
     poplar::ComputeSet cs);

/// Create vertices to cast elements of the specified \a src tensor to \a
/// dstType, returning the result as a new tensor. The vertices are added to the
/// specified compute set.
poplar::Tensor
cast(poplar::Graph &graph, poplar::Tensor src, const poplar::Type &dstType,
     poplar::ComputeSet cs, const std::string &debugPrefix = "");


// Helper function which checks the absolute error in the float Tensor \value
// when casting it to half precision.  The result is a Tensor of type bool
// which is true if the error is < \tolerance
poplar::Tensor
checkAccuracyInHalfPrecision(poplar::Graph &graph, const poplar::Tensor &value,
                             float tolerance,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix =  "");

}


#endif // popops_Cast_hpp
