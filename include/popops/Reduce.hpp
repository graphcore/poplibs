// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Reduce_hpp
#define popops_Reduce_hpp

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <poplar/OptionFlags.hpp>
#include <vector>

namespace popops {

/// Type of operation in a reduction
enum class Operation {
  ADD,
  MUL,
  MIN,
  MAX,
  LOGICAL_AND,
  LOGICAL_OR,
  SQUARE_ADD,
  // TODO: ABS_ADD
};

/// A reduce operation can optionally scale the output, and can also be an
/// "update", i.e. A += reduce(B) rather than A = reduce(B).
///
/// FullOperation stores that information, as well as the basic operation
/// being performed (add, mul, etc).
///
/// scale == 1.0f is treated as a special case and no scaling is applied.
///
struct ReduceParams {
  ReduceParams() = default;
  // Allow implicit convertion from a popops::Operation.
  ReduceParams(popops::Operation op, float scale = 1.0f, bool update = false)
    : op(op), scale(scale), update(update) {}

  popops::Operation op;
  float scale = 1.0f;
  bool update = false;
};

// Debug information about the reduction. This is internal currently.
struct ReductionDebug;

/// Reduce A in dimensions dims. params specifies the operation. Note that
/// currently scale and update are only valid with the Add operation, and they
/// cannot be used simultaneously.
///
/// Optionally a ReductionDebug object can be filled in with debug information
/// to help visualise and debug the reduction.
///
/// Internally this creates a new variable for the output then calls
/// reduceWithOutput(). The type of the output will be the same as the input.
///
/// The options parameter accepts the following:
///
///    'accumType.interTile' - The type to used for intermediate values
///                            between tiles (either 'float' or 'half').
///    'accumType.inVertex' - The type to used for intermediate values within
///                           a vertex (either 'float' or 'half').
/// If either of the above options are not set then the intermediate type will
/// default to either the input tensor element type or `float` if the input
/// is of type 'half' and the reduction operation benefits from
/// higher precision (e.g. add).
///
poplar::Tensor reduce(poplar::Graph &graph,
                      const poplar::Tensor &A,
                      const poplar::Type &outType,
                      const std::vector<std::size_t> &dims,
                      ReduceParams params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {},
                      ReductionDebug *debug = nullptr);

// An alias for reduce(graph, A, A.elementType(), ...)
poplar::Tensor reduce(poplar::Graph &graph,
                      const poplar::Tensor &A,
                      const std::vector<std::size_t> &dims,
                      ReduceParams params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {},
                      ReductionDebug *debug = nullptr);

/// This is similar to reduce() but allows you to specify the output.
/// If the tile mapping of `out` is not complete it will be set. Otherwise it
/// won't be changed.
poplar::Tensor reduceWithOutput(poplar::Graph &graph,
                                const poplar::Tensor &A,
                                const poplar::Tensor &out,
                                const std::vector<std::size_t> &dims,
                                ReduceParams params,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix = "",
                                const poplar::OptionFlags &options = {},
                                ReductionDebug *debug = nullptr);

}

#endif // popops_Reduce_hpp
