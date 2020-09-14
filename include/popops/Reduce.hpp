// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Define types of operations used in a reduce.
 *
 */

#ifndef popops_Reduce_hpp
#define popops_Reduce_hpp

#include "popops/Operation.hpp"
#include <poputil/exceptions.hpp>

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <poplar/OptionFlags.hpp>
#include <vector>

namespace popops {

/// Stores parameters for the reduce operation, as well as the basic
/// operation being performed (for example, \c add or \c mul).
struct ReduceParams {
  ReduceParams() = default;
  // Allow implicit conversion from a popops::Operation.
  ReduceParams(popops::Operation op, bool update = false)
      : op(op), update(update) {
    useScale = false;
  }

  /** Define the details of the reduce operation that will
   *  be performed by the reduce() and reduceWithOutput()
   *  functions.
   *
   * \param op    The reduce operation to use.
   * \param scale Can (optionally) scale the output.
   * \param update Specify that the output should be updated,
   *               where `out += reduce(in)` rather than
   *               `out = reduce(in)`.
   */
  ReduceParams(popops::Operation op, bool update, poplar::Tensor scale)
      : op(op), update(update), scale(scale) {
    useScale = true;
  }

  // Explicitly disable the old API to avoid accidental type conversion
  ReduceParams(popops::Operation op, float constantScale,
               bool update = false) = delete;

  popops::Operation op;
  bool update;
  bool useScale;
  poplar::Tensor scale;
};

/// Apply a reduction operation to a tensor.
/// \p scale and \p update are only valid with the \c ADD or \c SQUARE_ADD
/// operations.
///
/// Internally, this creates a new variable for the output then calls
/// reduceWithOutput(). The type of the output will be \p outType.
///
/// The options parameter accepts the following:
///
///  * **accumType.interTile** (float, half)
///
///    The type to use for intermediate values between tiles.
///
///  * **accumType.inVertex** (float, half)
///
///    The type to use for intermediate values within a vertex.
///
/// If either of the above options are not set then the intermediate type will
/// default to either the input tensor element type or float if the input
/// is of type half and the reduction operation benefits from
/// higher precision (for example, add).
///
/// The input and output types that are supported depend on the operation:
///
///   - `ADD`, `SQUARE_ADD`, `MUL`: float->float, half->half,
///     int->int, float->half, half->float
///   - `MAX`, `MIN`: float->float, half->half, int->int
///   - `LOGICAL_AND`, `LOGICAL_OR`: bool->bool
///
/// \param graph The graph to add the operation to.
/// \param in The tensor to be reduced.
/// \param outType The output type of the reduce operation.
/// \param dims The dimensions to reduce in.
/// \param prog The program sequence to add the operation to.
/// \param debugPrefix Identifying prefix for debugging information.
///
poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Type &outType,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});

/// \copybrief reduce
///
/// An alias for reduce(graph, in, in.elementType(), ...)
///
/// \copydetails reduce
poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});

/// \copybrief reduce
///
/// This is similar to reduce() but allows you to specify the output.
/// If the tile mapping of \p out is not complete it will be set. Otherwise it
/// won't be changed.
///
/// \copydetails reduce
void reduceWithOutput(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Tensor &out,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});

/// \copybrief reduce
///
/// These are alternate forms that add their vertices to a vector of compute
/// sets instead of a poplar::program::Sequence. The caller is expected to add
/// each compute set to a poplar::program::Sequence (in a
/// poplar::program::Execute) themselves, like this:
///
///     Sequence seq;
///     std::vector<ComputeSet> css;
///     auto A = reduce(..., css);
///     auto B = reduce(..., css);
///     for (const auto &cs : css) {
///       seq.add(Execute(cs));
///
/// This allows you to do multiple reductions in parallel. Note that the
/// reductions are not aware of each other, so it may be more efficient
/// to concatenate tensors and do a single reduction instead if they have the
/// same shape, operation, and input and output types.
///
/// \copydetails reduce
/// @{
poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Type &outType,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<poplar::ComputeSet> &css,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});

poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<poplar::ComputeSet> &css,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});

void reduceWithOutput(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Tensor &out,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<poplar::ComputeSet> &css,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {});
/// @}

} // namespace popops

#endif // popops_Reduce_hpp
