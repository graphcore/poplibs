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
/// \p scale and \p update are only valid with the \c ADD , \c SQUARE_ADD
/// or \c LOG_ADD operations.
/// \c LOG_ADD performs all arithmetic consistent with the input and output
/// being log probabilities.  In other words, the `update` is another log add
/// operation and the `scale` is a log multiply operation.
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
///   - `LOG_ADD` : float->float, half->half, float->half, half->float
///   - `MAX`, `MIN`: float->float, half->half, int->int
///   - `LOGICAL_AND`, `LOGICAL_OR`: bool->bool
///
/// \param graph The graph to add the operation to.
/// \param in The tensor to be reduced.
/// \param outType The output type of the reduce operation.
/// \param dims The dimensions to reduce in.
/// \param prog The program sequence to add the operation to.
/// \param debugContext Optional debug information.
///
poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Type &outType,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {});

/// \copybrief reduce
///
/// An alias for reduce(graph, in, in.elementType(), ...)
///
/// \copydetails reduce
poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
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
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {});

/// \copybrief reduce
///
/// \deprecated The reduce overloads that expect a vector of compute sets
///             are deprecated. Please use the reduceMany() function instead.
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
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {});

poplar::Tensor reduce(poplar::Graph &graph, const poplar::Tensor &in,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<poplar::ComputeSet> &css,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {});

void reduceWithOutput(poplar::Graph &graph, const poplar::Tensor &in,
                      const poplar::Tensor &out,
                      const std::vector<std::size_t> &dims, ReduceParams params,
                      std::vector<poplar::ComputeSet> &css,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {});
/// @}

/// The parameterisation of the inputs to a single reduction for the
/// reduceMany() function.
///
/// Please see the documentation for reduce() for a description of the struct
/// members.
struct SingleReduceOp {
  poplar::Tensor in;
  std::vector<std::size_t> dims;
  ReduceParams params;
  /// Note that if `useOutType` is `false` then the element type of `in` is
  /// used. Also note that `OutType` is ignored if the `outputs` vector is not
  /// empty when calling reduceMany().
  bool useOutType;
  poplar::Type outType;
  std::string debugName;

  SingleReduceOp(poplar::Tensor in, std::vector<std::size_t> dims,
                 ReduceParams params, poplar::Type outType,
                 std::string debugName = "")
      : in(std::move(in)), dims(std::move(dims)), params(std::move(params)),
        useOutType(true), outType(outType), debugName(std::move(debugName)) {}

  SingleReduceOp(poplar::Tensor in, std::vector<std::size_t> dims,
                 ReduceParams params, std::string debugName = "")
      : in(std::move(in)), dims(std::move(dims)), params(std::move(params)),
        useOutType(false), outType(poplar::BOOL),
        debugName(std::move(debugName)) {}
};

/// Perform many reductions (in parallel if possible).
///
/// Please see the documentation for reduce() for details of the common inputs.
///
/// \param reductions The inputs to each reduction to perform. The `outType`
///        attribute controls the element type of the output tensor if
///        \p outputs is empty, otherwise it is ignored. If \p outputs is empty
///        and `useOutType` is `false` then the output element type will be
///        set to the same element type as the corresponding `in` tensor.
/// \param outputs The tensors to store the output of the reductions. This may
///        be empty in which case `reduceMany` will create the tensors. If the
///        tile mapping is not set or not complete it will be set completely by
///        this function.
/// \exception poputils::poplibs_error If \p outputs is not empty then its size
///            must exactly match the size of reductions else an exception will
///            be thrown.
/// \exception poputils::poplibs_error If \p outputs is empty and any reduction
///            has `params.update` set to true then an exception will be thrown.
///            \p outputs is required to perform an update reduction.
void reduceMany(poplar::Graph &graph,
                const std::vector<SingleReduceOp> &reductions,
                std::vector<poplar::Tensor> &outputs,
                poplar::program::Sequence &prog,
                const poplar::DebugContext &debugContext = {},
                const poplar::OptionFlags &options = {});

} // namespace popops

#endif // popops_Reduce_hpp
