// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *  Functions for recurrent neural networks (RNN).
 */

#ifndef popnn_Rnn_hpp
#define popnn_Rnn_hpp

#include <cassert>
#include <cstdint>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/DebugInfo.hpp>

namespace popnn {
namespace rnn {

/** Structure of Recurrent Neural Network (RNN) parameters which allows for any
 *  customized implementation of the cellular part of the RNN.
 */
struct RnnParams {
  /// The datatype used for the RNN.
  poplar::Type dataType;

  /// The batch size.
  std::size_t batchSize;

  /// The maximum number of RNN time steps.
  std::size_t maxTimeSteps;

  /// \deprecated Use RnnParams.maxTimeSteps instead.
  std::size_t timeSteps;

  /// The run-time number of RNN time steps of dimension [`batchSize`]
  /// If this tensor is default constructed, the number of time steps
  /// for the sequence corresponding to each batch will be set
  /// according to the `maxTimeSteps` member.
  poplar::Tensor varTimeSteps;

  /// For each RNN layer, the layer size parameter needs to be specified for the
  /// input and the output. This is done using a 2-element vector where
  /// the first element is the input size and the second element is the
  /// output size of the RNN layer.
  std::vector<std::size_t> layerSizes;

  RnnParams(poplar::Type dataType, std::size_t batchSize, std::size_t timeSteps,
            std::vector<std::size_t> layerSizes);

  RnnParams(poplar::Type dataType, std::size_t batchSize,
            std::size_t maxTimeSteps, const poplar::Tensor &varTimeSteps,
            std::vector<std::size_t> layerSizes);

  // Return the maximum number of shards.
  std::size_t getMaxShards(const poplar::Graph &graph) const;

  // Return the number of bytes of the input per tile.
  std::size_t getInputBytesPerTile(const poplar::Graph &graph) const;

  // Return the number of bytes of the output per tile.
  std::size_t getOutputBytesPerTile(const poplar::Graph &graph) const;

  // Indicate that time steps are determined by tensor variable.
  bool variableTimeSteps() const;

  // Indicate that time steps are determined by tensor variable for each batch.
  bool batchVariableTimeSteps() const;
};

/** Create state tensor to be used in all recurrences of the RNN. The tensor
 *  shape is [`multiple`, `batchSize`, size] where size is determined by
 *  whether the state tensor is an input or output tensor, depending on \p
 *  isOutput. If the RNN is sharded, a tensor of this shape is created for each
 *  shard.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param isOutput        Flag that indicates that the state tensor will be
 *                        an output tensor. If false, indicates that this is an
 *                        input tensor.
 * \param multiple        The number of state variables that are
 *                        concatenated into a single state tensor.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return State tensor of shape [`multiple`, `batchSize`, size].
 */
poplar::Tensor
createInitialState(poplar::Graph &graph, const RnnParams &params, bool isOutput,
                   unsigned multiple, unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create recurrent tensor of shape [`timeSteps`, `batchSize`, `size`] suitable
 *  for slicing and/or sharding of the outermost dimension.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param size            The innermost dimension of the tensor.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Recurrent tensor of shape [`timeSteps`, `batchSize`, `size`].
 */
poplar::Tensor
createRecurrentTensor(poplar::Graph &graph, const RnnParams &params,
                      unsigned size, unsigned numShards,
                      const poplar::DebugContext &debugContext = {});

/**
 *  Create input tensor of shape [`timeSteps`, `batchSize`, `inputSize`]
 *  suitable for slicing and/or sharding of the outermost dimension.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Input tensor of shape [`timeSteps`, `batchSize`, `inputSize`].
 */
poplar::Tensor createInputTensor(poplar::Graph &graph, const RnnParams &params,
                                 unsigned numShards,
                                 const poplar::DebugContext &debugContext = {});

/** Create a standard output tensor of shape [`timeSteps`, `batchSize`,
 *  `outputSize`] suitable for slicing and/or sharding of the outermost
 *  dimension.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Output tensor of shape [`timeSteps`, `batchSize`, `outputSize`].
 */
poplar::Tensor
createOutputTensor(poplar::Graph &graph, const RnnParams &params,
                   unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create a single output tensor with \p multiple (standard) output tensors
 *  concatenated along the outermost (`timeSteps`) dimension.
 *
 *  The concatenated tensor is of shape [`multiple * timeSteps`, `batchSize`,
 *  `outputSize`] which is suitable for slicing and/or sharding along the
 *  outermost dimension.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param multiple        The number of standard output tensors to be
 *                        concatenated.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Output tensor of shape [`multiple * timeSteps`, `batchSize`,
 * `outputSize`].
 */
poplar::Tensor
createOutputTensor(poplar::Graph &graph, const RnnParams &params,
                   unsigned multiple, unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create a single-step shifted RNN tensor from an input tensor.
 *
 *  For an input tensor \p tBase, `n`-th iteration of the RNN tensor points to
 *  the `n-1`-th iteration of `tBase`. For the 0-th iteration of the RNN tensor,
 *  a copy is made from the provided tensor `tSingle` tensor.
 *
 * \param graph           The graph object.
 * \param params          The RNN parameters.
 * \param tBase           The tensor to shift.
 * \param tSingle         The tensor to be copied in the 0-th iteration.
 * \param prog            The program to add a tensor copy.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return RNN tensor which is a single-step shifted version of \p tBase.
 */
poplar::Tensor shiftRnnTensor(poplar::Graph &graph, const RnnParams &params,
                              const poplar::Tensor &tBase,
                              const poplar::Tensor &tSingle,
                              poplar::program::Sequence &prog,
                              unsigned numShards,
                              const poplar::DebugContext &debugContext = {});

/** Tensors required for processing a single time step.
 *
 * \param inputs             The input tensor sequences.
 * \param interimIn          The intermediate input sequence.
 * \param interimOut         The intermediate output sequence.
 * \param outputs            The output tensor sequences.
 */
struct RnnSlice {
  std::vector<poplar::Tensor> inputs;
  poplar::Tensor interimIn;
  poplar::Tensor interimOut;
  std::vector<poplar::Tensor> outputs;
};

/* Flags set per batch if the current step is within the batchwise step limit.
 * The component tensor(s) are of type `dataType` and shape [`batchSize`].
 */
struct RnnBatchwiseFlags {
  poplar::Tensor mask;
  poplar::Tensor inverse;

  bool valid() const { return mask.valid(); };
};

struct TimeStepState {
  poplar::Tensor begin;
  poplar::Tensor counter;
  poplar::Tensor variableSeqFlag;
};

/** Create loop body function for the given shard.
 *
 * \param graph              The graph object.
 * \param shardIdx           The tensor that specifies the starting sequence
 *                           index for the current shard.
 * \param seqIdx             The tensor that iterates over the range of input
 *                           sequences that are mapped on the current shard,
 *                           beginning from 0.
 * \param batchwiseFlags     Flags that indicate batches for which the current
 *                           step is within the batchwise step limit.
 * \param state              State tensors.
 * \param slice              The input/output tensors for a specific shard.
 * \param created            The output tensors which are created by this
 *                           function.
 * \param prog               The program initialization sequence.
 * \param dnai               Debug name and Id.
 *
 * \return  Loop body function for the given shard.
 */
using LoopBodyType = std::function<poplar::program::Sequence(
    poplar::Graph &graph, const TimeStepState &time, const RnnBatchwiseFlags &,
    std::vector<poplar::Tensor> &, const RnnSlice &slice,
    std::vector<poplar::Tensor> &, poplar::program::Sequence *,
    const poplar::DebugNameAndId &)>;

/** Create gather body function for the given shard.
 *
 * \param graph              The graph object
 * \param slice              The input/output tensors for a specific shard
 * \param stepsPerGather     The time step interval between calls to this
 *                           function.
 * \param prog               The program initialization sequence
 * \param dnai               Debug name and Id
 *
 * \return  Gather body function for the given shard.
 */
using GatherBodyType = std::function<poplar::program::Sequence(
    poplar::Graph &graph, const RnnSlice &slice, unsigned stepsPerGather,
    poplar::program::Sequence *, const poplar::DebugNameAndId &)>;

/**
 * Structure that associates a particular state tensor with a user-defined
 * output tensor. When passed to the Rnn() function, the state tensor for each
 * recurrence is stored in the tensor \p output.
 *
 * \param output            The output tensor which stores the state.
 * \param stateIndex        The index which identifies the state tensor which
 *                          will be stored in the output tensor.
 */
struct StateSequence {
  poplar::Tensor output;
  std::size_t stateIndex;
};

/** Run custom Recurrent Neural Net cell implementation recurrently.
 *
 * **RNN options**
 *
 *    * `codeReuse` (true, false) [=false]
 *
 *      If true, the custom RNN implementation defined by the \p loopFn
 *      parameter will be reused by every shard. If false, the RNN code is
 *      duplicated for every shard.
 *
 * \param graph              The graph to which the RNN cell belongs.
 * \param params             The RNN parameters.
 * \param reverse            Process tensors in reverse order, so beginning
 *                           with the last element.
 * \param initState          The state tensors that specify the initial states.
 * \param stateSequence      Optionally, specify that the recurrent updates of
 *                           a state tensor need to be stored in a user-defined
 *                           output tensor.
 * \param inputs             The input tensors for each recurrence.
 * \param *interimIn         Pointer to the intermediate inputs to cell
 *                           computation.
 * \param *interimOut        Pointer to the intermediate outputs from cell
 *                           computation.
 * \param output             The output tensor for each recurrence. Each tensor
 *                           must be defined prior to calling Rnn().
 * \param created            The output tensor that is allocated by the custom
 *                           implementation defined in \p loopFn.
 * \param prog               Program sequence.
 * \param loopFn             The loop body function for RNN cell computation
 *                           which is invoked for every shard.
 * \param numShards          The number of shards to be used.
 * \param options            The RNN implementation options. See createInput().
 * \param debugContext       Optional debug information.
 *
 */
std::vector<poplar::Tensor>
Rnn(poplar::Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<poplar::Tensor> &initState,
    const StateSequence &stateSequence,
    const std::vector<poplar::Tensor> &inputs, const poplar::Tensor *interimIn,
    poplar::Tensor *interimOut, const std::vector<poplar::Tensor> &outputs,
    const std::vector<poplar::Tensor> &created, poplar::program::Sequence &prog,
    const LoopBodyType &loopFn, unsigned numShards,
    poplar::OptionFlags &options,
    const poplar::DebugContext &debugContext = {});

/** Run custom Recurrent Neural Net cell callback at every time step in
 *  decrementing order. At each time step, create a temporary variable and pass
 *  it to the `gatherFn` callback function which gets called at a time step
 *  interval determined by the `stepsPerGather` parameter.
 *
 * **RNN options**
 *
 *    * `codeReuse` (true, false) [=false]
 *
 *      If true, the custom RNN implementation defined by the \p loopFn
 *      parameter will be reused by every shard. If false, the RNN code is
 *      duplicated for every shard.
 *
 * \param graph              The graph to which the RNN cell belongs.
 * \param params             The RNN parameters.
 * \param initState          The state tensors that specify the initial states.
 * \param stateSequence      Optionally, specify that the recurrent updates of
 *                           a state tensor need to be stored in a user-defined
 *                           output tensor.
 * \param inputs             The input tensors to the loop body function
 *                           `loopFn`
 * \param interimIn          The intermediate inputs to cell computation.
 * \param numTemps           The number of temporary variables of shape
 *                           [`batchSize`, size] per time step which are to be
 *                           passed to the `gatherFn` callback function.
 * \param prog               Program sequence.
 * \param loopFn             Function for RNN cell computation which is
 *                           invoked for every time step.
 * \param gatherInputs       The input tensors to the gather body function
 *                           `gatherFn`.
 * \param gatherFn           The gather body function which processes the
 *                           temporary buffer generated by the loop body
 *                           function `loopFn` with the time step interval
 *                           between calls determined by the \p stepsPerGather
 *                           parameter.
 * \param numShards          The number of shards to be used.
 * \param stepsPerGather     The time step interval used by the `gatherFn`
 *                           callback.
 * \param options            The RNN implementation options. See createInput().
 * \param debugContext       Optional debug information.
 *
 */
std::vector<poplar::Tensor>
Rnn(poplar::Graph &graph, const RnnParams &params,
    const std::vector<poplar::Tensor> &initState,
    const StateSequence &stateSequence,
    const std::vector<poplar::Tensor> &inputs, const poplar::Tensor &interimIn,
    const unsigned numTemps, poplar::program::Sequence &prog,
    const LoopBodyType &loopFn, const std::vector<poplar::Tensor> &gatherInputs,
    const GatherBodyType &gatherFn, unsigned numShards, unsigned stepsPerGather,
    poplar::OptionFlags &options,
    const poplar::DebugContext &debugContext = {});

} // namespace rnn
} // namespace popnn

#endif // #ifndef popnn_Rnn_hpp
