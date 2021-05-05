// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

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

/** Structure of  Recurrent Neural Network (RNN) parameters which allows for any
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

  /// The run-time number of RNN time steps of dimension `{batchSize}`
  /// If this tensor is default constructed, the number of time steps
  /// for the sequence corresponding to each batch will be set
  /// according to the `maxTimeSteps` member.
  poplar::Tensor varTimeSteps;

  /// For each RNN layer the layer size parameter need to be specified for the
  /// input and the output. This is done using a 2-element vector of which
  /// the first element is the input size and the second element is the
  /// output size of the RNN layer.
  std::vector<std::size_t> layerSizes;

  RnnParams(poplar::Type dataType, std::size_t batchSize, std::size_t timeSteps,
            std::vector<std::size_t> layerSizes);

  RnnParams(poplar::Type dataType, std::size_t batchSize,
            std::size_t maxTimeSteps, const poplar::Tensor &varTimeSteps,
            std::vector<std::size_t> layerSizes);

  // Return the maximum number of shards
  std::size_t getMaxShards(const poplar::Graph &graph) const;

  // Return the number of bytes of the input per tile
  std::size_t getInputBytesPerTile(const poplar::Graph &graph) const;

  // Return the number of bytes of the output per tile
  std::size_t getOutputBytesPerTile(const poplar::Graph &graph) const;

  // Check if time steps are determined by tensor variable
  bool variableTimeSteps() const;

  // Check if time steps are determined by tensor variable for each batch
  bool batchVariableTimeSteps() const;
};

/** Create state tensor to be used in all recurrences of the RNN. The tensor
 *  shape is {multiple, batchSize, size}. If the RNN happens to be sharded,
 *  a tensor of this shape is created for each shard.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param isOutput        Flag that indicates that the tensor is for output. If
 *                        the flag is false that indicates that this is an input
 * \param multiple        The number of state variables that are
 *                        concatenated into one single state tensor.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor of shape {multiple, batchSize, size}.
 */
poplar::Tensor
createInitialState(poplar::Graph &graph, const RnnParams &params, bool isOutput,
                   unsigned multiple, unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create tensor of shape {timeSteps, batchSize, size} which is suitable
 *  for slicing and/or sharding outermost dimension.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param size            The inner most dimension of the tensor
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor of shape {timeSteps, batchSize, size}.
 */
poplar::Tensor
createRecurrentTensor(poplar::Graph &graph, const RnnParams &params,
                      unsigned size, unsigned numShards,
                      const poplar::DebugContext &debugContext = {});

/**
 *  Create tensor of shape {timeSteps, batchSize, inputSize} which is suitable
 *  for slicing and/or sharding outermost dimension.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor of shape  {timeSteps, batchSize, inputSize}.
 */
poplar::Tensor createInputTensor(poplar::Graph &graph, const RnnParams &params,
                                 unsigned numShards,
                                 const poplar::DebugContext &debugContext = {});

/** Create tensor of shape {timeSteps, batchSize, outputSize} which is
 *  suitable for slicing and/or sharding outermost dimension.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor of shape  {timeSteps, batchSize, outputSize}.
 */
poplar::Tensor
createOutputTensor(poplar::Graph &graph, const RnnParams &params,
                   unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create tensor with size which is a multiple of an output tensor. The
 *  concatenated tensor is of shape {multiple * timeSteps, batchSize,
 *  outputSize} which is suitable for slicing and/or sharding along outermost
 *  dimension.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param multiple        Integer multiple of standard output tensor.
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor of shape  {timeSteps * multiple, batchSize, outputSize}.
 */
poplar::Tensor
createOutputTensor(poplar::Graph &graph, const RnnParams &params,
                   unsigned multiple, unsigned numShards,
                   const poplar::DebugContext &debugContext = {});

/** Create RNN tensor based on a provided 'tBase' tensor such that for the
 *  'n'th iteration the RNN tensor points to the 'n-1'th iteration of the tBase
 *  tensor. For the 0'th iteration of the RNN tensor, a copy is made from
 *  the provided 'tSingle' tensor.
 *
 * \param graph           Graph object.
 * \param params          The RNN parameters.
 * \param tBase           Tensor to shift
 * \param tSingle         Tensor to be copied to 0'th iteration
 * \param prog            The program to add tensor copy
 * \param numShards       The number of shards to be used.
 * \param debugContext    Debug information.
 *
 * \return Tensor which is single step shifted version of tBase.
 */
poplar::Tensor shiftRnnTensor(poplar::Graph &graph, const RnnParams &params,
                              const poplar::Tensor &tBase,
                              const poplar::Tensor &tSingle,
                              poplar::program::Sequence &prog,
                              unsigned numShards,
                              const poplar::DebugContext &debugContext = {});

/** Loop body function wrapper with the following arguments:
 *
 * \param graph              Graph Object
 * \param shardIdx           Tensor that specifies the starting sequence index
 *                           for the current shard.
 * \param seqIdx             Tensor that iterates over the range of input
 *                           sequences that are mapped on the current shard,
 *                           beginning from 0.
 * \param state              state tensors
 * \param inputs             Input tensors
 * \param interimIn          Collated interim input tensors
 * \param interimOut         Collated interim outputs tensors
 * \param output             Pre-defined output tensors
 * \param created            Output tensors which are created by this function.
 * \param prog               Program initialization sequence
 * \param dnai               Debug name and Id
 *
 * \return  Program for the given shard
 *
 */
using LoopBodyType = std::function<poplar::program::Sequence(
    poplar::Graph &graph, const poplar::Tensor &, const poplar::Tensor &,
    const poplar::Tensor &, std::vector<poplar::Tensor> &,
    const std::vector<poplar::Tensor> &, const poplar::Tensor &,
    poplar::Tensor &, std::vector<poplar::Tensor> &,
    std::vector<poplar::Tensor> &, poplar::program::Sequence *,
    const poplar::DebugNameAndId &)>;

/**
 * Structure that associates a particular state tensor with a user defined
 * output tensor. When passed to the Rnn() function the state tensor for each
 * recurrence is stored to the provided tensor.
 *
 * \param output            A tensor to which the state is to be stored
 * \param stateIndex        Index which identifies the state tensor which
 *                          is to form the output tensor.
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
 *      If true, the custom RNN implementation defined by the loopFn parameter
 *      will be reused by every shard. If false the RNN code is duplicated
 *      for every shard.
 *
 * \param graph              Graph to which the RNN cell belongs.
 * \param params             The parameters of the RNN.
 * \param reverse            Process tensors in reverse, i.e., beginning from
 *                           the last element.
 * \param initState          state tensors that specify the initial states.
 * \param stateSequence      Optionally specifies that the recurrent updates of
 *                           a state Tensor need to be stored to a user defined
 *                           output tensor.
 * \param inputs             Input tensors for each recurrence
 * \param *interimIn         Pointer to intermediate inputs to Cell computation.
 * \param *interimOut        Pointer to intermediate outputs from Cell
 *                           computation.
 * \param output             Output tensor for each recurrence. Each tensor
 *                           must be defined prior to calling  the Rnn function.
 * \param created            Output tensor that is allocated by the custom
 *                           implementation defined in the loopFn parameter.
 * \param prog               Program sequence.
 * \param loopFn             Function for RNN cell computation which is
 *                           invoked for every shard.
 * \param numShards          The number of shards to be used.
 * \param options            RNN implementation options. See createInput().
 * \param debugContext       Optional debug information.
 *
 */
std::vector<poplar::Tensor>
Rnn(poplar::Graph &graph, const RnnParams &params, bool reverse,
    const std::vector<poplar::Tensor> &initState,
    const StateSequence &stateSequence,
    const std::vector<poplar::Tensor> &inputs, const poplar::Tensor *interimIn,
    poplar::Tensor *interimOut, std::vector<poplar::Tensor> &outputs,
    std::vector<poplar::Tensor> &created, poplar::program::Sequence &prog,
    const LoopBodyType &loopFn, unsigned numShards,
    poplar::OptionFlags &options,
    const poplar::DebugContext &debugContext = {});

} // namespace rnn
} // namespace popnn

#endif // #ifndef popnn_Rnn_hpp
