// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_Lstm_hpp
#define popnn_Lstm_hpp

#include <poplin/MatMul.hpp>
#include <popnn/LstmDef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace lstm {

/** Structure representing the parameters of the LSTM.
 */
struct LstmParams {
  // The datatype of the LSTM
  poplar::Type dataType;
  // The batch size
  std::size_t batchSize;
  // The number of time steps in the sequence of the LSTM
  std::size_t timeSteps;
  // The number of neurons before and after each layer of the LSTM.
  // If the LSTM consists of N layers, then this should be a vector
  // of size N+1. The first element is the input size and each subsequent
  // element is the output size of the LSTM layer.
  std::vector<std::size_t> layerSizes;
  // If this parameter is set to false then the LSTM will skip the
  // calculation of weighted inputs (only useful for benchmarking).
  bool doInputWeightCalc = true;
  // If this parameter is set to false then the LSTM will skip the
  // calculation of the gradients of the inputs.
  bool calcInputGradients = true;
  LstmParams() = default;
  LstmParams(poplar::Type dataType,
             std::size_t batchSize,
             std::size_t timeSteps,
             std::vector<std::size_t> layerSizes);
};

/** Structure holding the initial state of a LSTM cell */
struct LstmInitialState {
  poplar::Tensor output;
  poplar::Tensor cellState;
};

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params);

uint64_t getBasicLstmCellBwdFlops(const LstmParams &params);

uint64_t getBasicLstmCellWuFlops(const LstmParams &params);

/** Create an input tensor of shape {numSteps, batchSize, inputSize} which is
 *  optimally mapped to multiply the whole input sequence in a single matrix
 *  multiply operation
 *
 * \param graph           Graph object
 * \param params          The LSTM parameters
 * \param name            String annotation
 * \param options         Any implementation/debug options for the LSTM
 * \param planningCache   A poplin matrix multiply planning cache
 *
 * \return A tensor created in the graph of shape  {timeSteps, batchSize,
 *         inputSize}
 */
poplar::Tensor
createInput(poplar::Graph &graph, const LstmParams &params,
            const std::string &name,
            const poplar::OptionFlags &options = {},
            poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create initial state that is fed into the LSTM call at the first timestep.
 *  It can be initialised by writing the the appropriate member or using
 *  zeroInitialState()
 *
 * \param graph           Graph object
 * \param params          The LSTM parameters
 * \param name            String annotation
 * \param options         Any implementation/debug options for the LSTM
 * \param planningCache   A poplin matrix multiply planning cache
 *
 * \return A tensor which is the state for the forward operation of the LSTM
 *         cell
 */
LstmInitialState
createInitialState(poplar::Graph &graph, const LstmParams &params,
                   const std::string &name,
                   const poplar::OptionFlags &options = {},
                   poplin::matmul::PlanningCache *planningCache = nullptr);

/** Initialize the forward state of an LSTM with zeros.
 *
 *  \param graph             Graph object
 *  \param initialState      The initial state to zero
 *  \param prog              The program to extend with the initialization
 *                           code
 *  \param debugPrefix       A debug string to prepend to debug indentifiers
 *                           in the added code.
 */
void zeroInitialState(poplar::Graph &graph,
                      const LstmInitialState &initialState,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

/** Returns the output tensor view from the forward state tensor
 *
 * \param fwdState        Forward state of the LSTM cell for a step
 *
 * \return Forward output activations of the LSTM cell given the state tensor
 *         for any step
 */
poplar::Tensor
getOutputFromFwdState(const poplar::Tensor &fwdState);

inline poplar::Tensor
getOutputFromFwdState(const LstmInitialState &fwdState) {
  return fwdState.output;
}

/** Returns the cell state tensor view from the forward state tensor
 *
 * \param fwdState        Forward state of the LSTM cell for a step
 *
 * \return Cell state of the LSTM cell given the state tensor for any step
 */
poplar::Tensor
getCellFromFwdState(const poplar::Tensor &fwdState);

inline poplar::Tensor
getCellFromFwdState(const LstmInitialState &fwdState) {
  return fwdState.cellState;
}

/** Structure holding all the weights (parameters) of an LSTM cell
 */
struct LstmWeights {
  poplar::Tensor inputWeights;
  poplar::Tensor outputWeights;
  poplar::Tensor biases;
};

/** Create the weights used to weight the input of an lstm.
 */
LstmWeights
createWeights(poplar::Graph &graph, const LstmParams &params,
              const std::string &name,
              const poplar::OptionFlags &options = {},
              poplin::matmul::PlanningCache *planningCache = nullptr);


/** Calculate the result of applying an LSTM across a sequence
 *
 * The LSTM is run for seqSize steps each with a batch of size batchSize and
 * input size inputSize and output size outputSize. The total number of units
 * within each LSTM cell is lstmUnits = BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param params        The parameters of the LSTM
 * \param stateInit     Initial state for the LSTM
 * \param in            The input tensor to the LSTM of dimension
 *                      [timesteps, batch, inputSize]
 * \param weights       The LSTM weights structure
 * \param fwdProg       Program sequence
 * \param debugPrefix   String used as prefix for compute sets
 * \param options       LSTM implementation options
 * \param planningCache The matmul planning cache
 *
 * \return sequence of lstm states where the outer dimension is the number of
 *         timesteps
 */
poplar::Tensor lstmFwd(poplar::Graph &graph,
                       const LstmParams &params,
                       const LstmInitialState &stateInit,
                       const poplar::Tensor &in,
                       const LstmWeights &weights,
                       poplar::program::Sequence &fwdProg,
                       const std::string &debugPrefix = "",
                       const poplar::OptionFlags &options = {},
                       poplin::matmul::PlanningCache *planningCache = nullptr);


/** Create backward gradient pass state. The state is typically created as the
 *  initialisation state for the backward and weight update passes.
 *
 *  The state itself is initialsed appropriately for operation as the first step
 *  in the backward pass
 *
 * \param graph           Graph object
 * \param batchSize       Number of batch elements
 * \param outputSize      Number of output activations
 * \param prog            Control program wherein the initialisation of state is
 *                        done
 * \param dType           Data type of the state
 * \param debugPrefix     String annotation
 *
 * \return created backward state tensor
 */
poplar::Tensor
createBwdState(poplar::Graph &graph, const LstmParams &params,
               const std::string &name,
               const poplar::OptionFlags &options = {},
               poplin::matmul::PlanningCache *planningCache = nullptr);

/** Initialize the backward state of an LSTM with zeros.
 *
 *  \param graph             Graph object
 *  \param bwdState          The back state tensor
 *  \param prog              The program to extend with the initialization
 *                           code
 *  \param debugPrefix       A debug string to prepend to debug indentifiers
 *                           in the added code.
 */
void initBwdState(poplar::Graph &graph, const poplar::Tensor &bwdState,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

/**
 *  Run LSTM backward pass. The backward pass executes in reverse order as
 *  compared to the forward pass. If the forward steps for a LSTM layer are sf =
 *  {0, 1, 2, ..., S - 1} then the backward steps run for sb = {S - 1, S - 2,
 *  .... , 1, 0}.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param params        The parameters of the LSTM
 * \param doWU          When true weight and bias delta updates are calculated
 * \param prog       Program sequence
 * \param fwdStateInit  Forward state tensor for initial step
 * \param fwdState      Forward state tensor for all steps [0:timeSteps)
 * \param weights       The LSTM weights structure
 * \param input         The input tensor to the LSTM of shape:
 *                         [timesteps, batch, inputSize]
 * \param outputGrad    The gradients of the output of shape:
 *                         [timesteps, batch, outputSize]
 * \param debugPrefix   String used as prefix for compute sets
 * \param options       LSTM implementation options
 * \param planningCache The matmul planning cache
 *
 * \return Returns four tensors:
 *         - gradients for previous layer
 *         - input weight deltas
 *         - output weight deltas
 *         - bias deltas
 * When doWU is false the weight and bias deltas are not calculated
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
  lstmBwd(
    poplar::Graph &graph, const LstmParams &params,
    bool doWU,
    poplar::program::Sequence &prog,
    const LstmInitialState &fwdStateInit,
    const poplar::Tensor &fwdState,
    const LstmWeights &weights,
    const poplar::Tensor &input,
    const poplar::Tensor &outputGrad,
    const poplar::Tensor &bwdState,
    const std::string &debugPrefix = "",
    const poplar::OptionFlags &options = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

} // namespace lstm
} // namespave popnn
#endif // popnn_Lstm_hpp
