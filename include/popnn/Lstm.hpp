// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popnn_Lstm_hpp
#define popnn_Lstm_hpp

#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/LstmDef.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/Rnn.hpp>

namespace popnn {
namespace lstm {

/**
 * Get the default order of the gates in a basic LSTM cell.
 * The default order is:
 * [Forget gate, Input gate, Candidate, Output Gate].
 */
const std::vector<BasicLstmCellUnit> getDefaultBasicLstmCellOrder();

/** Structure representing the parameters of the LSTM.
 */
struct LstmParams {
  rnn::RnnParams rnn;

  /// The datatype of the LSTM.
  /// \deprecated Use rnn.dataType instead.
  poplar::Type dataType;
  /// The batch size.
  /// \deprecated Use rnn.batchSize instead.
  std::size_t batchSize;
  /// The number of time steps in the sequence of the LSTM.
  /// \deprecated Use rnn.timeSteps instead.
  std::size_t timeSteps;
  /// The number of neurons before and after each layer of the LSTM.
  /// If the LSTM consists of N layers, then this should be a vector
  /// of size N+1. The first element is the input size and each subsequent
  /// element is the output size of the LSTM layer.
  /// \deprecated Use rnn.layerSizes instead.
  std::vector<std::size_t> layerSizes;
  /// If true the Lstm function returns the entire sequence of outputs,
  /// otherwise it returns just the final output.
  bool outputFullSequence = true;
  /// If this parameter is set to false then the LSTM will skip the
  /// calculation of weighted inputs (only useful for benchmarking).
  bool doInputWeightCalc = true;
  /// If this parameter is set to false then the LSTM will skip the
  /// calculation of the gradients of the inputs.
  bool calcInputGradients = true;
  /// The weight and bias tensors are concatenated tensors in terms of which
  /// gates they service. This option allows the user to specify the order of
  /// the gates in that outermost dimension.
  /// The default order is:
  /// [Forget gate, Input gate, Candidate, Output Gate].
  std::vector<BasicLstmCellUnit> cellOrder = getDefaultBasicLstmCellOrder();
  /// Activation function.
  NonLinearityType activation = NonLinearityType::TANH;
  /// Recurrent activation function.
  NonLinearityType recurrentActivation = NonLinearityType::SIGMOID;

  LstmParams(poplar::Type dataType, std::size_t batchSize,
             std::size_t timeSteps, std::vector<std::size_t> layerSizes,
             NonLinearityType activation = NonLinearityType::TANH,
             NonLinearityType recurrentActivation = NonLinearityType::SIGMOID);

  LstmParams(poplar::Type dataType, std::size_t batchSize,
             std::size_t maxTimeSteps, const poplar::Tensor &timeSteps,
             std::vector<std::size_t> layerSizes,
             NonLinearityType activation = NonLinearityType::TANH,
             NonLinearityType recurrentActivation = NonLinearityType::SIGMOID);
};

/**
 * Structure holding the state of a LSTM cell, or the gradients for the state
 * (depending on the context).
 */
struct LstmState {
  poplar::Tensor output;
  poplar::Tensor cellState;

  poplar::Tensor getAsTensor() const;
};

/**
 * Predict what matrix multiplications will be needed for the given parameters
 * and return list of corresponding matmul parameters and options.
 */
std::vector<std::pair<poplin::MatMulParams, poplar::OptionFlags>>
getMatMulPrePlanParameters(LstmParams params, poplar::OptionFlags opts);

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params);

uint64_t getBasicLstmCellBwdFlops(const LstmParams &params);

uint64_t getBasicLstmCellWuFlops(const LstmParams &params);

/** Create an input tensor of shape {numSteps, batchSize, inputSize} which is
 *  optimally mapped to multiply the whole input sequence in a single matrix
 *  multiply operation.
 *
 * **LSTM options**
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive)
 *
 *      See poplin::createWeights().
 *
 *    * `inferenceOnly` (true, false) [=false]
 *
 *      Sets convolution pass to INFERENCE_FWD if true; TRAINING_FWD otherwise.
 *      See poplin::createWeights().
 *
 *    * `partialsType` (half, float) [=float]
 *
 *      See poplin::createWeights().
 *
 *    * `weightAccumulatorsType` (half, float) [=data type of lstm]
 *
 *      Data type of the weight acccumulators for the lstms weight matrices
 *      and biases
 *
 *    * `preCalcWeights` (true, false) [=false]
 *
 *      If true, use one big matrix multiply before the recurrent calculation to
 *      perform the part of the calculation that only depends on the input
 *      sequence.
 *
 *    * `recomputationMode` (none, cellAndTanh, full) [=none]
 *
 *      * none: No recomputation in the backwards pass.
 *
 *      * cellAndTanh: Small amount of recomputation in the backwards pass,
 *        yielding some reduction in memory footprint for the layer.
 *
 *      * full: Recompute everything from the forward pass. Saves the most
 *        memory at the cost of an extra forward pass of cycles.
 *
 * \param graph           Graph object.
 * \param params          The LSTM parameters.
 * \param debugContext    Debug information.
 * \param options         Any implementation/debug options for the LSTM.
 * \param planningCache   A poplin matrix multiply planning cache.
 *
 * \return A tensor created in the graph of shape  {timeSteps, batchSize,
 *         inputSize}.
 */
poplar::Tensor
createInput(poplar::Graph &graph, const LstmParams &params,
            const poplar::DebugContext &debugContext,
            const poplar::OptionFlags &options = {},
            poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the initial output that can be combined with the initial cell state
 *  using a LstmState. This then can be fed into the LSTM call at the first
 *  timestep.
 *
 * \param graph           Graph object.
 * \param params          The LSTM parameters.
 * \param debugContext    Debug information.
 * \param options         Any implementation/debug options for the LSTM. See
 *                        createInput().
 * \param planningCache   A poplin matrix multiply planning cache.
 *
 * \return A tensor which is the cell state for the forward operation of the
 *         LSTM cell.
 */
poplar::Tensor
createInitialOutput(poplar::Graph &graph, const LstmParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options = {},
                    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the initial cell state that can be combined with the initial output
 *  using a LstmState. This then can be fed into the LSTM call at the first
 *  timestep.
 *
 * \param graph           Graph object.
 * \param params          The LSTM parameters.
 * \param debugContext    Debug information.
 * \param options         Any implementation/debug options for the LSTM. See
 *                        createInput().
 * \param planningCache   A poplin matrix multiply planning cache.
 *
 * \return A tensor which is the cell state for the forward operation of the
 *         LSTM cell.
 */
poplar::Tensor
createInitialCellState(poplar::Graph &graph, const LstmParams &params,
                       const poplar::DebugContext &debugContext,
                       const poplar::OptionFlags &options = {},
                       poplin::matmul::PlanningCache *planningCache = nullptr);

/** Creates the initial state (both output and cellState) that is fed into the
 *  LSTM call at the first timestep. It can be initialised by writing the
 *  appropriate member or using zeroInitialState().
 *
 * \param graph           Graph object.
 * \param params          The LSTM parameters.
 * \param debugContext    Debug information.
 * \param options         Any implementation/debug options for the LSTM. See
 *                        createInput().
 * \param planningCache   A poplin matrix multiply planning cache.
 *
 * \return A tensor which is the state for the forward operation of the LSTM
 *         cell.
 */
LstmState
createInitialState(poplar::Graph &graph, const LstmParams &params,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options = {},
                   poplin::matmul::PlanningCache *planningCache = nullptr);

/** Initialize the forward state of an LSTM with zeros.
 *
 *  \param graph             Graph object.
 *  \param initialState      The initial state to zero.
 *  \param prog              The program to extend with the initialization
 *                           code
 *  \param debugContext      Optional debug information.
 */
void zeroInitialState(poplar::Graph &graph, const LstmState &initialState,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {});

/**
 * Structure holding all the parameters of an LSTM cell, or the
 * deltas for those parameters (depending on the context).
 */
struct LstmWeights {
  poplar::Tensor inputWeights;
  poplar::Tensor outputWeights;
  poplar::Tensor biases;
};

/** Create the weights kernel used to weight the input of an LSTM.
 *  Returns the inputWeights and outputWeights.
 */
std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const LstmParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options = {},
                    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the weights biases.
 */
poplar::Tensor
createWeightsBiases(poplar::Graph &graph, const LstmParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options = {},
                    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the weights (both kernel and biases) used to weight the input of an
 *  LSTM.
 */
LstmWeights
createWeights(poplar::Graph &graph, const LstmParams &params,
              const poplar::DebugContext &debugContext,
              const poplar::OptionFlags &options = {},
              poplin::matmul::PlanningCache *planningCache = nullptr);

/** Calculate the result of applying an LSTM across a sequence.
 *
 * The LSTM is run for seqSize steps each with a batch of size batchSize and
 * input size inputSize and output size outputSize. The total number of units
 * within each LSTM cell is lstmUnits = BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph              Graph to which the LSTM cell belongs.
 * \param params             The parameters of the LSTM.
 * \param prog               Program sequence to add operations to.
 * \param stateInit          Initial state for the LSTM.
 * \param weights            The LSTM weights structure.
 * \param in                 The input tensor to the LSTM of dimension
 *                           [timesteps, batch, inputSize].
 * \param[out] intermediates Intermediate results that are retained in the
 *                           the forward pass of training for use in the
 *                           backward pass. This argument should be set to
 *                           null if we are only doing inference.
 * \param debugContext       Optional debug information.
 * \param options            LSTM implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The LSTM output. If the `outputFullSequence` option is false the
 *         output of the last timestep of shape [batch, outputSize] is returned
 *         and if the `outputFullSequence` option is true the output sequence
 *         for every time step is returned in shape
 *         [timeSteps, batch, outputSize].
 */
poplar::Tensor lstmFwd(poplar::Graph &graph, const LstmParams &params,
                       poplar::program::Sequence &prog,
                       const LstmState &stateInit, const LstmWeights &weights,
                       const poplar::Tensor &in, poplar::Tensor *intermediates,
                       const poplar::DebugContext &debugContext = {},
                       const poplar::OptionFlags &options = {},
                       poplin::matmul::PlanningCache *planningCache = nullptr);

/** \deprecated
 *  **deprecated** Use previously defined popnn::lstmFwd() instead.
 *
 * Calculate the result of applying an LSTM across a sequence.
 *
 * The LSTM is run for seqSize steps each with a batch of size batchSize and
 * input size inputSize and output size outputSize. The total number of units
 * within each LSTM cell is lstmUnits = BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph              Graph to which the LSTM cell belongs.
 * \param params             The parameters of the LSTM.
 * \param stateInit          Initial state for the LSTM.
 * \param in                 The input tensor to the LSTM of dimension
 *                           [timesteps, batch, inputSize].
 * \param weights            The LSTM weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           the forward pass of training for use in the
 *                           backward pass. This argument should be set to
 *                           null if we are only doing inference.
 * \param fwdProg            Program sequence.
 * \param debugContext       Optional debug information.
 * \param options            LSTM implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the LSTM and the final cell state.\n
 *         Depending on the outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [batch, outputSize] or it is the sequence of outputs for every
 *         timestep in the shape [timesteps, batch, outputSize].
 */
std::pair<poplar::Tensor, poplar::Tensor>
lstmFwd(poplar::Graph &graph, const LstmParams &params,
        const LstmState &stateInit, const poplar::Tensor &in,
        const LstmWeights &weights, poplar::Tensor *intermediates,
        poplar::program::Sequence &fwdProg,
        const poplar::DebugContext &debugContext = {},
        const poplar::OptionFlags &options = {},
        poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 *  Run LSTM backward pass. The backward pass executes in reverse order as
 *  compared to the forward pass. If the forward steps for a LSTM layer are sf =
 *  {0, 1, 2, ..., S - 1} then the backward steps run for sb = {S - 1, S - 2,
 *  .... , 1, 0}.
 *
 * \param graph              Graph to which the LSTM cell belongs.
 * \param params             The parameters of the LSTM.
 * \param prog               Program sequence.
 * \param fwdStateInit       Forward state tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The LSTM weights structure.
 * \param input              The input tensor to the LSTM of shape:
 *                           [timesteps, batch, inputSize].
 * \param output             The output tensor from the forward pass. Depending
 *                           on the outputFullSequence parameter this is either
 *                           the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or
 *                           it is a sequence output gradients for each
 *                           timestep.
 * \param lastCellStateGrad  The gradient of the last cell state - may be
 *                           null if there is no incoming gradient.
 * \param[out] *inputSeqGrad The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param[out] *bwdIntermediates Intermediates gradients that are retained in
 *                           the backward pass of training for use in the
 *                           weight update. This argument should be set to null
 *                           if you do not need to calculate weight deltas.
 *  \param debugContext       Optional debug information.
 * \param options            LSTM implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial state.
 */
LstmState lstmBwd(poplar::Graph &graph, const LstmParams &params,
                  poplar::program::Sequence &prog,
                  const LstmState &fwdStateInit,
                  const poplar::Tensor &fwdIntermediates,
                  const LstmWeights &weights, const poplar::Tensor &input,
                  const poplar::Tensor &output,
                  const poplar::Tensor &outputGrad,
                  const poplar::Tensor *lastCellStateGrad,
                  poplar::Tensor *inputGrad, poplar::Tensor *bwdIntermediates,
                  const poplar::DebugContext &debugContext = {},
                  const poplar::OptionFlags &options = {},
                  poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 *  Run a standalone weight update pass. Takes intermediates and gradients
 *  from the backward pass and calculates and returns weight deltas.
 *
 *  Note: If the time step limit is variable, the entries above the given time
 *        step limit must be explicitly set to zero in `fwdIntermediates`, in
 *        order for the weights to be correctly updated.
 *
 *  \param graph            Graph to which the LSTM cell belongs.
 *  \param params           The parameters of the LSTM.
 *  \param prog             Program sequence to add operations to.
 *  \param fwdStateInit     Forward state tensor for initial step.
 *  \param fwdIntermediates Intermediate results from the forward pass.
 *  \param bwdIntermediates Intermediate results from the backward pass.
 *  \param weights          The LSTM weights structure.
 *  \param input            The input tensor to the LSTM of shape:
 *                          [timesteps, batch, inputSize].
 *  \param output           The output tensor from the forward pass. Depending
 *                          on the outputFullSequence parameter this is either
 *                          the output for the last timestep or it is a
 *                          sequence of outputs for each timestep.
 *  \param debugContext     Optional debug information.
 *  \param options          LSTM implementation options. See createInput().
 *  \param planningCache    The matmul planning cache.
 *
 *  \return A set of weight gradients to sum with weights.
 */
LstmWeights lstmWU(poplar::Graph &graph, const LstmParams &params,
                   poplar::program::Sequence &prog,
                   const LstmState &fwdStateInit,
                   const poplar::Tensor &fwdIntermediates,
                   const poplar::Tensor &bwdIntermediates,
                   const LstmWeights &weights, const poplar::Tensor &input,
                   const poplar::Tensor &output,
                   const poplar::DebugContext &debugContext = {},
                   const poplar::OptionFlags &options = {},
                   poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 *  Run a combined LSTM backward and weight update pass. Use this combined
 *  backward and weight update pass in preference to `lstmBwd` and `lstmWU`
 *  separately in order to allow the most efficient implementation to be
 *  chosen if you do not need to split the operation.
 *
 *  Note: If the time step limit is variable, the entries above the given time
 *        step limit must be explicitly set to zero in `fwdIntermediates`, in
 *        order for the weights to be correctly updated.
 *
 * \param graph              Graph to which the LSTM cell belongs.
 * \param params             The parameters of the LSTM.
 * \param prog               Program sequence.
 * \param fwdStateInit       Forward state tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The LSTM weights structure.
 * \param input              The input tensor to the LSTM of shape:
 *                           [timesteps, batch, inputSize].
 * \param output             The output tensor from the forward pass. Depending
 *                           on the outputFullSequence parameter this is either
 *                           the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or
 *                           it is a sequence output gradients for each
 *                           timestep.
 * \param lastCellStateGrad  The gradient of the last cell state - may be
 *                           null if there is no incoming gradient.
 * \param[out] *inputSeqGrad The gradients of the inputs. May be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param debugContext       Optional debug information.
 * \param options            LSTM implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial state.
 */
LstmState lstmBwdWithWU(poplar::Graph &graph, const LstmParams &params,
                        poplar::program::Sequence &prog,
                        const LstmState &fwdStateInit,
                        const poplar::Tensor &fwdIntermediates,
                        const LstmWeights &weights, const poplar::Tensor &input,
                        const poplar::Tensor &output,
                        const poplar::Tensor &outputGrad,
                        const poplar::Tensor *lastCellStateGrad,
                        poplar::Tensor *inputGrad, LstmWeights &weightsGrad,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {},
                        poplin::matmul::PlanningCache *planningCache = nullptr);

} // namespace lstm
} // namespace popnn

#endif // popnn_Lstm_hpp
