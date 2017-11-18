#ifndef _popnn_Lstm_hpp_
#define _popnn_Lstm_hpp_

#include <popnn/LstmDef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace lstm {

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize,
                                  bool weighInput = true);

uint64_t getBasicLstmCellBwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize,
                                  bool calcInputGrad = true);

uint64_t getBasicLstmCellWuFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize);


/** Create an input tensor of shape {numSteps, batchSize, inputSize} which is
 *  optimally mapped to multiply the whole input sequence in a single matrix
 *  multiply operation
 *
 * \param graph           Graph object
 * \param sequenceSize    Sequence step size of the tensor to be created
 * \param batchSize       Number of batch elements
 * \param inputSize       Number of input activations
 * \param outputSize      Number of output activations. This parameter is needed
 *                        to create an optimal mapping for the matrix multiply
 * \param dType           Data type string (float or half)
 * \param inferenceOnly   Parameter if set selects a tile maping of the tensor
 *                        which is optimal for inference operation.
 * \param name            String annotation
 *
 * \return A tensor created in the graph of shape  {sequenceSize, batchSize,
 *         inputSize}
 */
poplar::Tensor
createInput(poplar::Graph &graph,
            unsigned sequenceSize,
            unsigned batchSize,
            unsigned inputSize,
            unsigned outputSize,
            const poplar::Type &dType,
            bool inferenceOnly = false,
            const std::string &name = "");

/** Create forward state which is typically the input state of the LSTM cell.
 *  The first call to the LSTM forward pass will be to feed this created state
 *  as the init state of the cell. The "previous Output" and "cell state" may be
 *  initialised externally or by this function. It can be initialised externally
 *  by using the appropriate tensor views \see getOutputFromFwdState and \see
 *  getCellFromFwdState. The initialisation of the state is added to to the
 *  control program passed as an argument to this function.
 *
 * \param graph           Graph object
 * \param batchSize       Number of batch elements
 * \param outputSize      Number of input activations
 * \param prog            Control program to which the initialisation if enabled
 *                        is added to
 * \param initState       Flag to indicate if state created must be initialised
 * \param dType           Data type of the state
 * \param inferenceOnly   If true, the state is created for an inference
 * \param debugPrefix     String annotation
 *
 * \return A tensor which is the state for the forward operation of the LSTM
 *         cell
 */
poplar::Tensor
createFwdState(poplar::Graph &graph,
               unsigned batchSize,
               unsigned outputSize,
               poplar::program::Sequence &prog,
               bool initState,
               const poplar::Type &dType,
               bool inferenceOnly,
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


/** Returns the cell state tensor view from the forward state tensor
 *
 * \param fwdState        Forward state of the LSTM cell for a step
 *
 * \return Cell state of the LSTM cell given the state tensor for any step
 */
poplar::Tensor
getCellFromFwdState(const poplar::Tensor &fwdState);


/** Create the weights used to weight the input of an lstm. If the
 *  'preweights' parameter is set to true then weights are created that
 *  are optimal when using the calcSequenceWeightedInputs() function.
 */
poplar::Tensor
createWeightsInput(poplar::Graph &graph,
                   unsigned sequenceSize,
                   unsigned batchSize,
                   unsigned inputSize,
                   unsigned outputSize,
                   bool preweights,
                   const poplar::Type &dType,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   bool inferenceOnly = false,
                   const std::string &name = "");

poplar::Tensor
createWeightsOutput(poplar::Graph &graph,
                    unsigned sequenceSize,
                    unsigned batchSize,
                    unsigned outputSize,
                    const poplar::Type &dType,
                    const poplar::Type &partialsType = poplar::FLOAT,
                    bool inferenceOnly = false,
                    const std::string &name = "");

/**
 * Basic LSTM cell without peephole connections. The cell performs the following
 * operations:
 *   1) Compute a forget gate gF as a function of prevAct and cellState
 *   2) Compute an output gate gO as a function of prevAct and cellState
 *   3) Compute an input gate gI as a function of prevAct and cellState
 *   4) Compute a new candidate c as a function of prevAct and cellState
 *   5) new cellState = cellState * gF + input * gI * c
 *   6) Compute output nonLinearity(new cellState)
 *   7) The final output is gO * (new CellState)
 *
 * The LSTM may be run for a given step size stepSize each with a batch of size
 * batchSize and input size inputSize and output size outputSize. The total
 * number of units within each LSTM cell is BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param in            Input of shape {sequenceSize, batchSize, inputSize}
 * \param biases        Biases for each of the units of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize}
 * \param prevOutAct    Output activation from previous step
 * \param prevCellState Cell state from previous step
 * \param weightsInput  Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, inputSize, outputSize}
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param prog          Program sequence
 * \param partialsType  Intermediate data type used in operations
 * \param inferenceOnly Set this to true if the forward pass is only for
 *                      inference
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output state tensor for every sequence step this function is executed
 *         for
 */
poplar::Tensor
basicLstmCellForwardPass(poplar::Graph &graph,
                         const poplar::Tensor &in,
                         const poplar::Tensor &bBiases,
                         const poplar::Tensor &prevOutAct,
                         const poplar::Tensor &prevCellState,
                         const poplar::Tensor &weightsInput,
                         const poplar::Tensor &weightsOutput,
                         poplar::program::Sequence &prog,
                         const poplar::Type &partialsType = poplar::FLOAT,
                         bool inferenceOnly = false,
                         const std::string &debugPrefix = "");

/** Precalculate weighted inputs for the entire sequence to be passed to a
 *  LSTM cell.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param in            Input of shape {sequenceSize, batchSize, inputSize}
 * \param weightsInput  Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, inputSize, outputSize}
 * \param prog          Program sequence
 * \param partialsType  Intermediatedata type used in operations
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return weighted gate inputs tensor of shape
 *         {NUM_UNITS, sequenceSize, batchSize, outputSize}
 */
poplar::Tensor
calcSequenceWeightedInputs(poplar::Graph  &graph,
                           const poplar::Tensor &in,
                           const poplar::Tensor &weightsInput,
                           poplar::program::Sequence &prog,
                           const poplar::Type &partialsType = poplar::FLOAT,
                           const std::string &debugPrefix = "");


/** Calculate the result of apply an LSTM across a sequence given that
 *  the inputs have already had weights applied to them.
 *
 * The LSTM may be run for a given step size stepSize each with a batch of size
 * batchSize and input size inputSize and outputSize outputSize. The total
 * number of units within each LSTM cell is lstmUnits =
 * BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param weightedIn    Input of shape {lstmUnits, sequenceSize, batchSize,
 *                      outputSize}
 * \param biases        Biases for each of the units of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize}
 * \param prevOutAct    Output activation from previous step
 * \param prevCellState Cell state from previous step
 * \param fwdState      State for the previous step of the LSTM cell
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param prog          Program sequence
 * \param partialsType  Intermediate data type used in operations
 * \param inferenceOnly Set this to true if the forward pass is only for
 *                      inference
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output state tensor for every sequence step this function is executed
 */
poplar::Tensor
basicLstmCellForwardPassWeightedInputs(
              poplar::Graph &graph,
              const poplar::Tensor &weightedInput,
              const poplar::Tensor &bBiases,
              const poplar::Tensor &prevOutAct,
              const poplar::Tensor &prevCellState,
              const poplar::Tensor &weightsOutput,
              poplar::program::Sequence &prog,
              const poplar::Type &partialsType = poplar::FLOAT,
              bool inferenceOnly = false,
              const std::string &debugPrefix = "");

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
createBwdState(poplar::Graph &graph,
               unsigned batchSize,
               unsigned outputSize,
               poplar::program::Sequence &prog,
               const poplar::Type &dType,
               const std::string &debugPrefix = "");

/** A single step of a LSTM backward pass with S sequence steps. The backward
 *  pass executes in reverse order as compared to the forward pass. If the
 *  forward steps for a LSTM layer are sf = {0, 1, 2, ..., S - 1} then the
 *  backward steps run for sb = {S - 1, S - 2, .... , 1, 0}.
 *
 *  Below, s is the step for which this function is called.
 *
 * \param graph           Graph object
 * \param gradNextLayer   Gradient from next layer for step for sequence step s
 * \param fwdStateThisStep Forward state tensor for step s
 * \param prevCellState    Cell state tensor for step s - 1. If s == 0, this
 *                         state is for step -1, which is the initial forward
 *                         state of an LSTM layer (typically created using
 *                         \see createFwdState and \see getCellFromFwdState)
 * \param bwdState         This is backward state at step s + 1. for s == S-1,
 *                         this state is for step S, which is the initial
 *                         backward state of an LSTM layer (typically created
 *                         using \createBackwardState)
 * \param weightsInput     Input weights tensor (created using
 *                         \createWeightsInput)
 * \param weightsOutput    Output weights tensor (created using
 *                         \createWeightsOutput
 * \param prog             Control program
 * \param partialsType     Data type of the intermediate precision
 * \param debugPrefix      String annotation
 *
 * \return Returns two tensors. The first tensor is the gradient of the input
 *         activation for this step s needed for the next layer. The second
 *         tensor is the backward state tensor for step s
 */
std::pair<poplar::Tensor, poplar::Tensor>
basicLstmBackwardStep(poplar::Graph &graph,
                      const poplar::Tensor &gradNextLayer,
                      const poplar::Tensor &fwdStateThisStep,
                      const poplar::Tensor &prevCellState,
                      const poplar::Tensor &bwdState,
                      const poplar::Tensor &weightsInput,
                      const poplar::Tensor &weightsOutput,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const std::string &fPrefix = "");

/** Same as \see basicLstmBackwardStep but without the input weight matrix.
 * The returned tensor is the backward state tensor for this step
 */
poplar::Tensor
basicLstmBackwardStep(poplar::Graph &graph,
                      const poplar::Tensor &gradNextLayer,
                      const poplar::Tensor &fwdStateThisStep,
                      const poplar::Tensor &prevCellState,
                      const poplar::Tensor &bwdState,
                      const poplar::Tensor &weightsOutput,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const std::string &fPrefix = "");

/** A single step of a LSTM param deltas update with S sequence steps. The param
 *  update pass can be run in any random order of steps spu = {0, 1, ..., S-1}
 *  as long as param update is computed only once for each step.
 *
 *  Below, s is the step for which this function is called.
 *
 * \param graph           Graph object
 * \param prevLayerActs   Previous layer activations for step s
 * \param prevStepActs    LSTM output Activations for step s-1. If s == 0, the
 *                        activations must be those created \createFwdState for
 *                        the first forward step
 * \param bwdState        Backward state for step s
 * \param weightsInputDeltaAcc  Input weights delta values are accumulated into
 *                              this tensor. It must be created, mapped and
 *                              initialised to zero
 * \param weightsOutputDeltaAcc Output weights delta values are accumulated into
 *                              this tensor. It must be created, mapped and
 *                              initialised to zero
 * \param biasDeltaAcc    Bias delta values are accumulated into this tensor. It
 *                        must be created, mapped and initialised to zero
 * \param prog            Control program
 * \param partialsType    Data type of intermediate calculations
 * \param debugPrefix     String annotation
 */
void
basicLstmParamUpdate(poplar::Graph &graph,
                     const poplar::Tensor &prevLayerActs,
                     const poplar::Tensor &prevStepActs,
                     const poplar::Tensor &bwdState,
                     const poplar::Tensor &weightsInputDeltaAcc,
                     const poplar::Tensor &weightsOutputDeltaAcc,
                     const poplar::Tensor &biasDeltaAcc,
                     poplar::program::Sequence &prog,
                     const poplar::Type &partialsType = poplar::FLOAT,
                     const std::string &debugPrefix = "");

/** Calculate the result of applying an LSTM across a sequence
 *
 * The LSTM is run for seqSize steps each with a batch of size batchSize and
 * input size inputSize and output size outputSize. The total number of units
 * within each LSTM cell is lstmUnits = BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param inferenceOnly Set this to true if the forward pass is only for
 *                      inference
 * \param fwdProg       Program sequence
 * \param fwdStateInit  Initial state for this layer
 * \param weightedIn    Input of shape {lstmUnits, sequenceSize, batchSize,
 *                      outputSize}, or nullptr if Wff is to be applied
 * \param biases        Biases for each of the units of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize}
 * \param weightsInput  Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, inputSize, outputSize}
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param prevLayerActs Output activation from previous step
 * \param dataType      Data type of the activations and weights
 * \param partialsType  Intermediate data type used in operations
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output state tensor for every sequence step this function is executed
 */
poplar::Tensor lstmFwdSequence(
                     poplar::Graph &graph,
                     bool inferenceOnly,
                     poplar::program::Sequence &fwdProg,
                     const poplar::Tensor &fwdStateInit,
                     const poplar::Tensor *weightedIn,
                     const poplar::Tensor &biases,
                     const poplar::Tensor &weightsInput,
                     const poplar::Tensor &weightsOutput,
                     const poplar::Tensor &prevLayerActs,
                     const poplar::Type &dataType,
                     const poplar::Type &partialsType = poplar::FLOAT,
                     const std::string &debugPrefix = "");

/**
 *  Run LSTM backward pass. The backward pass executes in reverse order as
 *  compared to the forward pass. If the forward steps for a LSTM layer are sf =
 *  {0, 1, 2, ..., S - 1} then the backward steps run for sb = {S - 1, S - 2,
 *  .... , 1, 0}.
 *
 * \param graph           Graph object
 * \param doWU            When true weight and bias delta updates are calculated
 * \param ignoreInputGradientCalc Do not calculate the gradients over the input
 *                        weights
 * \param prog            Control program
 * \param fwdStateInit    Forward state tensor for initial step
 * \param fwdState        Forward state tensor for all steps [0:seqSize)
 * \param biases          Biases for each of the units of shape
 *                        {BASIC_LSTM_CELL_NUM_UNITS, outputSize}. Used only
 *                        for shape and mapping
 * \param weightsInput    Input weights tensor (created using
 *                        \createWeightsInput)
 * \param weightsOutput   Output weights tensor (created using
 *                        \createWeightsOutput
 * \param prevLayerActs   Previous layer activations for step s
 * \param gradNextLayer   Gradient from next layer
 * \param bwdState        Initial backward state, typically created
 *                        using \createBackwardState
 * \param dataType        Data type of the weights and activations
 * \param partialsType    Data type of the intermediate precision
 * \param debugPrefix     String annotation
 *
 * \return Returns four tensors:
 *         - gradients for previous layer
 *         - input weight deltas
 *         - output weight deltas
 *         - bias deltas
 * When doWU is false the weight and bias deltas are not calculated
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
  lstmBwdSequence(
    poplar::Graph &graph,
    bool doWU,
    bool ignoreInputGradientCalc,
    poplar::program::Sequence &prog,
    const poplar::Tensor &fwdStateInit,
    const poplar::Tensor &fwdState,
    const poplar::Tensor &biases,
    const poplar::Tensor &weightsInput,
    const poplar::Tensor &weightsOutput,
    const poplar::Tensor &prevLayerActs,
    const poplar::Tensor &gradNextLayer,
    const poplar::Tensor &bwdState,
    const poplar::Type &dataType,
    const poplar::Type &partialsType,
    const std::string &debugPrefix = "");

} // namespace lstm
} // namespave popnn
#endif  // _popnn_Lstm_hpp_
