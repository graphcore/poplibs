// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_Lstm_hpp
#define poplibs_test_Lstm_hpp

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <popnn/LstmDef.hpp>
#include <popnn/NonLinearityDef.hpp>

namespace poplibs_test {
namespace lstm {

// Defines for state information in forward pass
#define LSTM_NUM_FWD_STATES 7
#define LSTM_FWD_STATE_ACTS_IDX 0
#define LSTM_FWD_STATE_CELL_STATE_IDX 1

// Defines for state information in backward pass
#define LSTM_NUM_BWD_STATES BASIC_LSTM_CELL_NUM_UNITS

/**
 * Compute the cell state and output of a basic non-fused LSTM cell (without
 * peephole connections). The cell state and outputs are concatented into a
 * single dimension.
 *
 * \param input               Input to the LSTM cell of dimension
 *                            [sequenceSize][batchSize][inputSize]
 * \param biases              Biases in the LSTM cell of dimension
 *                            [NUM_LSTM_UNITS][outputSize]
 * \param prevOutput          Previous output used in the first time step
 *                            [batchSize][outputSize]
 * \param weightsInput        Weights in the LSTM cell which weigh the input
 *                            sequence. It is of dimension
 *                            [NUM_LSTM_UNITS][inputSize][outputSize]
 * \param weightsOutput       Weights in the LSTM cell which weigh the output
 *                            sequence. It is of dimension
 *                            [NUM_LSTM_UNITS][outputSize][outputSize]
 * \param timeSteps           Optional time steps limit of shape [batchSize]
 * \param prevCellState       Initial cell state of shape
 *                            [batchSize][outputSize]
 * \param state               The forward state for all the sequence steps of
 *                            dimension
 *                            [LSTM_NUM_FWD_STATES][sequenceSize][batchSize]
 *                            [outputSize]
 * \param lastOutput          Output after the final time step, of shape
 *                            [batchSize][outputSize]
 * \param lastCellState       Final cell state, of shape
 *                            [batchSize][outputSize]
 * \param cellOrder           The order that the weights for each gate are
 *                            stored in the input.
 * \param activation          Activation function.
 * \param recurrentActivation Recurrent activation function.
 */
void basicLstmCellForwardPass(
    const boost::multi_array_ref<double, 3> input,
    const boost::multi_array_ref<double, 2> biases,
    const boost::multi_array_ref<double, 2> prevOutput,
    const boost::multi_array_ref<double, 3> weightsInput,
    const boost::multi_array_ref<double, 3> weightsOutput,
    const boost::optional<boost::multi_array_ref<unsigned, 1>> &timeSteps,
    boost::multi_array_ref<double, 2> prevCellState,
    boost::multi_array_ref<double, 4> state,
    boost::multi_array_ref<double, 2> lastOutput,
    boost::multi_array_ref<double, 2> lastCellState,
    const std::vector<BasicLstmCellUnit> &cellOrder,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation);

/** Run backward pass given forward sequence
 *
 * \param outputFullSequence  if true, the full sequence of outputs will be
 *                            returned, otherwise, only the output of the last
 *                            cell will be returned.
 * \param weightsInput        Input weights
 *                            shape: [NUM_LSTM_UNITS][input ch][output ch]
 * \param weightsOutput       Output weights
 *                            shape: [NUM_LSTM_UNITS][output ch][output ch]
 * \param gradsNextLayer      Gradients from next layer needed to compute
 *                            gradients for this layer.
 *                            shape: [sequence][batch][output ch]
 * \param prevCellState       Cell state of the initial step in the forward pass
 *                            shape: [batch][output ch]
 * \param fwdState            Forward state returned by \see
 *                            basicLstmCellForwardPass.
 *                            shape: [LSTM_NUM_FWD_STATES][sequence][batch]
 *                                   [output ch]
 * \param initOutputGrad      Gradient of output at the last step which is
 *                            used to initialise the time sequential output
 *                            gradient in the backward pass.
 *                            shape: [batch][output ch]
 * \param initCellStateGrad   Gradient of cell state at the last step which is
 *                            used to initialise the time sequential cell state
 *                            gradient in the backward pass.
 *                            shape: [batch][output ch]
 * \param timeSteps           Optional time steps limit of shape [batchSize]
 * \param bwdState            Backward state returned by this function
 *                            shape:[LSTM_NUM_BWD_STATES][sequence]
 *                                  [batch][output ch]
 * \param gradsPrevIn         Gradients for previous input computed by this
 *                            function shape: [sequence][batch][input ch]
 * \param gradsPrevOut        Gradients for previous output computed by this
 *                            function shape: [sequence][batch][output ch]
 * \param gradsPrevCellState  Gradients for previous cell state computed by this
 *                            function shape: [sequence][batch][output ch]
 * \param cellOrder           The order that the weights for each gate are
 *                            stored in the input.
 * \param activation          Activation function.
 * \param recurrentActivation Recurrent activation function.
 */
void basicLstmCellBackwardPass(
    bool outputFullSequence,
    const boost::multi_array_ref<double, 3> weightsInput,
    const boost::multi_array_ref<double, 3> weightsOutput,
    const boost::multi_array_ref<double, 3> gradsNextLayer,
    const boost::multi_array_ref<double, 2> prevCellState,
    const boost::multi_array_ref<double, 4> fwdState,
    const boost::optional<boost::multi_array_ref<double, 2>> initOutputGrad,
    const boost::optional<boost::multi_array_ref<double, 2>> initCellStateGrad,
    const boost::optional<boost::multi_array_ref<unsigned, 1>> &timeSteps,
    boost::multi_array_ref<double, 4> bwdState,
    boost::multi_array_ref<double, 3> gradsPrevIn,
    const boost::multi_array_ref<double, 2> gradPrevOut,
    const boost::multi_array_ref<double, 2> gradsPrevCellState,
    const std::vector<BasicLstmCellUnit> &cellOrder,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation);

/** Param update
 *
 * \param prevLayerActs   Activations from previous layer
 *                        shape: [sequence][batch][input ch]
 * \param fwdState        Forward state compute by \see basicLstmCellForwardPass
 *                        shape: [LSTM_NUM_FWD_STATES][sequence][batch][output]
 * \param outputActsInit  Initial activations for the forward pass
 *                        shape: [batch][output ch]
 * \param bwdState        Backward channel state generated by backward pass
 *                        shape" [LSTM_NUM_BWD_STATES][sequence][batch][output]
 * \param weightsInputDeltas  Weight deltas computed by this function
 *                            shape: [NUM_LSTM_UNITS][input ch][output ch]
 * \param weightsOutputDeltas Weight deltas computed by this function
 *                            shape: [NUM_LSTM_UNITS][output ch][output ch]
 *
 * \param biasDeltas      Bias deltas computed by this function
 *                        shape: [NUM_LSTM_UNITS][output ch]
 * \param cellOrder       The order that the weights for each gate are
 *                        stored in the input.
 */
void basicLstmCellParamUpdate(
    const boost::multi_array_ref<double, 3> prevLayerActs,
    const boost::multi_array_ref<double, 4> fwdState,
    const boost::multi_array_ref<double, 2> outputActsInit,
    const boost::multi_array_ref<double, 4> bwdState,
    boost::multi_array_ref<double, 3> weightsInputDeltas,
    boost::multi_array_ref<double, 3> weightsOutputDeltas,
    boost::multi_array_ref<double, 2> biasDeltas,
    const std::vector<BasicLstmCellUnit> &cellOrder);
} // namespace lstm
} // namespace poplibs_test

#endif // poplibs_test_Lstm_hpp
