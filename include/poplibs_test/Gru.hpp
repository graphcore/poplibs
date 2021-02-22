// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplibs_test_Gru_hpp
#define poplibs_test_Gru_hpp

#include <boost/multi_array.hpp>
#include <boost/optional.hpp>
#include <popnn/GruDef.hpp>
#include <popnn/NonLinearityDef.hpp>

namespace poplibs_test {
namespace gru {

// Defines for state information in forward pass
#define GRU_NUM_FWD_STATES 4
#define GRU_FWD_STATE_RESET_GATE_IDX 0
#define GRU_FWD_STATE_UPDATE_GATE_IDX 1
#define GRU_FWD_STATE_CANDIDATE_IDX 2
#define GRU_FWD_STATE_ACTS_IDX 3

// Defines for state information in backward pass
#define GRU_NUM_BWD_STATES BASIC_GRU_CELL_NUM_UNITS

/**
 * Compute the output of a basic non-fused GRU cell (without
 * peephole connections).
 *
 * \param input               Input to the GRU cell of dimension
 *                            [sequenceSize][batchSize][inputSize]
 * \param biases              Biases in the GRU cell of dimension
 *                            [BASIC_GRU_CELL_NUM_UNITS][outputSize]
 * \param prevOutput          Previous output used in the first time step
 *                            [batchSize][outputSize]
 * \param weightsInput        Weights in the GRU cell which weigh the input
 *                            sequence. It is of dimension
 *                            [BASIC_GRU_CELL_NUM_UNITS][inputSize][outputSize]
 * \param weightsOutput       Weights in the GRU cell which weigh the output
 *                            sequence. It is of dimension
 *                            [BASIC_GRU_CELL_NUM_UNITS][outputSize][outputSize]
 * \param state               The forward state for all the sequence steps of
 *                            dimension
 *                            [GRU_NUM_FWD_STATES][sequenceSize][batchSize]
 *                            [outputSize]
 * \param cellOrder           The order that the weights for each gate are
 *                            stored in the input
 * \param resetAfter          Whether to apply the reset gate after the matrix
 *                            multiplication
 * \param recurrantBiases     A second set of biases applied to the recurrant
 *                            side of each gate
 *                            Only used when resetAfter=true
 *                            shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
 * \param activation          Activation function.
 * \param recurrentActivation Recurrent activation function.
 */
void basicGruCellForwardPass(
    bool outputFullSequence, const boost::multi_array_ref<double, 3> input,
    const boost::multi_array_ref<double, 2> biases,
    const boost::multi_array_ref<double, 2> prevOutput,
    const boost::multi_array_ref<double, 3> weightsInput,
    const boost::multi_array_ref<double, 3> weightsOutput,
    const boost::optional<boost::multi_array_ref<double, 2>> &attScoresOpt,
    const boost::optional<boost::multi_array_ref<int, 1>> &realTimeStepsOpt,
    boost::multi_array_ref<double, 4> state,
    const std::vector<BasicGruCellUnit> &cellOrder, bool resetAfter = false,
    const boost::optional<boost::multi_array_ref<double, 2>> recurrantBiases =
        boost::none,
    const popnn::NonLinearityType activation = popnn::NonLinearityType::TANH,
    const popnn::NonLinearityType recurrentActivation =
        popnn::NonLinearityType::SIGMOID);

/** Run backward pass given forward sequence
 *
 * \param outputFullSequence if ture, the all sequence of outputs will be
                             returned, otherwise, only the output of the last
                             cell will be returned.
 * \param weightsInput    Input weights
 *                        shape: [BASIC_GRU_CELL_NUM_UNITS][input ch][output ch]
 * \param weightsOutput   Output weights
 *                        shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
                                 [output ch]
 * \param gradsNextLayer  Gradients from next layer needed to compute gradients
 *                        for this layer. shape: [sequence][batch][output ch]
 * \param fwdState        Forward state returned by basicGRUCellForwardPass.
 *                        shape: [GRU_NUM_FWD_STATES][sequence][batch]
 *                               [output ch]
 * \param outputActsInit  Initial activations for the forward pass
 *                        shape: [batch][output ch]
 * \param bwdState        Backward state returned by this function
 *                        shape:[GRU_NUM_BWD_STATES][sequence]
 *                              [batch][output ch]
 * \param gradsPrevLayer  Gradients for previous layer computed by this function
 *                        shape: [sequence][batch][input ch]
 * \param cellOrder       The order that the weights for each gate are
 *                        stored in the input
 * \param resetAfter      Whether the reset gate was applied after the matrix
 *                        multiplication
 * \param recurrantBiases     A second set of biases applied to the recurrant
 *                            side of each gate
 *                            Only used when resetAfter=true
 *                            shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
 * \param activation          Activation function.
 * \param recurrentActivation Recurrent activation function.
 */
void basicGruCellBackwardPass(
    bool outputFullSequence,
    const boost::multi_array_ref<double, 3> weightsInput,
    const boost::multi_array_ref<double, 3> weightsOutput,
    const boost::multi_array_ref<double, 3> gradsNextLayer,
    const boost::multi_array_ref<double, 4> fwdState,
    const boost::multi_array_ref<double, 2> outputActsInit,
    const boost::optional<boost::multi_array_ref<int, 1>> &realTimeStepsOpt,
    const boost::optional<boost::multi_array_ref<double, 2>> &attScoresOpt,
    const boost::optional<boost::multi_array_ref<double, 2>> &attScoresGradsOpt,
    boost::multi_array_ref<double, 4> bwdState,
    boost::multi_array_ref<double, 3> gradsPrevLayer,
    const std::vector<BasicGruCellUnit> &cellOrder, bool resetAfter = false,
    const boost::optional<boost::multi_array_ref<double, 2>> recurrantBiases =
        boost::none,
    const popnn::NonLinearityType activation = popnn::NonLinearityType::TANH,
    const popnn::NonLinearityType recurrentActivation =
        popnn::NonLinearityType::SIGMOID);

/** Param update
 *
 * \param prevLayerActs   Activations from previous layer
 *                        shape: [sequence][batch][input ch]
 * \param fwdState        Forward state compute by basicGRUCellForwardPass
 *                        shape: [GRU_NUM_FWD_STATES][sequence][batch][output]
 * \param outputActsInit  Initial activations for the forward pass
 *                        shape: [batch][output ch]
 * \param bwdState        Backward channel state generated by backward pass
 *                        shape: [GRU_NUM_BWD_STATES][sequence][batch][output]
 * \param weightsInputDeltas  Weight deltas computed by this function
 *                            shape: [BASIC_GRU_CELL_NUM_UNITS][input ch]
                                     [output ch]
 * \param weightsOutputDeltas Weight deltas computed by this function
 *                            shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
                                     [output ch]
 * \param biasDeltas      Bias deltas computed by this function
 *                        shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
 * \param cellOrder       The order that the weights for each gate are
 *                        stored in the input
 * \param resetAfter      Whether the reset gate was applied after the matrix
 *                        multiplication
 * \param recurrantBiasDeltas Bias deltas computed by this function for the
 *                            recurrant biases
 *                            Only used when resetAfter=true
 *                            shape: [BASIC_GRU_CELL_NUM_UNITS][output ch]
 */

void basicGruCellParamUpdate(
    const boost::multi_array_ref<double, 3> prevLayerActs,
    const boost::multi_array_ref<double, 4> fwdState,
    const boost::multi_array_ref<double, 2> outputActsInit,
    const boost::multi_array_ref<double, 4> bwdState,
    boost::multi_array_ref<double, 3> weightsInputDeltas,
    boost::multi_array_ref<double, 3> weightsOutputDeltas,
    boost::multi_array_ref<double, 2> biasDeltas,
    const std::vector<BasicGruCellUnit> &cellOrder, bool resetAfter = false,
    boost::optional<boost::multi_array_ref<double, 2>> recurrantBiasDeltas =
        boost::none);
} // namespace gru
} // namespace poplibs_test

#endif // poplibs_test_gru_hpp
