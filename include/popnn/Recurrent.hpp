// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
/** \file
 *  Functions for recurrent neural networks (RNN).
 */

#ifndef popnn_Recurrent_hpp
#define popnn_Recurrent_hpp

/*
 *  Vanilla RNN layer implementation:
 *
 *       ------             ----         ----------------
 * x -->| Wff |----------->| + |------->| Non linearity |----------> y
 *      ------             ----         ----------------        |
 *                          /\                                 |
 *                          |        ------                   |
 *                          --------| Wfb |<------------------
 *                                   -----
 *
 *
 *  In general, the RNN can be run over a set of sequence steps. The
 *  multiplication with Wff can be done in parallel for any subset or even the
 *  full set of sequence steps. The recurrent part must be done a step at a
 *  time.
 *
 *  In the code below:
 *  Wff is named weightsInput
 *  Wfb is named weightsFeedback
 */

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popnn/NonLinearity.hpp>

namespace poplin {
namespace matmul {
class PlanningCache;
}
} // namespace poplin

namespace popnn {
/// Functions for Recurrent Neural Networks (RNN)
namespace rnn {

/**
 * Predict what matrix multiplications will be needed for the given parameters
 * and return list of corresponding matmul parameters and options.
 */
std::vector<std::pair<poplin::MatMulParams, poplar::OptionFlags>>
getMatMulPrePlanParameters(std::size_t numSteps, std::size_t batchSize,
                           std::size_t inputSize, std::size_t outputSize,
                           const poplar::Type &dType,
                           const poplar::Type &partialsType = poplar::FLOAT,
                           bool inferenceOnly = false,
                           bool hasFeedforwardWeights = true);

/**
 * Compute the total floating point operations for the forward pass of RNN
 */
uint64_t getFwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool weightInput = true);
/**
 * Compute the total floating point operations for the backward pass of RNN
 */
uint64_t getBwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool calcInputGrad = true);
/**
 * Compute the total floating point operations for the weight update pass of RNN
 */
uint64_t getWuFlops(unsigned sequenceSize, unsigned batchSize,
                    unsigned inputSize, unsigned outputSize);

/**
 * Create a tensor which is input to a vanilla RNN. The layout of the tensor is
 * best for a multiplication of the input weight matrix with the given
 * number of steps.
 *
 * \param graph           The graph object.
 * \param numSteps        The number of steps used in the forward weighting of
 *                        the input.
 * \param batchSize       The number of batch elements.
 * \param inputSize       The size of the input for each sequence step.
 * \param outputSize      Output (hidden) size of each sequence element.
 * \param inferenceOnly   Indicates whether the RNN layer is for
 *                        inference only. If true, we can ignore the backwards
 *                        and weight update passes.
 * \param dType           Data type of the created tensor.
 * \param partialsType    Data type of intermediate calculations.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return Tensor of shape [`numSteps`, `batchSize`, `inputSize`]
 */
poplar::Tensor
createInput(poplar::Graph &graph, unsigned numSteps, unsigned batchSize,
            unsigned inputSize, unsigned outputSize, const poplar::Type &dType,
            const poplar::Type &partialsType = poplar::FLOAT,
            bool inferenceOnly = false,
            const poplar::DebugContext &debugContext = {},
            poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 * Create initial state for a vanilla RNN. The state, apart from the
 * activations, is initialised by the control program.
 *
 * The number of hidden states may depend on whether the RNN is used for
 * inference or training.
 *
 * \param graph           The graph object.
 * \param dType           Data type of the created tensor.
 * \param batchSize       The number of batch elements.
 * \param outputSize      Output (hidden) of each sequence element.
 * \param prog            The control program.
 * \param initState       If true, indicates that the state should be
 *                        initialised.
 * \param inferenceOnly   Indicates whether the RNN layer is for inference only.
 *                        If true, we can ignore backwards and weight
 *                        update passes.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return A 2D tensor of shape [`batchSize`, `outputSize`]
 */
poplar::Tensor createFwdState(
    poplar::Graph &graph, const poplar::Type &dType, unsigned batchSize,
    unsigned outputSize, poplar::program::Sequence &prog, bool initState,
    bool inferenceOnly, const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Extract previous output tensor from the hidden state. The returned tensor is
 *  a view of the tensor and can be used to initialise the tensor, if required.
 */
poplar::Tensor getOutputFromFwdState(const poplar::Tensor &fwdState);

/** Create the weights used to weight the input of a vanilla RNN layer. The best
 *  tile mapping of the weight tensor is when it matches the sequence size in
 *  the input activation tensor used to multiply the input weights.
 *
 * \param graph           The graph object.
 * \param sequenceSize    The number of sequence steps used in the forward
 *                        weighting of the input. The best tile mapping is when
 *                        this matches the sequence size of the input activation
 *                        tensor.
 * \param batchSize       The number of batch elements.
 * \param inputSize       Input size of each sequence.
 * \param outputSize      Output (hidden) size of each sequence.
 * \param dType           Data type of the created tensor.
 * \param partialsType    Data type of partial results in the computation.
 * \param inferenceOnly   Indicates whether the RNN layer is for inference only.
 *                        If true, we can ignore backwards and weight
 *                        update passes.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 */
poplar::Tensor createWeightsInput(
    poplar::Graph &graph, unsigned sequenceSize, unsigned batchSize,
    unsigned inputSize, unsigned outputSize, const poplar::Type &dType,
    const poplar::Type &partialsType = poplar::FLOAT,
    bool inferenceOnly = false, const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the weights used in the recurrent part of a vanilla RNN layer.
 *
 * \param graph           The graph object.
 * \param batchSize       The number of batch elements.
 * \param outputSize      Output (hidden) size of each sequence.
 * \param dType           Data type of the created tensor.
 * \param partialsType    Data type of partial results in the computation
 * \param inferenceOnly   Indicates whether the RNN layer is for inference only.
 *                        If true, we can ignore backwards and weight
 *                        update passes.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 */
poplar::Tensor createWeightsFeedback(
    poplar::Graph &graph, unsigned batchSize, unsigned outputSize,
    const poplar::Type &dType, const poplar::Type &partialsType = poplar::FLOAT,
    bool inferenceOnly = false, const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 * Perform the feedforward part of a RNN layer. The feedforward part of the RNN
 * layer must be followed by the feedback part to complete the RNN layer. In
 * other words, the output must be fed as the feedforward input to the feedback
 * part.
 * \see forwardIterate
 *
 * The following definitions apply:
 *   - `numSteps` is the number of sequence steps.
 *   - `batchSize` is the number of batch elements.
 *   - `inputSize` is the size of the input for each step.
 *   - `outputSize` is the size of the output for each step.
 *
 * \param graph           The graph object.
 * \param actIn           The input activation tensor with shape
 *                        [`numSteps`, `batchSize`, `inputSize`].
 * \param weights         Feedforward weights with shape [`outputSize`,
 *                        `inputSize`].
 * \param prog            Program sequence. Programs added by this
 *                        function are appended to this program sequence.
 * \param partialsType    Data type for intermediates.
 * \param inferenceOnly   Indicates whether the RNN layer is for inference only.
 *                        If true, we can ignore backwards and weight
 *                        update passes.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return                Output tensor with shape [`numSteps`, `batchSize`,
 *                        `outputSize`].
 */
poplar::Tensor forwardWeightInput(
    poplar::Graph &graph, const poplar::Tensor &actIn,
    const poplar::Tensor &weights, poplar::program::Sequence &prog,
    const poplar::Type &partialsType = poplar::FLOAT,
    bool inferenceOnly = false, const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 * Perform the feedback part of the RNN layer. The feedback part of the RNN
 * layer must be preceded by the feedforward part of the RNN layer to complete
 * the layer.
 * \see forwardWeightInput
 *
 * The following definitions apply:
 *   - `numSteps` is the number of steps.
 *   - `batchSize` is the number of batch elements.
 *   - `inputSize` is the size of the input for each step.
 *   - `outputSize` is the size of the output for each step.
 *
 * \param graph           The graph object.
 * \param feedFwdIn       Input to this function (output from the feedforward
 *                        part of the RNN layer.
 * \param initState       The initial state of the RNN layer, which means the
 *                        the previous output.
 * \param feedbackWeights Feedback weights.
 * \param biases          Biases.
 * \param prog            Program sequence. Programs added by this
 *                        function are appended to this program sequence.
 * \param nonLinearityType Non linearity used for the output activations
 * \param partialsType    Data type for intermediates.
 * \param inferenceOnly   Indicates whether the RNN layer is for inference only.
 *                        If true, we can ignore backwards and weight
 *                        update passes.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return                Output activations of the RNN layer.
 *
 */
poplar::Tensor forwardIterate(
    poplar::Graph &graph, const poplar::Tensor &feedFwdIn,
    const poplar::Tensor &initState, const poplar::Tensor &feedbackWeights,
    const poplar::Tensor &biases, poplar::program::Sequence &prog,
    popnn::NonLinearityType nonLinearityType,
    const poplar::Type &partialsType = poplar::FLOAT,
    bool inferenceOnly = false, const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create initial state for backward pass of a vanilla RNN.
 *
 * \param graph           The graph object.
 * \param dType           Data type of the created tensor.
 * \param batchSize       The number of batch elements processed.
 * \param outputSize      The number of output activations.
 * \param prog            Control program.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return Tile mapped initial state tensor.
 */
poplar::Tensor
createBwdState(poplar::Graph &graph, const poplar::Type &dType,
               unsigned batchSize, unsigned outputSize,
               poplar::program::Sequence &prog,
               const poplar::DebugContext &debugContext = {},
               poplin::matmul::PlanningCache *planningCache = nullptr);

/** Compute a single step of the backward pass of a vanilla RNN layer. Two
 *  gradient outputs are produced. The first gradient is at the input of the RNN
 *  layer for the step. The second gradient is at the adder and can be used to
 *  backward propagate through the earlier steps.
 */
/*       ------        ----           -----
 *  <---| Wfb |<------| + |<---------| NL |<------- (bwd:gradientOut
 *      ------        ----           -----               for final step)
 *                      | (bwd:gradientOut)
 *                     \|/
 *                   -----
 *                  | Wff |
 *                  ------
 *                     |
 *  Wfb are the feedback weights
 *  Wff are the input weights
 */
/**
 * \param graph           The graph object.
 * \param nextLayerGrad   Loss gradient fed as input to this step.
 * \param bwdState        Gradient state for the previous step.
 * \param actOut          Output activation.
 * \param weightsInput    Input weights.
 * \param weightsFeedback Feedback weights.
 * \param prog            Control program to which programs are added.
 * \param nonLinearityType Type of non-linearity.
 * \param partialsType    Data type used in intermediate calculations.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 *
 * \return A pair of tensors. The first tensor is the loss gradient at the input
 *         layer. The second tensor is the backward state needed to run the next
 *         backward step.
 */
std::pair<poplar::Tensor, poplar::Tensor> backwardGradientStep(
    poplar::Graph &graph, const poplar::Tensor &nextLayerGrad,
    const poplar::Tensor &bwdState, const poplar::Tensor &actOut,
    const poplar::Tensor &weightsInput, const poplar::Tensor &weightsFeedback,
    poplar::program::Sequence &prog, popnn::NonLinearityType nonLinearityType,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

// clang-format off
// To handle long lines in Doxygen for auto-linking of overloaded functions.
/** Same as backwardGradientStep(poplar::Graph&, const poplar::Tensor&, const poplar::Tensor&, const poplar::Tensor&, const poplar::Tensor&, const poplar::Tensor &, poplar::program::Sequence&, popnn::NonLinearityType, const poplar::Type&, const poplar::DebugContext&, poplin::matmul::PlanningCache*) with the difference that the input gradients are not computed.
 */
// clang-format on
poplar::Tensor backwardGradientStep(
    poplar::Graph &graph, const poplar::Tensor &nextLayerGrad,
    const poplar::Tensor &bwdState, const poplar::Tensor &actOut,
    const poplar::Tensor &weightsFeedback, poplar::program::Sequence &prog,
    popnn::NonLinearityType nonLinearityType,
    const poplar::Type &partialsType = poplar::FLOAT,
    const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Update parameter deltas for a vanilla RNN step.
 *  The parameter deltas updated are:
 *      - Feedback Weights
 *      - Input Weights
 *      - Bias
 *  The new deltas computed for this step are added to the accumulated deltas
 *  from previous steps. The caller must zero the accumulated tensors at the
 *  first call if the tensors to maintain the result are in-place.
 *
 * \param graph           The graph object.
 * \param bwdState        Gradient state for this step.
 * \param actIn           Input activations for this step.
 * \param prevOut         Previous RNN output activations for this step.
 * \param weightsInputDeltasAcc    Previous weights input deltas tensor. This
 *                                 tensor must be tile-mapped. The deltas from
 *                                 this step are added to this tensor.
 * \param weightsFeedbackDeltasAcc Previous feedback weights deltas tensor. This
 *                                 tensor must be tile-mapped. The deltas from
 *                                 this step are added to this tensor.
 * \param biasDeltasAcc   Previous bias deltas tensor. This tensor must be
 *                        tile-mapped. The deltas from this step are added to
 *                        this tensor.
 * \param prog            Control program to which programs are added.
 * \param partialsType    Data type used in intermediate calculations.
 * \param debugContext    Optional debug information.
 * \param planningCache   The matmul planning cache.
 */
void paramDeltaUpdate(poplar::Graph &graph, const poplar::Tensor &bwdState,
                      const poplar::Tensor &actIn,
                      const poplar::Tensor &prevOut,
                      poplar::Tensor &weightsInputDeltasAcc,
                      poplar::Tensor &weightsFeedbackDeltasAcc,
                      poplar::Tensor &biasDeltasAcc,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const poplar::DebugContext &debugContext = {},
                      poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 * Perform the forward part of the RNN layer. The feedback part of the RNN layer
 * must be preceded by the feedforward part of the RNN layer to complete the
 * layer.
 * \see forwardWeightInput
 *
 * The following definitions apply:
 *   - `numSteps` is the number of steps
 *   - `batchSize` is the number of batch elements.
 *   - `inputSize` is the size of the input for each step.
 *   - `outputSize` is the size of the output for each step.
 *
 * \param graph             The graph object.
 * \param prog              Control program.
 * \param fwdStateInit      Forward state tensor for initial step.
 * \param weightedIn        Preweighted input, or `nullptr` if `feedFwdWeights`
 *                          is to be applied.
 * \param biases            Biases.
 * \param feedFwdWeights    Input weights.
 * \param feedbackWeights   Feedback weights.
 * \param prevLayerActs     Activations from previous layer (output from
 *                          feedforward part of the RNN layer.
 * \param nonLinearityType  Type of non-linearity used for the output
 *                          activations.
 * \param partialsType      Data type for intermediates.
 * \param inferenceOnly     Indicates whether the RNN layer is for inference
 *                          only. If true, we can ignore backwards and weight
 *                          update passes.
 * \param debugContext      Optional debug information.
 * \param planningCache     The matmul planning cache.
 *
 * \return Forward state tensor for all steps [0:seqSize).
 */
poplar::Tensor rnnFwdSequence(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const poplar::Tensor &fwdStateInit, const poplar::Tensor *weightedIn,
    const poplar::Tensor &biases, const poplar::Tensor &feedFwdWeights,
    const poplar::Tensor &feedbackWeights, const poplar::Tensor &prevLayerActs,
    const popnn::NonLinearityType &nonLinearityType,
    const poplar::Type &partialsType, bool inferenceOnly,
    const poplar::DebugContext &debugContext = {},
    poplin::matmul::PlanningCache *planningCache = nullptr);

/**
 * Perform the feedback part of the RNN layer. The feedback part of the RNN
 * layer must be preceded by the feedforward part of the RNN layer to complete
 * the layer.
 * \see forwardWeightInput
 *
 * The following definitions apply:
 *   - `numSteps` is the number of steps.
 *   - `batchSize` is the number of batch elements.
 *   - `inputSize` is the size of the input for each step.
 *   - `outputSize` is the size of the output for each step.
 *
 * \param graph                     The graph object.
 * \param doWU                      Calculate weight updates.
 * \param ignoreInputGradientCalc   Do not calculate the gradients over the
 *                                  input weights.
 * \param prog                      Control program.
 * \param fwdStateInit              Forward state tensor for initial step.
 * \param fwdState                  Forward state tensor for all steps
 *                                  [0:seqSize).
 * \param biases                    Biases.
 * \param feedFwdWeights            Input weights.
 * \param feedbackWeights           Feedback weights.
 * \param outGradient               Gradient from next layer.
 * \param actIn                     Activations
 *                                  from the previous layer, so the output from
 *                                  the feedforward part of the RNN layer.
 * \param nonLinearityType          Type of non-linearity used for the output
 *                                  activations.
 * \param partialsType              Data type for intermediates.
 * \param debugContext              Optional debug information.
 * \param planningCache             The matmul planning cache.
 *
 * \return Returns four tensors:
 *         - gradients for the previous layer
 *         - input weight deltas
 *         - output weight deltas
 *         - bias deltas
 * When \p doWU is false, the weight and bias deltas are not calculated.
 *
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
rnnBwdSequence(poplar::Graph &graph, bool doWU, bool ignoreInputGradientCalc,
               poplar::program::Sequence &prog,
               const poplar::Tensor &fwdStateInit,
               const poplar::Tensor &fwdState, const poplar::Tensor &biases,
               const poplar::Tensor &feedFwdWeights,
               const poplar::Tensor &feedbackWeights,
               const poplar::Tensor &outGradient, const poplar::Tensor &actIn,
               const popnn::NonLinearityType &nonLinearityType,
               const poplar::Type &partialsType,
               const poplar::DebugContext &debugContext = {},
               poplin::matmul::PlanningCache *planningCache = nullptr);

} // namespace rnn
} // namespace popnn

#endif // popnn_Recurrent_hpp
