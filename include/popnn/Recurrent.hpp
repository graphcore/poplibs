// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_Recurrent_hpp
#define popnn_Recurrent_hpp

/**
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

namespace popnn {
namespace rnn {

/**
 * Compute the total flops for the forward pass of RNN
 */
uint64_t getFwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool weightInput = true);
/**
 * Compute the total flops for the backward pass of RNN
 */
uint64_t getBwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool calcInputGrad = true);
/**
 * Compute the total flops for the weight update pass of RNN
 */
uint64_t getWuFlops(unsigned sequenceSize, unsigned batchSize,
                    unsigned inputSize, unsigned outputSize);

/**
 * Create a tensor which is input to a vanilla RNN. The layout of the tensor is
 * best for a multiplication of the input weight matrix with the given
 * number of steps.
 *
 * \param graph           Graph object
 * \param numSteps        Number of steps used in the forward weighting of input
 * \param batchSize       Number of batch elements
 * \param inputSize       Size of the input for each sequence step
 * \param outputSize      Output(hidden) size of each sequence element
 * \param inferenceOnly   Parameter if set selects a tile mapping of the tensor
 *                        which is better for inference operation.
 * \param dType           Data type of the created tensor
 * \param partialsType    Data type of intermediate calculations
 * \param name            Name of the tensor
 *
 * \return Tensor of shape {numSteps, batchSize, inputSize}
 */
poplar::Tensor
createInput(poplar::Graph &graph,
            unsigned numSteps,
            unsigned batchSize,
            unsigned inputSize,
            unsigned outputSize,
            const poplar::Type &dType,
            const poplar::Type &partialsType = poplar::FLOAT,
            bool inferenceOnly = false,
            const std::string &name = "");

/**
 * Create initial state for a vanilla RNN. The state apart from the activations
 * are initialised by the control program
 *
 * The amount of hidden state may depend on whether the RNN is used for
 * inference or training.
 *
 * \param graph           Graph object
 * \param dType           data type of the created tensor
 * \param batchSize       Number of batch elements
 * \param outputSize      Output(hidden) of each sequence element
 * \param prog            Control program
 * \param initState       Initialise the state
 * \param inferenceOnly   Set this flag to true if the RNN layer is to be run
 *                        for inference only
 * \param debugPrefix     String annotation
 *
 * \return A 2D tensor of shape {batchSize, outputSize}
 */
poplar::Tensor createFwdState(poplar::Graph &graph,
                              const poplar::Type &dType,
                              unsigned batchSize,
                              unsigned outputSize,
                              poplar::program::Sequence &prog,
                              bool initState,
                              bool inferenceOnly,
                              const std::string &debugPrefix = "");

/** Extract prev output tensor from hidden state. The returned tensor is a
 *  view of tensor and can be used to initialise the tensor if required
 */
poplar::Tensor getOutputFromFwdState(const poplar::Tensor &fwdState);

/** Create the weights used to weight the input of a vanilla RNN layer. The tile
 *  mapping of the weight tensor is best for multiplication with a sequence
 *  size in the input activation tensor used to multiply with the input weights.
 *
 * \param graph           Graph object
 * \param sequenceSize    Number of sequence steps used in the forward weighting
 *                        of the input. The best tile mapping is when this
 *                        matches the sequence size of the input activation
 *                        tensor
 * \param batchSize       Number of batch elements
 * \param inputSize       Input size of each sequence
 * \param outputSize      Output(hidden) size of each sequence
 * \param dType           Data type of the created tensor
 * \param partialsType    Data type of partial results in the computation
 * \param inferenceOnly   Parameter if set selects a tile mapping of the tensor
 *                        which is best for inference operation.
 * \param namePrefix      A string description of the weights tensor
 */
poplar::Tensor
createWeightsInput(poplar::Graph &graph,
                   unsigned sequenceSize, unsigned batchSize,
                   unsigned inputSize, unsigned outputSize,
                   const poplar::Type &dType,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   bool inferenceOnly = false,
                   const std::string &namePrefix = "");


/** Create the weights used in the recurrent part of a vanilla RNN layer
 *
 * \param graph           Graph object
 * \param batchSize       Number of batch elements
 * \param outputSize      Output(hidden) size of each sequence
 * \param dType           Data type of the created tensor
 * \param partialsType    Data type of partial results in the computation
 * \param inferenceOnly   Parameter if set selects a tile mapping of the tensor
 *                        which is best for inference operation.
 * \param namePrefix      A string description of the created tensor
 */
poplar::Tensor
createWeightsFeedback(poplar::Graph &graph,
                      unsigned batchSize,
                      unsigned outputSize,
                      const poplar::Type &dType,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      bool inferenceOnly = false,
                      const std::string &namePrefix = "");

/**
 * Perform feedforward part of a RNN layer. The feedforward part of the RNN
 * layer must be followed by the feedback part to complete the RNN layer. i.e.
 * the output must be fed as the feedforward input to the feedback part.
 * \see rnnForwardIterate
 *
 * The following definitions are used below:
 *   numSteps is the number of sequence steps
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each step
 *   outputSize is the size of the output for each step
 *
 * \param graph           Graph pbject
 * \param actIn           Input activation tensor with shape
 *                        {numSteps, batchSize, inputSize}
 * \param weights         Feedforward weights with shape {outputSize, inputSize}
 * \param prog            Program sequence to which  programs added by this
 *                        function are appended to
 * \param partialsType    Data type for intermediates
 * \param debugPrefix     Debug prefix string
 *
 * \return                Output tensor with shape {numSteps, batchSize,
 *                        outputSize}
 */
poplar::Tensor
forwardWeightInput(poplar::Graph &graph,
                   const poplar::Tensor &actIn,
                   const poplar::Tensor &weights,
                   poplar::program::Sequence &prog,
                   const poplar::Type &partialsType = poplar::FLOAT,
                   const std::string &debugPrefix = "");

/**
 * Perform the feedback part of the RNN layer. The feedback part of the RNN
 * layer must be preceded by the feedforward part of the RNN layer to complete
 * the layer
 * \see rnnForwardWeightInput
 *
 * The following definitions are used below:
 *   numSteps is the number of steps
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each step
 *   outputSize is the size of the output for each step
 *
 * \param graph           Graph object
 * \param feedFwdIn       Input to this function (output from feedforward part
 *                        of the RNN layer
 * \param initState       The initial state of the RNN layer(i.e. the previous
 *                        output)
 * \param feedbackWeights Feedback weights
 * \param biases          Biases
 * \param prog            Program sequence to which programs added by
 *                        this function are appended to
 * \param nonLinearityType Non linearity used for the output activations
 * \param partialsType    Data type for intermediates
 * \param debugPrefix     Debug prefix string
 *
 * \return                Output activations of RNN layer
 *
 */
poplar::Tensor forwardIterate(poplar::Graph &graph,
                              const poplar::Tensor &feedFwdIn,
                              const poplar::Tensor &initState,
                              const poplar::Tensor &feedbackWeights,
                              const poplar::Tensor &biases,
                              poplar::program::Sequence &prog,
                              popnn::NonLinearityType nonLinearityType,
                              const poplar::Type &partialsType = poplar::FLOAT,
                              const std::string &debugPrefix = "");

/** Create initial state for backward pass of a vanilla RNN
 *
 * \param graph           Graph object
 * \param dType           Data type of the created tensor
 * \param batchSize       Number of batch elements processed
 * \param outputSize      Number of output activations
 * \param prog            Control program
 * \param debugPrefix     String annotation
 *
 * \return Tile mapped initial state tensor
 */
poplar::Tensor createBwdState(poplar::Graph &graph, const poplar::Type &dType,
                              unsigned batchSize, unsigned outputSize,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix = "");

/** Compute a single step of backward pass of a vanilla RNN layer. Two gradient
 *  outputs are produced. The first is at the input of the RNN layer for the
 *  step. The second is at the adder and can be used to backward propagate
 *  through the earlier steps.
 *
 *       ------        ----           -----
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
 *
 * \param graph           Graph object
 * \param nextLayerGrad   Loss gradient fed as input to this step
 * \param bwdState        Gradient state for previous step
 * \param actOut          Output activation
 * \param weightsInput    Input weights
 * \param weightsFeedback Feedback weights
 * \param prog            Control program to which to add programs to
 * \param nonLinearityType Type of non-linearity
 * \param firstStep       Set to true to indicate if first step in the backward
 *                        pass
 * \param partialsType    Data type used in intermediate calculations
 * \param debugPrefix     A string annotation
 *
 * \return std::pair<poplar::Tensor,poplar::Tensor>
 *         A pair of tensors. The first is the loss gradient at the the input
 *         layer. The second is the backward state needed to run the next
 *         backward  step
 */
std::pair<poplar::Tensor, poplar::Tensor>
backwardGradientStep(poplar::Graph &graph,
                     const poplar::Tensor &nextLayerGrad,
                     const poplar::Tensor &bwdState,
                     const poplar::Tensor &actOut,
                     const poplar::Tensor &weightsInput,
                     const poplar::Tensor &weightsFeedback,
                     poplar::program::Sequence &prog,
                     popnn::NonLinearityType nonLinearityType,
                     const poplar::Type &partialsType = poplar::FLOAT,
                     const std::string &debugPrefix = ""
                     );

/** Same as function above with the difference that the input gradients are
 *  not computed
 */
poplar::Tensor
backwardGradientStep(poplar::Graph &graph,
                     const poplar::Tensor &nextLayerGrad,
                     const poplar::Tensor &bwdState,
                     const poplar::Tensor &actOut,
                     const poplar::Tensor &weightsFeedback,
                     poplar::program::Sequence &prog,
                     popnn::NonLinearityType nonLinearityType,
                     const poplar::Type &partialsType = poplar::FLOAT,
                     const std::string &debugPrefix = ""
                     );


/** Update parameter deltas for a vanilla RNN step.
 *  The parameter deltas updated are:
 *      - Feedback Weights
 *      - Input Weights
 *      - Bias
 *  The new deltas computed for this step are added to the accumulated deltas
 *  from previous steps. The caller must zero the accumulated tensors at the
 *  first call if the tensors to maintain the result are in-place
 *
 * \param graph           Graph object
 * \param bwdState        Gradient state for this step
 * \param actIn           Input activations for this step
 * \param prevOut         Previous RNN output activations for this step
 * \param weightsInputDeltasAcc    Previous weights input deltas tensor. This
 *                                 tensor must be tile-mapped. The deltas from
 *                                 this step are added to this tensor
 * \param weightsFeedbackDeltasAcc Previous feedback weights deltas tensor. This
 *                                 tensor must be tile-mapped. The deltas from
 *                                 this step are added to this tensor
 * \param biasDeltasAcc   Previous bias deltas tensor. This tensor must be
 *                        tile-mapped. The deltas from this step are added to
 *                        this tensor
 * \param prog            Control program to which to add programs to
 * \param partialsType    Data type used in intermediate calculations
 * \param debugPrefix     String anotation
 */
void paramDeltaUpdate(poplar::Graph &graph,
                      const poplar::Tensor &bwdState,
                      const poplar::Tensor &actIn,
                      const poplar::Tensor &prevOut,
                      poplar::Tensor &weightsInputDeltasAcc,
                      poplar::Tensor &weightsFeedbackDeltasAcc,
                      poplar::Tensor &biasDeltasAcc,
                      poplar::program::Sequence &prog,
                      const poplar::Type &partialsType = poplar::FLOAT,
                      const std::string &debugPrefix = "");

/**
 * Perform the forward part of the RNN layer. The feedback part of the RNN layer
 * must be preceded by the feedforward part of the RNN layer to complete the
 * layer
 * \see rnnForwardWeightInput
 *
 * The following definitions are used below:
 *   numSteps is the number of steps
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each step
 *   outputSize is the size of the output for each step
 *
 * \param graph           Graph object
 * \param prog            Control program
 * \param fwdStateInit    Forward state tensor for initial step
 * \param weightedIn      Preweighted input, or nullptr if Wff is to be applied
 * \param biases          Biases
 * \param feedFwdWeights  Input weights Wff
 * \param feedbackWeights Feedback weights Wfb
 * \param prevLayerActs   Activations from previous layer (output from
 *                        feedforward part of the RNN layer
 * \param nonLinearityType Non linearity used for the output activations
 * \param partialsType    Data type for intermediates
 * \param debugPrefix     Debug prefix string
 *
 * \return Forward state tensor for all steps [0:seqSize)
 */
poplar::Tensor rnnFwdSequence(poplar::Graph &graph,
                              poplar::program::Sequence &prog,
                              const poplar::Tensor &fwdStateInit,
                              const poplar::Tensor *weightedIn,
                              const poplar::Tensor &biases,
                              const poplar::Tensor &feedFwdWeights,
                              const poplar::Tensor &feedbackWeights,
                              const poplar::Tensor &prevLayerActs,
                              const popnn::NonLinearityType &nonLinearityType,
                              const poplar::Type &partialsType,
                              const std::string &debugPrefix);

/**
 * Perform the feedback part of the RNN layer. The feedback part of the RNN
 * layer must be preceded by the feedforward part of the RNN layer to complete
 * the layer
 * \see rnnForwardWeightInput
 *
 * The following definitions are used below:
 *   numSteps is the number of steps
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each step
 *   outputSize is the size of the output for each step
 *
 * \param graph           Graph object
 * \param doWU            Calculate weight updates
 * \param ignoreInputGradientCalc Do not calculate the gradients over the input
 *                        weights
 * \param prog            Control program
 * \param fwdStateInit    Forward state tensor for initial step
 * \param fwdState        Forward state tensor for all steps [0:seqSize)
 * \param biases          Biases
 * \param feedFwdWeights  Input weights Wff
 * \param feedbackWeights Feedback weights Wfb
 * \param outGradient     Gradient from next layer
 * \param actIn           Activations from previous layer (output from
 *                        feedforward part of the RNN layer
 * \param nonLinearityType Non linearity used for the output activations
 * \param partialsType    Data type for intermediates
 * \param debugPrefix     Debug prefix string
 *
 * \return Returns four tensors:
 *         - gradients for previous layer
 *         - input weight deltas
 *         - output weight deltas
 *         - bias deltas
 * When doWU is false the weight and bias deltas are not calculated
 *
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
  rnnBwdSequence(poplar::Graph &graph,
                 bool doWU,
                 bool ignoreInputGradientCalc,
                 poplar::program::Sequence &prog,
                 const poplar::Tensor &fwdStateInit,
                 const poplar::Tensor &fwdState,
                 const poplar::Tensor &biases,
                 const poplar::Tensor &feedFwdWeights,
                 const poplar::Tensor &feedbackWeights,
                 const poplar::Tensor &outGradient,
                 const poplar::Tensor &actIn,
                 const popnn::NonLinearityType &nonLinearityType,
                 const poplar::Type &partialsType,
                 const std::string &debugPrefix);

} // namespace rnn
} // namespace popnn

#endif // popnn_Recurrent_hpp
