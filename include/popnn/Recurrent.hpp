#ifndef __popnn_Recurrent_hpp__
#define __popnn_Recurrent_hpp__

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
                     bool weightInput);

/**
 * Perform feedforward part of an RNN. The feedforward part of the RNN must
 * be followed by the feedback part to complete the RNN. i.e. the output must
 * be fed as the feedforward input to the feedback part.
 * \see rnnForwardIterate
 *
 * The following definitions are used below:
 *   sequenceSize if the sequence size
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each sequence
 *   outputSize is the size of the output for each sequence
 *
 * \param graph           Graph pbject
 * \param actIn           Input activation tensor with shape
 *                        {sequenceSize, batchSize, inputSize}
 * \param weights         Feedforward weights with shape {outputSize, inputSize}
 * \param prog            Program sequence to which any programs added by this
 *                        function are appended to
 * \param partialsTypeStr Data type for intermediates
 * \param debugPrefix     Debug prefix string
 *
 * \return                Output tensor with shape
 *                        {sequenceSize, batchSize, outputSize}
 */
poplar::Tensor forwardWeightInput(poplar::Graph &graph,
                                  poplar::Tensor actIn,
                                  poplar::Tensor weights,
                                  poplar::program::Sequence &prog,
                                  const std::string partialsTypeStr,
                                  const std::string &debugPrefix);

/**
 * Perform the feedback part of the RNN. The feedback part of the RNN must be
 * preceded by the feedforward part of the RNN to complete the RNN
 * \see rnnForwardWeightInput
 *
 * The following definitions are used below:
 *   sequenceSize if the sequence size
 *   batchSize is the batchSize
 *   inputSize is the size of the input for each sequence
 *   outputSize is the size of the output for each sequence
 *
 * \param graph           Graph object
 * \param feedFwdIn       Input to this function (output from feedforward part
 *                        of the RNN of shape
 *                        {sequenceSize, batchSize, outputSize}
 * \param initState       The initial state of the RNN (i.e. the previous
 *                        output) with shape {batchSize, outputSize}
 * \param feedbackWeights Feedback weights with shape {outputSize, outputsize}
 * \param biases          Biases of shape {outputSize}.
 * \prog                  Program sequence to which any programs added by this
 *                        function are appended to
 * \nonLinearityType      Non linearity used for the output activations
 * \partialsTypeStr       Data type for intermediates
 * \debugPrefix           Debug prefix string
 *
 * \return                Output of the RNN of shape
 *                        {outputSize, batchSize, outputSize}
 */
poplar::Tensor forwardIterate(poplar::Graph &graph,
                              poplar::Tensor feedFwdIn,
                              poplar::Tensor initState,
                              poplar::Tensor feedbackWeights,
                              poplar::Tensor biases,
                              poplar::program::Sequence &prog,
                              popnn::NonLinearityType nonLinearityType,
                              const std::string partialsTypeStr,
                              const std::string &debugPrefix);

} // namespace rnn
} // namespace popnn

#endif  // __popnn_Recurrent_hpp__
