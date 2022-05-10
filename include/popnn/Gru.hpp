// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for gated recurrent units.
 *
 */

#ifndef popnn_Gru_hpp
#define popnn_Gru_hpp

#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/GruDef.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popnn/Rnn.hpp>

namespace popnn {
namespace gru {

/**
 * Get the default order of the gates in a basic GRU cell.
 * The default order is:
 * [Reset gate, Update gate, Candidate].
 */
const std::vector<BasicGruCellUnit> getDefaultBasicGruCellOrder();

/** Structure representing the parameters of the GRU.
 */
struct GruParams {
  rnn::RnnParams rnn;

  // The datatype of the GRU.
  /// \deprecated Use rnn::RnnParams.dataType instead.
  poplar::Type dataType;
  /// The batch size
  /// \deprecated Use rnn::RnnParams.batchSize instead.
  std::size_t batchSize;
  /// The number of time steps in the sequence of the GRU.
  /// \deprecated Use rnn::RnnParams.maxTimeSteps instead.
  std::size_t timeSteps;
  /// The number of neurons for the input and output layer.
  /// \deprecated Use rnn::RnnParams.layerSizes instead.
  std::vector<std::size_t> layerSizes;
  /// If true the GRU function returns the entire sequence of outputs,
  /// otherwise it returns just the final output.
  bool outputFullSequence = true;
  /// If this parameter is set to false then the GRU will skip the
  /// calculation of the gradients of the inputs.
  bool calcInputGradients = true;
  /// The weights and biases for all of the layers being processed are
  /// concatenated in the outermost dimension of the weights and biases tensors.
  /// This option allows you to specify the order of the gates in that outermost
  /// dimension. The default order can be obtained with
  /// getDefaultBasicGruCellOrder().
  std::vector<BasicGruCellUnit> cellOrder = getDefaultBasicGruCellOrder();
  /// Controls whether the reset gate is applied before or after the candidate
  /// weights and biases.
  bool resetAfter = false;
  /// Activation function.
  NonLinearityType activation = NonLinearityType::TANH;
  /// Recurrent activation function.
  NonLinearityType recurrentActivation = NonLinearityType::SIGMOID;

  GruParams(poplar::Type dataType, std::size_t batchSize, std::size_t timeSteps,
            std::vector<std::size_t> layerSizes,
            NonLinearityType activation = NonLinearityType::TANH,
            NonLinearityType recurrentActivation = NonLinearityType::SIGMOID);

  GruParams(poplar::Type dataType, std::size_t batchSize,
            std::size_t maxTimeSteps, const poplar::Tensor &timeSteps,
            std::vector<std::size_t> layerSizes,
            NonLinearityType activation = NonLinearityType::TANH,
            NonLinearityType recurrentActivation = NonLinearityType::SIGMOID);

  GruParams(const GruParams &other);
};

uint64_t getBasicGruCellFwdFlops(const GruParams &params);

uint64_t getBasicGruCellBwdFlops(const GruParams &params);

uint64_t getBasicGruCellWuFlops(const GruParams &params);

/** Create an input tensor of shape [\p numSteps, \p batchSize, \p inputSize],
 *  that is optimally mapped to multiply the whole input sequence in a single
 *  matrix multiply operation.
 *
 * **GRU options**
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive).
 *
 *      See poplin::createWeights() for more information.
 *
 *    * `inferenceOnly` (true, false) [=true]
 *
 *      Sets convolution pass to `INFERENCE_FWD` if true; `TRAINING_FWD`
 *      otherwise. See the `pass` option in poplin::createWeights().
 *
 *    * `partialsType` (half, float) [=float]
 *
 *      See poplin::createWeights() for more information.
 *
 * \param graph           Graph object to add the tensor to.
 * \param params          The GRU parameters.
 * \param debugContext    Optional debug information.
 * \param options         Any implementation/debug options for the GRU.
 * \param planningCache   A poplin matrix multiply planning cache.
 *
 * \return                A tensor created in the graph of shape
 *                        [\p timeSteps, \p batchSize, \p inputSize].
 */
poplar::Tensor createInput(poplar::Graph &graph, const GruParams &params,
                           const poplar::DebugContext &debugContext,
                           const poplar::OptionFlags &options = {},
                           poplin::PlanningCache *planningCache = nullptr);

poplar::Tensor createInitialState(poplar::Graph &graph, const GruParams &params,
                                  const poplar::DebugContext &debugContext,
                                  const poplar::OptionFlags &options,
                                  poplin::PlanningCache *cache);
/**
 * Structure holding all the parameters of a GRU cell, or the
 * deltas for those parameters (depending on the context).
 */
struct GruWeights {
  poplar::Tensor inputWeights;
  poplar::Tensor outputWeights;
  poplar::Tensor biases;
};

/** Create the weights kernel used to weight the input and output
 *  of a GRU. Returns the \p inputWeights and \p outputWeights.
 */
std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const GruParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options = {},
                    poplin::PlanningCache *planningCache = nullptr);

/** Create the weights biases.
 */
poplar::Tensor
createWeightsBiases(poplar::Graph &graph, const GruParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options = {},
                    poplin::PlanningCache *planningCache = nullptr);

/** Create the weights (both kernel and biases) used to weight the input
 *  and output of a GRU.
 */
GruWeights createWeights(poplar::Graph &graph, const GruParams &params,
                         const poplar::DebugContext &debugContext,
                         const poplar::OptionFlags &options = {},
                         poplin::PlanningCache *planningCache = nullptr);

/** Create an attention tensor for AUGRU.
 */
poplar::Tensor createAttention(poplar::Graph &graph, const GruParams &params,
                               const poplar::DebugContext &debugContext,
                               const poplar::OptionFlags &options = {});

/** Calculate the result of applying a GRU across a sequence.
 *
 * The formulas for a GRU cell are:
 *
 * - \f$r_t = \operatorname{sigmoid}(w_r \times x_t + u_r \times h_{t-1} +
 * b_r)\f$
 * - \f$u_t = \operatorname{sigmoid}(w_u \times x_t + u_u \times h_{t-1} +
 * b_u)\f$
 * - \f$c_t = \tanh(w_c \times x_t + u_c \times (r_t \circ h_{t-1}) + b_c)\f$
 * - \f$h_t = u_t \circ h_{t-1} + (1 - u_t) \circ c_t\f$
 *
 * Where:
 *   - \f$\times\f$ is matrix multiplication
 *   - \f$\circ\f$ is Hadamard product
 *
 * The GRU is run for rnn::RnnParams.maxTimeSteps, each with a batch of size
 * \p batchSize and input size \p inputSize and output size \p outputSize. The
 * total number of units within each GRU cell is `BASIC_GRU_CELL_NUM_UNITS`.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param stateInit          Initial state for the GRU.
 * \param in                 The input tensor to the GRU of dimension
 *                           [\p timeSteps, \p batchSize, \p inputSize].
 * \param weights            The GRU weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           forward pass of training for use in the backward
 *                           pass. It includes the data for reset gate, update
 *                           gate, candidate, and output if
 *                           \p outputFullSequence is false.
 *                           This argument should be set to null if we
 *                           are only doing inference.
 * \param fwdProg            Program sequence.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the GRU.
 *         Depending on the \p outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [\p batchSize, \p outputSize] or it is the sequence of outputs for
 *         every timestep in the shape [\p timeSteps, \p batchSize,
 *         \p outputSize].
 */
poplar::Tensor gruFwd(poplar::Graph &graph, const GruParams &params,
                      const poplar::Tensor &stateInit, const poplar::Tensor &in,
                      const GruWeights &weights, poplar::Tensor *intermediates,
                      poplar::program::Sequence &fwdProg,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      poplin::PlanningCache *planningCache = nullptr);

/** Calculate the result of applying a GRU across a sequence.
 *
 * \deprecated Use previously defined gruFwd() instead.
 *
 * The formulas for a GRU cell are:
 *
 *   - \f$r_t = \operatorname{sigmoid}(w_r \times x_t + u_r \times h_{t-1} +
 * b_r)\f$
 *   - \f$u_t = \operatorname{sigmoid}(w_u \times x_t + u_u \times h_{t-1} +
 * b_u)\f$
 *   - \f$c_t = \tanh(w_c \times x_t + u_c \times (r_t \circ h_{t-1}) + b_c)\f$
 *   - \f$h_t = u_t \circ h_{t-1} + (1 - u_t) \circ c_t\f$
 *
 * Where:
 *   - \f$\times\f$ is matrix multiplication
 *   - \f$\circ\f$ is Hadamard product
 *
 * The GRU is run for rnn::RnnParams.maxTimeSteps, each with a batch of size
 * \p batchSize and input size \p inputSize and output size \p outputSize. The
 * total number of units within each GRU cell is `BASIC_GRU_CELL_NUM_UNITS`.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param stateInit          Initial state for the GRU.
 * \param in                 The input tensor to the GRU of dimension
 *                           [\p timeSteps, \p batchSize, \p inputSize].
 * \param realTimeSteps      A tensor containing real timesteps for each
 *                           sequence, of shape [\p batch].
 * \param weights            The GRU weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           forward pass of training for use in the backward
 *                           pass. It includes the data for reset gate, update
 *                           gate, candidate, and output if
 *                           \p outputFullSequence is false.
 *                           This argument should be set to null if we
 *                           are only doing inference.
 * \param fwdProg            Program sequence.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the GRU.
 *         Depending on the \p outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [\p batchSize, \p outputSize] or it is the sequence of outputs for
 *         every timestep in the shape [\p timeSteps, \p batchSize,
 *         \p outputSize].
 */
poplar::Tensor gruFwd(poplar::Graph &graph, const GruParams &params,
                      const poplar::Tensor &stateInit, const poplar::Tensor &in,
                      const poplar::Tensor &realTimeSteps,
                      const GruWeights &weights, poplar::Tensor *intermediates,
                      poplar::program::Sequence &fwdProg,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      poplin::PlanningCache *planningCache = nullptr);

/** Calculate the result of applying an AUGRU across a sequence.
 *
 * The formulas for a AUGRU cell are:
 *
 *   - \f$r_t = sigmod(w_r \times x_t + u_r \times h_{t-1} + b_r)\f$
 *   - \f$u_t = sigmod(w_u \times x_t + u_u \times h_{t-1} + b_u)\f$
 *   - \f$c_t = tanh(w_c \times x_t + u_c \times (r_t \circ h_{t-1}) + b_c)\f$
 *   - \f$u_t = (1 - a_t) \times u_t\f$
 *   - \f$h_t = u_t \circ h_{t-1} + (1 - u_t) \circ c_t\f$
 *
 * Where:
 *   - \f$\times\f$ is matrix multiplication
 *   - \f$\circ\f$ is Hadamard product
 *   - \f$a_t\f$ is a scalar
 *
 * The AUGRU is run for rnn::RnnParams.maxTimeSteps, each with a batch of size
 * \p batchSize and input size \p inputSize and output size \p outputSize. The
 * total number of units within each AUGRU cell is `BASIC_GRU_CELL_NUM_UNITS`.
 *
 * \param graph              Graph to which the AUGRU cell belongs.
 * \param params             The parameters of the AUGRU.
 * \param stateInit          Initial state for the AUGRU.
 * \param in                 The input tensor to the AUGRU of dimension
 *                           [\p timeSteps, \p batchSize, \p inputSize].
 * \param weights            The AUGRU weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           forward pass of training for use in the backward
 *                           pass. It includes the data for reset gate, update
 *                           gate, candidate, and output if
 *                           \p outputFullSequence is false.
 *                           This argument should be set to null if we
 *                           are only doing inference.
 * \param attScores          Attention for each timestep.
 * \param fwdProg            Program sequence.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the GRU.
 *         Depending on the \p outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [\p batchSize, \p outputSize] or it is the sequence of outputs for
 *         every timestep in the shape [\p timeSteps, \p batchSize,
 *         \p outputSize].
 */
poplar::Tensor auGruFwd(poplar::Graph &graph, const GruParams &params,
                        const poplar::Tensor &stateInit,
                        const poplar::Tensor &in, const GruWeights &weights,
                        poplar::Tensor *intermediates,
                        const poplar::Tensor &attScores,
                        poplar::program::Sequence &fwdProg,
                        const poplar::DebugContext &debugContext = {},
                        const poplar::OptionFlags &options = {},
                        poplin::PlanningCache *planningCache = nullptr);

/** Calculate the result of applying an AUGRU across a sequence.
 *
 * \deprecated Use previously defined auGruFwd() instead.
 *
 * The formulas for a AUGRU cell are:
 *
 *   - \f$r_t = sigmod(w_r \times x_t + u_r \times h_{t-1} + b_r)\f$
 *   - \f$u_t = sigmod(w_u \times x_t + u_u \times h_{t-1} + b_u)\f$
 *   - \f$c_t = tanh(w_c \times x_t + u_c \times (r_t \circ h_{t-1}) + b_c)\f$
 *   - \f$u_t = (1 - a_t) \times u_t\f$
 *   - \f$h_t = u_t \circ h_{t-1} + (1 - u_t) \circ c_t\f$
 *
 * Where:
 *   - \f$\times\f$ is matrix multiplication
 *   - \f$\circ\f$ is Hadamard product
 *   - \f$a_t\f$ is a scalar
 *
 * The AUGRU is run for rnn::RnnParams.maxTimeSteps, each with a batch of size
 * \p batchSize and input size \p inputSize and output size \p outputSize. The
 * total number of units within each AUGRU cell is `BASIC_GRU_CELL_NUM_UNITS`.
 *
 * \param graph              Graph to which the AUGRU cell belongs.
 * \param params             The parameters of the AUGRU.
 * \param stateInit          Initial state for the AUGRU.
 * \param in                 The input tensor to the AUGRU of dimension
 *                           [\p timeSteps, \p batchSize, \p inputSize].
 * \param realTimeSteps      A tensor containing real timesteps for each
 *                           sequence, of shape [\p batch].
 * \param weights            The AUGRU weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           forward pass of training for use in the backward
 *                           pass. It includes the data for reset gate, update
 *                           gate, candidate, and output if
 *                           \p outputFullSequence is false.
 *                           This argument should be set to null if we
 *                           are only doing inference.
 * \param attScores          Attention for each timestep.
 * \param fwdProg            Program sequence.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the GRU.
 *         Depending on the \p outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [\p batchSize, \p outputSize] or it is the sequence of outputs for
 *         every timestep in the shape [\p timeSteps, \p batchSize,
 *         \p outputSize].
 */
poplar::Tensor
auGruFwd(poplar::Graph &graph, const GruParams &params,
         const poplar::Tensor &stateInit, const poplar::Tensor &in,
         const poplar::Tensor &realTimeSteps, const GruWeights &weights,
         poplar::Tensor *intermediates, const poplar::Tensor &attScores,
         poplar::program::Sequence &fwdProg,
         const poplar::DebugContext &debugContext = {},
         const poplar::OptionFlags &options = {},
         poplin::PlanningCache *planningCache = nullptr);

/**
 *  Run GRU backward pass. The backward pass executes in reverse order compared
 *  to the forward pass. If the forward steps for a GRU layer are sf =
 *  {0, 1, 2, ..., S - 1} then the backward steps run for sb = {S - 1, S - 2,
 *  .... , 1, 0}.
 *
 * \param graph               Graph to which the GRU cell belongs.
 * \param params              The parameters of the GRU.
 * \param prog                Program sequence.
 * \param fwdOutputInit       Forward output tensor for initial step.
 * \param fwdIntermediatesSeq Intermediates results from the forward pass.
 * \param weights             The GRU weights structure.
 * \param fwdInputSeq         The input tensor to the GRU, of shape
 *                            [\p timeSteps, \p batchSize, \p inputSize]
 * \param fwdOutput           The output tensor from the forward pass. Depending
 *                            on the \p outputFullSequence parameter this is
 *                            either the output for the last timestep or it is a
 *                            sequence of outputs for each timestep.
 * \param gradLayerNext       The gradients of the output. Depending on the
 *                            \p outputFullSequence parameter this is either the
 *                            gradient of the output for the last timestep or
 *                            it is a sequence output gradients for each
 *                            timestep.
 * \param[out] *inputGrad     The gradients of the inputs - may be null if
 *                            this information is not required.
 * \param[out] *bwdIntermediates Intermediates gradients that are retained in
 *                           the backward pass of training for use in the
 *                           weight update. It includes the derivatives for
 *                           reset gate, update gate, and candidate.
 *                           This argument should be set to null if you do not
 *                           need to calculate weight deltas.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor gruBwd(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediatesSeq, const GruWeights &weights,
    const poplar::Tensor &fwdInputSeq, const poplar::Tensor &fwdOutput,
    const poplar::Tensor &gradLayerNext, poplar::Tensor *inputGrad,
    poplar::Tensor *bwdIntermediates, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

/** Run GRU backward pass.
 *
 *  \deprecated Use previously defined popnn::gruBwd() instead.
 *
 *  The backward pass executes in reverse order compared
 *  to the forward pass. If the forward steps for a GRU layer are sf =
 *  {0, 1, 2, ..., S - 1} then the backward steps run for sb = {S - 1, S - 2,
 *  .... , 1, 0}.
 *
 * \param graph               Graph to which the GRU cell belongs.
 * \param params              The parameters of the GRU.
 * \param prog                Program sequence.
 * \param fwdOutputInit       Forward output tensor for initial step.
 * \param fwdIntermediatesSeq Intermediates results from the forward pass.
 * \param weights             The GRU weights structure.
 * \param realTimeSteps       A tensor containing real timesteps for each
 *                            sequence, with shape [\p batch].
 * \param fwdInputSeq         The input tensor to the GRU, of shape
 *                            [\p timeSteps, \p batchSize, \p inputSize]
 * \param fwdOutput           The output tensor from the forward pass. Depending
 *                            on the \p outputFullSequence parameter this is
 *                            either the output for the last timestep or it is a
 *                            sequence of outputs for each timestep.
 * \param gradLayerNext       The gradients of the output. Depending on the
 *                            \p outputFullSequence parameter this is either the
 *                            gradient of the output for the last timestep or
 *                            it is a sequence output gradients for each
 *                            timestep.
 * \param[out] *inputGrad     The gradients of the inputs - may be null if
 *                            this information is not required.
 * \param[out] *bwdIntermediates Intermediates gradients that are retained in
 *                           the backward pass of training for use in the
 *                           weight update. It includes the derivatives for
 *                           reset gate, update gate, and candidate.
 *                           This argument should be set to null if you do not
 *                           need to calculate weight deltas.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor
gruBwd(poplar::Graph &graph, const GruParams &params,
       poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
       const poplar::Tensor &fwdIntermediatesSeq, const GruWeights &weights,
       const poplar::Tensor &fwdInputSeq, const poplar::Tensor &realTimeSteps,
       const poplar::Tensor &fwdOutput, const poplar::Tensor &gradLayerNext,
       poplar::Tensor *inputGrad, poplar::Tensor *bwdIntermediates,
       const poplar::DebugContext &debugContext,
       const poplar::OptionFlags &options_,
       poplin::PlanningCache *planningCache);

/**
 *  Run AUGRU backward pass. The backward pass executes in reverse order
 * compared to the forward pass. If the forward steps for an AUGRU layer are
 *  sf = {0, 1, 2, ..., S - 1} then the backward steps run for
 *  sb = {S - 1, S - 2, .... , 1, 0}.
 *
 * \param graph               Graph to which the AUGRU cell belongs.
 * \param params              The parameters of the AUGRU.
 * \param prog                Program sequence.
 * \param fwdOutputInit       Forward output tensor for initial step.
 * \param fwdIntermediatesSeq Intermediates results from the forward pass.
 * \param weights             The AUGRU weights structure.
 * \param fwdInputSeq         The input tensor to the AUGRU, of shape
 *                            [\p timeSteps, \p batchSize, \p inputSize]
 * \param fwdOutput           The output tensor from the forward pass. Depending
 *                            on the \p outputFullSequence parameter this is
 *                            either the output for the last timestep or it is a
 *                            sequence of outputs for each timestep.
 * \param gradLayerNext       The gradients of the output. Depending on the
 *                            \p outputFullSequence parameter this is either the
 *                            gradient of the output for the last timestep or
 *                            it is a sequence output gradients for each
 *                            timestep.
 * \param[out] *inputGrad     The gradients of the inputs - may be null if
 *                            this information is not required.
 * \param[out] *bwdIntermediates Intermediates gradients that are retained in
 *                           the backward pass of training for use in the
 *                           weight update. It includes the derivatives for
 *                           reset gate, update gate, and candidate.
 *                           This argument should be set to null if you do not
 *                           need to calculate weight deltas.
 * \param attentions         Attentions for each timestep.
 * \param[out] attentionsGrad Gradients for attentions.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor auGruBwd(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediatesSeq, const GruWeights &weights,
    const poplar::Tensor &fwdInputSeq, const poplar::Tensor &fwdOutput,
    const poplar::Tensor &gradLayerNext, poplar::Tensor *inputGrad,
    poplar::Tensor *bwdIntermediates, const poplar::Tensor &attentions,
    poplar::Tensor *attentionsGrad, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

/** Run AUGRU backward pass.
 *
 *  \deprecated Use previously defined auGruBwd() instead.
 *
 *  The backward pass executes in reverse order
 *  compared to the forward pass. If the forward steps for an AUGRU layer are
 *  sf = {0, 1, 2, ..., S - 1} then the backward steps run for
 *  sb = {S - 1, S - 2, .... , 1, 0}.
 *
 * \param graph               Graph to which the AUGRU cell belongs.
 * \param params              The parameters of the AUGRU.
 * \param prog                Program sequence.
 * \param fwdOutputInit       Forward output tensor for initial step.
 * \param fwdIntermediatesSeq Intermediates results from the forward pass.
 * \param weights             The AUGRU weights structure.
 * \param fwdInputSeq         The input tensor to the AUGRU, of shape
 *                            [\p timeSteps, \p batchSize, \p inputSize]
 * \param realTimeSteps       A tensor containing real timesteps for each
 *                            sequence, of shape [\p batch].
 * \param fwdOutput           The output tensor from the forward pass. Depending
 *                            on the \p outputFullSequence parameter this is
 *                            either the output for the last timestep or it is a
 *                            sequence of outputs for each timestep.
 * \param gradLayerNext       The gradients of the output. Depending on the
 *                            \p outputFullSequence parameter this is either the
 *                            gradient of the output for the last timestep or
 *                            it is a sequence output gradients for each
 *                            timestep.
 * \param[out] *inputGrad     The gradients of the inputs - may be null if
 *                            this information is not required.
 * \param[out] *bwdIntermediates Intermediates gradients that are retained in
 *                           the backward pass of training for use in the
 *                           weight update. It includes the derivatives for
 *                           reset gate, update gate, and candidate.
 *                           This argument should be set to null if you do not
 *                           need to calculate weight deltas.
 * \param attentions         Attentions for each timestep.
 * \param[out] attentionsGrad Gradients for attentions.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor
auGruBwd(poplar::Graph &graph, const GruParams &params,
         poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
         const poplar::Tensor &fwdIntermediatesSeq, const GruWeights &weights,
         const poplar::Tensor &fwdInputSeq, const poplar::Tensor &realTimeSteps,
         const poplar::Tensor &fwdOutput, const poplar::Tensor &gradLayerNext,
         poplar::Tensor *inputGrad, poplar::Tensor *bwdIntermediates,
         const poplar::Tensor &attentions, poplar::Tensor *attentionsGrad,
         const poplar::DebugContext &debugContext,
         const poplar::OptionFlags &options_,
         poplin::PlanningCache *planningCache);

/**
 * Run a standalone weight update pass. Takes intermediates and gradients from
 * the backward pass and calculates and returns weight deltas.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in `fwdIntermediates`, in
 *       order for the weights to be correctly updated.
 *
 * \param graph            Graph to which the GRU cell belongs.
 * \param params           The parameters of the GRU.
 * \param prog             Program sequence to add operations to.
 * \param fwdOutputInit    Forward output tensor for initial step.
 * \param fwdIntermediates Intermediate results from the forward pass.
 * \param bwdIntermediates Intermediate results from the backward pass.
 * \param weights          The GRU weights structure.
 * \param input            The input tensor to the GRU, of shape
 *                          [\p timeSteps, \p batchSize, \p inputSize]
 * \param output           The output tensor from the forward pass. Depending
 *                         on the \p outputFullSequence parameter this is either
 *                         the output for the last timestep or it is a
 *                         sequence of outputs for each timestep.
 * \param debugContext     Optional debug information.
 * \param options          GRU implementation options.
 *                         See createInput().
 * \param planningCache    The matmul planning cache.
 *
 * \return A set of weight gradients to sum with weights.
 */
GruWeights gruWU(poplar::Graph &graph, const GruParams &params,
                 poplar::program::Sequence &prog,
                 const poplar::Tensor &fwdOutputInit,
                 const poplar::Tensor &fwdIntermediates,
                 const poplar::Tensor &bwdIntermediates,
                 const GruWeights &weights, const poplar::Tensor &input,
                 const poplar::Tensor &output,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options_,
                 poplin::PlanningCache *planningCache);

/**
 * Run a standalone weight update pass. Takes intermediates and gradients from
 * the backward pass and calculates and returns weight deltas.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in \p fwdIntermediates, in
 *       order for the weights to be correctly updated.
 *
 * \param graph            Graph to which the GRU cell belongs.
 * \param params           The parameters of the GRU.
 * \param prog             Program sequence to add operations to.
 * \param fwdOutputInit    Forward output tensor for initial step.
 * \param fwdIntermediates Intermediate results from the forward pass.
 * \param bwdIntermediates Intermediate results from the backward pass.
 * \param weights          The GRU weights structure.
 * \param input            The input tensor to the GRU, of shape
 *                          [\p timeSteps, \p batchSize, \p inputSize]
 * \param output           The output tensor from the forward pass. Depending
 *                         on the \p outputFullSequence parameter this is either
 *                         the output for the last timestep or it is a
 *                         sequence of outputs for each timestep.
 * \param debugContext      Optional debug information.
 * \param options          GRU implementation options.
 *                         See createInput().
 * \param planningCache    The matmul planning cache.
 *
 * \return A set of weight gradients to sum with weights.
 */
GruWeights auGruWU(poplar::Graph &graph, const GruParams &params,
                   poplar::program::Sequence &prog,
                   const poplar::Tensor &fwdOutputInit,
                   const poplar::Tensor &fwdIntermediates,
                   const poplar::Tensor &bwdIntermediates,
                   const GruWeights &weights, const poplar::Tensor &input,
                   const poplar::Tensor &output,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_,
                   poplin::PlanningCache *planningCache);

/**
 * Run a combined GRU backward and weight update pass. Use this combined
 * backward and weight update pass in preference to gruBwd() and gruWU()
 * separately in order to allow the most efficient implementation to be chosen
 * if you do not need to split the operation.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in `fwdIntermediates`, in
 *       order for the weights to be correctly updated.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param prog               Program sequence.
 * \param fwdOutputInit      Forward output tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The GRU weights structure.
 * \param input              The input tensor to the GRU, of shape
 *                           [\p timeSteps, \p batchSize, \p inputSize]
 * \param output             The output tensor from the forward pass. Depending
 *                           on the \p outputFullSequence parameter this is
 *                           either the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or it
 *                           is a sequence output gradients for each timestep.
 * \param[out] *inputGrad    The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor gruBwdWithWU(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediates, const GruWeights &weights,
    const poplar::Tensor &input, const poplar::Tensor &output,
    const poplar::Tensor &outputGrad, poplar::Tensor *inputGrad,
    GruWeights &weightsGrad, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

/** Run a combined GRU backward and weight update pass.
 *
 * \deprecated Use previously defined gruBwdWithWU() instead.
 *
 * Use this combined backward and weight update pass in preference to gruBwd()
 * and gruWU() separately in order to allow the most efficient implementation to
 * be chosen if you do not need to split the operation.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in `fwdIntermediates`, in
 *       order for the weights to be correctly updated.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param prog               Program sequence.
 * \param fwdOutputInit      Forward output tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The GRU weights structure.
 * \param input              The input tensor to the GRU, of shape
 *                           [\p timeSteps, \p batchSize, \p inputSize]
 * \param realTimeSteps      A tensor containing real timesteps for each
 *                           sequence, of shape [\p batch].
 * \param output             The output tensor from the forward pass. Depending
 *                           on the \p outputFullSequence parameter this is
 *                           either the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or it
 *                           is a sequence output gradients for each timestep.
 * \param[out] *inputGrad    The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor gruBwdWithWU(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediates, const GruWeights &weights,
    const poplar::Tensor &input, const poplar::Tensor &realTimeSteps,
    const poplar::Tensor &output, const poplar::Tensor &outputGrad,
    poplar::Tensor *inputGrad, GruWeights &weightsGrad,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

/**
 * Run a combined AUGRU backward and weight update pass. Use this combined
 * backward and weight update pass in preference to auGruBwd() and auGruWU()
 * separately in order to allow the most efficient implementation to be chosen
 * if you do not need to split the operation.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in \p fwdIntermediates, in
 *       order for the weights to be correctly updated.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param prog               Program sequence.
 * \param fwdOutputInit      Forward output tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The GRU weights structure.
 * \param input              The input tensor to the GRU, of shape
 *                           [\p timeSteps, \p batchSize, \p inputSize]
 * \param output             The output tensor from the forward pass. Depending
 *                           on the \p outputFullSequence parameter this is
 *                           either the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or it
 *                           is a sequence output gradients for each timestep.
 * \param[out] *inputGrad    The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param attentions         Attention for each timestep.
 * \param[out] attentionsGrad Gradients for attentions.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor auGruBwdWithWU(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediates, const GruWeights &weights,
    const poplar::Tensor &input, const poplar::Tensor &output,
    const poplar::Tensor &outputGrad, poplar::Tensor *inputGrad,
    GruWeights &weightsGrad, const poplar::Tensor &attentions,
    poplar::Tensor *attentionsGrad, const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

/** Run a combined AUGRU backward and weight update pass.
 *
 * \deprecated Use previously defined auGruBwdWithWU() instead.
 *
 * Use this combined backward and weight update pass in preference to auGruBwd()
 * and auGruWU() separately in order to allow the most efficient implementation
 * to be chosen if you do not need to split the operation.
 *
 * Note: If the timestep limit is variable, the entries above the given time
 *       step limit must be explicitly set to zero in `fwdIntermediates`, in
 *       order for the weights to be correctly updated.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param prog               Program sequence.
 * \param fwdOutputInit      Forward output tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The GRU weights structure.
 * \param input              The input tensor to the GRU, of shape
 *                           [\p timeSteps, \p batchSize, \p inputSize].
 * \param realTimeSteps      A tensor containing real timesteps for each
 *                           sequence, of shape [\p batch].
 * \param output             The output tensor from the forward pass. Depending
 *                           on the \p outputFullSequence parameter this is
 *                           either the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           \p outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or it
 *                           is a sequence output gradients for each timestep.
 * \param[out] *inputGrad    The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param attentions         Attention for each timestep.
 * \param[out] attentionsGrad Gradients for attentions.
 * \param debugContext       Optional debug information.
 * \param options            GRU implementation options.
 *                           See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor auGruBwdWithWU(
    poplar::Graph &graph, const GruParams &params,
    poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
    const poplar::Tensor &fwdIntermediates, const GruWeights &weights,
    const poplar::Tensor &input, const poplar::Tensor &realTimeSteps,
    const poplar::Tensor &output, const poplar::Tensor &outputGrad,
    poplar::Tensor *inputGrad, GruWeights &weightsGrad,
    const poplar::Tensor &attentions, poplar::Tensor *attentionsGrad,
    const poplar::DebugContext &debugContext,
    const poplar::OptionFlags &options_, poplin::PlanningCache *planningCache);

} // namespace gru
} // namespace popnn

#endif // popnn_Gru_hpp
