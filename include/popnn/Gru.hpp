// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popnn_Gru_hpp
#define popnn_Gru_hpp

#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/GruDef.hpp>

namespace popnn {
namespace gru {

/** Structure representing the parameters of the GRU.
 */
struct GruParams {
  // The datatype of the GRU
  poplar::Type dataType;
  // The batch size
  std::size_t batchSize;
  // The number of time steps in the sequence of the GRU
  std::size_t timeSteps;
  // The number of neurons for the input and output layer
  std::vector<std::size_t> layerSizes;
  // If true the GRU function returns the entire sequence of outputs,
  // otherwise it returns just the final output.
  bool outputFullSequence = true;
  // If this parameter is set to false then the GRU will skip the
  // calculation of the gradients of the inputs.
  bool calcInputGradients = true;
  GruParams() = default;
  GruParams(poplar::Type dataType, std::size_t batchSize, std::size_t timeSteps,
            std::vector<std::size_t> layerSizes);
  GruParams(const GruParams &other);
};

uint64_t getBasicGruCellFwdFlops(const GruParams &params);

uint64_t getBasicGruCellBwdFlops(const GruParams &params);

uint64_t getBasicGruCellWuFlops(const GruParams &params);

/** Create an input tensor of shape [numSteps, batchSize, inputSize] which is
 *  optimally mapped to multiply the whole input sequence in a single matrix
 *  multiply operation
 *
 * **GRU options**
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive)
 *
 *      See createWeights().
 *
 *    * `inferenceOnly` (true, false) [=true]
 *
 *      Sets convolution pass to INFERENCE_FWD if true; TRAINING_FWD otherwise.
 *      See createWeights().
 *
 *    * `partialsType` (half, float) [=float]
 *
 *      See createWeights().
 *
 * \param graph           Graph object
 * \param params          The GRU parameters
 * \param name            String annotation
 * \param options         Any implementation/debug options for the GRU
 * \param planningCache   A poplin matrix multiply planning cache
 *
 * \return                A tensor created in the graph of shape:
 *                        [timeSteps, batchSize, inputSize]
 */
poplar::Tensor
createInput(poplar::Graph &graph, const GruParams &params,
            const std::string &name, const poplar::OptionFlags &options = {},
            poplin::matmul::PlanningCache *planningCache = nullptr);

poplar::Tensor createInitialState(poplar::Graph &graph, const GruParams &params,
                                  const std::string &debugPrefix,
                                  const poplar::OptionFlags &options,
                                  poplin::matmul::PlanningCache *cache);
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
 *  of a GRU. Returns the inputWeights and outputWeights.
 */
std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const GruParams &params,
                    const std::string &name,
                    const poplar::OptionFlags &options = {},
                    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the weights biases.
 */
poplar::Tensor
createWeightsBiases(poplar::Graph &graph, const GruParams &params,
                    const std::string &name,
                    const poplar::OptionFlags &options = {},
                    poplin::matmul::PlanningCache *planningCache = nullptr);

/** Create the weights (both kernel and biases) used to weight the input
 *  and output of a GRU.
 */
GruWeights
createWeights(poplar::Graph &graph, const GruParams &params,
              const std::string &name, const poplar::OptionFlags &options = {},
              poplin::matmul::PlanningCache *planningCache = nullptr);

/** Calculate the result of applying a GRU across a sequence
 *
 * The following are the formulas for a GRU cell:
 *
 * r_t = sigmod(w_r * x_t + u_r * h_t-1 + b_r)
 * u_t = sigmod(w_u * x_t + u_u * h_t-1 + b_u)
 * c_t = tanh  (w_c * x_t + u_c * (r_t x h_t-1) + b_c)
 * h_t = u_t x h_t-1 + (1 - u_t) x c_t
 *
 * notes: * is matrix multiplication
 *        x is Hadamard product
 *
 * The GRU is run for seqSize steps each with a batch of size batchSize and
 * input size inputSize and output size outputSize. The total number of units
 * within each GRU cell is BASIC_GRU_CELL_NUM_UNITS.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param stateInit          Initial state for the GRU.
 * \param in                 The input tensor to the GRU of dimension
 *                           [timesteps, batch, inputSize]
 * \param weights            The GRU weights structure.
 * \param[out] intermediates Intermediate results that are retained in the
 *                           forward pass of training for use in the backward
 *                           pass. It includes the data for reset gate, update
 *                           gate, candidate, and output if outputFullSequence
 *                           is false. This argument should be set to null if we
 *                           are only doing inference.
 * \param fwdProg            Program sequence.
 * \param debugPrefix        String used as prefix for compute sets.
 * \param options            GRU implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The output of the GRU.
 *         Depending on the outputFullSequence parameter the output tensor is
 *         either the output of the last timestep in the shape
 *         [batch, outputSize] or it is the sequence of outputs for every
 *         timestep in the shape [timesteps, batch, outputSize]
 */
poplar::Tensor gruFwd(poplar::Graph &graph, const GruParams &params,
                      const poplar::Tensor &stateInit, const poplar::Tensor &in,
                      const GruWeights &weights, poplar::Tensor *intermediates,
                      poplar::program::Sequence &fwdProg,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {},
                      poplin::matmul::PlanningCache *planningCache = nullptr);

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
 * \param fwdInputSeq         The input tensor to the GRU of shape:
 *                            [timesteps, batch, inputSize]
 * \param fwdOutput           The output tensor from the forward pass. Depending
 *                            on the outputFullSequence parameter this is either
 *                            the output for the last timestep or it is a
 *                            sequence of outputs for each timestep.
 * \param gradLayerNext       The gradients of the output. Depending on the
 *                            outputFullSequence parameter this is either the
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
 * \param debugPrefix        String used as prefix for compute sets.
 * \param options            GRU implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor
gruBwd(poplar::Graph &graph, const GruParams &params,
       poplar::program::Sequence &prog, const poplar::Tensor &fwdOutputInit,
       const poplar::Tensor &fwdIntermediatesSeq, const GruWeights &weights,
       const poplar::Tensor &fwdInputSeq, const poplar::Tensor &fwdOutput,
       const poplar::Tensor &gradLayerNext, poplar::Tensor *inputGrad,
       poplar::Tensor *bwdIntermediates, const std::string &debugPrefix,
       const poplar::OptionFlags &options_,
       poplin::matmul::PlanningCache *planningCache);

/**
 * Run a standalone weight update pass. Takes intermediates and gradients from
 * the backward pass and calculates and returns weight deltas.
 *
 * \param graph            Graph to which the GRU cell belongs.
 * \param params           The parameters of the GRU.
 * \param prog             Program sequence to add operations to.
 * \param fwdOutputInit    Forward output tensor for initial step.
 * \param fwdIntermediates Intermediate results from the forward pass.
 * \param bwdIntermediates Intermediate results from the backward pass.
 * \param weights          The GRU weights structure.
 * \param input            The input tensor to the GRU of shape:
 *                          [timesteps, batch, inputSize]
 * \param output           The output tensor from the forward pass. Depending
 *                         on the outputFullSequence parameter this is either
 *                         the output for the last timestep or it is a
 *                         sequence of outputs for each timestep.
 * \param debugPrefix      String used as a prefix to compute sets and
 *                         tensors added to the graph.
 * \param options          GRU implementation options. See createInput().
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
                 const poplar::Tensor &output, const std::string &debugPrefix,
                 const poplar::OptionFlags &options_,
                 poplin::matmul::PlanningCache *planningCache);

/**
 * Run a combined GRU backward and weight update pass. Use this combined
 * backward and weight update pass in preference to `gruBwd` and `gruWU`
 * separately in order to allow the most efficient implementation to be chosen
 * if you do not need to split the operation.
 *
 * \param graph              Graph to which the GRU cell belongs.
 * \param params             The parameters of the GRU.
 * \param prog               Program sequence.
 * \param fwdOutputInit      Forward output tensor for initial step.
 * \param fwdIntermediates   Intermediates results from the forward pass.
 * \param weights            The GRU weights structure.
 * \param input              The input tensor to the GRU of shape:
 *                           [timesteps, batch, inputSize]
 * \param output             The output tensor from the forward pass. Depending
 *                           on the outputFullSequence parameter this is either
 *                           the output for the last timestep or it is a
 *                           sequence of outputs for each timestep.
 * \param outputGrad         The gradients of the output. Depending on the
 *                           outputFullSequence parameter this is either the
 *                           gradient of the output for the last timestep or it
 *                           is a sequence output gradients for each timestep.
 * \param[out] *inputGrad    The gradients of the inputs - may be null if
 *                           this information is not required.
 * \param weightsGrad        A set of weight deltas to sum with weights.
 * \param debugPrefix        String used as prefix for compute sets.
 * \param options            GRU implementation options. See createInput().
 * \param planningCache      The matmul planning cache.
 *
 * \return The gradient of the initial output.
 */
poplar::Tensor
gruBwdWithWU(poplar::Graph &graph, const GruParams &params,
             poplar::program::Sequence &prog,
             const poplar::Tensor &fwdOutputInit,
             const poplar::Tensor &fwdIntermediates, const GruWeights &weights,
             const poplar::Tensor &input, const poplar::Tensor &output,
             const poplar::Tensor &outputGrad, poplar::Tensor *inputGrad,
             GruWeights &weightsGrad, const std::string &debugPrefix,
             const poplar::OptionFlags &options_,
             poplin::matmul::PlanningCache *planningCache);

} // namespace gru
} // namespace popnn

#endif // popnn_Gru_hpp
