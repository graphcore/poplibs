#ifndef _popnn_Lstm_hpp_
#define _popnn_Lstm_hpp_

#include <popnn/LstmDef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace lstm {

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize);


poplar::Tensor createInput(poplar::Graph &graph, const std::string &dType,
                           unsigned sequenceSize, unsigned batchSize,
                           unsigned inputSize, unsigned outputSize,
                           const std::string &name = "");

/** Create the weights used to weight the input of an lstm. If the
 *  'preweights' parameter is set to true then weights are created that
 *  are optimal when using the calcSequenceWeightedInputs() function.
 */
poplar::Tensor
createWeightsInput(poplar::Graph &graph, const std::string &dType,
                   const std::string &partialsType,
                   const poplar::Tensor &prevAct, unsigned outputSize,
                   bool preweights);

poplar::Tensor
createWeightsOutput(poplar::Graph &graph, const std::string &dType,
                    const std::string &partialsType,
                    const poplar::Tensor &cellState,
                    unsigned outputSize);

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
 * batchSize and input size inputSize and outputSize outputSize. The total
 * number of units within each LSTM cell is BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param in            Input of shape {sequenceSize, batchSize, inputSize}
 * \param biases        Biases for each of the units of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize}
 * \param weightsInput  Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, inputSize, outputSize}
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param cellState     Updated cell state of the LSTM cell after sequenceSize
 *                      steps of shape {batchSize, outputSize}
 * \param prog          Program sequence
 * \param partialsTypeStr Intermediate data type used in operations
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output tensor of shape {sequenceSize, batchSize, outputSize}
 */
poplar::Tensor basicLstmCellForwardPass(poplar::Graph &graph,
                                        const poplar::Tensor &in,
                                        const poplar::Tensor &bBiases,
                                        const poplar::Tensor &prevOutput,
                                        const poplar::Tensor &weightsInput,
                                        const poplar::Tensor &weightsOutput,
                                        const poplar::Tensor &cellState,
                                        poplar::program::Sequence &prog,
                                        const std::string partialsTypeStr,
                                        const std::string &debugPrefix);

/** Precalculate weighted inputs for the entire sequence to be passed to a
 *  LSTM cell.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param in            Input of shape {sequenceSize, batchSize, inputSize}
 * \param weightsInput  Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, inputSize, outputSize}
 * \param prog          Program sequence
 * \param partialsTypeStr Intermediatedata type used in operations
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
                           const std::string partialsTypeStr,
                           const std::string &debugPrefix);


/** Calculate the result of apply an LSTM across a sequence given that
 *  the inputs have already had weights applied to them.
 *
 * The LSTM may be run for a given step size stepSize each with a batch of size
 * batchSize and input size inputSize and outputSize outputSize. The total
 * number of units within each LSTM cell is BASIC_LSTM_CELL_NUM_UNITS.
 *
 * \param graph         Graph to which the LSTM cell belongs to
 * \param weightedIn    Input of shape {sequenceSize, batchSize, outputSize}
 * \param biases        Biases for each of the units of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize}
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param cellState     Updated cell state of the LSTM cell after sequenceSize
 *                      steps of shape {batchSize, outputSize}
 * \param prog          Program sequence
 * \param partialsTypeStr Intermediate data type used in operations
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output tensor of shape {sequenceSize, batchSize, outputSize}
 */
poplar::Tensor
basicLstmCellForwardPassWeightedInputs(poplar::Graph &graph,
                                       const poplar::Tensor &weightedInput,
                                       const poplar::Tensor &bBiases,
                                       const poplar::Tensor &prevOutput,
                                       const poplar::Tensor &weightsOutput,
                                       const poplar::Tensor &cellState,
                                       poplar::program::Sequence &prog,
                                       const std::string partialsTypeStr,
                                       const std::string &debugPrefix);


} // namespace lstm
} // namespave popnn
#endif  // _popnn_Lstm_hpp_
