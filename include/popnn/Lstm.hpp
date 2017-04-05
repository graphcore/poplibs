#ifndef _popnn_Lstm_hpp_
#define _popnn_Lstm_hpp_

#include <popnn/LstmDef.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace lstm {

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize);


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
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, inputSize}
 * \param weightsOutput Input weights for each of the unit in the cell of shape
 *                      {BASIC_LSTM_CELL_NUM_UNITS, outputSize, outputSize}
 * \param cellState     Updated cell state of the LSTM cell after sequenceSize
 *                      steps of shape {batchSize, outputSize}
 * \param prog          Program sequence
 * \param partialsTypeStr Intermediate data type used in operations
 * \param debugPrefix   String used as prefix for compute sets
 *
 * \return output tensor of shape {outputSize, batchSize, outputSize}
 */
poplar::Tensor basicLstmCellForwardPass(poplar::Graph  &graph,
                                        poplar::Tensor in,
                                        poplar::Tensor bBiases,
                                        poplar::Tensor prevOutput,
                                        poplar::Tensor weightsInput,
                                        poplar::Tensor weightsOutput,
                                        poplar::Tensor cellState,
                                        poplar::program::Sequence &prog,
                                        const std::string partialsTypeStr,
                                        const std::string &debugPrefix);
} // namespace lstm
} // namespave popnn
#endif  // _popnn_Lstm_hpp_
