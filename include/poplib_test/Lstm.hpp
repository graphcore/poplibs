#ifndef __poplib_test_Lstm_hpp_
#define __poplib_test_Lstm_hpp_

#include <popnn/LstmDef.hpp>
#include <boost/multi_array.hpp>

namespace poplib_test {
namespace lstm {

/**
 * Compute the cell state and output of a basic non-fused LSTM cell (without
 * peephole connections). The cell state and outputs are concatented into a
 * single dimension.
 *
 * \param input               Input to the LSTM cell of dimension
 *                            [sequenceSize][batchSize][inputSize]
 * \param weightsInput        Weights in the LSTM cell which weigh the input
 *                            sequence. It is of dimension
 *                            [NUM_LSTM_UNITS][inputSize][outputSize]
 * \param weightsOutput       Weights in the LSTM cell which weigh the output
 *                            sequence. It is of dimension
 *                            [NUM_LSTM_UNITS][outputSize][outputSize]
 * \param biases              Biases in the LSTM cell of dimension
 *                            [NUM_LSTM_UNITS][outputSize]
 * \param prevOutput          Previous output used in the first time step
 *                            [bastchSize][outputSize]
 * \param cellState           Cell state, Is updated at every step
 *                            [batchSize][outputSize]
 * \param output              The output for all thecsequence steps of dimension
 *                            [sequenceSize][batchSize][outputSize]
 */
void basicLstmCellForwardPass(
                   const boost::multi_array_ref<double, 3> input,
                   const boost::multi_array_ref<double, 2> biases,
                   const boost::multi_array_ref<double, 2> prevOutput,
                   const boost::multi_array_ref<double, 3> weightsInput,
                   const boost::multi_array_ref<double, 3> weightsOutput,
                   boost::multi_array_ref<double, 2>       cellState,
                   boost::multi_array_ref<double, 3>       output);

} // poplib_lstm
} // ref

#endif // __poplib_test_Lstm_hpp_
