#include <poplib_test/GeneralMatrixMultiply.hpp>
#include <poplib_test/GeneralMatrixAdd.hpp>
#include <poplib_test/Lstm.hpp>
#include <poplib_test/NonLinearity.hpp>
#include <boost/multi_array.hpp>
#include <cassert>

using IndexRange = boost::multi_array_types::index_range;
using Array2dRef = boost::multi_array_ref<double, 2>;
using Array2d    = boost::multi_array<double, 2>;
using Array3dRef = boost::multi_array_ref<double, 3>;
using Array3d    = boost::multi_array<double, 3>;

/**
 * Process a given unit type within an LSTM given its weights and biases.
 * The non-linearity is also specified although it may be derived from the unit
 */
static void processBasicLstmUnit(const Array2dRef        prevOutput,
                                 const Array2dRef        input,
                                 const Array3dRef        weightsInput,
                                 const Array3dRef        weightsOutput,
                                 const Array2dRef        biases,
                                 Array2dRef              output,
                                 enum BasicLstmCellUnit  lstmUnit,
                                 popnn::NonLinearityType nonLinearityType) {
  const auto batchSize = prevOutput.shape()[0];
  const auto outputSize = prevOutput.shape()[1];

  /* split weight into two parts:
   * 1) part which weighs only the previous output
   * 2) part which weighs only the input
   */
  Array2d weightsOutputUnit = weightsOutput[lstmUnit];
  Array2d weightsInputUnit = weightsInput[lstmUnit];

  poplib_test::gemm::generalMatrixMultiply(prevOutput, weightsOutputUnit,
                                           output, output, 1.0, 0, false,
                                           false);
  poplib_test::gemm::generalMatrixMultiply(input, weightsInputUnit,
                                           output, output, 1.0, 1.0, false,
                                           false);

  /* add bias */
  for (auto b = 0U; b != batchSize; ++b) {
    for (auto i = 0U; i != outputSize; ++i) {
      output[b][i] += biases[lstmUnit][i];
    }
  }

  /* apply non-linearity */
  poplib_test::nonLinearity(nonLinearityType, output);
}

void poplib_test::lstm::basicLstmCellForwardPass(const Array3dRef input,
                                                 const Array2dRef biases,
                                                 const Array2dRef prevOutput,
                                                 const Array3dRef weightsInput,
                                                 const Array3dRef weightsOutput,
                                                 Array2dRef       cellState,
                                                 Array3dRef       output) {
  const auto sequenceSize = output.shape()[0];
  const auto batchSize = output.shape()[1];
  const auto outputSize = output.shape()[2];
#ifndef NDEBUG
  const auto inputSize = input.shape()[2];
#endif
  assert(weightsInput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsInput.shape()[1] == inputSize);
  assert(weightsInput.shape()[2] == outputSize);
  assert(weightsOutput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.shape()[1] == outputSize);
  assert(weightsOutput.shape()[2] == outputSize);
  assert(cellState.shape()[0] == batchSize);
  assert(cellState.shape()[1] == outputSize);
  assert(biases.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(biases.shape()[1] == outputSize);
  assert(prevOutput.shape()[0] == batchSize);
  assert(prevOutput.shape()[1] == outputSize);

  for (auto s = 0U; s != sequenceSize; ++s) {
    Array2d ysm1 = s == 0 ? output[s] : output[s - 1];
    Array2dRef prevOutputThisStep = s == 0 ? prevOutput : ysm1;
    Array2d inputThisStep = input[s];

    /* forget gate */
    Array2d forgetGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(prevOutputThisStep, inputThisStep, weightsInput,
                         weightsOutput, biases, forgetGate,
                         BASIC_LSTM_CELL_FORGET_GATE,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID);

    /* input gate */
    Array2d inputGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(prevOutputThisStep, inputThisStep, weightsInput,
                         weightsOutput, biases, inputGate,
                         BASIC_LSTM_CELL_INPUT_GATE,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID);

    /* new candidate contribution to this cell */
    Array2d candidate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(prevOutputThisStep, inputThisStep, weightsInput,
                         weightsOutput, biases,  candidate,
                         BASIC_LSTM_CELL_CANDIDATE,
                         popnn::NonLinearityType::NON_LINEARITY_TANH);

    /* output gate */
    Array2d outputGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(prevOutputThisStep, inputThisStep, weightsInput,
                         weightsOutput, biases, outputGate,
                         BASIC_LSTM_CELL_OUTPUT_GATE,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID);


    poplib_test::gemm::hadamardProduct(forgetGate, cellState, cellState);
    poplib_test::gemm::hadamardProduct(inputGate, candidate, candidate);
    poplib_test::axpby::add(cellState, candidate, cellState);

    /* need to maintain the cell state for next step */
    Array2d outputThisStep = cellState;

    poplib_test::nonLinearity(popnn::NonLinearityType::NON_LINEARITY_TANH,
                              outputThisStep);
    poplib_test::gemm::hadamardProduct(outputThisStep, outputGate,
                                       outputThisStep);

    output[s] = outputThisStep;
  }
}
