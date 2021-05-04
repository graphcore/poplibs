// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplibs_support/logging.hpp"
#include <boost/multi_array.hpp>
#include <cassert>
#include <poplibs_test/GeneralMatrixAdd.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Lstm.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <unordered_map>

// Fwd state array indices
#define LSTM_FWD_STATE_FORGET_GATE 2
#define LSTM_FWD_STATE_CAND_TANH 3
#define LSTM_FWD_STATE_INPUT_GATE 4
#define LSTM_FWD_STATE_OUTPUT_GATE 5
#define LSTM_FWD_STATE_OUTPUT_TANH 6

using IndexRange = boost::multi_array_types::index_range;
using Array1dRef = boost::multi_array_ref<double, 1>;
using Array1dRefUNSIGNED = boost::multi_array_ref<unsigned, 1>;
using Array2dRef = boost::multi_array_ref<double, 2>;
using Array2d = boost::multi_array<double, 2>;
using Array3dRef = boost::multi_array_ref<double, 3>;
using Array4dRef = boost::multi_array_ref<double, 4>;
using Array3d = boost::multi_array<double, 3>;

using namespace poplibs_support;
using namespace poplibs_test;

static void matrixZero(boost::multi_array_ref<double, 2> matA) {
  std::fill(matA.data(), matA.data() + matA.num_elements(), 0.0);
}

/**
 * Process a given unit type within an LSTM given its weights and biases.
 * The non-linearity is also specified although it may be derived from the unit
 */
static void processBasicLstmUnit(const Array2dRef prevOutput,
                                 const Array2dRef input,
                                 const Array3dRef weightsInput,
                                 const Array3dRef weightsOutput,
                                 const Array2dRef biases, Array2dRef output,
                                 unsigned lstmUnitOffset,
                                 popnn::NonLinearityType nonLinearityType) {
  const auto batchSize = prevOutput.shape()[0];
  const auto outputSize = prevOutput.shape()[1];

  /* split weight into two parts:
   * 1) part which weighs only the previous output
   * 2) part which weighs only the input
   */
  Array2d weightsOutputUnit = weightsOutput[lstmUnitOffset];
  Array2d weightsInputUnit = weightsInput[lstmUnitOffset];

  gemm::generalMatrixMultiply(prevOutput, weightsOutputUnit, output, output,
                              1.0, 0, false, false);
  gemm::generalMatrixMultiply(input, weightsInputUnit, output, output, 1.0, 1.0,
                              false, false);
  /* add bias */
  for (auto b = 0U; b != batchSize; ++b) {
    for (auto i = 0U; i != outputSize; ++i) {
      output[b][i] += biases[lstmUnitOffset][i];
    }
  }

  /* apply non-linearity */
  nonLinearity(nonLinearityType, output);
}

/**
 * Apply mask to gates.
 */
static void applySeqMask(const boost::optional<Array1dRefUNSIGNED> &timeSteps,
                         const unsigned step, Array2dRef ionput) {
  if (timeSteps) {
    const auto batchSize = ionput.shape()[0];
    const auto outputSize = ionput.shape()[1];
    for (auto b = 0U; b != batchSize; ++b) {
      for (auto i = 0U; i != outputSize; ++i) {
        auto limit = (*timeSteps)[timeSteps->size() > 1 ? b : 0];
        if (step >= limit) {
          ionput[b][i] = 0;
        }
      }
    }
  }
}

/**
 * Update output for batches that have not reached iteration limit
 */
static void
copyIfStepWithinRange(const boost::optional<Array1dRefUNSIGNED> &timeSteps,
                      const unsigned step,
                      const boost::optional<Array2dRef> &current,
                      const Array2dRef update, Array2dRef dst) {
  const auto batchSize = dst.shape()[0];
  const auto outputSize = dst.shape()[1];
  for (auto b = 0U; b != batchSize; ++b) {
    for (auto i = 0U; i != outputSize; ++i) {
      auto limit = (*timeSteps)[(timeSteps->size() > 1) ? b : 0];
      if (step < limit) {
        dst[b][i] = update[b][i];
      } else if (current) {
        dst[b][i] = (*current)[b][i];
      }
    }
  }
}

static void
copyIfStepWithinRange(const boost::optional<Array1dRefUNSIGNED> &timeSteps,
                      const unsigned step, const Array2dRef update,
                      Array2dRef dst) {
  copyIfStepWithinRange(timeSteps, step, {}, update, dst);
}

static std::unordered_map<BasicLstmCellUnit, unsigned>
getCellMapping(const std::vector<BasicLstmCellUnit> &cellOrder) {
  // build a mapping of the order that the gates are stored in.
  std::unordered_map<BasicLstmCellUnit, unsigned> cellMapping;
  for (unsigned i = 0; i < cellOrder.size(); ++i) {
    auto gate = cellOrder.at(i);
    cellMapping.insert(std::make_pair(gate, i));
  }

  return cellMapping;
}

void poplibs_test::lstm::basicLstmCellForwardPass(
    const Array3dRef input, const Array2dRef biases,
    const Array2dRef prevOutput, const Array3dRef weightsInput,
    const Array3dRef weightsOutput,
    const boost::optional<Array1dRefUNSIGNED> &timeSteps,
    Array2dRef prevCellState, Array4dRef state, Array2dRef lastOutput,
    Array2dRef lastCellState, const std::vector<BasicLstmCellUnit> &cellOrder,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation) {
  const auto sequenceSize = state.shape()[1];
  const auto batchSize = state.shape()[2];
  const auto outputSize = state.shape()[3];
#ifndef NDEBUG
  const auto inputSize = input.shape()[2];
#endif
  assert(state.shape()[0] == LSTM_NUM_FWD_STATES);
  assert(weightsInput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsInput.shape()[1] == inputSize);
  assert(weightsInput.shape()[2] == outputSize);
  assert(weightsOutput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.shape()[1] == outputSize);
  assert(weightsOutput.shape()[2] == outputSize);
  assert(prevCellState.shape()[0] == batchSize);
  assert(prevCellState.shape()[1] == outputSize);
  assert(biases.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(biases.shape()[1] == outputSize);
  assert(prevOutput.shape()[0] == batchSize);
  assert(prevOutput.shape()[1] == outputSize);

  auto cellMapping = getCellMapping(cellOrder);

  Array2d nextOutput = prevOutput;
  Array2d nextCellState = prevCellState;
  for (auto s = 0U; s != sequenceSize; ++s) {
    Array2d inputThisStep = input[s];

    auto cellState = nextCellState;

    /* forget gate */
    Array2d forgetGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                         biases, forgetGate,
                         cellMapping.at(BASIC_LSTM_CELL_FORGET_GATE),
                         recurrentActivation);
    applySeqMask(timeSteps, s, forgetGate);
    state[LSTM_FWD_STATE_FORGET_GATE][s] = forgetGate;

    /* input gate */
    Array2d inputGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                         biases, inputGate,
                         cellMapping.at(BASIC_LSTM_CELL_INPUT_GATE),
                         recurrentActivation);
    applySeqMask(timeSteps, s, inputGate);
    state[LSTM_FWD_STATE_INPUT_GATE][s] = inputGate;

    /* new candidate contribution to this cell */
    Array2d candidate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                         biases, candidate,
                         cellMapping.at(BASIC_LSTM_CELL_CANDIDATE), activation);
    applySeqMask(timeSteps, s, candidate);
    state[LSTM_FWD_STATE_CAND_TANH][s] = candidate;

    /* output gate */
    Array2d outputGate(boost::extents[batchSize][outputSize]);
    processBasicLstmUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                         biases, outputGate,
                         cellMapping.at(BASIC_LSTM_CELL_OUTPUT_GATE),
                         recurrentActivation);
    applySeqMask(timeSteps, s, outputGate);
    state[LSTM_FWD_STATE_OUTPUT_GATE][s] = outputGate;

    poplibs_test::gemm::hadamardProduct(forgetGate, cellState, cellState);
    poplibs_test::gemm::hadamardProduct(inputGate, candidate, candidate);
    poplibs_test::axpby::add(cellState, candidate, cellState);

    /* need to maintain the cell state for next step */
    Array2d outputThisStep = cellState;
    nonLinearity(activation, outputThisStep);
    state[LSTM_FWD_STATE_OUTPUT_TANH][s] = outputThisStep;
    gemm::hadamardProduct(outputThisStep, outputGate, outputThisStep);

    if (timeSteps) {
      copyIfStepWithinRange(timeSteps, s, outputThisStep, nextOutput);
      copyIfStepWithinRange(timeSteps, s, cellState, nextCellState);
    } else {
      nextOutput = outputThisStep;
      nextCellState = cellState;
    }
    state[LSTM_FWD_STATE_ACTS_IDX][s] = outputThisStep;
    state[LSTM_FWD_STATE_CELL_STATE_IDX][s] = cellState;
  }

  // Save final state
  lastOutput = nextOutput;
  lastCellState = nextCellState;
}

static void computeGradients(const Array2dRef weightIn,
                             const Array2dRef weightPrev, const Array2dRef grad,
                             Array2dRef gradIn, Array2dRef gradPrev, bool acc) {
  double k = acc ? 1.0 : 0.0;
  gemm::generalMatrixMultiply(grad, weightIn, gradIn, gradIn, 1.0, k, false,
                              true);
  gemm::generalMatrixMultiply(grad, weightPrev, gradPrev, gradPrev, 1.0, k,
                              false, true);
}

void poplibs_test::lstm::basicLstmCellBackwardPass(
    bool outputFullSequence, const Array3dRef weightsInput,
    const Array3dRef weightsOutput, const Array3dRef gradsNextLayer,
    const Array2dRef prevCellState, const Array4dRef fwdState,
    const boost::optional<Array1dRefUNSIGNED> &timeSteps, Array4dRef bwdState,
    Array3dRef gradsPrevLayer, Array2dRef lastGradLayerOut,
    Array2dRef lastGradCellState,
    const std::vector<BasicLstmCellUnit> &cellOrder,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation) {
  const auto sequenceSize = fwdState.shape()[1];
  const auto batchSize = fwdState.shape()[2];
  const auto outputSize = fwdState.shape()[3];
  const auto inputSize = gradsPrevLayer.shape()[2];

  assert(fwdState.shape()[0] == LSTM_NUM_FWD_STATES);
  assert(bwdState.shape()[0] == LSTM_NUM_BWD_STATES);
  assert(weightsInput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsInput.shape()[1] == inputSize);
  assert(weightsInput.shape()[2] == outputSize);
  assert(weightsOutput.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.shape()[1] == outputSize);
  assert(weightsOutput.shape()[2] == outputSize);
  assert(prevCellState.shape()[0] == batchSize);
  assert(prevCellState.shape()[1] == outputSize);
  assert(fwdState.shape()[1] == sequenceSize);
  assert(fwdState.shape()[2] == batchSize);
  assert(fwdState.shape()[3] == outputSize);
  assert(bwdState.shape()[1] == sequenceSize);
  assert(bwdState.shape()[2] == batchSize);
  assert(bwdState.shape()[3] == outputSize);
  assert(gradsNextLayer.shape()[0] == sequenceSize);
  assert(gradsNextLayer.shape()[1] == batchSize);
  assert(gradsNextLayer.shape()[2] == outputSize);
  assert(gradsPrevLayer.shape()[0] == sequenceSize);
  assert(gradsPrevLayer.shape()[1] == batchSize);

  auto cellMapping = getCellMapping(cellOrder);

  // gradient of cell state for this step
  Array2d prevGradCellState(boost::extents[batchSize][outputSize]);
  for (auto it = prevGradCellState.data(),
            end = prevGradCellState.data() + prevGradCellState.num_elements();
       it != end; ++it) {
    *it = 0;
  }

  // gradient of output of this step
  Array2d gradOutput(boost::extents[batchSize][outputSize]);
  matrixZero(gradOutput);
  for (auto it = prevGradCellState.data(),
            end = prevGradCellState.data() + prevGradCellState.num_elements();
       it != end; ++it) {
    *it = 0;
  }

  for (auto i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    Array2d gradOut(boost::extents[batchSize][outputSize]);
    Array2d sumGradOut(boost::extents[batchSize][outputSize]);

    if (outputFullSequence) {
      gradOut = gradsNextLayer[s];
    } else {
      // Only the last layer receive the gradient
      if (s == sequenceSize - 1)
        gradOut = gradsNextLayer[0];
      else {
        matrixZero(gradOut);
      }
    }
    auto gradCellState = prevGradCellState;
    axpby::add(gradOut, gradOutput, sumGradOut);

    Array2d actOutGate = fwdState[LSTM_FWD_STATE_OUTPUT_GATE][s];
    Array2d gradAtOTanhInp(boost::extents[batchSize][outputSize]);
    gemm::hadamardProduct(actOutGate, sumGradOut, gradAtOTanhInp);

    Array2d actTanhOutGate = fwdState[LSTM_FWD_STATE_OUTPUT_TANH][s];
    Array2d gradAtOutGate(boost::extents[batchSize][outputSize]);
    ;

    gemm::hadamardProduct(actTanhOutGate, sumGradOut, gradAtOutGate);

    bwdNonLinearity(activation, actTanhOutGate, gradAtOTanhInp);

    bwdNonLinearity(recurrentActivation, actOutGate, gradAtOutGate);

    Array2dRef gradAtCellStateSum = gradAtOTanhInp;
    axpby::add(gradAtOTanhInp, gradCellState, gradAtCellStateSum);

    Array2d actInpGate = fwdState[LSTM_FWD_STATE_INPUT_GATE][s];
    Array2d gradAtCand(boost::extents[batchSize][outputSize]);
    ;
    gemm::hadamardProduct(actInpGate, gradAtCellStateSum, gradAtCand);
    Array2d actCand = fwdState[LSTM_FWD_STATE_CAND_TANH][s];
    Array2d gradAtInpGate(boost::extents[batchSize][outputSize]);
    ;
    gemm::hadamardProduct(actCand, gradAtCellStateSum, gradAtInpGate);
    bwdNonLinearity(activation, actCand, gradAtCand);
    bwdNonLinearity(recurrentActivation, actInpGate, gradAtInpGate);

    Array2d actForgetGate = fwdState[LSTM_FWD_STATE_FORGET_GATE][s];
    gemm::hadamardProduct(actForgetGate, gradAtCellStateSum, gradCellState);

    Array2d pCellAct(boost::extents[batchSize][outputSize]);

    if (s == 0) {
      pCellAct = prevCellState;
    } else {
      pCellAct = fwdState[LSTM_FWD_STATE_CELL_STATE_IDX][s - 1];
    }
    Array2d gradAtForgetGate(boost::extents[batchSize][outputSize]);
    ;

    gemm::hadamardProduct(pCellAct, gradAtCellStateSum, gradAtForgetGate);
    bwdNonLinearity(recurrentActivation, actForgetGate, gradAtForgetGate);

    Array2d gradIn(boost::extents[batchSize][inputSize]);
    ;
    Array2d weightsInUnit =
        weightsInput[cellMapping.at(BASIC_LSTM_CELL_FORGET_GATE)];
    Array2d weightsOutUnit =
        weightsOutput[cellMapping.at(BASIC_LSTM_CELL_FORGET_GATE)];
    Array2d nextGradOut(boost::extents[batchSize][outputSize]);
    computeGradients(weightsInUnit, weightsOutUnit, gradAtForgetGate, gradIn,
                     nextGradOut, false);
    weightsInUnit = weightsInput[cellMapping.at(BASIC_LSTM_CELL_INPUT_GATE)];
    weightsOutUnit = weightsOutput[cellMapping.at(BASIC_LSTM_CELL_INPUT_GATE)];
    computeGradients(weightsInUnit, weightsOutUnit, gradAtInpGate, gradIn,
                     nextGradOut, true);
    weightsInUnit = weightsInput[cellMapping.at(BASIC_LSTM_CELL_OUTPUT_GATE)];
    weightsOutUnit = weightsOutput[cellMapping.at(BASIC_LSTM_CELL_OUTPUT_GATE)];
    computeGradients(weightsInUnit, weightsOutUnit, gradAtOutGate, gradIn,
                     nextGradOut, true);
    weightsInUnit = weightsInput[cellMapping.at(BASIC_LSTM_CELL_CANDIDATE)];
    weightsOutUnit = weightsOutput[cellMapping.at(BASIC_LSTM_CELL_CANDIDATE)];
    computeGradients(weightsInUnit, weightsOutUnit, gradAtCand, gradIn,
                     nextGradOut, true);

    if (timeSteps) {
      if (outputFullSequence) {
        copyIfStepWithinRange(timeSteps, s, nextGradOut, gradOutput);
      } else {
        copyIfStepWithinRange(timeSteps, s, sumGradOut, nextGradOut,
                              gradOutput);
      }
      copyIfStepWithinRange(timeSteps, s, gradCellState, prevGradCellState);
    } else {
      gradOutput = nextGradOut;
      prevGradCellState = gradCellState;
    }
    gradsPrevLayer[s] = gradIn;

    // save bwd state for weight update
    bwdState[BASIC_LSTM_CELL_FORGET_GATE][s] = gradAtForgetGate;
    bwdState[BASIC_LSTM_CELL_INPUT_GATE][s] = gradAtInpGate;
    bwdState[BASIC_LSTM_CELL_OUTPUT_GATE][s] = gradAtOutGate;
    bwdState[BASIC_LSTM_CELL_CANDIDATE][s] = gradAtCand;
  }
  lastGradLayerOut = gradOutput;
  lastGradCellState = prevGradCellState;
}

void poplibs_test::lstm::basicLstmCellParamUpdate(
    const Array3dRef prevLayerActs, const Array4dRef fwdState,
    const Array2dRef outputActsInit, const Array4dRef bwdState,
    Array3dRef weightsInputDeltas, Array3dRef weightsOutputDeltas,
    Array2dRef biasDeltas, const std::vector<BasicLstmCellUnit> &cellOrder) {
  const auto sequenceSize = prevLayerActs.shape()[0];
  const auto batchSize = prevLayerActs.shape()[1];
  const auto inputSize = prevLayerActs.shape()[2];
  const auto outputSize = fwdState.shape()[3];

  assert(fwdState.shape()[0] == LSTM_NUM_FWD_STATES);
  assert(fwdState.shape()[1] == sequenceSize);
  assert(fwdState.shape()[2] == batchSize);
  assert(outputActsInit.shape()[0] == batchSize);
  assert(outputActsInit.shape()[1] == outputSize);
  assert(bwdState.shape()[0] == LSTM_NUM_BWD_STATES);
  assert(bwdState.shape()[1] == sequenceSize);
  assert(bwdState.shape()[2] == batchSize);
  assert(bwdState.shape()[3] == outputSize);
  assert(weightsInputDeltas.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsInputDeltas.shape()[1] == inputSize);
  assert(weightsInputDeltas.shape()[2] == outputSize);
  assert(weightsOutputDeltas.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutputDeltas.shape()[1] == outputSize);
  assert(weightsOutputDeltas.shape()[2] == outputSize);
  assert(biasDeltas.shape()[0] == BASIC_LSTM_CELL_NUM_UNITS);
  assert(biasDeltas.shape()[1] == outputSize);

  auto cellMapping = getCellMapping(cellOrder);

  for (auto it = weightsInputDeltas.data(),
            end = weightsInputDeltas.data() + weightsInputDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }
  for (auto it = weightsOutputDeltas.data(),
            end =
                weightsOutputDeltas.data() + weightsOutputDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }
  for (auto it = biasDeltas.data(),
            end = biasDeltas.data() + biasDeltas.num_elements();
       it != end; ++it) {
    *it = 0;
  }

  for (auto i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    Array2d outActs(boost::extents[batchSize][outputSize]);
    if (s == 0) {
      outActs = outputActsInit;
    } else {
      outActs = fwdState[LSTM_FWD_STATE_ACTS_IDX][s - 1];
    }
    Array2d inActs = prevLayerActs[s];
    for (auto i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
      const auto unit = static_cast<BasicLstmCellUnit>(i);

      Array2d grad = bwdState[i][s];
      Array2d wInputDeltasUnit(boost::extents[inputSize][outputSize]);

      gemm::generalMatrixMultiply(inActs, grad, wInputDeltasUnit,
                                  wInputDeltasUnit, 1.0, 0, true, false);
      for (auto ic = 0u; ic != inputSize; ++ic) {
        for (auto oc = 0u; oc != outputSize; ++oc) {
          weightsInputDeltas[cellMapping.at(unit)][ic][oc] +=
              wInputDeltasUnit[ic][oc];
        }
      }
      Array2d wOutputDeltasUnit(boost::extents[outputSize][outputSize]);

      gemm::generalMatrixMultiply(outActs, grad, wOutputDeltasUnit,
                                  wOutputDeltasUnit, 1.0, 0, true, false);
      for (auto oc1 = 0u; oc1 != outputSize; ++oc1) {
        for (auto oc2 = 0u; oc2 != outputSize; ++oc2) {
          weightsOutputDeltas[cellMapping.at(unit)][oc1][oc2] +=
              wOutputDeltasUnit[oc1][oc2];
        }
      }

      for (auto oc = 0u; oc != outputSize; ++oc) {
        for (auto b = 0u; b != batchSize; ++b) {
          biasDeltas[cellMapping.at(unit)][oc] += grad[b][oc];
        }
      }
    }
  }
}
