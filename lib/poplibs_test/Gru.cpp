// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <boost/multi_array.hpp>
#include <cassert>
#include <poplibs_test/GeneralMatrixAdd.hpp>
#include <poplibs_test/GeneralMatrixMultiply.hpp>
#include <poplibs_test/Gru.hpp>
#include <poplibs_test/NonLinearity.hpp>
#include <unordered_map>

// #define DEBUG_TENSOR

using IndexRange = boost::multi_array_types::index_range;
using Array1dRef = boost::multi_array_ref<double, 1>;
using Array1dRefUNSIGNED = boost::multi_array_ref<unsigned, 1>;
using Array2dRef = boost::multi_array_ref<double, 2>;
using Array2d = boost::multi_array<double, 2>;
using Array3dRef = boost::multi_array_ref<double, 3>;
using Array4dRef = boost::multi_array_ref<double, 4>;
using Array3d = boost::multi_array<double, 3>;

using namespace poplibs_test;

// Fwd state array indices
#define GRU_FWD_STATE_RESET_GATE 0
#define GRU_FWD_STATE_UPDATE_GATE 1
#define GRU_FWD_STATE_CANDIDATE 2
#define GRU_FWD_STATE_OUTPUT 3

#define GRU_BWD_STATE_RESET_GATE 0
#define GRU_BWD_STATE_UPDATE_GATE 1
#define GRU_BWD_STATE_CANDIDATE 2

static void matrixOne(boost::multi_array_ref<double, 2> matA) {
  std::fill(matA.data(), matA.data() + matA.num_elements(), 1.0);
}

static void matrixZero(boost::multi_array_ref<double, 2> matA) {
  std::fill(matA.data(), matA.data() + matA.num_elements(), 0.0);
}

static void matrixZero(boost::multi_array_ref<double, 3> matA) {
  std::fill(matA.data(), matA.data() + matA.num_elements(), 0.0);
}

/**
 * Process a given unit type within an GRU given its weights and biases.
 * The non-linearity is also specified although it may be derived from the unit
 */
static void processBasicGruUnit(const Array2dRef prevOutput,
                                const Array2dRef input,
                                const Array3dRef weightsInput,
                                const Array3dRef weightsOutput,
                                const Array2dRef biases, Array2dRef output,
                                unsigned gruUnitOffset,
                                popnn::NonLinearityType nonLinearityType) {
  const auto batchSize = prevOutput.shape()[0];
  const auto outputSize = prevOutput.shape()[1];

  /* split weight into two parts:
   * 1) part which weighs only the previous output
   * 2) part which weighs only the input
   */
  Array2d weightsOutputUnit = weightsOutput[gruUnitOffset];
  Array2d weightsInputUnit = weightsInput[gruUnitOffset];

  gemm::generalMatrixMultiply(prevOutput, weightsOutputUnit, output, false,
                              false);
  gemm::generalMatrixMultiply(input, weightsInputUnit, output, output, 1.0, 1.0,
                              false, false);
  /* add bias */
  for (auto b = 0U; b != batchSize; ++b) {
    for (auto i = 0U; i != outputSize; ++i) {
      output[b][i] += biases[gruUnitOffset][i];
    }
  }

  /* apply non-linearity */
  nonLinearity(nonLinearityType, output);
}

static std::unordered_map<BasicGruCellUnit, unsigned>
getCellMapping(const std::vector<BasicGruCellUnit> &cellOrder) {
  // build a mapping of the order that the gates are stored in.
  std::unordered_map<BasicGruCellUnit, unsigned> cellMapping;
  for (unsigned i = 0; i < cellOrder.size(); ++i) {
    auto gate = cellOrder.at(i);
    cellMapping.insert(std::make_pair(gate, i));
  }

  return cellMapping;
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

/**
 * Apply attention to update gate.
 */
static void applyAtt(const boost::optional<Array2dRef> attScores,
                     const int step, Array2dRef ionput) {
  if (attScores) {
    const auto batchSize = ionput.shape()[0];
    const auto outputSize = ionput.shape()[1];
    for (auto b = 0U; b != batchSize; ++b) {
      auto score = 1.0 - (*attScores)[b][step];
      for (auto i = 0U; i != outputSize; ++i) {
        ionput[b][i] *= score;
      }
    }
  }
}

static void applyAttBwd(const boost::optional<Array2dRef> attScores,
                        const Array2dRef u, const int step, Array2dRef d_u,
                        Array2dRef u0, Array2dRef attGrad) {
  if (attScores) {
    const auto batchSize = d_u.shape()[0];
    const auto outputSize = d_u.shape()[1];
    for (auto b = 0U; b != batchSize; ++b) {
      auto d_u_scale = 1.0f - (*attScores)[b][step];
      auto u0_scale = 1.0 / d_u_scale;
      attGrad[b][step] = 0;
      for (auto i = 0U; i != outputSize; ++i) {
        u0[b][i] = u[b][i] * u0_scale;
        attGrad[b][step] -= u0[b][i] * d_u[b][i];
        d_u[b][i] *= d_u_scale;
      }
    }
  }
}

static void printMatrix2d(FILE *fp, std::string msg, Array2dRef in) {
  if (!fp)
    return;

  fprintf(fp, "%s: {\n", msg.c_str());
  unsigned matRows = in.shape()[0];
  unsigned matCols = in.shape()[1];

  for (auto r = 0U; r != matRows; ++r) {
    fprintf(fp, " {");
    for (auto c = 0U; c != matCols; ++c) {
      if (c != matCols - 1)
        fprintf(fp, "%f,", in[r][c]);
      else
        fprintf(fp, "%f}\n", in[r][c]);
    }
  }
  fprintf(fp, "}\n");
}

static void printMatrix3d(FILE *fp, std::string msg, Array3dRef in) {
  if (!fp)
    return;

  fprintf(fp, "%s: {\n", msg.c_str());
  unsigned matRows = in.shape()[0];
  unsigned matCols = in.shape()[1];
  unsigned matInner = in.shape()[2];

  for (auto r = 0U; r != matRows; ++r) {
    fprintf(fp, " {\n");
    for (auto c = 0U; c != matCols; ++c) {
      fprintf(fp, "  {");
      for (auto i = 0U; i != matInner; ++i) {
        if (i != matInner - 1)
          fprintf(fp, "%f,", in[r][c][i]);
        else
          fprintf(fp, "%f}\n", in[r][c][i]);
      }
    }
    fprintf(fp, " }\n");
  }
  fprintf(fp, "}\n");
}

void poplibs_test::gru::basicGruCellForwardPass(
    const Array3dRef input, const Array2dRef biases,
    const Array2dRef prevOutput, const Array3dRef weightsInput,
    const Array3dRef weightsOutput,
    const boost::optional<boost::multi_array_ref<double, 2>> &attScoresOpt,
    const boost::optional<boost::multi_array_ref<unsigned, 1>> &timeSteps,
    Array4dRef state, Array2dRef lastOutput,
    const std::vector<BasicGruCellUnit> &cellOrder, bool resetAfter,
    const boost::optional<Array2dRef> recurrantBiases,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation) {
  const auto sequenceSize = state.shape()[1];
  const auto batchSize = state.shape()[2];
  const auto outputSize = state.shape()[3];
#ifndef NDEBUG
  const auto inputSize = input.shape()[2];
#endif
  assert(state.shape()[0] == GRU_NUM_FWD_STATES);
  assert(weightsInput.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsInput.shape()[1] == inputSize);
  assert(weightsInput.shape()[2] == outputSize);
  assert(weightsOutput.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsOutput.shape()[1] == outputSize);
  assert(weightsOutput.shape()[2] == outputSize);
  assert(biases.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(biases.shape()[1] == outputSize);
  assert(prevOutput.shape()[0] == batchSize);
  assert(prevOutput.shape()[1] == outputSize);
  assert(lastOutput.shape()[0] == batchSize);
  assert(lastOutput.shape()[1] == outputSize);
  assert(recurrantBiases.is_initialized() == resetAfter);
  if (resetAfter) {
    assert(recurrantBiases.get().shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
    assert(recurrantBiases.get().shape()[1] == outputSize);
  }

  auto cellMapping = getCellMapping(cellOrder);

  FILE *fp = NULL;
#ifdef DEBUG_TENSOR
  fp = fopen("fwd.txt", "w");
#endif

  printMatrix3d(fp, "fwd weightsInput", weightsInput);
  printMatrix3d(fp, "fwd weightsOutput", weightsOutput);
  printMatrix2d(fp, "fwd bias", biases);
  if (resetAfter) {
    printMatrix2d(fp, "fwd recurrant bias", recurrantBiases.get());
  }

  Array2d totalBiases(boost::extents[BASIC_GRU_CELL_NUM_UNITS][outputSize]);
  if (resetAfter) {
    /* Sum biases for use in update gate and reset gate */
    poplibs_test::axpby::add(biases, recurrantBiases.get(), totalBiases);
  }

  Array2d nextOutput = prevOutput;
  for (auto s = 0U; s != sequenceSize; ++s) {
    if (fp)
      fprintf(fp, "fwd Loop: {%d}\n", s);
    Array2d inputThisStep = input[s];

    printMatrix2d(fp, "fwd h_prev", nextOutput); // prevOutputThisStep);
    printMatrix2d(fp, "fwd input", inputThisStep);

    /* update gate */
    Array2d updateGate(boost::extents[batchSize][outputSize]);
    processBasicGruUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                        (resetAfter ? totalBiases : biases), updateGate,
                        cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE),
                        recurrentActivation);
    applyAtt(attScoresOpt, s, updateGate);
    applySeqMask(timeSteps, s, updateGate);
    state[GRU_FWD_STATE_UPDATE_GATE_IDX][s] = updateGate;

    /* reset gate */
    Array2d resetGate(boost::extents[batchSize][outputSize]);
    processBasicGruUnit(nextOutput, inputThisStep, weightsInput, weightsOutput,
                        (resetAfter ? totalBiases : biases), resetGate,
                        cellMapping.at(BASIC_GRU_CELL_RESET_GATE),
                        recurrentActivation);
    applySeqMask(timeSteps, s, resetGate);
    state[GRU_FWD_STATE_RESET_GATE_IDX][s] = resetGate;

    /* candidate */
    Array2d candidate(boost::extents[batchSize][outputSize]);
    Array2d tmp1(boost::extents[batchSize][outputSize]);

    if (resetAfter) {
      auto candidateOffset = cellMapping.at(BASIC_GRU_CELL_CANDIDATE);
      /* apply weights to previous output */
      Array2d &&w_c_out = weightsOutput[candidateOffset];
      poplibs_test::gemm::generalMatrixMultiply(nextOutput, w_c_out, tmp1,
                                                false, false);
      /* add recurrant bias */
      for (auto b = 0U; b != batchSize; ++b) {
        for (auto i = 0U; i != outputSize; ++i) {
          tmp1[b][i] += recurrantBiases.get()[candidateOffset][i];
        }
      }
      /* apply reset gate */
      poplibs_test::gemm::hadamardProduct(resetGate, tmp1, candidate);
      /* apply weights to input */
      Array2d &&w_c_in = weightsInput[candidateOffset];
      poplibs_test::gemm::generalMatrixMultiply(
          inputThisStep, w_c_in, candidate, candidate, 1.0, 1.0, false, false);
      /* add bias */
      for (auto b = 0U; b != batchSize; ++b) {
        for (auto i = 0U; i != outputSize; ++i) {
          candidate[b][i] += biases[candidateOffset][i];
        }
      }
      /* apply non-linearity */
      nonLinearity(activation, candidate);
    } else {
      poplibs_test::gemm::hadamardProduct(resetGate, nextOutput, tmp1);
      processBasicGruUnit(tmp1, inputThisStep, weightsInput, weightsOutput,
                          (resetAfter ? totalBiases : biases), candidate,
                          cellMapping.at(BASIC_GRU_CELL_CANDIDATE), activation);
    }
    applySeqMask(timeSteps, s, candidate);

    state[GRU_FWD_STATE_CANDIDATE_IDX][s] = candidate;

    printMatrix2d(fp, "fwd resetGate", resetGate);
    printMatrix2d(fp, "fwd updateGate", updateGate);
    printMatrix2d(fp, "fwd candidate", candidate);

    /* output */
    Array2d matOne(boost::extents[batchSize][outputSize]);
    matrixOne(matOne);

    Array2d updateGateComp(boost::extents[batchSize][outputSize]);
    poplibs_test::axpby::add(matOne, updateGate, updateGateComp, 1.0, -1.0);
    Array2d s1(boost::extents[batchSize][outputSize]);
    Array2d s2(boost::extents[batchSize][outputSize]);
    poplibs_test::gemm::hadamardProduct(updateGate, nextOutput, s1);
    poplibs_test::gemm::hadamardProduct(updateGateComp, candidate, s2);

    Array2d outputThisStep(boost::extents[batchSize][outputSize]);
    poplibs_test::axpby::add(s1, s2, outputThisStep);
    applySeqMask(timeSteps, s, outputThisStep);

    if (timeSteps) {
      copyIfStepWithinRange(timeSteps, s, outputThisStep, nextOutput);
    } else {
      nextOutput = outputThisStep;
    }

    state[GRU_FWD_STATE_ACTS_IDX][s] = outputThisStep;
    printMatrix2d(fp, "fwd output", outputThisStep);
  }

  // Save final state
  lastOutput = nextOutput;

  if (fp)
    fclose(fp);
}

static Array2d getSlice(Array2d &in, int offset, int size) {
  int batchSize = in.shape()[0];

  Array2d out(boost::extents[batchSize][size]);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < size; j++) {
      out[i][j] = in[i][j + offset];
    }
  }

  return out;
}

static Array2d concatMatrix2D(const Array2d matA, const Array2d matB,
                              int dimension) {
  const auto matARows = matA.shape()[0];
  const auto matACols = matA.shape()[1];

  const auto matBRows = matB.shape()[0];
  const auto matBCols = matB.shape()[1];

  if (dimension == 0) {
    Array2d matC(boost::extents[matARows + matBRows][matACols]);
    if (matACols != matBCols)
      return matC;
    for (unsigned int i = 0; i < matARows; i++) {
      for (unsigned int j = 0; j < matACols; j++) {
        matC[i][j] = matA[i][j];
      }
    }
    for (unsigned int i = 0; i < matBRows; i++) {
      for (unsigned int j = 0; j < matBCols; j++) {
        matC[i + matARows][j] = matB[i][j];
      }
    }
    return matC;
  } else if (dimension == 1) {
    Array2d matC(boost::extents[matARows][matACols + matBCols]);
    if (matARows != matBRows)
      return matC;
    for (unsigned int i = 0; i < matARows; i++) {
      for (unsigned int j = 0; j < matACols; j++) {
        matC[i][j] = matA[i][j];
      }
    }
    for (unsigned int i = 0; i < matBRows; i++) {
      for (unsigned int j = 0; j < matBCols; j++) {
        matC[i][j + matACols] = matB[i][j];
      }
    }

    return matC;
  } else {
    // not implemented
    assert(0);
    Array2d matC(boost::extents[matARows][matACols + matBCols]);
    return matC;
  }
}

void poplibs_test::gru::basicGruCellBackwardPass(
    bool outputFullSequence, const Array3dRef weightsInput,
    const Array3dRef weightsOutput, const Array3dRef gradsNextLayer,
    const Array4dRef fwdState, const Array2dRef outputActsInit,
    const boost::optional<Array1dRefUNSIGNED> &timeSteps,
    const boost::optional<Array2dRef> &attScoresOpt,
    const boost::optional<Array2dRef> &attScoresGradsOpt, Array4dRef bwdState,
    Array3dRef gradsPrevIn, Array2dRef gradsPrevOut,
    const std::vector<BasicGruCellUnit> &cellOrder, bool resetAfter,
    const boost::optional<Array2dRef> recurrantBiases,
    const popnn::NonLinearityType activation,
    const popnn::NonLinearityType recurrentActivation) {
  const auto sequenceSize = fwdState.shape()[1];
  const auto batchSize = fwdState.shape()[2];
  const auto outputSize = fwdState.shape()[3];
  const auto inputSize = gradsPrevIn.shape()[2];

  assert(fwdState.shape()[0] == GRU_NUM_FWD_STATES);
  assert(bwdState.shape()[0] == GRU_NUM_BWD_STATES);
  assert(weightsInput.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsInput.shape()[1] == inputSize);
  assert(weightsInput.shape()[2] == outputSize);
  assert(weightsOutput.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsOutput.shape()[1] == outputSize);
  assert(weightsOutput.shape()[2] == outputSize);
  assert(fwdState.shape()[1] == sequenceSize);
  assert(fwdState.shape()[2] == batchSize);
  assert(fwdState.shape()[3] == outputSize);
  assert(bwdState.shape()[1] == sequenceSize);
  assert(bwdState.shape()[2] == batchSize);
  assert(bwdState.shape()[3] == outputSize);
  assert(gradsNextLayer.shape()[0] == sequenceSize);
  assert(gradsNextLayer.shape()[1] == batchSize);
  assert(gradsNextLayer.shape()[2] == outputSize);
  assert(gradsPrevIn.shape()[0] == sequenceSize);
  assert(gradsPrevIn.shape()[1] == batchSize);
  assert(gradsPrevIn.shape()[2] == inputSize);
  assert(gradsPrevOut.shape()[0] == batchSize);
  assert(gradsPrevOut.shape()[1] == outputSize);
  assert(recurrantBiases.is_initialized() == resetAfter);
  if (resetAfter) {
    assert(recurrantBiases.get().shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
    assert(recurrantBiases.get().shape()[1] == outputSize);
  }

  auto cellMapping = getCellMapping(cellOrder);

  // gradient of output of this step
  Array2d gradOutput(boost::extents[batchSize][outputSize]);
  matrixZero(gradOutput);

  Array2d matOne(boost::extents[batchSize][outputSize]);
  matrixOne(matOne);

  const Array2d w_c_in = weightsInput[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)];
  const Array2d w_c_out =
      weightsOutput[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)];
  Array2d w_ru = concatMatrix2D(
      concatMatrix2D(weightsInput[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)],
                     weightsOutput[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)],
                     0),
      concatMatrix2D(weightsInput[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)],
                     weightsOutput[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)],
                     0),
      1);
  FILE *fp = NULL;
#ifdef DEBUG_TENSOR
  fp = fopen("bwd.txt", "w");
#endif

  for (unsigned i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    if (fp)
      fprintf(fp, "bwd Loop: {%ld}\n", (long int)s);

    Array2d d_h(boost::extents[batchSize][outputSize]);
    Array2d gradOut(boost::extents[batchSize][outputSize]);

    if (outputFullSequence)
      gradOut = gradsNextLayer[s];
    else {
      // Only the last layer receive the gradient
      if (s == sequenceSize - 1)
        gradOut = gradsNextLayer[0];
      else {
        matrixZero(gradOut);
      }
    }

    printMatrix2d(fp, "bwd outGrad", gradOutput);
    printMatrix2d(fp, "bwd gradNextLayer", gradOut);
    axpby::add(gradOut, gradOutput, d_h);

    Array2d u = fwdState[GRU_FWD_STATE_UPDATE_GATE][s];
    Array2d r = fwdState[GRU_FWD_STATE_RESET_GATE][s];
    Array2d c = fwdState[GRU_FWD_STATE_CANDIDATE][s];

    printMatrix2d(fp, "bwd d_h", d_h);
    printMatrix2d(fp, "bwd r", r);
    printMatrix2d(fp, "bwd u", u);
    printMatrix2d(fp, "bwd c", c);

    Array2d u_comp(boost::extents[batchSize][outputSize]);
    poplibs_test::axpby::add(matOne, u, u_comp, 1.0, -1.0);
    Array2d d_c(boost::extents[batchSize][outputSize]);
    gemm::hadamardProduct(u_comp, d_h, d_c);
    bwdNonLinearity(activation, c, d_c);
    applySeqMask(timeSteps, s, d_c);

    Array2d h_prev(boost::extents[batchSize][outputSize]);
    if (s == 0) {
      h_prev = outputActsInit;
    } else {
      h_prev = fwdState[GRU_FWD_STATE_ACTS_IDX][s - 1];
    }

    printMatrix2d(fp, "bwd h_prev", h_prev);

    Array2d h_prev_c(boost::extents[batchSize][outputSize]);
    poplibs_test::axpby::add(h_prev, c, h_prev_c, 1.0, -1.0);
    Array2d d_u(boost::extents[batchSize][outputSize]);
    gemm::hadamardProduct(d_h, h_prev_c, d_u);

    if (attScoresOpt) {
      Array2d u0(boost::extents[batchSize][outputSize]);
      applyAttBwd(attScoresOpt, u, s, d_u, u0, *attScoresGradsOpt);
      bwdNonLinearity(recurrentActivation, u0, d_u);
      printMatrix2d(fp, "bwd d_u", d_u);
      printMatrix2d(fp, "bwd attGrad", *attScoresGradsOpt);
    } else {
      bwdNonLinearity(recurrentActivation, u, d_u);
      printMatrix2d(fp, "bwd d_u", d_u);
    }

    Array2d d_r(boost::extents[batchSize][outputSize]);
    Array2d d_x2(boost::extents[batchSize][inputSize]);
    Array2d d_h_prev2(boost::extents[batchSize][outputSize]);
    gemm::generalMatrixMultiply(d_c, w_c_in, d_x2, false, true);
    if (resetAfter) {
      Array2d h_prev2(boost::extents[batchSize][outputSize]);
      Array2d d_cr(boost::extents[batchSize][outputSize]);
      gemm::generalMatrixMultiply(h_prev, w_c_out, h_prev2, false, false);
      for (unsigned i = 0; i < batchSize; i++) {
        for (unsigned j = 0; j < outputSize; j++) {
          h_prev2[i][j] +=
              recurrantBiases
                  .get()[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)][j];
        }
      }
      gemm::hadamardProduct(d_c, h_prev2, d_r);
      gemm::hadamardProduct(d_c, r, d_cr);
      gemm::generalMatrixMultiply(d_cr, w_c_out, d_h_prev2, false, true);
    } else {
      Array2d d_h_prev2r(boost::extents[batchSize][outputSize]);
      // [2nd_component_of_d_x d_h_prevr] = d_c X w_c^T
      gemm::generalMatrixMultiply(d_c, w_c_out, d_h_prev2r, false, true);
      gemm::hadamardProduct(d_h_prev2r, h_prev, d_r);
      gemm::hadamardProduct(d_h_prev2r, r, d_h_prev2);
    }
    bwdNonLinearity(recurrentActivation, r, d_r);

    // [1st_component_of_d_x 1st_component_of_d_h_prev] = [d_r d_u] X w_ru^T
    Array2d d_r_d_u = concatMatrix2D(d_r, d_u, 1);
    Array2d d_x1_h_prev1(boost::extents[batchSize][inputSize + outputSize]);
    gemm::generalMatrixMultiply(d_r_d_u, w_ru, d_x1_h_prev1, false, true);
    Array2d d_x1 = getSlice(d_x1_h_prev1, 0, inputSize);
    Array2d d_h_prev1 = getSlice(d_x1_h_prev1, inputSize, outputSize);

    Array2d d_x(boost::extents[batchSize][inputSize]);
    poplibs_test::axpby::add(d_x1, d_x2, d_x);

    Array2d d_h_prev(boost::extents[batchSize][outputSize]);
    {
      gemm::hadamardProduct(d_h, u, d_h_prev);
      poplibs_test::axpby::add(d_h_prev, d_h_prev1, d_h_prev, 1.0, 1.0);
      poplibs_test::axpby::add(d_h_prev, d_h_prev2, d_h_prev, 1.0, 1.0);
    }
    if (timeSteps) {
      if (outputFullSequence) {
        copyIfStepWithinRange(timeSteps, s, d_h_prev, gradOutput);
      } else {
        copyIfStepWithinRange(timeSteps, s, d_h, d_h_prev, gradOutput);
      }
    } else {
      gradOutput = d_h_prev;
    }
    gradsPrevIn[s] = d_x;

    // save bwd state for weight update
    bwdState[BASIC_GRU_CELL_UPDATE_GATE][s] = d_u;
    bwdState[BASIC_GRU_CELL_RESET_GATE][s] = d_r;
    bwdState[BASIC_GRU_CELL_CANDIDATE][s] = d_c;
  }
  gradsPrevOut = gradOutput;
  if (fp)
    fclose(fp);
}

void poplibs_test::gru::basicGruCellParamUpdate(
    const Array3dRef prevLayerActs, const Array4dRef fwdState,
    const Array2dRef outputActsInit, const Array4dRef bwdState,
    Array3dRef weightsInputDeltas, Array3dRef weightsOutputDeltas,
    Array2dRef biasDeltas, const std::vector<BasicGruCellUnit> &cellOrder,
    bool resetAfter, boost::optional<Array2dRef> recurrantBiasDeltas) {
  const auto sequenceSize = prevLayerActs.shape()[0];
  const auto batchSize = prevLayerActs.shape()[1];
  const auto inputSize = prevLayerActs.shape()[2];
  const auto outputSize = fwdState.shape()[3];

  assert(fwdState.shape()[0] == GRU_NUM_FWD_STATES);
  assert(fwdState.shape()[1] == sequenceSize);
  assert(fwdState.shape()[2] == batchSize);
  assert(outputActsInit.shape()[0] == batchSize);
  assert(outputActsInit.shape()[1] == outputSize);
  assert(bwdState.shape()[0] == GRU_NUM_BWD_STATES);
  assert(bwdState.shape()[1] == sequenceSize);
  assert(bwdState.shape()[2] == batchSize);
  assert(bwdState.shape()[3] == outputSize);
  assert(weightsInputDeltas.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsInputDeltas.shape()[1] == inputSize);
  assert(weightsInputDeltas.shape()[2] == outputSize);
  assert(weightsOutputDeltas.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsOutputDeltas.shape()[1] == outputSize);
  assert(weightsOutputDeltas.shape()[2] == outputSize);
  assert(biasDeltas.shape()[0] == BASIC_GRU_CELL_NUM_UNITS);
  assert(biasDeltas.shape()[1] == outputSize);
  assert(recurrantBiasDeltas.is_initialized() == resetAfter);

  auto cellMapping = getCellMapping(cellOrder);

  matrixZero(weightsInputDeltas);
  matrixZero(weightsOutputDeltas);
  matrixZero(biasDeltas);
  if (resetAfter) {
    matrixZero(recurrantBiasDeltas.get());
  }
  /*
    d_w_r = x_h_prev^T * d_r

    d_w_u = x_h_prev^T * d_u

    d_w_c = x_h_prevr^T * d_c_bar

    d_b_ru = sum of d_r_bar_u_bar along axis = 0

    d_b_c = sum of d_c_bar along axis = 0
  */
  for (unsigned i = sequenceSize; i != 0; --i) {
    const auto s = i - 1;
    Array2d h_prev(boost::extents[batchSize][outputSize]);
    if (s == 0) {
      h_prev = outputActsInit;
    } else {
      h_prev = fwdState[GRU_FWD_STATE_ACTS_IDX][s - 1];
    }
    Array2d x = prevLayerActs[s];
    Array2d x_h_prev = concatMatrix2D(x, h_prev, 1);
    Array2d r = fwdState[GRU_FWD_STATE_RESET_GATE_IDX][s];

    Array2d d_r = bwdState[BASIC_GRU_CELL_RESET_GATE][s];
    Array2d d_u = bwdState[BASIC_GRU_CELL_UPDATE_GATE][s];
    Array2d d_c = bwdState[BASIC_GRU_CELL_CANDIDATE][s];
    Array2d d_cr(boost::extents[batchSize][outputSize]);
    Array2d d_w_r(boost::extents[inputSize + outputSize][outputSize]);
    Array2d d_w_u(boost::extents[inputSize + outputSize][outputSize]);
    Array2d d_w_c(boost::extents[inputSize + outputSize][outputSize]);
    gemm::generalMatrixMultiply(x_h_prev, d_r, d_w_r, true, false);
    gemm::generalMatrixMultiply(x_h_prev, d_u, d_w_u, true, false);

    if (resetAfter) {
      Array2d d_w_c_in(boost::extents[inputSize][outputSize]);
      Array2d d_w_c_out(boost::extents[outputSize][outputSize]);

      gemm::hadamardProduct(d_c, r, d_cr);
      gemm::generalMatrixMultiply(h_prev, d_cr, d_w_c_out, true, false);
      gemm::generalMatrixMultiply(x, d_c, d_w_c_in, true, false);
      d_w_c = concatMatrix2D(d_w_c_in, d_w_c_out, 0);
    } else {
      Array2d h_prevr(boost::extents[batchSize][outputSize]);
      gemm::hadamardProduct(h_prev, r, h_prevr);
      Array2d x_h_prevr = concatMatrix2D(x, h_prevr, 1);
      gemm::generalMatrixMultiply(x_h_prevr, d_c, d_w_c, true, false);
    }

    for (unsigned int m = 0; m < inputSize; m++) {
      for (unsigned int n = 0; n < outputSize; n++) {
        weightsInputDeltas[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)][m][n] +=
            d_w_r[m][n];
        weightsInputDeltas[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)][m][n] +=
            d_w_u[m][n];
        weightsInputDeltas[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)][m][n] +=
            d_w_c[m][n];
      }
    }
    for (unsigned int m = 0; m < outputSize; m++) {
      for (unsigned int n = 0; n < outputSize; n++) {
        weightsOutputDeltas[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)][m][n] +=
            d_w_r[m + inputSize][n];
        weightsOutputDeltas[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)][m][n] +=
            d_w_u[m + inputSize][n];
        weightsOutputDeltas[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)][m][n] +=
            d_w_c[m + inputSize][n];
      }
    }
    for (unsigned int m = 0; m < outputSize; m++) {
      for (unsigned int n = 0; n < batchSize; n++) {
        biasDeltas[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)][m] += d_r[n][m];
        biasDeltas[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)][m] += d_u[n][m];
        biasDeltas[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)][m] += d_c[n][m];

        if (resetAfter) {
          recurrantBiasDeltas
              .get()[cellMapping.at(BASIC_GRU_CELL_RESET_GATE)][m] += d_r[n][m];
          recurrantBiasDeltas
              .get()[cellMapping.at(BASIC_GRU_CELL_UPDATE_GATE)][m] +=
              d_u[n][m];
          recurrantBiasDeltas
              .get()[cellMapping.at(BASIC_GRU_CELL_CANDIDATE)][m] += d_cr[n][m];
        }
      }
    }
  }
}
