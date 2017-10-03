#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <popstd/HadamardProduct.hpp>
#include <popstd/VertexTemplates.hpp>
#include <popnn/Lstm.hpp>
#include <popstd/Operations.hpp>
#include <popconv/Convolution.hpp>
#include <popstd/Zero.hpp>
#include <popreduce/Reduce.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;
using namespace popnn;

// Tensor elements maintained in forward state. The number of elements is a
// function of the amount of recomputation done in the backward pass
enum FwdStateTensorElems {
  LSTM_FWD_STATE_OUTPUT_ACTS = 0,
  LSTM_FWD_STATE_CELL_STATE,
  LSTM_NUM_FWD_STATES_INFERENCE,
  LSTM_FWD_STATE_ACTS_FORGET_GATE = LSTM_NUM_FWD_STATES_INFERENCE,
  LSTM_FWD_STATE_ACTS_INPUT_GATE,
  LSTM_FWD_STATE_ACTS_CAND_TANH,
  LSTM_FWD_STATE_ACTS_OUTPUT_GATE,
  LSTM_FWD_STATE_ACTS_OUTPUT_TANH,
  LSTM_NUM_FWD_STATES_TRAINING
};

// Tensor elements maintained in backward state. The number of elements is a
// function of the amount of recomputation done in the weight update pass
enum BwdStateTensorElems {
  LSTM_BWD_STATE_GRAD_CELL_STATE = 0,
  LSTM_BWD_STATE_GRAD_ACT_GRAD,
  LSTM_BWD_STATE_GRAD_FORGET_GATE,
  LSTM_BWD_STATE_GRAD_INPUT_GATE,
  LSTM_BWD_STATE_GRAD_CANDIDATE,
  LSTM_BWD_STATE_GRAD_OUTPUT_GATE,
  LSTM_NUM_BWD_STATES
};

// Flatten a 3D tensor to a 2D tensor such that the innermost dimension is the
// product of outputs(or inputs) and units
static Tensor flattenUnits(const Tensor &t) {
  return t.dimShuffle({1, 0, 2}).reshape({t.dim(1), t.dim(0) * t.dim(2)});
}

// unflatten a 2D tensor which has units flattened in it's innermost dimension.
// The resultant 3D tensor view has the unit dimension as the outermost
// dimension
static Tensor unflattenUnits(const Tensor &t) {
  return t.reshape({ t.dim(0), BASIC_LSTM_CELL_NUM_UNITS,
                     t.dim(1) / BASIC_LSTM_CELL_NUM_UNITS})
          .dimShuffle({1, 0, 2});
}

static Tensor getFwdState(const Tensor &fwdState, unsigned idx) {
  const auto rank = fwdState.rank();
  if (rank != 3 && rank != 4) {
    throw popstd::poplib_error("Unexpected state tensor dimensions");
  }
  if (rank == 3) {
    assert(fwdState.dim(0) == LSTM_NUM_FWD_STATES_INFERENCE ||
           fwdState.dim(0) == LSTM_NUM_FWD_STATES_TRAINING);
    return fwdState[idx];
  } else {
    assert(fwdState.dim(1) == LSTM_NUM_FWD_STATES_INFERENCE ||
           fwdState.dim(1) == LSTM_NUM_FWD_STATES_TRAINING);
    return fwdState.slice(idx, idx + 1, 1).squeeze({1});
  }
}

static Tensor getBwdState(const Tensor &bwdState, unsigned idx) {
  const auto rank = bwdState.rank();
  if (rank != 3 && rank != 4) {
    throw popstd::poplib_error("Unexpected state tensor dimensions");
  }
  assert(idx < LSTM_NUM_BWD_STATES);
  if (rank == 3) {
    return bwdState[idx];
  } else {
    return bwdState.slice(idx, idx + 1, 1).squeeze({1});
  }
}

static void
applyGateNonlinearities(Graph &graph,
                        Tensor &t,
                        Sequence &prog,
                        const std::string &debugStr) {
  auto sigmoidIn = concat({t[BASIC_LSTM_CELL_INPUT_GATE],
                           t[BASIC_LSTM_CELL_FORGET_GATE],
                           t[BASIC_LSTM_CELL_OUTPUT_GATE]});
  auto cs = graph.addComputeSet(debugStr + "/OutputGate");
  nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
               sigmoidIn, cs, debugStr);
  nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_TANH,
               t[BASIC_LSTM_CELL_CANDIDATE], cs, debugStr);
  prog.add(Execute(cs));
}

// Computes the output before nonlinearities to all the units are applies
static Tensor
basicLstmUnitsNlInputPreWeighted(Graph &graph,
                                       Tensor weightedIn,
                                       Tensor prevOutput,
                                       Tensor weightsOutput,
                                       Sequence &prog,
                                       MatMulOptions &mmOpt,
                                       const std::string debugStr) {
  assert(weightedIn.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto output =
      unflattenUnits(matMul(graph, prevOutput, flattenUnits(weightsOutput),
                     prog, debugStr + "/WeighOutput", mmOpt));
  addTo(graph, output, weightedIn, 1.0, prog, debugStr + "/AddWeightedOutputs");
  return output;
}

// Computes the output before nonlinearities to all the units are applied
static Tensor
basicLstmUnitsNlInput(Graph &graph,
                      Tensor prevAct,
                      Tensor prevOutput,
                      Tensor weightsInput,
                      Tensor weightsOutput,
                      Sequence &prog,
                      MatMulOptions &mmOpt,
                      const std::string &debugStr) {
  assert(weightsInput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto prodInp =
      unflattenUnits(matMul(graph, prevAct, flattenUnits(weightsInput),
                            prog, debugStr + "/WeighInput", mmOpt));
  return
      basicLstmUnitsNlInputPreWeighted(graph, prodInp, prevOutput,
                                       weightsOutput, prog, mmOpt, debugStr);
}

// Add bias and compute LSTM output and update cellState given output of all
// the gates
static Tensor
basicLstmComputeOutput(Graph &graph,
                       Tensor &gatesOutput,
                       const Tensor &cellState,
                       Tensor &bBiases,
                       Sequence &prog,
                       bool saveExtraState,
                       const std::string &debugStr) {
  assert(gatesOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = gatesOutput[BASIC_LSTM_CELL_FORGET_GATE];
  auto candidate = gatesOutput[BASIC_LSTM_CELL_CANDIDATE];
  auto outputGate = gatesOutput[BASIC_LSTM_CELL_OUTPUT_GATE];
  auto inputGate = gatesOutput[BASIC_LSTM_CELL_INPUT_GATE];
  const auto dType = cellState.elementType();
  addTo(graph, gatesOutput, bBiases, 1.0, prog, debugStr + "/AddBias");
  applyGateNonlinearities(graph, gatesOutput, prog, debugStr);

  auto prod = mul(graph, concat(forgetGate, candidate),
                  concat(cellState, inputGate), prog,
                  debugStr + "/{Forget + Input}Gate");

  auto updatedCellState = prod.slice(0, forgetGate.dim(0));
  auto updatedCandidate = prod.slice(forgetGate.dim(0),
                                     forgetGate.dim(0) + candidate.dim(0));

  addTo(graph, updatedCellState, updatedCandidate, 1.0, prog,
        debugStr + "/AddCellCand");
  auto tanhOutput = popstd::tanh(graph, updatedCellState, prog, debugStr);
  auto output = mul(graph, tanhOutput, outputGate, prog,
                    debugStr + "/OutputGate");
  Tensor state = concat(output.expand({0}), updatedCellState.expand({0}));
  if (saveExtraState) {
    state = concat(state, forgetGate.expand({0}));
    state = concat(state, inputGate.expand({0}));
    state = concat(state, candidate.expand({0}));
    state = concat(state, outputGate.expand({0}));
    state = concat(state, tanhOutput.expand({0}));
  }
  return state;
}

namespace popnn {
namespace lstm {

Tensor createInput(Graph &graph,
                   unsigned sequenceSize,
                   unsigned batchSize,
                   unsigned inputSize,
                   unsigned outputSize,
                   const std::string &dType,
                   bool inferenceOnly,
                   const std::string &name) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = "float";
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  auto fcOutputSize = BASIC_LSTM_CELL_NUM_UNITS * outputSize;
  auto fcInputSize = inputSize;
  auto fcBatchSize = sequenceSize * batchSize;
  auto in = createMatMulInputLHS(graph, dType,
                                 {fcBatchSize, fcInputSize},
                                 {fcInputSize, fcOutputSize},
                                 name, mmOpt);
  return in.reshape({sequenceSize, batchSize, inputSize});
}

Tensor createFwdState(Graph &graph,
                      unsigned batchSize,
                      unsigned outputSize,
                      Sequence &prog,
                      bool initState,
                      const std::string &dType,
                      bool inferenceOnly,
                      const std::string &debugPrefix) {
  auto stateDims = inferenceOnly ? LSTM_NUM_FWD_STATES_INFERENCE :
                   LSTM_NUM_FWD_STATES_TRAINING;
  auto state =
    graph.addTensor(dType, {stateDims, batchSize, outputSize},
                    debugPrefix + "/fwdStateOut");
  for (auto i = 0; i != stateDims; ++i) {
    mapTensorLinearly(graph, state[i]);
  }

  if (initState) {
    zero(graph, state, prog, debugPrefix);
  } else {
    // zero internal state
    if (state.dim(0) > LSTM_NUM_FWD_STATES_INFERENCE) {
      zero(graph, state.slice(LSTM_NUM_FWD_STATES_INFERENCE,
                              LSTM_NUM_FWD_STATES_TRAINING, 0),
                   prog, debugPrefix);
    }
  }
  return state;
}

Tensor createBwdState(Graph &graph,
                      unsigned batchSize,
                      unsigned outputSize,
                      Sequence &prog,
                      const std::string &dType,
                      const std::string &debugPrefix) {
  auto state =
    graph.addTensor(dType, {LSTM_NUM_BWD_STATES, batchSize, outputSize},
                    debugPrefix + "/BwdState");
  for (auto i = 0; i != LSTM_NUM_BWD_STATES; ++i) {
    mapTensorLinearly(graph, state[i]);
  }
  zero(graph, state.slice(0, 2), prog, debugPrefix);
  return state;
}

Tensor getOutputFromFwdState(const Tensor &fwdState) {
  return getFwdState(fwdState, LSTM_FWD_STATE_OUTPUT_ACTS);
}

Tensor getCellFromFwdState(const Tensor &fwdState) {
  return getFwdState(fwdState, LSTM_FWD_STATE_CELL_STATE);
}

Tensor createWeightsInput(Graph &graph,
                          unsigned seqSize,
                          unsigned batchSize,
                          unsigned inputSize,
                          unsigned outputSize,
                          bool preweights,
                          const std::string &dType,
                          const std::string &partialsType,
                          bool inferenceOnly,
                          const std::string &name
                          ) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  std::vector<std::size_t> aShape(2);
  aShape[0] = preweights ? seqSize * batchSize : batchSize;
  aShape[1] = inputSize;

  auto weightsInput =
      createMatMulInputRHS(graph, dType,
                           aShape,
                           {inputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                           name + "/weightsIn",
                           mmOpt);
  return unflattenUnits(weightsInput);
}

Tensor createWeightsOutput(Graph &graph,
                           unsigned seqSize,
                           unsigned batchSize,
                           unsigned outputSize,
                           const std::string &dType,
                           const std::string &partialsType,
                           bool inferenceOnly,
                           const std::string &name
                           ) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  auto weightsOutput =
      createMatMulInputRHS(graph, dType,
                           {batchSize, outputSize},
                           {outputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                           "weightsOut",
                           mmOpt);
  return unflattenUnits(weightsOutput);
}

Tensor
calcSequenceWeightedInputs(Graph &graph,
                           const Tensor &in_,
                           const Tensor &weightsInput_,
                           program::Sequence &prog,
                           const std::string &partialsTypeStr,
                           const std::string &debugPrefix) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  auto sequenceSize = in_.dim(0);
  auto batchSize = in_.dim(1);
  auto inputSize = in_.dim(2);
  auto in = in_.reshape({sequenceSize * batchSize, inputSize});
  auto outputSize = weightsInput_.dim(2);
  auto weightsInput = flattenUnits(weightsInput_);
  return matMul(graph, in, weightsInput,
                prog, debugPrefix + "/Lstm/CalcWeighedInput", mmOpt)
           .reshape({sequenceSize, batchSize, BASIC_LSTM_CELL_NUM_UNITS,
                     outputSize})
           .dimShuffle({2, 0, 1, 3});
}


static Tensor
basicLstmCellForwardPassImpl(Graph &graph,
                             const Tensor &in_,
                             const Tensor &biases,
                             const Tensor &prevOutputAct,
                             const Tensor &prevCellState,
                             const Tensor *weightsInput,
                             const Tensor &weightsOutput,
                             Sequence &prog,
                             const std::string &partialsTypeStr,
                             bool inferenceOnly,
                             const std::string &debugPrefix) {
  unsigned sequenceSize;
  const unsigned outputSize = prevCellState.dim(1);
  const unsigned batchSize = prevCellState.dim(0);
  Tensor in = in_;

  if (weightsInput == nullptr) {
    sequenceSize = in.dim(1);
    assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
    assert(weightsOutput.dim(1) == outputSize);
    assert(weightsOutput.dim(2) == outputSize);
    in =  in_.dimShuffle({1, 0, 2, 3});
  } else {
    sequenceSize = in.dim(0);
#ifndef NDEBUG
    const unsigned inputSize = in.dim(2);
#endif
    assert(weightsInput->dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
    assert(weightsInput->dim(1) == inputSize);
    assert(weightsInput->dim(2) == outputSize);

    assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
    assert(weightsOutput.dim(1) == outputSize);
    assert(weightsOutput.dim(2) == outputSize);
  }

  const auto dType = in.elementType();

  auto bBiases = graph.addTensor(dType, {0, batchSize, outputSize}, "bbiases");
  for (unsigned u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto unitBias = biases[u].broadcast(batchSize, 0)
                             .reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  mmOpt.cache = &cache;
  unsigned stateDims = inferenceOnly ? LSTM_NUM_FWD_STATES_INFERENCE :
                                       LSTM_NUM_FWD_STATES_TRAINING;
  Tensor stateOut
    = graph.addTensor(dType, {0, stateDims, batchSize, outputSize}, "stateOut");

  for (auto s = 0U; s != sequenceSize; ++s) {
    const std::string baseStr = debugPrefix
                                + "/BasicLstmCell/"
                                + std::to_string(s);

    auto prevOutputActThisStep = s == 0 ? prevOutputAct :
                                          getOutputFromFwdState(stateOut[s-1]);
    auto prevCellStateThisStep = s == 0 ? prevCellState :
                                          getCellFromFwdState(stateOut[s-1]);
    Tensor unitsOutput;
    if (weightsInput == nullptr) {
      unitsOutput =
        basicLstmUnitsNlInputPreWeighted(graph,
                                         in[s],
                                         prevOutputActThisStep,
                                         weightsOutput,
                                         prog, mmOpt,
                                         baseStr + "/ProcessUnits");
    } else {
      unitsOutput =
        basicLstmUnitsNlInput(graph, in[s],
                              prevOutputActThisStep,
                              *weightsInput,
                              weightsOutput,
                              prog, mmOpt,
                              baseStr + "/ProcessUnits");
    }
    if (s == 0) {
      for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
        graph.setTileMapping(biases[u],
                             graph.getTileMapping(unitsOutput[u][0]));
      }
    }
    auto newState =
      basicLstmComputeOutput(graph, unitsOutput, prevCellStateThisStep, bBiases,
                             prog, !inferenceOnly, baseStr);
    stateOut = append(stateOut, newState);
  }
  return stateOut;
}

Tensor
basicLstmCellForwardPassWeightedInputs(Graph &graph,
                                       const Tensor &weightedIn,
                                       const Tensor &biases,
                                       const Tensor &prevOutputAct,
                                       const Tensor &prevCellState,
                                       const Tensor &weightsOutput,
                                       Sequence &prog,
                                       const std::string &partialsTypeStr,
                                       bool inferenceOnly,
                                       const std::string &debugPrefix) {
  return
    basicLstmCellForwardPassImpl(graph, weightedIn, biases, prevOutputAct,
                                 prevCellState, nullptr, weightsOutput,
                                 prog, partialsTypeStr, inferenceOnly,
                                 debugPrefix);
}

Tensor
basicLstmCellForwardPass(Graph &graph,
                         const Tensor &in,
                         const Tensor &biases,
                         const Tensor &prevOutputAct,
                         const Tensor &prevCellState,
                         const Tensor &weightsInput,
                         const Tensor &weightsOutput,
                         Sequence &prog,
                         const std::string &partialsTypeStr,
                         bool inferenceOnly,
                         const std::string &debugPrefix) {
  return
    basicLstmCellForwardPassImpl(graph, in, biases, prevOutputAct,
                                 prevCellState, &weightsInput, weightsOutput,
                                 prog, partialsTypeStr, inferenceOnly,
                                 debugPrefix);
}

std::pair<Tensor, Tensor>
basicLstmBackwardStep(Graph &graph,
                      const Tensor &gradNextLayer,
                      const Tensor &fwdStateThisStep,
                      const Tensor &prevCellState,
                      const Tensor &bwdState,
                      const Tensor &weightsInput,
                      const Tensor &weightsOutput,
                      Sequence &prog,
                      const std::string &partialsTypeStr,
                      const std::string &debugPrefix) {
  const auto fPrefix = debugPrefix + "/LstmBwd";
  auto gradSum =
    popstd::add(graph, getBwdState(bwdState, LSTM_BWD_STATE_GRAD_ACT_GRAD),
                gradNextLayer, prog, fPrefix + "/AddActGrads");
  auto actOutputGate =
    getFwdState(fwdStateThisStep, LSTM_FWD_STATE_ACTS_OUTPUT_GATE);
  auto actOutputTanh =
    getFwdState(fwdStateThisStep, LSTM_FWD_STATE_ACTS_OUTPUT_TANH);
  auto t =
    mul(graph, concat({actOutputGate, actOutputTanh}), gradSum.broadcast(2, 0),
        prog, fPrefix + "/MulOGate");
  auto gradAtOTanhInput = t.slice(0, gradSum.dim(0));
  auto gradAtOutputGateInput = t.slice(gradSum.dim(0), 2 * gradSum.dim(0));

  auto cs1 = graph.addComputeSet(fPrefix + "/OutputGate");
  auto gradAtOTanhOutput =
    nonLinearityInputGradient(graph, NonLinearityType::NON_LINEARITY_TANH,
                              actOutputTanh, gradAtOTanhInput, cs1,
                              fPrefix + "/OuputTanh");
  auto gradOutputGate =
    nonLinearityInputGradient(graph, NonLinearityType::NON_LINEARITY_SIGMOID,
                              actOutputGate, gradAtOutputGateInput, cs1,
                              fPrefix + "/OutputGate");
  prog.add(Execute(cs1));

  auto gradCellState = getBwdState(bwdState, LSTM_BWD_STATE_GRAD_CELL_STATE);

  addTo(graph, gradAtOTanhOutput, gradCellState, prog,
        fPrefix + "/AddCellState");
  auto actInputGate =
    getFwdState(fwdStateThisStep, LSTM_FWD_STATE_ACTS_INPUT_GATE);
  auto actCandidate =
    getFwdState(fwdStateThisStep, LSTM_FWD_STATE_ACTS_CAND_TANH);
  auto actForgetGate =
    getFwdState(fwdStateThisStep, LSTM_FWD_STATE_ACTS_FORGET_GATE);
  auto t1 =
    mul(graph,
        concat({actInputGate, actCandidate, prevCellState, actForgetGate}),
        gradAtOTanhOutput.broadcast(4, 0), prog, fPrefix);

  const auto batchSize = gradAtOTanhOutput.dim(0);
  auto gradAtCandTanhInput = t1.slice(0, batchSize);
  auto gradAtInputGateInput = t1.slice(batchSize, 2 * batchSize);
  auto gradAtForgetGateInput = t1.slice(2 * batchSize, 3 * batchSize);
  auto newGradCellState = t1.slice(3 * batchSize, 4 * batchSize);

  auto cs2 = graph.addComputeSet(fPrefix + "/{Input+Candidate}Gate");
  auto gradInputGate =
    nonLinearityInputGradient(graph, NonLinearityType::NON_LINEARITY_SIGMOID,
                              actInputGate, gradAtInputGateInput, cs2,
                              fPrefix + "/InputGate");
  auto gradCandidate =
    nonLinearityInputGradient(graph, NonLinearityType::NON_LINEARITY_TANH,
                              actCandidate, gradAtCandTanhInput, cs2,
                              fPrefix + "/Cand");
  auto gradForgetGate =
    nonLinearityInputGradient(graph, NonLinearityType::NON_LINEARITY_SIGMOID,
                              actForgetGate, gradAtForgetGateInput, cs2,
                              fPrefix + "/Cand");
  prog.add(Execute(cs2));

  auto gradUnits = concat({gradForgetGate.expand({0}),
                           gradInputGate.expand({0}),
                           gradCandidate.expand({0}),
                           gradOutputGate.expand({0})});

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.fullyConnectedPass = FullyConnectedPass::BWD;
  mmOpt.cache = &cache;
  auto gradientIn =
    matMul(graph,
           flattenUnits(gradUnits),
           flattenUnits(weightsInput).transpose(),
           prog,
           fPrefix + "/InputGrad", mmOpt);
  auto gradientPrevStep =
    matMul(graph,
           flattenUnits(gradUnits),
           flattenUnits(weightsOutput).transpose(),
           prog,
           fPrefix + "/PrevStepGrad", mmOpt);

  // update state
  auto newState = concat({newGradCellState.expand({0}),
                          gradientPrevStep.expand({0}),
                          gradForgetGate.expand({0}),
                          gradInputGate.expand({0}),
                          gradCandidate.expand({0}),
                          gradOutputGate.expand({0})});
  return std::make_pair(gradientIn, newState);
}

void
basicLstmParamUpdate(Graph &graph,
                     const Tensor &prevLayerActs,
                     const Tensor &prevStepActs,
                     const Tensor &bwdState,
                     const Tensor &weightsInputDeltaAcc,
                     const Tensor &weightsOutputDeltaAcc,
                     const Tensor &biasDeltaAcc,
                     Sequence &prog,
                     const std::string &partialsTypeStr,
                     const std::string &debugPrefix) {
  const auto fPrefix = debugPrefix + "/LstmDeltas";
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.fullyConnectedPass = FullyConnectedPass::WU;
  mmOpt.cache = &cache;
  auto gradUnits =
    concat({getBwdState(bwdState, LSTM_BWD_STATE_GRAD_FORGET_GATE).expand({0}),
            getBwdState(bwdState, LSTM_BWD_STATE_GRAD_INPUT_GATE).expand({0}),
            getBwdState(bwdState, LSTM_BWD_STATE_GRAD_CANDIDATE).expand({0}),
            getBwdState(bwdState,LSTM_BWD_STATE_GRAD_OUTPUT_GATE).expand({0})});

  matMulAcc(graph,
            concat(flattenUnits(weightsInputDeltaAcc),
                   flattenUnits(weightsOutputDeltaAcc)),
            1.0,
            concat(prevLayerActs.transpose(), prevStepActs.transpose()),
            flattenUnits(gradUnits),
            prog,
            fPrefix + "/Wi",
            mmOpt);
  popreduce::reduceAcc(graph, biasDeltaAcc, 1.0,
                       gradUnits.dimShuffle({1, 0, 2}), prog, fPrefix +"/Bias");
}

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize,
                                  bool weighInput) {
  uint64_t multsWeighInp = weighInput ?
      static_cast<uint64_t>(inputSize) * outputSize * batchSize * sequenceSize :
      0;
  uint64_t multsWeighOut =
      static_cast<uint64_t>(outputSize) * outputSize * batchSize * sequenceSize;

  uint64_t addsWeighInp  = weighInput ?
      static_cast<uint64_t>(inputSize - 1) * outputSize * batchSize
                                           * sequenceSize : 0;
  uint64_t addsWeighOut  =
      static_cast<uint64_t>(outputSize - 1) * outputSize * batchSize
                                            * sequenceSize;
  uint64_t hadamardProd =
      3 * static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  uint64_t cellStateAdd =
      static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;

  return 4 * (multsWeighInp + multsWeighOut + addsWeighInp + addsWeighOut)
         + hadamardProd + cellStateAdd;
}

uint64_t getBasicLstmCellBwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize) {
  uint64_t addFlopsUnit = sequenceSize * batchSize * outputSize;
  uint64_t multFlopsUnit = sequenceSize * batchSize * outputSize;
  uint64_t matMulFlops =  4 * static_cast<uint64_t>(sequenceSize) * batchSize
                           * outputSize * (inputSize + outputSize);
  uint64_t matMulAddFlops = 4 * static_cast<uint64_t>(sequenceSize) * batchSize
                            * outputSize * (inputSize + outputSize - 2);
  // A total of 5 non linearity derivatives each with two flops
  uint64_t nonlinearityGradCycles = addFlopsUnit * 5 * 2;

  uint64_t totalFlops = 2 * static_cast<uint64_t>(addFlopsUnit)
                        + nonlinearityGradCycles
                        + 6 * multFlopsUnit + matMulFlops
                        + matMulAddFlops + 6 * (sequenceSize * batchSize *
                                                (inputSize + outputSize));
  return totalFlops;
}

uint64_t getBasicLstmCellWuFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize) {
  uint64_t prevLayerActsFlops = 4 * static_cast<uint64_t>(inputSize)
                                * outputSize * batchSize * sequenceSize
                               + 4 * inputSize * outputSize * (batchSize - 1)
                                   * sequenceSize;
  uint64_t thisLayerActsFlops = 4 * static_cast<uint64_t>(outputSize)
                                  * outputSize * batchSize * sequenceSize
                               + 4 * outputSize * outputSize * (batchSize - 1)
                                 * sequenceSize;
  uint64_t biasFlops = 4 * (batchSize - 1) * outputSize * sequenceSize;
  return prevLayerActsFlops + thisLayerActsFlops + biasFlops;
}


} // namespace lstm
} // namespace popnn
