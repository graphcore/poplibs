#include <popnn/Lstm.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <poputil/VertexTemplates.hpp>
#include <popops/ElementWise.hpp>
#include <poplin/Convolution.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include "poplibs_support/OptionParsing.hpp"
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace poputil;
using namespace popnn;
using namespace popops;

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
    throw poputil::poplib_error("Unexpected state tensor dimensions");
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
    throw poputil::poplib_error("Unexpected state tensor dimensions");
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
  nonLinearity(graph, popnn::NonLinearityType::SIGMOID,
               sigmoidIn, cs, debugStr);
  nonLinearity(graph, popnn::NonLinearityType::TANH,
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
                                       OptionFlags &mmOpt,
                                       matmul::PlanningCache *cache,
                                       const std::string debugStr) {
  assert(weightedIn.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto output =
      unflattenUnits(matMul(graph, prevOutput, flattenUnits(weightsOutput),
                     prog, debugStr + "/WeighOutput", mmOpt, cache));
  addInPlace(graph, output, weightedIn, prog, debugStr + "/AddWeightedOutputs");
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
                      OptionFlags &mmOpt,
                      matmul::PlanningCache *cache,
                      const std::string &debugStr) {
  assert(weightsInput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto prodInp =
      unflattenUnits(matMul(graph, prevAct, flattenUnits(weightsInput),
                            prog, debugStr + "/WeighInput", mmOpt, cache));
  return
      basicLstmUnitsNlInputPreWeighted(graph, prodInp, prevOutput,
                                       weightsOutput, prog,
                                       mmOpt, cache, debugStr);
}

namespace popnn {
namespace lstm {

LstmParams::LstmParams(poplar::Type dataType,
                       std::size_t batchSize,
                       std::size_t timeSteps,
                       std::vector<std::size_t> layerSizes) :
  dataType(std::move(dataType)),
  batchSize(batchSize), timeSteps(timeSteps),
  layerSizes(std::move(layerSizes)) {}

struct LstmOpts {
  bool inferenceOnly;
  bool preCalcWeights;
  poplar::Type partialsType;
};

std::map<std::string, poplar::Type> partialsTypeMap {
  { "half", poplar::HALF },
  { "float", poplar::FLOAT }
};

static LstmOpts parseOptions(const OptionFlags &options) {
  LstmOpts lstmOpts;
  lstmOpts.inferenceOnly = false;
  lstmOpts.preCalcWeights = false;
  lstmOpts.partialsType = poplar::FLOAT;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec lstmSpec{
    { "inferenceOnly", OptionHandler::createWithBool(
      lstmOpts.inferenceOnly) },
    { "preCalcWeights", OptionHandler::createWithBool(
      lstmOpts.preCalcWeights) },
    { "partialsType", OptionHandler::createWithEnum(
      lstmOpts.partialsType, partialsTypeMap) }
  };
  options.list([&](const std::string &option, const std::string &value) {
    lstmSpec.parse(option, value);
  });
  return lstmOpts;
}

static void validateParams(const LstmParams &params) {
  if (params.layerSizes.size() != 2) {
    throw poplib_error("Invalid LSTM params (layerSize != 2)");
  }
}

Tensor createInput(Graph &graph, const LstmParams &params,
                   const std::string &name,
                   const OptionFlags &options,
                   matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);
  OptionFlags mmOpt{
    { "partialsType", opt.partialsType.toString() },
    { "fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                                "TRAINING_FWD" }
  };

  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  auto fcOutputSize = BASIC_LSTM_CELL_NUM_UNITS * outputSize;
  auto fcInputSize = inputSize;
  if (opt.preCalcWeights) {
    auto fcBatchSize = params.timeSteps * params.batchSize;
    auto in = createMatMulInputLHS(graph, params.dataType,
                                   {fcBatchSize, fcInputSize},
                                   {fcInputSize, fcOutputSize},
                                   name, mmOpt, cache);
    return in.reshape({params.timeSteps, params.batchSize, inputSize});
  } else {
    std::vector<Tensor> input;
    auto fcBatchSize = params.batchSize;
    auto in = createMatMulInputLHS(graph, params.dataType,
                                   {fcBatchSize, fcInputSize},
                                   {fcInputSize, fcOutputSize},
                                   name, mmOpt, cache);
    input.emplace_back(in.expand({0}));
    for (unsigned i = 1; i != params.timeSteps; ++i) {
      input.emplace_back(graph.clone(input[0],
                                     "input[" + std::to_string(i) + "]"));
    }
    return concat(input);
  }
}

Tensor createFwdState(Graph &graph, const LstmParams &params,
                      const std::string &debugPrefix,
                      const OptionFlags &options,
                      matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);
  auto stateDims = opt.inferenceOnly ? LSTM_NUM_FWD_STATES_INFERENCE :
                   LSTM_NUM_FWD_STATES_TRAINING;

  auto outputSize = params.layerSizes[1];
  auto state =
    graph.addVariable(params.dataType, {stateDims, params.batchSize,
                                        outputSize},
                      debugPrefix + "/fwdStateOut");
  for (auto i = 0; i != stateDims; ++i) {
    mapTensorLinearly(graph, state[i]);
  }
  return state;
}

void initFwdState(Graph &graph, const Tensor &state, bool zeroTrainingState,
                  Sequence &prog,
                  const std::string &debugPrefix) {
  if (zeroTrainingState) {
    zero(graph, state, prog, debugPrefix);
  } else {
    // zero internal state
    if (state.dim(0) > LSTM_NUM_FWD_STATES_INFERENCE) {
      zero(graph, state.slice(LSTM_NUM_FWD_STATES_INFERENCE,
                              LSTM_NUM_FWD_STATES_TRAINING, 0),
           prog, debugPrefix);
    }
  }
}

Tensor createBwdState(Graph &graph, const LstmParams &params,
                      const std::string &debugPrefix,
                      const OptionFlags &options,
                      matmul::PlanningCache *cache) {
  validateParams(params);
  auto outputSize = params.layerSizes[1];
  auto state =
    graph.addVariable(params.dataType,
                      {LSTM_NUM_BWD_STATES, params.batchSize, outputSize},
                      debugPrefix + "/BwdState");
  for (auto i = 0; i != LSTM_NUM_BWD_STATES; ++i) {
    mapTensorLinearly(graph, state[i]);
  }
  return state;
}

void initBwdState(Graph &graph, const Tensor &state,
                  Sequence &prog,
                  const std::string &debugPrefix) {
  zero(graph, state.slice(0, 2), prog, debugPrefix);
}

Tensor getOutputFromFwdState(const Tensor &fwdState) {
  return getFwdState(fwdState, LSTM_FWD_STATE_OUTPUT_ACTS);
}

Tensor getCellFromFwdState(const Tensor &fwdState) {
  return getFwdState(fwdState, LSTM_FWD_STATE_CELL_STATE);
}

LstmWeights
createWeights(Graph &graph, const LstmParams &params,
              const std::string &name,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);

  LstmWeights weights;
  OptionFlags mmOpt{
    { "partialsType", opt.partialsType.toString() },
    { "fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" }
  };
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  std::vector<std::size_t> aShape(2);
  aShape[0] = opt.preCalcWeights ? params.timeSteps * params.batchSize
                                 : params.batchSize;
  aShape[1] = inputSize;

  if (params.doInputWeightCalc) {
    auto weightsInput =
        createMatMulInputRHS(graph, params.dataType,
                             aShape,
    {inputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             name + "/weightsIn",
                             mmOpt, cache);
    weights.inputWeights = unflattenUnits(weightsInput);
  }
  auto weightsOutput =
      createMatMulInputRHS(graph, params.dataType,
                           {params.batchSize, outputSize},
                           {outputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                           "weightsOut",
                           mmOpt, cache);
  weights.outputWeights = unflattenUnits(weightsOutput);

  auto biases = graph.addVariable(params.dataType,
                                  {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                  "biases");
  mapTensorLinearly(graph, biases);
  weights.biases = biases;
  return weights;
}

static Tensor
calcSequenceWeightedInputs(Graph &graph,
                           const Tensor &in_,
                           const Tensor &weightsInput_,
                           program::Sequence &prog,
                           const Type &partialsType,
                           const std::string &debugPrefix,
                           matmul::PlanningCache *cache) {
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() }
  };
  auto sequenceSize = in_.dim(0);
  auto batchSize = in_.dim(1);
  auto inputSize = in_.dim(2);
  auto in = in_.reshape({sequenceSize * batchSize, inputSize});
  auto outputSize = weightsInput_.dim(2);
  auto weightsInput = flattenUnits(weightsInput_);
  return matMul(graph, in, weightsInput,
                prog, debugPrefix + "/Lstm/CalcWeighedInput", mmOpt, cache)
           .reshape({sequenceSize, batchSize, BASIC_LSTM_CELL_NUM_UNITS,
                     outputSize})
           .dimShuffle({0, 2, 1, 3});
}

struct LstmRecurrentState {
  Tensor output;
  Tensor cellState;

  Tensor getAsTensor() const {
    return concat(output.expand({0}), cellState.expand({0}));
  }
};

struct LstmInternalState {
  Tensor forgetGate;
  Tensor inputGate;
  Tensor candidate;
  Tensor outputGate;
  Tensor tanhOutput;

  Tensor getAsTensor() const {
    return concat({
      forgetGate.expand({0}),
      inputGate.expand({0}),
      candidate.expand({0}),
      outputGate.expand({0}),
      tanhOutput.expand({0})
    });
  }
};

static std::pair<LstmRecurrentState,LstmInternalState>
basicLstmCellForwardPass(Graph &graph,
                         const Tensor &in_,
                         const Tensor &biases,
                         const LstmRecurrentState &prevState,
                         const Tensor *weightsInput,
                         const Tensor &weightsOutput,
                         Sequence &prog,
                         const Type &partialsType,
                         bool inferenceOnly,
                         const std::string &debugPrefix,
                         matmul::PlanningCache *cache) {
  auto prevCellState = prevState.cellState;
  auto prevOutput = prevState.output;
  const unsigned outputSize = prevCellState.dim(1);
  const unsigned batchSize = prevCellState.dim(0);
  Tensor in = in_;

  if (weightsInput) {
#ifndef NDEBUG
    const unsigned inputSize = in.dim(1);
#endif
    assert(weightsInput->dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
    assert(weightsInput->dim(1) == inputSize);
    assert(weightsInput->dim(2) == outputSize);

  }
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

  const auto dType = in.elementType();

  auto bBiases = graph.addVariable(dType, {0, batchSize, outputSize},
                                   "bbiases");
  for (unsigned u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto unitBias = biases[u].broadcast(batchSize, 0)
                             .reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" }
  };
  const std::string baseStr = debugPrefix
                              + "/BasicLstmCell";

  Tensor unitsOutput;
  if (weightsInput == nullptr) {
    unitsOutput =
      basicLstmUnitsNlInputPreWeighted(graph,
                                       in,
                                       prevOutput,
                                       weightsOutput,
                                       prog, mmOpt, cache,
                                       baseStr + "/ProcessUnits");
  } else {
    unitsOutput =
      basicLstmUnitsNlInput(graph, in,
                            prevOutput,
                            *weightsInput,
                            weightsOutput,
                            prog, mmOpt, cache,
                            baseStr + "/ProcessUnits");
  }
  for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    graph.setTileMapping(biases[u],
                         graph.getTileMapping(unitsOutput[u][0]));
  }
  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[BASIC_LSTM_CELL_FORGET_GATE];
  auto candidate = unitsOutput[BASIC_LSTM_CELL_CANDIDATE];
  auto outputGate = unitsOutput[BASIC_LSTM_CELL_OUTPUT_GATE];
  auto inputGate = unitsOutput[BASIC_LSTM_CELL_INPUT_GATE];
  addInPlace(graph, unitsOutput, bBiases, prog, baseStr + "/AddBias");
  applyGateNonlinearities(graph, unitsOutput, prog, baseStr);

  auto prod = mul(graph, concat(forgetGate, candidate),
                  concat(prevCellState, inputGate), prog,
                  baseStr + "/{Forget + Input}Gate");

  auto updatedCellState = prod.slice(0, forgetGate.dim(0));
  auto updatedCandidate = prod.slice(forgetGate.dim(0),
                                     forgetGate.dim(0) + candidate.dim(0));

  addInPlace(graph, updatedCellState, updatedCandidate, prog,
             baseStr + "/AddCellCand");
  auto tanhOutput = popops::tanh(graph, updatedCellState, prog, baseStr);
  auto output = mul(graph, tanhOutput, outputGate, prog,
                    baseStr + "/OutputGate");
  LstmRecurrentState recurrentState = {output, updatedCellState};
  LstmInternalState internalState = {
    forgetGate,
    inputGate,
    candidate,
    outputGate,
    tanhOutput
  };
  return {recurrentState, internalState};
}

Tensor getRetainedState(const LstmRecurrentState &recurrentState,
                        const LstmInternalState &internalState,
                        const LstmOpts &opt) {
  if (opt.inferenceOnly) {
    return recurrentState.getAsTensor();
  }
  return concat({
    recurrentState.getAsTensor(),
    internalState.getAsTensor()
  });
}

Tensor lstmFwd(Graph &graph,
               const LstmParams &params,
               const Tensor &fwdStateInit,
               const Tensor &prevLayerActs,
               const LstmWeights &weights,
               Sequence &fwdProg,
               const std::string &debugPrefix,
               const OptionFlags &options,
               poplin::matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);

  Tensor weightedIn;
  if (!params.doInputWeightCalc) {
    weightedIn =
      graph.addVariable(params.dataType, {params.timeSteps,
                                          BASIC_LSTM_CELL_NUM_UNITS,
                                          params.batchSize,
                                          params.layerSizes[1]},
                                          "dummyWeightedIn");
    for (unsigned s = 0; s < params.timeSteps; ++s) {
      mapTensorLinearly(graph, weightedIn[s]);
    }
  } else if (opt.preCalcWeights) {
    weightedIn = calcSequenceWeightedInputs(graph, prevLayerActs,
                                            weights.inputWeights,
                                            fwdProg, opt.partialsType,
                                            debugPrefix + "/lstm/weightInputs",
                                            cache);
  }

  // loop counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1},
                                  debugPrefix + "/seqIdx");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, fwdProg, debugPrefix + "/seqIdx");

  // state for current layer, start from initialiser
  auto fwdStateInitCopy = duplicate(graph, fwdStateInit, fwdProg);
  LstmRecurrentState thisState = {
    fwdStateInitCopy[LSTM_FWD_STATE_OUTPUT_ACTS],
    fwdStateInitCopy[LSTM_FWD_STATE_CELL_STATE]
  };

  unsigned seqSize = prevLayerActs.dim(0);
  // core lstm loop
  auto loop = Sequence();
  bool useWeightedIn = !params.doInputWeightCalc || opt.preCalcWeights;
  Tensor fwdInput;
  const Tensor *inputWeightsPtr;
  if (useWeightedIn) {
    fwdInput = popops::dynamicSlice(
      graph, weightedIn, seqIdx, {0}, {1}, loop,
      debugPrefix + "/lstmWeighted")[0];
    inputWeightsPtr = nullptr;
  } else {
    fwdInput = popops::dynamicSlice(
      graph, prevLayerActs, seqIdx, {0}, {1}, loop,
      debugPrefix + "/lstm")[0];
    inputWeightsPtr = &weights.inputWeights;
  }
  LstmRecurrentState newState;
  LstmInternalState internalState;
  std::tie(newState, internalState) =
      basicLstmCellForwardPass(
          graph, fwdInput, weights.biases,
          thisState,
          inputWeightsPtr, weights.outputWeights,
          loop, opt.partialsType, opt.inferenceOnly, debugPrefix,
          cache);
  auto retainedState = getRetainedState(newState, internalState, opt);
  // all output sequence elements take the same mapping so will only
  // require on-tile copies
  auto retainedStateSeq =
      graph.addVariable(params.dataType,
                        {seqSize, retainedState.dim(0), retainedState.dim(1),
                         retainedState.dim(2)},
                         debugPrefix + "/fwdState");
  for (unsigned i = 0; i != seqSize; ++i) {
    graph.setTileMapping(retainedStateSeq[i],
                         graph.getTileMapping(retainedState));
  }
  auto thisStateTensor = thisState.getAsTensor();
  auto newStateTensor = newState.getAsTensor();
  graph.setTileMapping(thisStateTensor, graph.getTileMapping(newStateTensor));
  loop.add(Copy(newStateTensor, thisStateTensor));

  popops::dynamicUpdate(
      graph, retainedStateSeq, retainedState.expand({0}), seqIdx, {0}, {1},
      loop, debugPrefix + "/lstmUpdateState");

  addInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxIncr");

  fwdProg.add(WriteUndef(retainedStateSeq));
  fwdProg.add(Repeat(seqSize, loop));
  return retainedStateSeq;
}

static std::pair<Tensor, Tensor>
BackwardStepImpl(Graph &graph,
                      const Tensor &gradNextLayer,
                      const Tensor &fwdStateThisStep,
                      const Tensor &prevCellState,
                      const Tensor &bwdState,
                      const Tensor *weightsInput,
                      const Tensor *weightsOutput,
                      Sequence &prog,
                      const Type &partialsType,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmBwd";
  auto gradSum =
    popops::add(graph, getBwdState(bwdState, LSTM_BWD_STATE_GRAD_ACT_GRAD),
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
    nonLinearityInputGradient(graph, NonLinearityType::TANH,
                              actOutputTanh, gradAtOTanhInput, cs1,
                              fPrefix + "/OuputTanh");
  auto gradOutputGate =
    nonLinearityInputGradient(graph, NonLinearityType::SIGMOID,
                              actOutputGate, gradAtOutputGateInput, cs1,
                              fPrefix + "/OutputGate");
  prog.add(Execute(cs1));

  auto gradCellState = getBwdState(bwdState, LSTM_BWD_STATE_GRAD_CELL_STATE);

  addInPlace(graph, gradAtOTanhOutput, gradCellState, prog,
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
    nonLinearityInputGradient(graph, NonLinearityType::SIGMOID,
                              actInputGate, gradAtInputGateInput, cs2,
                              fPrefix + "/InputGate");
  auto gradCandidate =
    nonLinearityInputGradient(graph, NonLinearityType::TANH,
                              actCandidate, gradAtCandTanhInput, cs2,
                              fPrefix + "/Cand");
  auto gradForgetGate =
    nonLinearityInputGradient(graph, NonLinearityType::SIGMOID,
                              actForgetGate, gradAtForgetGateInput, cs2,
                              fPrefix + "/Cand");
  prog.add(Execute(cs2));

  auto gradUnits = concat({gradForgetGate.expand({0}),
                           gradInputGate.expand({0}),
                           gradCandidate.expand({0}),
                           gradOutputGate.expand({0})});

  Tensor gradientIn;
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", "TRAINING_BWD" }
  };

  if (weightsInput != nullptr) {
    gradientIn =
    matMul(graph,
           flattenUnits(gradUnits),
           flattenUnits(*weightsInput).transpose(),
           prog,
           fPrefix + "/InputGrad", mmOpt, cache);
  }
  auto gradientPrevStep =
    matMul(graph,
           flattenUnits(gradUnits),
           flattenUnits(*weightsOutput).transpose(),
           prog,
           fPrefix + "/PrevStepGrad", mmOpt, cache);

  // update state
  auto newState = concat({newGradCellState.expand({0}),
                          gradientPrevStep.expand({0}),
                          gradForgetGate.expand({0}),
                          gradInputGate.expand({0}),
                          gradCandidate.expand({0}),
                          gradOutputGate.expand({0})});
  return std::make_pair(gradientIn, newState);
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
                      const Type &partialsType,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  Tensor gradientIn, gradAtPrevOutput;
  return
    BackwardStepImpl(graph, gradNextLayer, fwdStateThisStep, prevCellState,
                     bwdState, &weightsInput, &weightsOutput, prog,
                     partialsType, debugPrefix, cache);
}

Tensor
basicLstmBackwardStep(Graph &graph,
                      const Tensor &gradNextLayer,
                      const Tensor &fwdStateThisStep,
                      const Tensor &prevCellState,
                      const Tensor &bwdState,
                      const Tensor &weightsOutput,
                      Sequence &prog,
                      const Type &partialsType,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  Tensor gradientIn, gradAtPrevOutput;
  std::tie(gradientIn, gradAtPrevOutput) =
    BackwardStepImpl(graph, gradNextLayer, fwdStateThisStep, prevCellState,
                     bwdState, nullptr, &weightsOutput, prog,
                     partialsType, debugPrefix, cache);
  return gradAtPrevOutput;
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
                     const Type &partialsType,
                     const std::string &debugPrefix,
                     matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmDeltas";
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", "TRAINING_WU" }
  };
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
            mmOpt, cache);

  popops::reduceWithOutput(graph, gradUnits, biasDeltaAcc, {1},
                           {popops::Operation::ADD, 1.0f, true},
                           prog, fPrefix +"/Bias");
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
  lstmBwd(
    Graph &graph, const LstmParams &params,
    bool doWU,
    Sequence &prog,
    const Tensor &fwdStateInit,
    const Tensor &fwdState,
    const LstmWeights &weights,
    const Tensor &prevLayerActs,
    const Tensor &gradLayerNext,
    const Tensor &bwdState,
    const std::string &debugPrefix,
    const OptionFlags &options,
    poplin::matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);
  auto &weightsInput = weights.inputWeights;
  auto &weightsOutput = weights.outputWeights;
  auto &biases = weights.biases;

  Tensor gradPrevLayer, weightsInputDeltasAcc, weightsOutputDeltasAcc,
         biasDeltasAcc;
  if (doWU) {
    weightsInputDeltasAcc =
      graph.clone(weightsInput, "WeightsInputDeltasAcc");
    weightsOutputDeltasAcc =
      graph.clone(weightsOutput, "WeightsOutputDeltasAcc");
    biasDeltasAcc = graph.clone(biases, "biasDeltasAcc");
    popops::zero(graph, weightsInputDeltasAcc, prog);
    popops::zero(graph, weightsOutputDeltasAcc, prog);
    popops::zero(graph, biasDeltasAcc, prog);
  }

  unsigned seqSize = gradLayerNext.dim(0);
  // sequence down-counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start = graph.addConstant(UNSIGNED_INT, {1}, seqSize - 1);
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  // input state from the previous layer
  Tensor fwdStateS = duplicate(graph, fwdState[seqSize - 1],
                               prog);
  graph.setTileMapping(fwdStateS, graph.getTileMapping(fwdStateInit));
  auto gradSize = params.calcInputGradients ? weightsInput.dim(1)
                                            : weightsOutput.dim(2);

  // output gradient to previous layer
  gradPrevLayer = graph.addVariable(params.dataType,
                                    {seqSize, gradLayerNext.dim(1), gradSize},
                                    debugPrefix + "/gradPrevLayer");
  auto loop = Sequence();
  {
    Tensor gradPrevLayerS, bwdStateUpdated;
    // fwdStateM1 is an offset version of fwdState
    Tensor fwdStateM1 = concat(fwdStateInit.expand({0}),
                               fwdState.slice(0, fwdState.dim(0) - 1));

    Tensor fwdStateM1Copy = duplicate(graph, fwdStateM1, prog);
    for (unsigned s = 0; s != fwdStateM1Copy.dim(0); ++s)
      graph.setTileMapping(fwdStateM1Copy[s],
                           graph.getTileMapping(fwdStateS));
    Tensor fwdStateM1S =
      dynamicSlice(graph, fwdStateM1Copy, seqIdx, {0}, {1}, loop,
                   debugPrefix + "/fwdStateM1").squeeze({0});
    Tensor outGradientShufS =
      dynamicSlice(graph, gradLayerNext, seqIdx, {0}, {1}, loop,
                   debugPrefix + "/gradLayerNext").squeeze({0});
    Tensor cellState = popnn::lstm::getCellFromFwdState(fwdStateM1S);
    if (!params.calcInputGradients) {
      bwdStateUpdated = popnn::lstm::basicLstmBackwardStep(
        graph, outGradientShufS, fwdStateS, cellState, bwdState,
        weightsOutput, loop,
        opt.partialsType, debugPrefix, cache);
      for (unsigned s = 0; s != seqSize; ++s)
        mapTensorLinearly(graph, gradPrevLayer[s]);
    } else {
      std::tie(gradPrevLayerS, bwdStateUpdated) =
        popnn::lstm::basicLstmBackwardStep(
          graph, outGradientShufS, fwdStateS, cellState, bwdState,
          weightsInput, weightsOutput, loop,
          opt.partialsType, debugPrefix, cache);
      gradPrevLayerS = gradPrevLayerS.expand({0});
      for (unsigned s = 0; s != seqSize; ++s)
        graph.setTileMapping(gradPrevLayer[s],
                             graph.getTileMapping(gradPrevLayerS));
      dynamicUpdate(graph, gradPrevLayer, gradPrevLayerS,
                    seqIdx, {0}, {1}, loop,
                    debugPrefix + "/gradPrevLayer");
    }
    if (doWU) {
      Tensor actsOutS = popnn::lstm::getOutputFromFwdState(fwdStateM1S);
      Tensor actsInS = dynamicSlice(graph, prevLayerActs,seqIdx, {0}, {1}, loop,
                                    debugPrefix + "/prevLayerActs")
                                   .squeeze({0});
      popnn::lstm::basicLstmParamUpdate(
        graph, actsInS, actsOutS,
        bwdStateUpdated,
        weightsInputDeltasAcc, weightsOutputDeltasAcc, biasDeltasAcc,
        loop, opt.partialsType, debugPrefix, cache);
    }
    loop.add(Copy(fwdStateM1S, fwdStateS));
    loop.add(Copy(bwdStateUpdated, bwdState));
    subInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxDecr");
  }
  prog.add(Repeat(seqSize, loop));
  return std::tie(gradPrevLayer, weightsInputDeltasAcc,
                  weightsOutputDeltasAcc, biasDeltasAcc);
};

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  auto weighInput = params.doInputWeightCalc;
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

uint64_t getBasicLstmCellBwdFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  auto calcInputGrad = params.calcInputGradients;
  uint64_t addFlopsUnit = sequenceSize * batchSize * outputSize;
  uint64_t multFlopsUnit = sequenceSize * batchSize * outputSize;
  uint64_t matMulFlops =  4 * static_cast<uint64_t>(sequenceSize) * batchSize *
                          outputSize * (inputSize * calcInputGrad + outputSize);
  uint64_t matMulAddFlops = 4 * static_cast<uint64_t>(sequenceSize) * batchSize
                            * outputSize *
                            ((inputSize - 1) * calcInputGrad + outputSize - 1);
  // A total of 5 non linearity derivatives each with two flops
  uint64_t nonlinearityGradCycles = addFlopsUnit * 5 * 2;

  uint64_t totalFlops = 2 * static_cast<uint64_t>(addFlopsUnit)
                        + nonlinearityGradCycles
                        + 6 * multFlopsUnit + matMulFlops
                        + matMulAddFlops
                        // adding 4 gradients
                        + 3 * static_cast<uint64_t>(sequenceSize) * batchSize
                            * outputSize
                        + 3 * static_cast<uint64_t>(sequenceSize) * batchSize
                            * inputSize * calcInputGrad;
  return totalFlops;
}

uint64_t getBasicLstmCellWuFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  uint64_t prevLayerActsFlops = 4 * static_cast<uint64_t>(inputSize)
                                * outputSize * batchSize * sequenceSize
                               + 4 * static_cast<uint64_t>(inputSize) *
                                 outputSize * (batchSize - 1) * sequenceSize;
  uint64_t thisLayerActsFlops = 4 * static_cast<uint64_t>(outputSize)
                                  * outputSize * batchSize * sequenceSize
                               + 4 * static_cast<uint64_t>(outputSize) *
                                 outputSize * (batchSize - 1)
                                 * sequenceSize;
  uint64_t biasFlops = 4 * (batchSize - 1) * static_cast<uint64_t>(outputSize)
                         * sequenceSize;
  return prevLayerActsFlops + thisLayerActsFlops + biasFlops;
}


} // namespace lstm
} // namespace popnn
