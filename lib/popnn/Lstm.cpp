#include "RnnUtil.hpp"
#include <popnn/Lstm.hpp>

using namespace popnn::Rnn;

// Tensor elements maintained in forward state. The number of elements is a
// function of the amount of recomputation done in the backward pass
enum FwdIntermediates {
  // Saved unless doing full recomputation
  LSTM_FWD_INTERMEDIATE_FORGET_GATE,
  LSTM_FWD_INTERMEDIATE_INPUT_GATE,
  LSTM_FWD_INTERMEDIATE_CAND_TANH,
  LSTM_FWD_INTERMEDIATE_OUTPUT_GATE,

  // Saved unless doing fast/full recomputation
  LSTM_FWD_INTERMEDIATE_OUTPUT_TANH,
  LSTM_FWD_INTERMEDIATE_PREV_CELL_STATE,

  // Saved if `outputFullSequence` is not set i.e. outputs aren't already
  // saved as part of the forward pass output.
  // TODO: Never currently recomputed even if !outputFullSequence
  LSTM_FWD_INTERMEDIATE_OUTPUT
};

// Tensor elements maintained in backward state. The number of elements is a
// function of the amount of recomputation done in the weight update pass
enum BwdStateTensorElems {
  LSTM_BWD_STATE_GRAD_CELL_STATE = 0,
  LSTM_BWD_STATE_GRAD_ACT_GRAD,
  LSTM_NUM_BWD_STATES
};

static void
applyGateNonlinearities(Graph &graph,
                        const Tensor &t,
                        Sequence &prog,
                        const std::string &debugStr) {
  auto sigmoidIn = concat({t[BASIC_LSTM_CELL_INPUT_GATE],
                           t[BASIC_LSTM_CELL_FORGET_GATE],
                           t[BASIC_LSTM_CELL_OUTPUT_GATE]});
  auto cs = graph.addComputeSet(debugStr + "/OutputGate");
  nonLinearityInPlace(graph, popnn::NonLinearityType::SIGMOID,
                      sigmoidIn, cs, debugStr);
  nonLinearityInPlace(graph, popnn::NonLinearityType::TANH,
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
                     prog, debugStr + "/WeighOutput", mmOpt, cache),
                     BASIC_LSTM_CELL_NUM_UNITS);
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
  auto weights = concat(weightsInput, weightsOutput, 1);
  return
      unflattenUnits(matMul(graph, concat(prevAct, prevOutput, 1),
                            flattenUnits(weights), prog,
                            debugStr + "/Weigh", mmOpt, cache),
                     BASIC_LSTM_CELL_NUM_UNITS);
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

enum class LstmRecomputationMode {
  // No recomputation in the backwards pass.
  None,
  // Small amount of recomputation in the backwards pass, yielding
  // some reduction in memory footprint for the layer.
  CellAndTanh,
  // Recompute everything from the forward pass. Saves the most memory
  // at the cost of an extra forward pass of cycles.
  Full
};

struct LstmOpts {
  bool inferenceOnly;
  bool preCalcWeights;
  poplar::Type partialsType;
  LstmRecomputationMode recomputationMode;
  boost::optional<double> availableMemoryProportion;
};

std::map<std::string, poplar::Type> partialsTypeMap {
  { "half", poplar::HALF },
  { "float", poplar::FLOAT }
};

std::map<std::string, LstmRecomputationMode> recomputationModeMap {
  { "none", LstmRecomputationMode::None },
  { "cellAndTanh", LstmRecomputationMode::CellAndTanh },
  { "full", LstmRecomputationMode::Full }
};

static OptionFlags getMMOpts(const LstmOpts &lstmOpts) {
  OptionFlags mmOpts = {
    { "partialsType", lstmOpts.partialsType.toString() },
  };
  if (lstmOpts.availableMemoryProportion) {
    mmOpts.set("availableMemoryProportion",
               std::to_string(lstmOpts.availableMemoryProportion.get()));
  }
  return mmOpts;
}

static LstmOpts parseOptions(const OptionFlags &options) {
  LstmOpts lstmOpts;
  lstmOpts.inferenceOnly = false;
  lstmOpts.preCalcWeights = false;
  lstmOpts.partialsType = poplar::FLOAT;
  lstmOpts.recomputationMode = LstmRecomputationMode::None;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec lstmSpec{
    { "inferenceOnly", OptionHandler::createWithBool(
      lstmOpts.inferenceOnly) },
    { "preCalcWeights", OptionHandler::createWithBool(
      lstmOpts.preCalcWeights) },
    { "partialsType", OptionHandler::createWithEnum(
      lstmOpts.partialsType, partialsTypeMap) },
    { "recomputationMode", OptionHandler::createWithEnum(
      lstmOpts.recomputationMode, recomputationModeMap) },
    { "availableMemoryProportion", OptionHandler::createWithDouble(
      lstmOpts.availableMemoryProportion) },
  };
  for (const auto &entry : options) {
    lstmSpec.parse(entry.first, entry.second);
  }
  return lstmOpts;
}

static void validateParams(const LstmParams &params) {
  if (params.layerSizes.size() != 2) {
    throw poplibs_error("Invalid LSTM params (layerSize != 2)");
  }
}

/// Create and map a tensor for a sequence of outputs from a LSTM layer.
/// The sequence length is taken from \a sequenceLength parameter, not the
/// \a params structure.
static Tensor
createOutputTensor(Graph &graph,
                   const LstmParams &params,
                   unsigned sequenceLength,
                   const std::string &name) {
  const auto outputSize = params.layerSizes[1];
  const auto batchSize = params.batchSize;
  // TODO take output grouping from matmul operation.
  const auto outputGrouping = gcd(16UL, outputSize);
  const auto numGroups = (outputSize * batchSize) / outputGrouping;
  auto output =
      createDynamicSliceTensor(graph, params.dataType, sequenceLength,
                               numGroups, outputGrouping, name)
      .reshapePartial(1, 2, {outputSize / outputGrouping, batchSize})
      .dimRoll(1, 2)
      .flatten(2, 4);
  return output;
}

Tensor createInput(Graph &graph, const LstmParams &params,
                   const std::string &name,
                   const OptionFlags &options,
                   matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                                      "TRAINING_FWD");

  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  if (opt.preCalcWeights) {
    auto fcOutputSize = BASIC_LSTM_CELL_NUM_UNITS * outputSize;
    auto fcInputSize = inputSize;
    auto fcBatchSize = params.timeSteps * params.batchSize;
    auto in = createMatMulInputLHS(graph, params.dataType,
                                   {fcBatchSize, fcInputSize},
                                   {fcInputSize, fcOutputSize},
                                   name, mmOpt, cache);
    return in.reshape({params.timeSteps, params.batchSize, inputSize});
  } else {
    const auto batchSize = params.batchSize;
    // TODO take input grouping from matmul operation.
    const auto inputGrouping = gcd(16UL, inputSize);
    const auto numInputGroups = (inputSize * batchSize) / inputGrouping;
    auto in = createDynamicSliceTensor(graph, params.dataType,
                                       params.timeSteps, numInputGroups,
                                       inputGrouping,
                                       name);
    return
        in.reshapePartial(1, 2, {inputSize / inputGrouping, batchSize})
          .dimRoll(1, 2)
          .flatten(2, 4);
  }
}

static poplar::Tensor
createStateTensor(Graph &graph, const LstmParams &params,
                  const std::string &name, const OptionFlags &options,
                  matmul::PlanningCache *cache) {
  validateParams(params);
  return createOutputTensor(graph, params, 1, name).squeeze({0});
}

poplar::Tensor
createInitialOutput(Graph &graph, const LstmParams &params,
                    const std::string &debugPrefix,
                    const OptionFlags &options,
                    matmul::PlanningCache *cache) {
  return createStateTensor(graph, params, debugPrefix + "/initialOutput",
                           options, cache);
}

poplar::Tensor
createInitialCellState(Graph &graph, const LstmParams &params,
                       const std::string &debugPrefix,
                       const OptionFlags &options,
                       matmul::PlanningCache *cache) {
  return createStateTensor(graph, params, debugPrefix + "/initialCellState",
                           options, cache);

}

LstmState createInitialState(Graph &graph, const LstmParams &params,
                             const std::string &debugPrefix,
                             const OptionFlags &options,
                             matmul::PlanningCache *cache) {
  auto initialOutput = createInitialOutput(graph, params, debugPrefix, options,
                                           cache);
  auto initialCellState = graph.clone(initialOutput,
                                      debugPrefix + "/initialCellState");
  return {initialOutput, initialCellState};
}

void zeroInitialState(Graph &graph, const LstmState &state,
                      Sequence &prog, const std::string &debugPrefix) {
  zero(graph, concat(state.output, state.cellState), prog, debugPrefix);
}

std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const LstmParams &params,
                    const std::string &name,
                    const poplar::OptionFlags &options,
                    poplin::matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                                      "TRAINING_FWD");
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  poplar::Tensor inputWeights;
  poplar::Tensor outputWeights;
  if (opt.preCalcWeights) {
    if (params.doInputWeightCalc) {
      std::vector<std::size_t> aShape(2);
      aShape[0] = opt.preCalcWeights ? params.timeSteps * params.batchSize
                                     : params.batchSize;
      aShape[1] = inputSize;
      auto weightsInput =
          createMatMulInputRHS(graph, params.dataType,
                               aShape,
      {inputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                               name + "/weightsIn",
                               mmOpt, cache);
      inputWeights = unflattenUnits(weightsInput, BASIC_LSTM_CELL_NUM_UNITS);
    }
    auto weightsOutput =
        createMatMulInputRHS(graph, params.dataType,
                             {params.batchSize, outputSize},
                             {outputSize,
                              BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             name + "/weightsOut",
                             mmOpt, cache);
    outputWeights = unflattenUnits(weightsOutput, BASIC_LSTM_CELL_NUM_UNITS);
  } else {
    auto weights =
        createMatMulInputRHS(graph, params.dataType,
                             {params.batchSize, inputSize + outputSize},
                             {inputSize + outputSize,
                              BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             name + "/weights",
                             mmOpt, cache);
    inputWeights = unflattenUnits(weights.slice(0, inputSize),
                                  BASIC_LSTM_CELL_NUM_UNITS);
    outputWeights = unflattenUnits(weights.slice(inputSize,
                                                 inputSize + outputSize),
                                   BASIC_LSTM_CELL_NUM_UNITS);
  }
  return {inputWeights, outputWeights};
}

/** Create the weights biases.
 */
poplar::Tensor
createWeightsBiases(poplar::Graph &graph, const LstmParams &params,
                    const std::string &name,
                    const OptionFlags &,
                    poplin::matmul::PlanningCache *) {
  validateParams(params);
  auto outputSize = params.layerSizes[1];
  auto biases = graph.addVariable(params.dataType,
                                  {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                  name + "/biases");
  mapTensorLinearly(graph, biases);
  return biases;
}

LstmWeights
createWeights(Graph &graph, const LstmParams &params,
              const std::string &name,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {

  LstmWeights lstmWeights;
  std::tie(lstmWeights.inputWeights, lstmWeights.outputWeights) =
    createWeightsKernel(graph, params, name, options, cache);
  lstmWeights.biases = createWeightsBiases(graph, params, name, options, cache);
  return lstmWeights;
}

static Tensor
calcSequenceWeightedInputs(Graph &graph,
                           const Tensor &in_,
                           const Tensor &weightsInput_,
                           program::Sequence &prog,
                           const LstmOpts &opt,
                           const std::string &debugPrefix,
                           matmul::PlanningCache *cache) {
  auto mmOpt = getMMOpts(opt);
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

Tensor LstmState::getAsTensor() const {
  return concat({
    output.expand({0}),
    cellState.expand({0})
  });
}

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

static const char *getUnitName(BasicLstmCellUnit unit) {
  switch (unit) {
  default: POPLIB_UNREACHABLE();
  case BASIC_LSTM_CELL_FORGET_GATE: return "ForgetGate";
  case BASIC_LSTM_CELL_INPUT_GATE: return "InputGate";
  case BASIC_LSTM_CELL_CANDIDATE: return "Candidate";
  case BASIC_LSTM_CELL_OUTPUT_GATE: return "OutputGate";
  }
}

static void rearrangeUnitsOutputFwd(Graph &graph, Tensor outputUnits,
                                    Tensor outputUnitsRearranged,
                                    Sequence &prog,
                                    const std::string &debugPrefix) {
  const auto outputGrouping =
      detectInnermostGrouping(graph, outputUnitsRearranged);
  // Typically the matrix multiplication result is laid out in memory such
  // that innermost dimension is groups batch elements. Try to rearrange the
  // result so the innermost dimension of the underlying memory is groups of the
  // specified number of outputs.
  outputUnits =
      unflattenUnits(
        tryGroupedPartialTranspose(graph, flattenUnits(outputUnits),
                                   outputGrouping, prog, debugPrefix),
        BASIC_LSTM_CELL_NUM_UNITS);
  prog.add(Copy(outputUnits, outputUnitsRearranged));
}

static void
lstmCellForwardPassCalcUnits(Graph &graph,
                             const Tensor &in,
                             const Tensor &biases,
                             const LstmState &prevState,
                             const Tensor *weightsInput,
                             const Tensor &weightsOutput,
                             Sequence &prog,
                             const LstmOpts &opt,
                             bool inferenceOnly,
                             const Tensor &unitsOutputRearranged,
                             const std::string &baseStr,
                             matmul::PlanningCache *cache) {
  auto prevCellState = prevState.cellState;
  auto prevOutput = prevState.output;
  const unsigned outputSize = prevOutput.dim(1);
  const unsigned batchSize = prevOutput.dim(0);

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
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" :
                                                  "TRAINING_FWD");

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

  // Rearrange the output of the matrix multiplication so each output unit
  // arranged the same as the cell state. This avoids the rearrangement
  // during the subsequent binary operations.
  rearrangeUnitsOutputFwd(graph, unitsOutput, unitsOutputRearranged,
                          prog, baseStr);

  for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    graph.setTileMapping(biases[u],
                         graph.getTileMapping(unitsOutputRearranged[u][0]));
  }
  addInPlace(graph, unitsOutputRearranged, bBiases, prog, baseStr + "/AddBias");
  applyGateNonlinearities(graph, unitsOutputRearranged, prog, baseStr);
}

static std::pair<LstmState,LstmInternalState>
basicLstmCellForwardPass(Graph &graph,
                         const Tensor &in,
                         const Tensor &biases,
                         const LstmState &prevState,
                         const Tensor *weightsInput,
                         const Tensor &weightsOutput,
                         Sequence &prog,
                         const LstmOpts &opt,
                         bool inferenceOnly,
                         const std::string &debugPrefix,
                         matmul::PlanningCache *cache) {
  auto prevCellState = prevState.cellState;
  const std::string baseStr = debugPrefix
                              + "/BasicLstmCell";

  std::vector<Tensor> toConcat;
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    toConcat.push_back(
      graph.clone(prevCellState,
                  debugPrefix + "/" +
                  getUnitName(BasicLstmCellUnit(i)) +
                  "Rearranged").expand({0})
    );
  }
  auto unitsOutput = concat(toConcat);
  lstmCellForwardPassCalcUnits(graph, in, biases, prevState, weightsInput,
                               weightsOutput, prog, opt,
                               inferenceOnly, unitsOutput, baseStr, cache);

  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[BASIC_LSTM_CELL_FORGET_GATE];
  auto candidate = unitsOutput[BASIC_LSTM_CELL_CANDIDATE];
  auto outputGate = unitsOutput[BASIC_LSTM_CELL_OUTPUT_GATE];
  auto inputGate = unitsOutput[BASIC_LSTM_CELL_INPUT_GATE];
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
  LstmState recurrentState = {output, updatedCellState};
  LstmInternalState internalState = {
    forgetGate,
    inputGate,
    candidate,
    outputGate,
    tanhOutput
  };
  return {recurrentState, internalState};
}

static void
basicLstmCellForwardPassInPlace(Graph &graph,
                                const Tensor &in,
                                const Tensor &biases,
                                const LstmState &state,
                                const Tensor *weightsInput,
                                const Tensor &weightsOutput,
                                Sequence &prog,
                                const LstmOpts &opt,
                                bool inferenceOnly,
                                const std::string &debugPrefix,
                                matmul::PlanningCache *cache) {
  auto cellState = state.cellState;
  auto output = state.output;
  const std::string baseStr = debugPrefix
                              + "/BasicLstmCell";

  std::vector<Tensor> toConcat;
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    if (i == BASIC_LSTM_CELL_OUTPUT_GATE) {
      toConcat.push_back(output.expand({0}));
    } else {
      toConcat.push_back(
        graph.clone(cellState,
                    debugPrefix + "/" +
                    getUnitName(BasicLstmCellUnit(i)) +
                    "Rearranged").expand({0})
      );
    }
  }
  auto unitsOutput = concat(toConcat);
  lstmCellForwardPassCalcUnits(graph, in, biases, state, weightsInput,
                               weightsOutput, prog, opt,
                               inferenceOnly, unitsOutput, baseStr, cache);

  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[BASIC_LSTM_CELL_FORGET_GATE];
  auto candidate = unitsOutput[BASIC_LSTM_CELL_CANDIDATE];
  auto outputGate = unitsOutput[BASIC_LSTM_CELL_OUTPUT_GATE];
  auto inputGate = unitsOutput[BASIC_LSTM_CELL_INPUT_GATE];
  using namespace popops::expr;
  mulInPlace(graph, concat(cellState, candidate),
             concat(forgetGate, inputGate), prog,
             baseStr + "/{Forget + Input}Gate");
  addInPlace(graph, cellState, candidate, prog,
             baseStr + "/AddCellCand");
  mapInPlace(graph, Mul(_1, Tanh(_2)), {outputGate, cellState}, prog,
             baseStr + "/CalcNextOutput");
}

static Tensor getFwdIntermediatesToSave(const LstmState &state,
                                        const LstmState &newState,
                                        const LstmInternalState &internalState,
                                        const LstmOpts &options,
                                        const LstmParams &params) {
  Tensor intermediates;
  switch(options.recomputationMode) {
    case LstmRecomputationMode::None:
      intermediates = concat({internalState.forgetGate.expand({0}),
                              internalState.inputGate.expand({0}),
                              internalState.candidate.expand({0}),
                              internalState.outputGate.expand({0}),
                              internalState.tanhOutput.expand({0}),
                              state.cellState.expand({0})});
      break;
    case LstmRecomputationMode::CellAndTanh:
      intermediates = concat({internalState.forgetGate.expand({0}),
                              internalState.inputGate.expand({0}),
                              internalState.candidate.expand({0}),
                              internalState.outputGate.expand({0})});
      break;
    case LstmRecomputationMode::Full:
    default:
      throw poputil::poplibs_error("Unhandled recomputation type");
  }

  if (!params.outputFullSequence) {
    // TODO: It may be cheaper to save the previous output rather than
    // the output for the current step here for the backward pass so that
    // when we aren't saving the full output sequence we can avoid
    // unrolling the last step in the backward pass.
    intermediates = concat(intermediates, newState.output.expand({0}));
  }
  return intermediates;
}

static Tensor getSavedFwdIntermediate(const Tensor &fwdIntermediates,
                                      const LstmParams &params,
                                      const LstmOpts &options,
                                      FwdIntermediates intermediate) {
  auto recompType = options.recomputationMode;
  int index = intermediate;
  if (intermediate >= LSTM_FWD_INTERMEDIATE_OUTPUT &&
      (recompType == LstmRecomputationMode::CellAndTanh ||
       recompType == LstmRecomputationMode::Full)) {
    assert(index >= (LSTM_FWD_INTERMEDIATE_OUTPUT -
                     LSTM_FWD_INTERMEDIATE_OUTPUT_TANH));
    index -= (LSTM_FWD_INTERMEDIATE_OUTPUT -
              LSTM_FWD_INTERMEDIATE_OUTPUT_TANH);
  }
  if (intermediate >= LSTM_FWD_INTERMEDIATE_OUTPUT_TANH &&
      recompType == LstmRecomputationMode::Full) {
    assert(index >= (LSTM_FWD_INTERMEDIATE_OUTPUT_TANH -
                     LSTM_FWD_INTERMEDIATE_FORGET_GATE));
    index -= (LSTM_FWD_INTERMEDIATE_OUTPUT_TANH -
              LSTM_FWD_INTERMEDIATE_FORGET_GATE);
  }
  assert(index < fwdIntermediates.dim(0));
  return fwdIntermediates[index];
}

static Tensor
reconstructIntermediatesFromRecomputed(const Tensor &savedIntermediates,
                                       const Tensor &recomputedIntermediates,
                                       const LstmParams &params,
                                       const LstmOpts &options) {
  switch (options.recomputationMode) {
    case LstmRecomputationMode::None:
      return savedIntermediates;
    case LstmRecomputationMode::CellAndTanh: {
      auto intermediates =
        concat(savedIntermediates.slice(LSTM_FWD_INTERMEDIATE_FORGET_GATE,
                                        LSTM_FWD_INTERMEDIATE_OUTPUT_TANH),
               recomputedIntermediates);
      if (!params.outputFullSequence) {
        auto output =
          getSavedFwdIntermediate(savedIntermediates, params, options,
                                  LSTM_FWD_INTERMEDIATE_OUTPUT);
        intermediates = concat(intermediates, output.expand({0}));
      }
      return intermediates;
    }
    case LstmRecomputationMode::Full:
    default:
      throw poputil::poplibs_error("Unhandled recomputation type");
  }

  POPLIB_UNREACHABLE();
}

std::pair<Tensor, Tensor>
lstmFwd(Graph &graph,
        const LstmParams &params,
        const LstmState &fwdStateInit,
        const Tensor &prevLayerActs,
        const LstmWeights &weights,
        Tensor *intermediatesSeq,
        program::Sequence &fwdProg,
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
                                            fwdProg, opt,
                                            debugPrefix + "/lstm/weightInputs",
                                            cache);
  }

  // loop counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1},
                                  debugPrefix + "/seqIdx");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, fwdProg, debugPrefix + "/initSeqIdx");

  // state for current layer, start from initialiser
  LstmState state = {
    duplicate(graph, fwdStateInit.output, fwdProg,
              debugPrefix + "/fwdOutputState"),
    duplicate(graph, fwdStateInit.cellState, fwdProg,
              debugPrefix + "/fwdCellState")
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
  if (intermediatesSeq) {
    LstmState newState;
    LstmInternalState internalState;
    std::tie(newState, internalState) =
        basicLstmCellForwardPass(
            graph, fwdInput, weights.biases,
            state,
            inputWeightsPtr, weights.outputWeights,
            loop, opt, opt.inferenceOnly, debugPrefix,
            cache);
    auto intermediates =
      getFwdIntermediatesToSave(state, newState, internalState, opt, params);
    const auto numIntermediates = intermediates.dim(0);
    *intermediatesSeq =
          createOutputTensor(graph, params,
                             seqSize * numIntermediates,
                             debugPrefix + "/fwdIntermediatesSeq")
                      .reshapePartial(0, 1, {seqSize, numIntermediates});
    auto intermediatesRearranged =
        createOutputTensor(graph, params,
                           numIntermediates,
                           debugPrefix + "/fwdIntermediatesRearranged");
    loop.add(Copy(intermediates, intermediatesRearranged));
    fwdProg.add(WriteUndef(*intermediatesSeq));
    popops::dynamicUpdate(
        graph, *intermediatesSeq, intermediatesRearranged.expand({0}), seqIdx,
        {0}, {1}, loop, debugPrefix + "/lstmUpdateIntermediates");

    auto stateTensor = state.getAsTensor();
    auto newStateTensor = newState.getAsTensor();
    graph.setTileMapping(stateTensor, graph.getTileMapping(newStateTensor));
    loop.add(Copy(newStateTensor, stateTensor));
  } else {
    basicLstmCellForwardPassInPlace(
        graph, fwdInput, weights.biases,
        state,
        inputWeightsPtr, weights.outputWeights,
        loop, opt, opt.inferenceOnly, debugPrefix,
        cache);
  }
  Tensor outputSeq;
  if (params.outputFullSequence) {
    outputSeq = createOutputTensor(graph, params, seqSize,
                                   debugPrefix + "/Output");
    fwdProg.add(WriteUndef(outputSeq));
    popops::dynamicUpdate(
        graph, outputSeq, state.output.expand({0}), seqIdx, {0}, {1},
        loop, debugPrefix + "/updateOutputSeq");
  }
  addInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxIncr");
  fwdProg.add(Repeat(seqSize, loop));
  return {params.outputFullSequence ? outputSeq : state.output,
          state.cellState};
}

static std::tuple<LstmState, Tensor, Tensor>
backwardStepImpl(Graph &graph,
                 const Tensor *gradNextLayer,
                 const Tensor &fwdIntermediates,
                 const LstmState &stateGrad,
                 const Tensor *weightsInput,
                 const Tensor &weightsOutput,
                 Sequence &initProg,
                 Sequence &prog,
                 const LstmOpts &opt,
                 const std::string &debugPrefix,
                 matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmBwd";
  auto outputGrad = stateGrad.output;
  auto outputGroupingIntoLayer = detectInnermostGrouping(graph, outputGrad);
  if (gradNextLayer) {
    outputGrad =
      popops::add(graph, outputGrad, *gradNextLayer, prog,
                  fPrefix + "/AddActGrads");
  }
  auto actOutputGate =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT_GATE];
  auto actOutputTanh =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT_TANH];
  auto prevCellState =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_PREV_CELL_STATE];
  auto t =
    mul(graph, concat({actOutputGate, actOutputTanh}),
        outputGrad.broadcast(2, 0), prog, fPrefix + "/MulOGate");
  auto gradAtOTanhInput = t.slice(0, outputGrad.dim(0));
  auto gradAtOutputGateInput = t.slice(outputGrad.dim(0),
                                       2 * outputGrad.dim(0));

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

  auto gradCellState = stateGrad.cellState;

  addInPlace(graph, gradAtOTanhOutput, gradCellState, prog,
             fPrefix + "/AddCellState");
  auto actInputGate =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_INPUT_GATE];
  auto actCandidate =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_CAND_TANH];
  auto actForgetGate =
      fwdIntermediates[LSTM_FWD_INTERMEDIATE_FORGET_GATE];
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

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  auto grads = flattenUnits(gradUnits);
  Tensor weightsTransposed;
  if (weightsInput == nullptr) {
    weightsTransposed = flattenUnits(weightsOutput)
              .transpose();
  } else {
    weightsTransposed = flattenUnits(concat(*weightsInput, weightsOutput, 1))
                        .transpose();
  }

  weightsTransposed =
      preArrangeMatMulInputRHS(graph, grads.shape(), weightsTransposed,
                               initProg, fPrefix + "/PreArrangeWeights",
                               mmOpt, cache);

  Tensor gradientIn, gradientPrevStep;
  if (weightsInput) {
    auto inputSize = weightsInput->dim(1);
    auto outputSize = weightsOutput.dim(1);
    auto out =
      matMul(graph, grads, weightsTransposed, prog,
             fPrefix + "/{Prev + Input}Grad", mmOpt, cache);
    out = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer,
                                     prog, fPrefix);
    gradientIn = out.slice(0, inputSize, 1);
    gradientPrevStep = out.slice(inputSize, inputSize + outputSize, 1);
  } else {
    gradientPrevStep =
      matMul(graph, grads, weightsTransposed, prog, fPrefix + "/PrevStepGrad",
             mmOpt, cache);
    gradientPrevStep =
        tryGroupedPartialTranspose(graph, gradientPrevStep,
                                   detectInnermostGrouping(graph, outputGrad),
                                   prog, fPrefix);
  }

  return std::make_tuple(LstmState{gradientPrevStep, newGradCellState},
                         gradientIn,
                         concat({gradForgetGate.expand({0}),
                                 gradInputGate.expand({0}),
                                 gradCandidate.expand({0}),
                                 gradOutputGate.expand({0})}));
}



std::tuple<LstmState, Tensor, Tensor>
basicLstmBackwardStep(Graph &graph,
                      const Tensor *gradNextLayer,
                      const Tensor &fwdIntermediates,
                      const LstmState &stateGrad,
                      const Tensor &weightsInput,
                      const Tensor &weightsOutput,
                      Sequence &initProg,
                      Sequence &prog,
                      const LstmOpts &opt,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  return
    backwardStepImpl(graph, gradNextLayer, fwdIntermediates,
                     stateGrad, &weightsInput, weightsOutput, initProg, prog,
                     opt, debugPrefix, cache);
}

std::pair<LstmState, Tensor>
basicLstmBackwardStep(Graph &graph,
                      const Tensor *gradNextLayer,
                      const Tensor &fwdIntermediates,
                      const LstmState &stateGrad,
                      const Tensor &weightsOutput,
                      Sequence &initProg,
                      Sequence &prog,
                      const LstmOpts &opt,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  LstmState prevStateGrad;
  Tensor bwdIntermediates;
  std::tie(prevStateGrad, std::ignore, bwdIntermediates) =
    backwardStepImpl(graph, gradNextLayer, fwdIntermediates,
                     stateGrad, nullptr, weightsOutput, initProg, prog,
                     opt, debugPrefix, cache);
  return std::make_pair(prevStateGrad, bwdIntermediates);
}

/// Add the partial weight gradients from this timestep to the accumulated
/// weight gradients. Once all the gradients have been accumulated call
/// basicLstmParamUpdateFinal() to do any final accumulation / rearrangement
/// that is required.
static void
basicLstmParamUpdate(Graph &graph,
                     const Tensor &prevLayerActs,
                     const Tensor &prevStepActs,
                     const Tensor &bwdIntermediates,
                     const LstmWeights &weightGrads,
                     Sequence &prog,
                     const LstmOpts &opt,
                     const std::string &debugPrefix,
                     matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmDeltas";
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_WU");
  matMulAcc(graph,
            concat(flattenUnits(weightGrads.inputWeights),
                   flattenUnits(weightGrads.outputWeights)),
            1.0,
            concat(prevLayerActs.transpose(), prevStepActs.transpose()),
            flattenUnits(bwdIntermediates),
            prog,
            fPrefix + "/Wi",
            mmOpt, cache);

  // We defer the reduction across the batch to later.
  popops::addInPlace(graph, weightGrads.biases, bwdIntermediates, prog,
                     fPrefix + "/Bias");
}

static LstmWeights
basicLstmParamUpdateFinal(Graph &graph,
                          const LstmWeights &weights,
                          const LstmWeights &weightGrads,
                          Sequence &prog,
                          const std::string &debugPrefix) {
  // The accumulated bias gradients still has a batch axis that we must
  // accumulate over - do this now.
  auto biasGrad =
    graph.clone(weights.biases, debugPrefix + "/biasGrad");
  popops::reduceWithOutput(graph, weightGrads.biases, biasGrad, {1},
                           {popops::Operation::ADD},
                           prog, debugPrefix + "/FinalBiasReduction");
  auto finalWeightGrads = weightGrads;
  finalWeightGrads.biases = biasGrad;
  return finalWeightGrads;
}

/// Create variables used to accumulate gradients of the weights in the
/// backward pass.
static LstmWeights
createWeightAccumulators(Graph &graph, const LstmWeights &weights,
                         const Tensor &bwdIntermediates,
                         const LstmOpts &options,
                         const std::string &debugPrefix) {
  LstmWeights weightAccs;
  if (options.preCalcWeights) {
    weightAccs.inputWeights =
        graph.clone(weights.inputWeights,
                    debugPrefix + "/inputWeightsDeltaAcc");
    weightAccs.outputWeights =
      graph.clone(weights.outputWeights,
                  debugPrefix + "/outputWeightsDeltaAcc");
  } else {
    // inputWeights and outputWeights are slices of the one variable. Clone
    // them together as it results in a less complex tensor expression.
    auto concatenated = concat(flattenUnits(weights.inputWeights),
                               flattenUnits(weights.outputWeights));
    auto weightsDeltaAcc = graph.clone(concatenated,
                                       debugPrefix + "/weightsDeltaAcc");
    const auto inputSize = weights.inputWeights.dim(1);
    const auto outputSize = weights.outputWeights.dim(1);
    weightAccs.inputWeights =
        unflattenUnits(weightsDeltaAcc.slice(0, inputSize),
                       BASIC_LSTM_CELL_NUM_UNITS);
    weightAccs.outputWeights =
        unflattenUnits(weightsDeltaAcc.slice(inputSize,
                                             inputSize + outputSize),
                       BASIC_LSTM_CELL_NUM_UNITS);
  }
  // We delay reducing across the batch until after we have accumulated
  // gradients from each timestep and therefore the bias accumlator still has
  // a batch axis. This amortizes the cost of reducing over the batch which
  // otherwise can be significant.
  weightAccs.biases =
    graph.clone(bwdIntermediates, debugPrefix + "/bwdIntermediatesAcc");
  return weightAccs;
}

static void
zeroWeightAccumulators(Graph &graph, program::Sequence &prog,
                       const LstmWeights &weightsAcc,
                       const LstmOpts &options) {
  if (options.preCalcWeights) {
    popops::zero(graph,
                 concat({weightsAcc.inputWeights.flatten(),
                         weightsAcc.outputWeights.flatten(),
                         weightsAcc.biases.flatten()}),
                 prog);
  } else {
    // inputWeights and outputWeights are slices of the one variable.
    // Recombining them means reorderToSimplify() in popops::zero() works a lot
    // better.
    auto concatenated = concat(flattenUnits(weightsAcc.inputWeights),
                               flattenUnits(weightsAcc.outputWeights));
    popops::zero(graph,
                 concat({concatenated.flatten(),
                         weightsAcc.biases.flatten()}),
                 prog);

  }
}

// Is it beneficial memory-wise to interleave weight update with
// backwards pass.
static bool interleavedWUIsBeneficial(const LstmParams &params) {
  const auto batchSize = params.batchSize;
  const auto inputSize = params.layerSizes[0];
  const auto outputSize = params.layerSizes[1];
  // Total elements needed for transposed weights.
  const auto totalTransposeParams =
    (inputSize + outputSize) * outputSize * BASIC_LSTM_CELL_NUM_UNITS;
  // Total elements needed for unit gradients for weight update if
  // not interleaved with backpropagation.
  const auto totalBwdIntermediates =
    batchSize * outputSize * BASIC_LSTM_CELL_NUM_UNITS * params.timeSteps;
  return totalTransposeParams <= totalBwdIntermediates;
}

static Tensor recomputeCellAndTanhImpl(Graph &graph, const LstmParams &params,
                                       const LstmOpts &options,
                                       const LstmState &fwdStateInit,
                                       const Tensor &fwdIntermediatesSeq,
                                       program::Sequence &prog,
                                       const std::string &debugPrefix) {
  unsigned seqSize = params.timeSteps;

  // sequence counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, prog, debugPrefix + "/initSeqIdx");

  std::size_t numToRecompute =
    LSTM_FWD_INTERMEDIATE_OUTPUT - LSTM_FWD_INTERMEDIATE_OUTPUT_TANH;
  auto recomputedIntermediatesSeq =
    createOutputTensor(graph, params, seqSize * numToRecompute,
                       debugPrefix + "/recomputedIntermediates")
    .reshapePartial(0, 1, {seqSize, numToRecompute});

  auto loop = Sequence();
  {
    auto savedIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1}, loop,
                   debugPrefix + "/getSavedIntermediates").squeeze({0});

    auto forgetGate =
      getSavedFwdIntermediate(savedIntermediates, params, options,
                              LSTM_FWD_INTERMEDIATE_FORGET_GATE);
    auto candidate =
      getSavedFwdIntermediate(savedIntermediates, params, options,
                              LSTM_FWD_INTERMEDIATE_CAND_TANH);
    auto outputGate =
      getSavedFwdIntermediate(savedIntermediates, params, options,
                              LSTM_FWD_INTERMEDIATE_OUTPUT_GATE);
    auto inputGate =
      getSavedFwdIntermediate(savedIntermediates, params, options,
                              LSTM_FWD_INTERMEDIATE_INPUT_GATE);

    auto prevCellState =
      graph.clone(forgetGate, debugPrefix + "/prevCellState");
    prog.add(Copy(fwdStateInit.cellState, prevCellState));

    // Recompute cell state and tanh
    Tensor newCellState, newTanhOutput;
    {
      auto prod = mul(graph, concat(forgetGate, candidate),
                      concat(prevCellState, inputGate), loop,
                      debugPrefix + "/{Forget + Input}Gate");

      newCellState = prod.slice(0, forgetGate.dim(0));
      auto updatedCandidate = prod.slice(forgetGate.dim(0),
                                         forgetGate.dim(0) + candidate.dim(0));
      addInPlace(graph, newCellState, updatedCandidate, loop,
                 debugPrefix + "/AddCellCand");
      newTanhOutput = popops::tanh(graph, newCellState, loop,
                                   debugPrefix + "/TanhCellState");
    }

    auto rearrangedIntermediates =
      createOutputTensor(graph, params, numToRecompute,
                         debugPrefix + "/recomputedIntermediatesRearranged");
    loop.add(Copy(concat(newTanhOutput.expand({0}), prevCellState.expand({0})),
                  rearrangedIntermediates));
    loop.add(Copy(newCellState, prevCellState));
    prog.add(WriteUndef(recomputedIntermediatesSeq));
    dynamicUpdate(graph, recomputedIntermediatesSeq,
                  rearrangedIntermediates.expand({0}), seqIdx,
                  {0}, {1}, loop, debugPrefix + "/storeRecomputed");

    addInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxIncr");
  }
  prog.add(Repeat(seqSize, loop));

  return recomputedIntermediatesSeq;
}

static Tensor recomputeAndGetFwdIntermediates(
    Graph &graph,
    const LstmState &fwdStateInit,
    const Tensor &fwdIntermediatesSeq,
    const LstmParams &params,
    const LstmOpts &options,
    program::Sequence &recomputeProg,
    const std::string &recomputePrefix,
    program::Sequence &sliceProg,
    const Tensor &sliceIdx,
    const std::string &slicePrefix) {
  Tensor savedSlice;
  Tensor recomputedSlice;
  switch (options.recomputationMode) {
    case LstmRecomputationMode::None:
    {
      // No recomputation needed, we need only slice the existing forward
      // intermediates.
      savedSlice =
        dynamicSlice(graph, fwdIntermediatesSeq, sliceIdx, {0}, {1},
                     sliceProg, slicePrefix).squeeze({0});
      break;
    }
    case LstmRecomputationMode::CellAndTanh:
    {
      auto recomputedIntermediatesSeq =
        recomputeCellAndTanhImpl(graph, params, options, fwdStateInit,
                                 fwdIntermediatesSeq, recomputeProg,
                                 recomputePrefix);
      savedSlice =
        dynamicSlice(graph, fwdIntermediatesSeq, sliceIdx,
                     {0}, {1}, sliceProg, slicePrefix).squeeze({0});
      recomputedSlice =
        dynamicSlice(graph, recomputedIntermediatesSeq, sliceIdx,
                     {0}, {1}, sliceProg, slicePrefix).squeeze({0});
      break;
    }
    case LstmRecomputationMode::Full:
      // TODO: Unimplemented
      // fallthrough
    default:
      throw poplibs_error("Unhandled recomputation type");
  }
  return reconstructIntermediatesFromRecomputed(savedSlice,
                                                recomputedSlice,
                                                params, options);
}

// Perform an LSTM backward pass.
// Optionally return the intermediates from the backward pass (sequence
// cell unit gradients), or calculate weight gradients directly during
// this pass interleaved with the backward pass.
static LstmState
lstmBwdImpl(Graph &graph, const LstmParams &params,
            program::Sequence &prog,
            const LstmState &fwdStateInit,
            const Tensor &fwdIntermediatesSeq,
            const LstmWeights &weights,
            const Tensor &fwdInputSeq,
            const Tensor &fwdOutput,
            const Tensor &gradLayerNext,
            const Tensor *lastCellStateGradPtr,
            Tensor *inputGradSeq,
            Tensor *bwdIntermediatesPtr,
            LstmWeights *weightsGrad,
            const std::string &debugPrefix,
            const LstmOpts &options,
            poplin::matmul::PlanningCache *cache) {
  auto &weightsInput = weights.inputWeights;
  auto &weightsOutput = weights.outputWeights;

  unsigned seqSize = params.timeSteps;
  // sequence down-counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start =
      graph.addConstant(UNSIGNED_INT, {1}, seqSize - 1, debugPrefix + "/start");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(start, 0);
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  const auto batchSize = params.batchSize;

  Tensor gradLayerNextRearranged =
    createOutputTensor(graph, params, seqSize,
                       debugPrefix + "/gradLayerNextRearranged");
  prog.add(Copy(gradLayerNext, gradLayerNextRearranged));

  auto lastOutGrad =
      createOutputTensor(graph, params, 1, debugPrefix + "/outGrad")[0];
  if (params.outputFullSequence) {
    zero(graph, lastOutGrad, prog, debugPrefix + "/initLastOutGrad");
  } else {
    prog.add(Copy(gradLayerNext, lastOutGrad));
  }
  auto lastCellStateGrad =
      createOutputTensor(graph, params, 1, debugPrefix + "/cellStateGrad")[0];
  if (lastCellStateGradPtr) {
    prog.add(Copy(*lastCellStateGradPtr, lastCellStateGrad));
  } else {
    zero(graph, lastCellStateGrad, prog, debugPrefix + "/initCellStateGrad");
  }
  LstmState stateGrads = {
    lastOutGrad,
    lastCellStateGrad,
  };

  auto sliceIntermediates = Sequence();
  auto sliceOutput = Sequence();

  Tensor fwdIntermediates = recomputeAndGetFwdIntermediates(
      graph, fwdStateInit, fwdIntermediatesSeq, params, options,
      prog, debugPrefix + "/recomputeFwdIntermediates",
      sliceIntermediates, seqIdx,
      debugPrefix + "/getFwdIntermediates");

  Tensor prevStepOut;
  if (weightsGrad) {
    if (params.outputFullSequence) {
      prevStepOut =
        dynamicSlice(graph, fwdOutput, seqIdx, {0}, {1}, sliceOutput,
                     debugPrefix + "/getPrevStepOut").squeeze({0});
    } else {
      prevStepOut = fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT];
    }
  }

  prog.add(sliceIntermediates);
  prog.add(sliceOutput);

  auto loop = Sequence();
  auto bwdLoopBody = Sequence();
  auto wuLoopBody = Sequence();
  {
    LstmState newStateGrads;
    Tensor bwdIntermediates;
    Tensor gradLayerNextThisStep;
    Tensor *gradLayerNextThisStepPtr = nullptr;
    if (params.outputFullSequence) {
      gradLayerNextThisStep =
        dynamicSlice(graph, gradLayerNextRearranged, seqIdx, {0}, {1},
                     bwdLoopBody, debugPrefix + "/gradLayerNext").squeeze({0});
      gradLayerNextThisStepPtr = &gradLayerNextThisStep;
    }
    if (inputGradSeq) {
      Tensor inputGrad;
      std::tie(newStateGrads, inputGrad, bwdIntermediates) =
        popnn::lstm::basicLstmBackwardStep(
          graph, gradLayerNextThisStepPtr, fwdIntermediates, stateGrads,
          weightsInput, weightsOutput, prog, bwdLoopBody,
          options, debugPrefix, cache);
      const auto inputSize = inputGrad.dim(1);
      const auto inputGrouping = gcd(16UL, inputSize);
      const auto numInputGroups = inputSize / inputGrouping;
      *inputGradSeq =
          createDynamicSliceTensor(graph, inputGrad.elementType(),
              seqSize, numInputGroups * batchSize, inputGrouping,
              debugPrefix + "/inputGradSeq")
          .reshapePartial(1, 2, {numInputGroups, batchSize})
          .dimRoll(1, 2)
          .flatten(2, 4);
      auto inputGradRearranged =
          createDynamicSliceTensor(graph, inputGrad.elementType(),
              1, numInputGroups * batchSize, inputGrouping,
              debugPrefix + "/inputGradRearranged")
          .reshapePartial(1, 2, {numInputGroups, batchSize})
          .dimRoll(1, 2)
          .flatten(2, 4)[0];
      bwdLoopBody.add(Copy(inputGrad, inputGradRearranged));
      prog.add(WriteUndef(*inputGradSeq));
      dynamicUpdate(graph, *inputGradSeq, inputGradRearranged.expand({0}),
                    seqIdx, {0}, {1}, bwdLoopBody,
                    debugPrefix + "/gradLayerPrev");
    } else {
      std::tie(newStateGrads, bwdIntermediates) =
          basicLstmBackwardStep(graph, gradLayerNextThisStepPtr,
                                fwdIntermediates, stateGrads, weightsOutput,
                                prog, bwdLoopBody, options,
                                debugPrefix, cache);
    }

    // If bwdIntermediatesPtr is given, create a sequence containing gradients
    // for each cell unit in each step.
    if (bwdIntermediatesPtr) {
      *bwdIntermediatesPtr =
        createOutputTensor(graph, params,
                           seqSize * BASIC_LSTM_CELL_NUM_UNITS,
                           debugPrefix + "/bwdIntermediates")
        .reshapePartial(0, 1, {seqSize, BASIC_LSTM_CELL_NUM_UNITS});
      auto bwdIntermediatesRearranged =
        createOutputTensor(graph, params, BASIC_LSTM_CELL_NUM_UNITS,
                           debugPrefix + "/bwdIntermediatesRearranged");
      bwdLoopBody.add(Copy(bwdIntermediates, bwdIntermediatesRearranged));
      prog.add(WriteUndef(*bwdIntermediatesPtr));
      dynamicUpdate(graph, *bwdIntermediatesPtr,
                    bwdIntermediatesRearranged.expand({0}),
                    seqIdx, {0}, {1}, bwdLoopBody,
                    debugPrefix + "/bwdIntermediates");
    }
    Tensor prevLayerOut;
    if (weightsGrad) {
      prevLayerOut =
        dynamicSlice(graph, fwdInputSeq, seqIdx, {0}, {1}, bwdLoopBody,
                     debugPrefix + "/prevLayerActs").squeeze({0});
    }
    bwdLoopBody.add(
        Copy(newStateGrads.getAsTensor(), stateGrads.getAsTensor()));
    subInPlace(graph, seqIdx, one, bwdLoopBody, debugPrefix + "/seqIdxDecr");

    loop.add(bwdLoopBody);
    loop.add(sliceIntermediates);
    loop.add(sliceOutput);

    if (weightsGrad) {
      *weightsGrad = createWeightAccumulators(graph, weights, bwdIntermediates,
                                              options, debugPrefix);
      zeroWeightAccumulators(graph, prog, *weightsGrad, options);

      basicLstmParamUpdate(
        graph, prevLayerOut, prevStepOut, bwdIntermediates,
        *weightsGrad, wuLoopBody, options, debugPrefix, cache);
    }
    loop.add(wuLoopBody);
  }

  // TODO: Last loop iteration is unrolled here to insert copy instead of slice
  // even when we don't need weightsGrad. It would be a minor optimisation in
  // this case to do the full loop in one.
  prog.add(Repeat(seqSize - 1, loop));
  prog.add(bwdLoopBody);
  if (weightsGrad) {
    prog.add(Copy(fwdStateInit.output, prevStepOut));
    prog.add(wuLoopBody);
    *weightsGrad = basicLstmParamUpdateFinal(graph, weights, *weightsGrad,
                                             prog, debugPrefix);
  }

  return stateGrads;
}

LstmState lstmBwd(Graph &graph, const LstmParams &params,
                  program::Sequence &prog,
                  const LstmState &fwdStateInit,
                  const Tensor &fwdIntermediatesSeq,
                  const LstmWeights &weights,
                  const Tensor &fwdInputSeq,
                  const Tensor &fwdOutput,
                  const Tensor &gradLayerNext,
                  const Tensor *lastCellStateGradPtr,
                  Tensor *inputGrad,
                  Tensor *bwdIntermediates,
                  const std::string &debugPrefix,
                  const OptionFlags &options_,
                  poplin::matmul::PlanningCache *planningCache) {
  validateParams(params);
  auto options = parseOptions(options_);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                       (inputGrad ? "non null" : "null") +
                       " if and only if params.calcInputGradients is " +
                       (inputGrad ? "true" : "false"));
  }
  return lstmBwdImpl(graph, params, prog, fwdStateInit, fwdIntermediatesSeq,
                     weights, fwdInputSeq, fwdOutput, gradLayerNext,
                     lastCellStateGradPtr, inputGrad, bwdIntermediates,
                     nullptr, debugPrefix, std::move(options), planningCache);
}

static LstmWeights
lstmWUImpl(Graph &graph, const LstmParams &params,
           program::Sequence &prog,
           const LstmState &fwdStateInit,
           const Tensor &fwdIntermediatesSeq,
           const Tensor &bwdIntermediatesSeq,
           const LstmWeights &weights,
           const Tensor &input,
           const Tensor &output,
           const std::string &debugPrefix,
           const LstmOpts &options,
           poplin::matmul::PlanningCache *planningCache) {
  LstmWeights weightGrads =
    createWeightAccumulators(graph, weights, bwdIntermediatesSeq[0], options,
                             debugPrefix);
  zeroWeightAccumulators(graph, prog, weightGrads, options);

  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start = graph.addConstant(
        UNSIGNED_INT, {1}, params.timeSteps - 1, debugPrefix + "/start");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(start, 0);
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  auto sliceOutput = Sequence();
  Tensor prevStepOut;
  if (params.outputFullSequence) {
    prevStepOut =
      dynamicSlice(graph, output, seqIdx, {0}, {1}, sliceOutput,
                   debugPrefix + "/getPrevStepOut").squeeze({0});
  } else {
    // TODO: If for full recomputation we want to recompute the output also,
    // that will need to be accounted for here as this info won't be part
    // of the intermediates.
    auto prevFwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1}, sliceOutput,
                   debugPrefix + "/getFwdIntermediates").squeeze({0});
    prevStepOut =
      getSavedFwdIntermediate(prevFwdIntermediates, params, options,
                              LSTM_FWD_INTERMEDIATE_OUTPUT);
  }

  auto loop = Sequence();
  auto sliceLoopBody = Sequence();
  auto wuLoopBody = Sequence();
  {
    // Dynamic slice required state per-step
    auto prevLayerOut =
      dynamicSlice(graph, input, seqIdx, {0}, {1}, sliceLoopBody,
                   debugPrefix + "/prevLayerActs").squeeze({0});
    auto bwdIntermediates =
      dynamicSlice(graph, bwdIntermediatesSeq, seqIdx, {0}, {1}, sliceLoopBody,
                   debugPrefix + "/getBwdIntermediates").squeeze({0});
    subInPlace(graph, seqIdx, one, sliceLoopBody, debugPrefix + "/seqIdxDecr");
    loop.add(sliceLoopBody);
    loop.add(sliceOutput);

    basicLstmParamUpdate(
      graph, prevLayerOut, prevStepOut, bwdIntermediates,
      weightGrads, wuLoopBody, options,
      debugPrefix, planningCache);
    loop.add(wuLoopBody);
  }
  prog.add(Repeat(params.timeSteps - 1, loop));
  prog.add(sliceLoopBody);
  prog.add(Copy(fwdStateInit.output, prevStepOut));
  prog.add(wuLoopBody);

  weightGrads =
      basicLstmParamUpdateFinal(graph, weights, weightGrads, prog, debugPrefix);

  return weightGrads;
}

LstmWeights lstmWU(Graph &graph, const LstmParams &params,
                   program::Sequence &prog,
                   const LstmState &fwdStateInit,
                   const Tensor &fwdIntermediates,
                   const Tensor &bwdIntermediates,
                   const LstmWeights &weights,
                   const Tensor &input,
                   const Tensor &output,
                   const std::string &debugPrefix,
                   const poplar::OptionFlags &options_,
                   poplin::matmul::PlanningCache *planningCache) {
  validateParams(params);
  auto options = parseOptions(options_);
  return lstmWUImpl(graph, params, prog, fwdStateInit, fwdIntermediates,
                    bwdIntermediates, weights, input, output, debugPrefix,
                    std::move(options), planningCache);
}

LstmState lstmBwdWithWU(poplar::Graph &graph, const LstmParams &params,
                        poplar::program::Sequence &prog,
                        const LstmState &fwdStateInit,
                        const poplar::Tensor &fwdIntermediates,
                        const LstmWeights &weights,
                        const poplar::Tensor &input,
                        const poplar::Tensor &output,
                        const poplar::Tensor &outputGrad,
                        const poplar::Tensor *lastCellStateGrad,
                        poplar::Tensor *inputGrad,
                        LstmWeights &weightsGrad,
                        const std::string &debugPrefix,
                        const poplar::OptionFlags &options_,
                        poplin::matmul::PlanningCache *planningCache) {
  validateParams(params);
  auto options = parseOptions(options_);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                       (inputGrad ? "non null" : "null") +
                       " if and only if params.calcInputGradients is " +
                       (inputGrad ? "true" : "false"));
  }

  bool interleaveWU = interleavedWUIsBeneficial(params);
  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  LstmState stateGrads =
    lstmBwdImpl(graph, params, prog, fwdStateInit, fwdIntermediates, weights,
                input, output, outputGrad, lastCellStateGrad, inputGrad,
                interleaveWU ? nullptr : &bwdIntermediates,
                interleaveWU ? &weightsGrad : nullptr,
                debugPrefix, options, planningCache);

  if (!interleaveWU) {
    weightsGrad = lstmWUImpl(graph, params, prog, fwdStateInit,
                             fwdIntermediates, bwdIntermediates, weights,
                             input, output, debugPrefix,
                             std::move(options), planningCache);
  }

  return stateGrads;
}

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  auto weighInput = params.doInputWeightCalc;
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.

  uint64_t multsWeighInp = weighInput ?
      static_cast<uint64_t>(inputSize) * 4 * outputSize * batchSize *
                             sequenceSize * 2 : 0;
  uint64_t multsWeighOut =
      static_cast<uint64_t>(outputSize) * 4 * outputSize * batchSize *
      sequenceSize * 2;

  // We ignore FLOPs for bias addition - in theory we could initialize the
  // accumulators with the biases during the matrix multipliciation.
  uint64_t mulFlops =
      3 * static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  uint64_t addFlops =
      static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  return multsWeighInp + multsWeighOut + addFlops + mulFlops;
}

uint64_t getBasicLstmCellBwdFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
  auto calcInputGrad = params.calcInputGradients;
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.

  uint64_t mulFlops =
      static_cast<uint64_t>(sequenceSize) * 6 * batchSize * outputSize;
  uint64_t inputGradFlops =
      calcInputGrad ?  static_cast<uint64_t>(inputSize) * 4 * outputSize *
                       batchSize * sequenceSize * 2 : 0;
  uint64_t outputGradFlops =
      static_cast<uint64_t>(outputSize) * 4 * outputSize * batchSize *
      sequenceSize * 2;
  return mulFlops + inputGradFlops + outputGradFlops;
}

uint64_t getBasicLstmCellWuFlops(const LstmParams &params) {
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];

  uint64_t weightFlops =
      static_cast<uint64_t>(inputSize + outputSize) * 4 * outputSize *
                           batchSize * sequenceSize * 2;
  uint64_t biasFlops =
      static_cast<uint64_t>(outputSize) * 4 * batchSize * sequenceSize * 2;
  return weightFlops + biasFlops;
}


} // namespace lstm
} // namespace popnn
