// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include <poplibs_support/logging.hpp>
#include <popnn/Lstm.hpp>
#include <popops/Cast.hpp>

#include "RnnUtil.hpp"
#include "poplin/FullyConnected.hpp"

using namespace poplar;
using namespace poplar::program;

using namespace poplin;
using namespace popnn::Rnn;
using namespace popops;
using namespace poputil;
using namespace poplibs_support;

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
  // TODO: T12908 Add support for recomputation.
  LSTM_FWD_INTERMEDIATE_OUTPUT
};

// Tensor elements maintained in backward state. The number of elements is a
// function of the amount of recomputation done in the weight update pass
enum BwdStateTensorElems {
  LSTM_BWD_STATE_GRAD_CELL_STATE = 0,
  LSTM_BWD_STATE_GRAD_ACT_GRAD,
  LSTM_NUM_BWD_STATES
};

static void applyGateNonlinearities(Graph &graph, const Tensor &t,
                                    Sequence &prog,
                                    const std::vector<std::size_t> &cellIndices,
                                    const DebugNameAndId &dnai) {
  auto sigmoidIn = concat({t[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]],
                           t[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]],
                           t[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]]});
  auto cs = graph.addComputeSet({dnai, "OutputGate"});
  nonLinearityInPlace(graph, popnn::NonLinearityType::SIGMOID, sigmoidIn, cs,
                      {dnai});
  nonLinearityInPlace(graph, popnn::NonLinearityType::TANH,
                      t[cellIndices[BASIC_LSTM_CELL_CANDIDATE]], cs, {dnai});
  prog.add(Execute(cs, {dnai}));
}

// Computes the output before nonlinearities to all the units are applies
static Tensor basicLstmUnitsNlInputPreWeighted(
    Graph &graph, Tensor weightedIn, Tensor prevOutput, Tensor weightsOutput,
    Sequence &prog, OptionFlags &mmOpt, matmul::PlanningCache *cache,
    const DebugNameAndId &dnai) {
  assert(weightedIn.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto output =
      unflattenUnits(matMul(graph, prevOutput, flattenUnits(weightsOutput),
                            prog, {dnai, "WeighOutput"}, mmOpt, cache),
                     BASIC_LSTM_CELL_NUM_UNITS);
  addInPlace(graph, output, weightedIn, prog, {dnai, "AddWeightedOutputs"});
  return output;
}

// Computes the output before nonlinearities to all the units are applied
static Tensor basicLstmUnitsNlInput(Graph &graph, Tensor prevAct,
                                    Tensor prevOutput, Tensor weightsInput,
                                    Tensor weightsOutput, Sequence &prog,
                                    OptionFlags &mmOpt,
                                    matmul::PlanningCache *cache,
                                    const DebugNameAndId &dnai) {
  assert(weightsInput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto weights = concat(weightsInput, weightsOutput, 1);
  return unflattenUnits(matMul(graph, concat(prevAct, prevOutput, 1),
                               flattenUnits(weights), prog, {dnai, "Weigh"},
                               mmOpt, cache),
                        BASIC_LSTM_CELL_NUM_UNITS);
}

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const BasicLstmCellUnit &t) {
  switch (t) {
  case BASIC_LSTM_CELL_FORGET_GATE:
    return poplar::ProfileValue("BASIC_LSTM_CELL_FORGET_GATE");
  case BASIC_LSTM_CELL_INPUT_GATE:
    return poplar::ProfileValue("BASIC_LSTM_CELL_INPUT_GATE");
  case BASIC_LSTM_CELL_CANDIDATE:
    return poplar::ProfileValue("BASIC_LSTM_CELL_CANDIDATE");
  case BASIC_LSTM_CELL_OUTPUT_GATE:
    return poplar::ProfileValue("BASIC_LSTM_CELL_OUTPUT_GATE");
  case BASIC_LSTM_CELL_NUM_UNITS:
    return poplar::ProfileValue("BASIC_LSTM_CELL_NUM_UNITS");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}

template <>
poplar::ProfileValue toProfileValue(const popnn::lstm::LstmParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"rnn", toProfileValue(t.rnn)});
  v.insert({"outputFullSequence", toProfileValue(t.outputFullSequence)});
  v.insert({"doInputWeightCalc", toProfileValue(t.doInputWeightCalc)});
  v.insert({"calcInputGradients", toProfileValue(t.calcInputGradients)});
  v.insert({"cellOrder", toProfileValue(t.cellOrder)});
  return v;
}

template <>
poplar::ProfileValue toProfileValue(const popnn::lstm::LstmState &t) {
  poplar::ProfileValue::Map v;
  v.insert({"output", toProfileValue(t.output)});
  v.insert({"cellState", toProfileValue(t.cellState)});
  return v;
}

template <>
poplar::ProfileValue toProfileValue(const popnn::lstm::LstmWeights &t) {
  poplar::ProfileValue::Map v;
  v.insert({"inputWeights", toProfileValue(t.inputWeights)});
  v.insert({"outputWeights", toProfileValue(t.outputWeights)});
  v.insert({"biases", toProfileValue(t.biases)});
  return v;
}

} // namespace poputil

namespace popnn {
namespace lstm {

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
  poplar::Type accumulatorsType;
  LstmRecomputationMode recomputationMode;
  boost::optional<double> availableMemoryProportion;
  std::size_t numShards;
};

std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                    {"float", poplar::FLOAT}};

std::map<std::string, LstmRecomputationMode> recomputationModeMap{
    {"none", LstmRecomputationMode::None},
    {"cellAndTanh", LstmRecomputationMode::CellAndTanh},
    {"full", LstmRecomputationMode::Full}};

static OptionFlags getMMOpts(const LstmOpts &lstmOpts) {
  OptionFlags mmOpts = {
      {"partialsType", lstmOpts.partialsType.toString()},
  };
  if (lstmOpts.availableMemoryProportion) {
    mmOpts.set("availableMemoryProportion",
               std::to_string(lstmOpts.availableMemoryProportion.get()));
  }
  return mmOpts;
}

static LstmOpts parseOptions(const OptionFlags &options,
                             const poplar::Type defaultAccType) {

  LstmOpts lstmOpts;
  lstmOpts.inferenceOnly = false;
  lstmOpts.preCalcWeights = false;
  lstmOpts.partialsType = poplar::FLOAT;
  lstmOpts.accumulatorsType =
      defaultAccType; // this will default to float in future
  lstmOpts.recomputationMode = LstmRecomputationMode::None;
  lstmOpts.numShards = 1;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec lstmSpec{
      {"inferenceOnly", OptionHandler::createWithBool(lstmOpts.inferenceOnly)},
      {"preCalcWeights",
       OptionHandler::createWithBool(lstmOpts.preCalcWeights)},
      {"partialsType",
       OptionHandler::createWithEnum(lstmOpts.partialsType, partialsTypeMap)},
      {"weightAccumulatorsType",
       OptionHandler::createWithEnum(lstmOpts.accumulatorsType,
                                     partialsTypeMap)},
      {"recomputationMode",
       OptionHandler::createWithEnum(lstmOpts.recomputationMode,
                                     recomputationModeMap)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(lstmOpts.availableMemoryProportion)},
      {"numShards", OptionHandler::createWithInteger(lstmOpts.numShards)},
  };
  for (const auto &entry : options) {
    lstmSpec.parse(entry.first, entry.second);
  }
  return lstmOpts;
}

static void validateParams(const LstmParams &params) {
  if (params.rnn.layerSizes.size() != 2) {
    throw poplibs_error("Invalid LSTM params (layerSize != 2)");
  }
}

static poplar::OptionFlags toFwdPassMatMulOptions(LstmOpts lstmOpts) {
  poplar::OptionFlags flags = {
      {"fullyConnectedPass",
       lstmOpts.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD"},
      {"partialsType", lstmOpts.partialsType.toString()}};
  if (lstmOpts.availableMemoryProportion) {
    flags.set("availableMemoryProportion",
              std::to_string(*lstmOpts.availableMemoryProportion));
  }
  return flags;
}

const std::vector<BasicLstmCellUnit> getDefaultBasicLstmCellOrder() {
  return {BASIC_LSTM_CELL_FORGET_GATE, BASIC_LSTM_CELL_INPUT_GATE,
          BASIC_LSTM_CELL_CANDIDATE, BASIC_LSTM_CELL_OUTPUT_GATE};
}

std::vector<std::pair<poplin::MatMulParams, poplar::OptionFlags>>
getMatMulPrePlanParameters(LstmParams params, poplar::OptionFlags opts) {
  const auto lstmOpts = parseOptions(opts, params.rnn.dataType);
  const auto mmFwdOpts = toFwdPassMatMulOptions(lstmOpts);

  const auto groupSize = 1;
  const auto batchSize = params.rnn.batchSize;
  const auto inputSize = 2 * params.rnn.layerSizes[0]; // We concat the weights
  const auto outputSize =
      BASIC_LSTM_CELL_NUM_UNITS * params.rnn.layerSizes[1]; // One for each cell

  const auto matmuls = poplin::fc::getMatMulPrePlanParameters(
      {groupSize, batchSize, inputSize, outputSize}, mmFwdOpts,
      params.rnn.dataType, lstmOpts.inferenceOnly);
  return matmuls;
}

static Tensor createInput(Graph &graph, const LstmParams &params,
                          const DebugNameAndId &dnai, const LstmOpts &opt,
                          matmul::PlanningCache *cache) {
  validateParams(params);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");

  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  if (opt.preCalcWeights) {
    auto fcOutputSize = BASIC_LSTM_CELL_NUM_UNITS * outputSize;
    auto fcInputSize = inputSize;
    auto fcBatchSize = params.rnn.timeSteps * params.rnn.batchSize;
    auto in = createMatMulInputLHS(
        graph, params.rnn.dataType, {fcBatchSize, fcInputSize},
        {fcInputSize, fcOutputSize}, {dnai}, mmOpt, cache);
    return in.reshape({params.rnn.timeSteps, params.rnn.batchSize, inputSize});
  } else {
    return rnn::createInputTensor(graph, params.rnn, opt.numShards,
                                  {dnai, "input"});
  }
}

Tensor createInput(Graph &graph, const LstmParams &params,
                   const poplar::DebugContext &debugContext,
                   const OptionFlags &options, matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  auto output = createInput(graph, params, {di},
                            parseOptions(options, params.rnn.dataType), cache);
  di.addOutput(output);
  return output;
}

LstmState createInitialState(Graph &graph, const LstmParams &params,
                             const poplar::DebugContext &debugContext,
                             const OptionFlags &options,
                             matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));
  auto opt = parseOptions(options, params.rnn.dataType);

  auto initialOutput =
      rnn::createInitialState(graph, params.rnn, true, 1, opt.numShards,
                              {di, "initialOutput"})
          .squeeze({0});
  auto initialCellState =
      rnn::createInitialState(graph, params.rnn, true, 1, opt.numShards,
                              {di, "initialCellState"})
          .squeeze({0});
  LstmState outputs = {initialOutput, initialCellState};
  di.addOutputs(DI_ARGS(outputs));
  return outputs;
}

void zeroInitialState(Graph &graph, const LstmState &state, Sequence &prog,
                      const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(state));

  zero(graph, concat(state.output, state.cellState), prog, {di});
}

std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const LstmParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options,
                    poplin::matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  validateParams(params);
  auto opt = parseOptions(options, params.rnn.dataType);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  poplar::Tensor inputWeights;
  poplar::Tensor outputWeights;
  if (opt.preCalcWeights) {
    if (params.doInputWeightCalc) {
      std::vector<std::size_t> aShape(2);
      aShape[0] = params.rnn.timeSteps * params.rnn.batchSize;
      aShape[1] = inputSize;
      auto weightsInput = createMatMulInputRHS(
          graph, params.rnn.dataType, aShape,
          {inputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
          {di, "weightsIn"}, mmOpt, cache);
      inputWeights = unflattenUnits(weightsInput, BASIC_LSTM_CELL_NUM_UNITS);
    }
    auto weightsOutput = createMatMulInputRHS(
        graph, params.rnn.dataType, {params.rnn.batchSize, outputSize},
        {outputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
        {di, "weightsOut"}, mmOpt, cache);
    outputWeights = unflattenUnits(weightsOutput, BASIC_LSTM_CELL_NUM_UNITS);
  } else {
    auto weights = createMatMulInputRHS(
        graph, params.rnn.dataType,
        {params.rnn.batchSize, inputSize + outputSize},
        {inputSize + outputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
        {di, "weights"}, mmOpt, cache);
    inputWeights =
        unflattenUnits(weights.slice(0, inputSize), BASIC_LSTM_CELL_NUM_UNITS);
    outputWeights =
        unflattenUnits(weights.slice(inputSize, inputSize + outputSize),
                       BASIC_LSTM_CELL_NUM_UNITS);
  }

  // rearrange the outermost dimension according to the cellOrder parameter.

  di.addOutputs(DI_ARGS(inputWeights, outputWeights));
  return std::make_pair(std::move(inputWeights), std::move(outputWeights));
}

/** Create the weights biases.
 */
poplar::Tensor createWeightsBiases(poplar::Graph &graph,
                                   const LstmParams &params,
                                   const poplar::DebugContext &debugContext,
                                   const OptionFlags &,
                                   poplin::matmul::PlanningCache *) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params));

  validateParams(params);
  auto outputSize = params.rnn.layerSizes[1];
  auto biases = graph.addVariable(params.rnn.dataType,
                                  {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                  {di, "biases"});
  mapTensorLinearly(graph, biases);
  di.addOutputs(DI_ARGS(biases));
  return biases;
}

LstmWeights createWeights(Graph &graph, const LstmParams &params,
                          const poplar::DebugContext &debugContext,
                          const OptionFlags &options,
                          poplin::matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  LstmWeights lstmWeights;
  std::tie(lstmWeights.inputWeights, lstmWeights.outputWeights) =
      createWeightsKernel(graph, params, {di}, options, cache);
  lstmWeights.biases = createWeightsBiases(graph, params, {di}, options, cache);
  di.addOutputs(DI_ARGS(lstmWeights));
  return lstmWeights;
}

static Tensor calcSequenceWeightedInputs(Graph &graph, const Tensor &in_,
                                         const Tensor &weightsInput_,
                                         program::Sequence &prog,
                                         const LstmOpts &opt,
                                         const DebugNameAndId &dnai,
                                         matmul::PlanningCache *cache) {
  auto mmOpt = getMMOpts(opt);
  auto sequenceSize = in_.dim(0);
  auto batchSize = in_.dim(1);
  auto inputSize = in_.dim(2);
  auto in = in_.reshape({sequenceSize * batchSize, inputSize});
  auto outputSize = weightsInput_.dim(2);
  auto weightsInput = flattenUnits(weightsInput_);
  return matMul(graph, in, weightsInput, prog, {dnai, "Lstm/CalcWeighedInput"},
                mmOpt, cache)
      .reshape({sequenceSize, batchSize, BASIC_LSTM_CELL_NUM_UNITS, outputSize})
      .dimShuffle({0, 2, 1, 3});
}

Tensor LstmState::getAsTensor() const {
  return concat({output.expand({0}), cellState.expand({0})});
}

struct LstmInternalState {
  Tensor forgetGate;
  Tensor inputGate;
  Tensor candidate;
  Tensor outputGate;
  Tensor tanhOutput;

  Tensor getAsTensor() const {
    return concat({forgetGate.expand({0}), inputGate.expand({0}),
                   candidate.expand({0}), outputGate.expand({0}),
                   tanhOutput.expand({0})});
  }
};

static std::string getUnitName(BasicLstmCellUnit unit) {
  switch (unit) {
  default:
    POPLIB_UNREACHABLE();
  case BASIC_LSTM_CELL_FORGET_GATE:
    return "ForgetGate";
  case BASIC_LSTM_CELL_INPUT_GATE:
    return "InputGate";
  case BASIC_LSTM_CELL_CANDIDATE:
    return "Candidate";
  case BASIC_LSTM_CELL_OUTPUT_GATE:
    return "OutputGate";
  }
}

static void rearrangeUnitsOutputFwd(Graph &graph, Tensor outputUnits,
                                    Tensor outputUnitsRearranged,
                                    Sequence &prog,
                                    const DebugNameAndId &dnai) {
  const auto outputGrouping =
      detectInnermostGrouping(graph, outputUnitsRearranged);
  // Typically the matrix multiplication result is laid out in memory such
  // that innermost dimension is groups batch elements. Try to rearrange the
  // result so the innermost dimension of the underlying memory is groups of the
  // specified number of outputs.
  outputUnits = unflattenUnits(
      tryGroupedPartialTranspose(graph, flattenUnits(outputUnits),
                                 outputGrouping, prog, {dnai}),
      BASIC_LSTM_CELL_NUM_UNITS);
  prog.add(Copy(outputUnits, outputUnitsRearranged, false, {dnai}));
}

static void lstmCellForwardPassCalcUnits(
    Graph &graph, const Tensor &in, const Tensor &biases,
    const LstmState &prevState, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const LstmOpts &opt,
    bool inferenceOnly, const Tensor &unitsOutputRearranged,
    const std::vector<std::size_t> &cellIndices, const DebugNameAndId &dnai,
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

  auto bBiases =
      graph.addVariable(dType, {0, batchSize, outputSize}, {dnai, "bbiases"});
  for (unsigned u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto unitBias =
        biases[u].broadcast(batchSize, 0).reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");

  Tensor unitsOutput;
  if (weightsInput == nullptr) {
    unitsOutput = basicLstmUnitsNlInputPreWeighted(
        graph, in, prevOutput, weightsOutput, prog, mmOpt, cache,
        {dnai, "ProcessUnits"});
  } else {
    unitsOutput = basicLstmUnitsNlInput(graph, in, prevOutput, *weightsInput,
                                        weightsOutput, prog, mmOpt, cache,
                                        {dnai, "ProcessUnits"});
  }

  // Rearrange the output of the matrix multiplication so each output unit
  // arranged the same as the cell state. This avoids the rearrangement
  // during the subsequent binary operations.
  rearrangeUnitsOutputFwd(graph, unitsOutput, unitsOutputRearranged, prog,
                          {dnai});

  for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    graph.setTileMapping(biases[u],
                         graph.getTileMapping(unitsOutputRearranged[u][0]));
  }
  addInPlace(graph, unitsOutputRearranged, bBiases, prog, {dnai, "AddBias"});
  applyGateNonlinearities(graph, unitsOutputRearranged, prog, cellIndices,
                          {dnai});
}

static std::pair<LstmState, LstmInternalState> basicLstmCellForwardPass(
    Graph &graph, const Tensor &in, const Tensor &biases,
    const LstmState &prevState, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const LstmOpts &opt,
    bool inferenceOnly, const std::vector<BasicLstmCellUnit> &cellOrder,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  const auto &prevCellState = prevState.cellState;
  const std::string baseStr = "BasicLstmCell";

  std::vector<Tensor> toConcat;
  toConcat.reserve(BASIC_LSTM_CELL_NUM_UNITS);
  assert(cellOrder.size() == BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    const auto unit = cellOrder.at(i);
    toConcat.push_back(
        graph.clone(prevCellState, {dnai, getUnitName(unit) + "Rearranged"})
            .expand({0}));
  }

  // build reverse mapping of cellOrder
  std::vector<std::size_t> cellIndices(BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i < cellOrder.size(); ++i) {
    cellIndices[cellOrder[i]] = i;
  }

  auto unitsOutput = concat(toConcat);
  lstmCellForwardPassCalcUnits(
      graph, in, biases, prevState, weightsInput, weightsOutput, prog, opt,
      inferenceOnly, unitsOutput, cellIndices, {dnai, baseStr}, cache);
  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]];
  auto candidate = unitsOutput[cellIndices[BASIC_LSTM_CELL_CANDIDATE]];
  auto outputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]];
  auto inputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]];
  auto prod = mul(graph, concat(forgetGate, candidate),
                  concat(prevCellState, inputGate), prog,
                  {dnai, baseStr + "/{Forget + Input}Gate"});

  auto updatedCellState = prod.slice(0, forgetGate.dim(0));
  auto updatedCandidate =
      prod.slice(forgetGate.dim(0), forgetGate.dim(0) + candidate.dim(0));

  addInPlace(graph, updatedCellState, updatedCandidate, prog,
             {dnai, baseStr + "/AddCellCand"});
  auto tanhOutput =
      popops::tanh(graph, updatedCellState, prog, {dnai, baseStr});
  auto output =
      mul(graph, tanhOutput, outputGate, prog, {dnai, baseStr + "/OutputGate"});
  LstmState recurrentState = {output, updatedCellState};
  LstmInternalState internalState = {forgetGate, inputGate, candidate,
                                     outputGate, tanhOutput};
  return {recurrentState, internalState};
}

static void basicLstmCellForwardPassInPlace(
    Graph &graph, const Tensor &in, const Tensor &biases,
    const LstmState &state, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const LstmOpts &opt,
    bool inferenceOnly, const std::vector<BasicLstmCellUnit> &cellOrder,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  auto cellState = state.cellState;
  auto output = state.output;
  const std::string baseStr = "BasicLstmCell";

  std::vector<Tensor> toConcat;
  toConcat.reserve(BASIC_LSTM_CELL_NUM_UNITS);
  assert(cellOrder.size() == BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    const auto unit = cellOrder.at(i);
    if (unit == BASIC_LSTM_CELL_OUTPUT_GATE) {
      toConcat.push_back(output.expand({0}));
    } else {
      toConcat.push_back(
          graph.clone(cellState, {dnai, getUnitName(unit) + "Rearranged"})
              .expand({0}));
    }
  }

  // build reverse mapping of cellOrder
  std::vector<std::size_t> cellIndices(BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i < cellOrder.size(); ++i) {
    cellIndices[cellOrder[i]] = i;
  }

  auto unitsOutput = concat(toConcat);

  lstmCellForwardPassCalcUnits(
      graph, in, biases, state, weightsInput, weightsOutput, prog, opt,
      inferenceOnly, unitsOutput, cellIndices, {dnai, baseStr}, cache);

  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]];
  auto candidate = unitsOutput[cellIndices[BASIC_LSTM_CELL_CANDIDATE]];
  auto outputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]];
  auto inputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]];
  using namespace popops::expr;
  mulInPlace(graph, concat(cellState, candidate), concat(forgetGate, inputGate),
             prog, {dnai, baseStr + "/{Forget + Input}Gate"});
  addInPlace(graph, cellState, candidate, prog,
             {dnai, baseStr + "/AddCellCand"});
  mapInPlace(graph, _1 * Tanh(_2), {outputGate, cellState}, prog,
             {dnai, baseStr + "/CalcNextOutput"});
}

unsigned getNumFwdIntermediatesToSave(const LstmOpts &options) {
  if (options.recomputationMode == LstmRecomputationMode::None) {
    return 6;
  } else if (options.recomputationMode == LstmRecomputationMode::CellAndTanh) {
    return 4;
  }
  throw poputil::poplibs_error("Unhandled recomputation type");
  return 0;
}

static Tensor getFwdIntermediatesToSave(const LstmState &state,
                                        const LstmState &newState,
                                        const LstmInternalState &internalState,
                                        const LstmOpts &options,
                                        const LstmParams &params) {
  Tensor intermediates;
  switch (options.recomputationMode) {
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
    // TODO: T12910 It may be cheaper to save the previous output rather than
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
    assert(index >=
           (LSTM_FWD_INTERMEDIATE_OUTPUT - LSTM_FWD_INTERMEDIATE_OUTPUT_TANH));
    index -= (LSTM_FWD_INTERMEDIATE_OUTPUT - LSTM_FWD_INTERMEDIATE_OUTPUT_TANH);
  }
  if (intermediate >= LSTM_FWD_INTERMEDIATE_OUTPUT_TANH &&
      recompType == LstmRecomputationMode::Full) {
    assert(index >= (LSTM_FWD_INTERMEDIATE_OUTPUT_TANH -
                     LSTM_FWD_INTERMEDIATE_FORGET_GATE));
    index -=
        (LSTM_FWD_INTERMEDIATE_OUTPUT_TANH - LSTM_FWD_INTERMEDIATE_FORGET_GATE);
  }
  assert(index < int(fwdIntermediates.dim(0)));
  return fwdIntermediates[index];
}

static Tensor reconstructIntermediatesFromRecomputed(
    const Tensor &savedIntermediates, const Tensor &recomputedIntermediates,
    const LstmParams &params, const LstmOpts &options) {
  switch (options.recomputationMode) {
  case LstmRecomputationMode::None:
    return savedIntermediates;
  case LstmRecomputationMode::CellAndTanh: {
    auto intermediates =
        concat(savedIntermediates.slice(LSTM_FWD_INTERMEDIATE_FORGET_GATE,
                                        LSTM_FWD_INTERMEDIATE_OUTPUT_TANH, 1),
               recomputedIntermediates, 1);
    if (!params.outputFullSequence) {
      auto output = getSavedFwdIntermediate(savedIntermediates, params, options,
                                            LSTM_FWD_INTERMEDIATE_OUTPUT);
      intermediates = concat(intermediates, output.expand({1}), 1);
    }
    return intermediates;
  }
  case LstmRecomputationMode::Full:
  default:
    throw poputil::poplibs_error("Unhandled recomputation type");
  }

  POPLIB_UNREACHABLE();
}

LstmParams::LstmParams(poplar::Type dataType, std::size_t batchSize,
                       std::size_t timeSteps,
                       std::vector<std::size_t> layerSizes)
    : rnn(dataType, batchSize, timeSteps, layerSizes), timeSteps(timeSteps) {}

std::pair<Tensor, Tensor>
lstmFwd(Graph &graph, const LstmParams &params, const LstmState &fwdStateInit,
        const Tensor &prevLayerActs, const LstmWeights &weights,
        Tensor *intermediatesSeq, program::Sequence &fwdProg,
        const poplar::DebugContext &debugContext, const OptionFlags &options,
        poplin::matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(prevLayerActs, weights, intermediatesSeq,
                            fwdStateInit, params, options, cache));

  validateParams(params);
  auto opt = parseOptions(options, params.rnn.dataType);

  Tensor weightedIn;
  if (!params.doInputWeightCalc) {
    weightedIn =
        graph.addVariable(params.rnn.dataType,
                          {params.rnn.timeSteps, BASIC_LSTM_CELL_NUM_UNITS,
                           params.rnn.batchSize, params.rnn.layerSizes[1]},
                          {di, "dummyWeightedIn"});
    for (unsigned s = 0; s < params.rnn.timeSteps; ++s) {
      mapTensorLinearly(graph, weightedIn[s]);
    }
  } else if (opt.preCalcWeights) {
    weightedIn = calcSequenceWeightedInputs(graph, prevLayerActs,
                                            weights.inputWeights, fwdProg, opt,
                                            {di, "lstm/weightInputs"}, cache);
  }
  std::vector<Tensor> initState = {fwdStateInit.output.expand({0}),
                                   fwdStateInit.cellState.expand({0})};
  auto shardingLoop = [&graph, &weights, &params, &opt, &cache](
                          unsigned shardIndex, std::vector<Tensor> &fwdState,
                          const std::vector<Tensor> &inputs,
                          const Tensor &seqIdx, const Tensor &interimIn,
                          Tensor &interimOut, std::vector<Tensor> &outputs,
                          program::Sequence &initProg,
                          const DebugNameAndId &dnai) {
    auto loop = Sequence({}, {dnai});
    auto &fwdInput = inputs[0];
    LstmState state = {fwdState[0].squeeze({0}), fwdState[1].squeeze({0})};
    bool useWeightedIn = !params.doInputWeightCalc || opt.preCalcWeights;
    const Tensor *inputWeightsPtr =
        useWeightedIn ? nullptr : &weights.inputWeights;
    if (interimOut.valid()) {
      LstmState newState;
      LstmInternalState internalState;
      std::tie(newState, internalState) = basicLstmCellForwardPass(
          graph, fwdInput, weights.biases, state, inputWeightsPtr,
          weights.outputWeights, loop, opt, opt.inferenceOnly, params.cellOrder,
          {dnai}, cache);
      auto fwdIntermediates = getFwdIntermediatesToSave(
          state, newState, internalState, opt, params);
      loop.add(Copy(fwdIntermediates, interimOut, false, {dnai}));
      auto newStateTensor = newState.getAsTensor();
      auto stateTensor = concat(fwdState);
      loop.add(Copy(newStateTensor, stateTensor, false, {dnai}));
    } else {
      basicLstmCellForwardPassInPlace(graph, fwdInput, weights.biases, state,
                                      inputWeightsPtr, weights.outputWeights,
                                      loop, opt, opt.inferenceOnly,
                                      params.cellOrder, {dnai}, cache);
    }
    return loop;
  };
  bool useWeightedIn = !params.doInputWeightCalc || opt.preCalcWeights;
  auto input = useWeightedIn ? weightedIn : prevLayerActs;
  auto numIntermediates =
      intermediatesSeq ? getNumFwdIntermediatesToSave(opt) : 0;
  boost::optional<rnn::StateSequence> stateSequence;
  if (params.outputFullSequence) {
    stateSequence = rnn::StateSequence{
        rnn::createOutputTensor(graph, params.rnn, opt.numShards,
                                {di, "output"}),
        0};
  }
  if (intermediatesSeq) {
    *intermediatesSeq =
        rnn::createOutputTensor(graph, params.rnn, numIntermediates,
                                opt.numShards, {di, "fwdIntermediatesSeq"})
            .reshapePartial(0, 1, {params.rnn.timeSteps, numIntermediates});
    fwdProg.add(WriteUndef(*intermediatesSeq, {di}));
  }
  std::vector<Tensor> dummyOutputs;
  auto updatedState =
      rnn::Rnn(graph, params.rnn, false, initState, stateSequence, {input},
               nullptr, intermediatesSeq, dummyOutputs, fwdProg, shardingLoop,
               opt.numShards, {di, "rnn"});
  updatedState[0] = params.outputFullSequence ? stateSequence->output
                                              : updatedState[0].squeeze({0});
  std::pair<Tensor, Tensor> outputs = {updatedState[0],
                                       updatedState[1].squeeze({0})};
  return outputs;
}

static std::tuple<LstmState, Tensor, Tensor>
backwardStepImpl(Graph &graph, const Tensor *gradNextLayer,
                 const Tensor &fwdIntermediates, const LstmState &stateGrad,
                 const Tensor *weightsInput, const Tensor &weightsOutput,
                 Sequence &initProg, Sequence &prog, const LstmOpts &opt,
                 const std::vector<BasicLstmCellUnit> &cellOrder,
                 const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  const std::string fPrefix = "LstmBwd";
  auto outputGrad = stateGrad.output;
  auto outputGroupingIntoLayer = detectInnermostGrouping(graph, outputGrad);
  if (gradNextLayer) {
    outputGrad = popops::add(graph, outputGrad, *gradNextLayer, prog,
                             {dnai, fPrefix + "/AddActGrads"});
  }
  auto actOutputGate = fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT_GATE];
  auto actOutputTanh = fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT_TANH];
  auto prevCellState = fwdIntermediates[LSTM_FWD_INTERMEDIATE_PREV_CELL_STATE];
  auto t = mul(graph, concat({actOutputGate, actOutputTanh}),
               outputGrad.broadcast(2, 0), prog, {dnai, fPrefix + "/MulOGate"});
  auto gradAtOTanhInput = t.slice(0, outputGrad.dim(0));
  auto gradAtOutputGateInput =
      t.slice(outputGrad.dim(0), 2 * outputGrad.dim(0));

  auto cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
  auto gradAtOTanhOutput = nonLinearityInputGradient(
      graph, NonLinearityType::TANH, actOutputTanh, gradAtOTanhInput, cs1,
      {dnai, fPrefix + "/OuputTanh"});
  auto gradOutputGate = nonLinearityInputGradient(
      graph, NonLinearityType::SIGMOID, actOutputGate, gradAtOutputGateInput,
      cs1, {dnai, fPrefix + "/OutputGate"});
  prog.add(Execute(cs1, {dnai}));

  auto gradCellState = stateGrad.cellState;

  addInPlace(graph, gradAtOTanhOutput, gradCellState, prog,
             {dnai, fPrefix + "/AddCellState"});
  auto actInputGate = fwdIntermediates[LSTM_FWD_INTERMEDIATE_INPUT_GATE];
  auto actCandidate = fwdIntermediates[LSTM_FWD_INTERMEDIATE_CAND_TANH];
  auto actForgetGate = fwdIntermediates[LSTM_FWD_INTERMEDIATE_FORGET_GATE];
  auto t1 = mul(
      graph, concat({actInputGate, actCandidate, prevCellState, actForgetGate}),
      gradAtOTanhOutput.broadcast(4, 0), prog, {dnai, fPrefix});

  const auto batchSize = gradAtOTanhOutput.dim(0);
  auto gradAtCandTanhInput = t1.slice(0, batchSize);
  auto gradAtInputGateInput = t1.slice(batchSize, 2 * batchSize);
  auto gradAtForgetGateInput = t1.slice(2 * batchSize, 3 * batchSize);
  auto newGradCellState = t1.slice(3 * batchSize, 4 * batchSize);

  auto cs2 = graph.addComputeSet({dnai, fPrefix + "/{Input+Candidate}Gate"});
  auto gradInputGate = nonLinearityInputGradient(
      graph, NonLinearityType::SIGMOID, actInputGate, gradAtInputGateInput, cs2,
      {dnai, fPrefix + "/InputGate"});
  auto gradCandidate = nonLinearityInputGradient(
      graph, NonLinearityType::TANH, actCandidate, gradAtCandTanhInput, cs2,
      {dnai, fPrefix + "/Cand"});
  auto gradForgetGate = nonLinearityInputGradient(
      graph, NonLinearityType::SIGMOID, actForgetGate, gradAtForgetGateInput,
      cs2, {dnai, fPrefix + "/Cand"});
  prog.add(Execute(cs2, {dnai}));

  const auto gradUnits = [&] {
    std::vector<Tensor> gradUnitsT;
    for (const auto cell : cellOrder) {
      switch (cell) {
      case BASIC_LSTM_CELL_FORGET_GATE:
        gradUnitsT.push_back(gradForgetGate.expand({0}));
        break;
      case BASIC_LSTM_CELL_INPUT_GATE:
        gradUnitsT.push_back(gradInputGate.expand({0}));
        break;
      case BASIC_LSTM_CELL_CANDIDATE:
        gradUnitsT.push_back(gradCandidate.expand({0}));
        break;
      case BASIC_LSTM_CELL_OUTPUT_GATE:
        gradUnitsT.push_back(gradOutputGate.expand({0}));
        break;
      case BASIC_LSTM_CELL_NUM_UNITS:
        assert(false);
      }
    }
    return concat(std::move(gradUnitsT));
  }();

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  auto grads = flattenUnits(gradUnits);
  Tensor weightsTransposed;
  if (weightsInput == nullptr) {
    weightsTransposed = flattenUnits(weightsOutput).transpose();
  } else {
    weightsTransposed =
        flattenUnits(concat(*weightsInput, weightsOutput, 1)).transpose();
  }

  weightsTransposed = preArrangeMatMulInputRHS(
      graph, grads.shape(), weightsTransposed, initProg,
      {dnai, fPrefix + "/PreArrangeWeights"}, mmOpt, cache);

  Tensor gradientIn, gradientPrevStep;
  if (weightsInput) {
    auto inputSize = weightsInput->dim(1);
    auto outputSize = weightsOutput.dim(1);
    auto out = matMul(graph, grads, weightsTransposed, prog,
                      {dnai, fPrefix + "/{Prev + Input}Grad"}, mmOpt, cache);
    out = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer, prog,
                                     {dnai, fPrefix});
    gradientIn = out.slice(0, inputSize, 1);
    gradientPrevStep = out.slice(inputSize, inputSize + outputSize, 1);
  } else {
    gradientPrevStep = matMul(graph, grads, weightsTransposed, prog,
                              {dnai, fPrefix + "/PrevStepGrad"}, mmOpt, cache);
    gradientPrevStep = tryGroupedPartialTranspose(
        graph, gradientPrevStep, detectInnermostGrouping(graph, outputGrad),
        prog, {dnai, fPrefix});
  }

  return std::make_tuple(LstmState{gradientPrevStep, newGradCellState},
                         gradientIn, gradUnits);
}

std::tuple<LstmState, Tensor, Tensor> basicLstmBackwardStep(
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const LstmState &stateGrad, const Tensor &weightsInput,
    const Tensor &weightsOutput, Sequence &initProg, Sequence &prog,
    const LstmOpts &opt, const std::vector<BasicLstmCellUnit> &cellOrder,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  return backwardStepImpl(graph, gradNextLayer, fwdIntermediates, stateGrad,
                          &weightsInput, weightsOutput, initProg, prog, opt,
                          cellOrder, {dnai}, cache);
}

std::pair<LstmState, Tensor> basicLstmBackwardStep(
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const LstmState &stateGrad, const Tensor &weightsOutput, Sequence &initProg,
    Sequence &prog, const LstmOpts &opt,
    const std::vector<BasicLstmCellUnit> &cellOrder, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  LstmState prevStateGrad;
  Tensor bwdIntermediates;
  std::tie(prevStateGrad, std::ignore, bwdIntermediates) = backwardStepImpl(
      graph, gradNextLayer, fwdIntermediates, stateGrad, nullptr, weightsOutput,
      initProg, prog, opt, cellOrder, {dnai}, cache);
  return std::make_pair(prevStateGrad, bwdIntermediates);
}

/// Add the partial weight gradients from this timestep to the accumulated
/// weight gradients. Once all the gradients have been accumulated call
/// basicLstmParamUpdateFinal() to do any final accumulation / rearrangement
/// that is required.
static void basicLstmParamUpdate(Graph &graph, const Tensor &prevLayerActs,
                                 const Tensor &prevStepActs,
                                 const Tensor &bwdIntermediates,
                                 LstmWeights &weightGrads, Sequence &prog,
                                 const LstmOpts &opt,
                                 const DebugNameAndId &dnai,
                                 matmul::PlanningCache *cache) {
  logging::popnn::debug("basicLstmParamUpdate begin {}", dnai.getPathName());
  const std::string fPrefix = "LstmDeltas";
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_WU");
  matMulAcc(graph,
            concat(flattenUnits(weightGrads.inputWeights),
                   flattenUnits(weightGrads.outputWeights)),
            1.0, concat(prevLayerActs.transpose(), prevStepActs.transpose()),
            flattenUnits(bwdIntermediates), prog, {dnai, fPrefix + "/Wi"},
            mmOpt, cache);

  if (bwdIntermediates.elementType() != weightGrads.biases.elementType()) {
    using namespace popops::expr;
    popops::mapInPlace(graph,
                       Add(_1, Cast(_2, weightGrads.biases.elementType())),
                       {weightGrads.biases, bwdIntermediates}, prog,
                       {dnai, fPrefix + "/basicLstmParamUpdate"});

  } else {
    // We defer the reduction across the batch to later.
    popops::addInPlace(graph, weightGrads.biases, bwdIntermediates, prog,
                       {dnai, fPrefix + "/Bias"});
  }
  logging::popnn::debug("basicLstmParamUpdate end {}", dnai.getPathName());
}

static LstmWeights basicLstmParamUpdateFinal(Graph &graph,
                                             const LstmWeights &weights,
                                             const LstmWeights &weightGrads,
                                             Sequence &prog,
                                             const DebugNameAndId &dnai) {
  // The accumulated bias gradients still has a batch axis that we must
  // accumulate over - do this now.
  logging::popnn::debug("basicLstmParamUpdateFinal begin {}",
                        dnai.getPathName());
  auto biasGrad = graph.clone(weightGrads.biases.elementType(), weights.biases,
                              {dnai, "biasGrad"});
  popops::reduceWithOutput(graph, weightGrads.biases, biasGrad, {1},
                           {popops::Operation::ADD}, prog,
                           {dnai, "FinalBiasReduction"});
  auto finalWeightGrads = weightGrads;
  finalWeightGrads.biases = biasGrad;
  logging::popnn::debug("basicLstmParamUpdateFinal end {}", dnai.getPathName());
  return finalWeightGrads;
}

/// Create variables used to accumulate gradients of the weights in the
/// backward pass.
static LstmWeights createWeightAccumulators(Graph &graph,
                                            const LstmWeights &weights,
                                            const Tensor &bwdIntermediates,
                                            const LstmOpts &options,
                                            const DebugNameAndId &dnai) {
  logging::popnn::debug("Create weightAccumulators of type {}",
                        options.accumulatorsType.toString());
  LstmWeights weightAccs;
  if (options.preCalcWeights) {
    weightAccs.inputWeights =
        graph.clone(options.accumulatorsType, weights.inputWeights,
                    {dnai, "inputWeightsDeltaAcc"});
    weightAccs.outputWeights =
        graph.clone(options.accumulatorsType, weights.outputWeights,
                    {dnai, "outputWeightsDeltaAcc"});
  } else {
    // inputWeights and outputWeights are slices of the one variable. Clone
    // them together as it results in a less complex tensor expression.
    auto concatenated = concat(flattenUnits(weights.inputWeights),
                               flattenUnits(weights.outputWeights));
    auto weightsDeltaAcc = graph.clone(options.accumulatorsType, concatenated,
                                       {dnai, "weightsDeltaAcc"});
    const auto inputSize = weights.inputWeights.dim(1);
    const auto outputSize = weights.outputWeights.dim(1);
    weightAccs.inputWeights = unflattenUnits(
        weightsDeltaAcc.slice(0, inputSize), BASIC_LSTM_CELL_NUM_UNITS);
    weightAccs.outputWeights =
        unflattenUnits(weightsDeltaAcc.slice(inputSize, inputSize + outputSize),
                       BASIC_LSTM_CELL_NUM_UNITS);
  }
  // We delay reducing across the batch until after we have accumulated
  // gradients from each timestep and therefore the bias accumulator still has
  // a batch axis. This amortizes the cost of reducing over the batch which
  // otherwise can be significant.
  weightAccs.biases = graph.clone(options.accumulatorsType, bwdIntermediates,
                                  {dnai, "biasesDeltaAcc"});
  logging::popnn::debug("Create weightAccumulators end");
  return weightAccs;
}

static void zeroWeightAccumulators(Graph &graph, program::Sequence &prog,
                                   const LstmWeights &weightsAcc,
                                   const LstmOpts &options,
                                   const DebugNameAndId &dnai) {
  logging::popnn::debug("zero weight accumulators");
  if (options.preCalcWeights) {
    popops::zero(graph,
                 concat({weightsAcc.inputWeights.flatten(),
                         weightsAcc.outputWeights.flatten(),
                         weightsAcc.biases.flatten()}),
                 prog, {dnai, "zeroWeightAccumulators"});
  } else {
    // inputWeights and outputWeights are slices of the one variable.
    // Recombining them means reorderToSimplify() in popops::zero() works a lot
    // better.
    auto concatenated = concat(flattenUnits(weightsAcc.inputWeights),
                               flattenUnits(weightsAcc.outputWeights));
    popops::zero(graph,
                 concat({concatenated.flatten(), weightsAcc.biases.flatten()}),
                 prog, {dnai, "zeroWeightAccumulators"});
  }
}

// Is it beneficial memory-wise to interleave weight update with
// backwards pass.
static bool interleavedWUIsBeneficial(const LstmParams &params) {
  const auto batchSize = params.rnn.batchSize;
  const auto inputSize = params.rnn.layerSizes[0];
  const auto outputSize = params.rnn.layerSizes[1];
  // Total elements needed for transposed weights.
  const auto totalTransposeParams =
      (inputSize + outputSize) * outputSize * BASIC_LSTM_CELL_NUM_UNITS;
  // Total elements needed for unit gradients for weight update if
  // not interleaved with backpropagation.
  const auto totalBwdIntermediates =
      batchSize * outputSize * BASIC_LSTM_CELL_NUM_UNITS * params.rnn.timeSteps;
  return totalTransposeParams <= totalBwdIntermediates;
}

static Tensor recomputeCellAndTanhImpl(Graph &graph, const LstmParams &params,
                                       const LstmOpts &options,
                                       const LstmState &fwdStateInit,
                                       const Tensor &fwdIntermediatesSeq,
                                       program::Sequence &prog,
                                       const DebugNameAndId &dnai) {
  auto shardingLoop = [&graph, &params, &options](
                          unsigned shardIndex, std::vector<Tensor> &shardState,
                          const std::vector<Tensor> &inputs,
                          const Tensor &seqIdx, const Tensor &interimIn,
                          Tensor &interimOut, std::vector<Tensor> &outputs,
                          program::Sequence &initProg,
                          const DebugNameAndId &dnai) {
    auto loop = Sequence({}, {dnai});
    auto prevCellState = shardState[0].squeeze({0});
    auto forgetGate = getSavedFwdIntermediate(
        interimIn, params, options, LSTM_FWD_INTERMEDIATE_FORGET_GATE);
    auto candidate = getSavedFwdIntermediate(interimIn, params, options,
                                             LSTM_FWD_INTERMEDIATE_CAND_TANH);
    auto outputGate = getSavedFwdIntermediate(
        interimIn, params, options, LSTM_FWD_INTERMEDIATE_OUTPUT_GATE);
    auto inputGate = getSavedFwdIntermediate(interimIn, params, options,
                                             LSTM_FWD_INTERMEDIATE_INPUT_GATE);

    // Recompute cell state and tanh
    Tensor newCellState, newTanhOutput;
    {
      auto prod = mul(graph, concat(forgetGate, candidate),
                      concat(prevCellState, inputGate), loop,
                      {dnai, "{Forget + Input}Gate"});

      newCellState = prod.slice(0, forgetGate.dim(0));
      auto updatedCandidate =
          prod.slice(forgetGate.dim(0), forgetGate.dim(0) + candidate.dim(0));
      addInPlace(graph, newCellState, updatedCandidate, loop,
                 {dnai, "AddCellCand"});
      newTanhOutput =
          popops::tanh(graph, newCellState, loop, {dnai, "TanhCellState"});
    }

    loop.add(Copy(concat(newTanhOutput.expand({0}), prevCellState.expand({0})),
                  outputs[0], false, {dnai}));
    loop.add(Copy(newCellState, prevCellState, false, {dnai}));
    return loop;
  };

  std::size_t numToRecompute =
      LSTM_FWD_INTERMEDIATE_OUTPUT - LSTM_FWD_INTERMEDIATE_OUTPUT_TANH;
  std::vector<Tensor> recomputedIntermediatesSeq{rnn::createOutputTensor(
      graph, params.rnn, numToRecompute, options.numShards,
      {dnai, "recomputedIntermediates"})};
  std::vector<Tensor> initState = {fwdStateInit.cellState.expand({0})};
  boost::optional<rnn::StateSequence> stateSequence{boost::none};
  rnn::Rnn(graph, params.rnn, false, initState, stateSequence, {},
           &fwdIntermediatesSeq, nullptr, recomputedIntermediatesSeq, prog,
           shardingLoop, options.numShards, {dnai, "rnn"});
  return recomputedIntermediatesSeq[0].reshapePartial(
      0, 1, {params.rnn.timeSteps, numToRecompute});
}

static Tensor recomputeFwdIntermediates(Graph &graph,
                                        const LstmState &fwdStateInit,
                                        const Tensor &fwdIntermediatesSeq,
                                        const LstmParams &params,
                                        const LstmOpts &options,
                                        program::Sequence &recomputeProg,
                                        const DebugNameAndId &recomputeDnai) {
  Tensor recomputedIntermediatesSeq;
  switch (options.recomputationMode) {
  case LstmRecomputationMode::None: {
    break;
  }
  case LstmRecomputationMode::CellAndTanh: {
    recomputedIntermediatesSeq = recomputeCellAndTanhImpl(
        graph, params, options, fwdStateInit, fwdIntermediatesSeq,
        recomputeProg, {recomputeDnai});
    break;
  }
  case LstmRecomputationMode::Full:
    // TODO: T12911 Implement this case.
    // fallthrough
  default:
    throw poplibs_error("Unhandled recomputation type");
  }
  return recomputedIntermediatesSeq;
}

// Perform an LSTM backward pass.
// Optionally return the intermediates from the backward pass (sequence
// cell unit gradients), or calculate weight gradients directly during
// this pass interleaved with the backward pass.
static LstmState
lstmBwdImpl(Graph &graph, const LstmParams &params, program::Sequence &prog,
            const LstmState &fwdStateInit, const Tensor &fwdIntermediatesSeq,
            const LstmWeights &weights, const Tensor &fwdInputSeq,
            const Tensor &fwdOutput, const Tensor &gradLayerNext,
            const Tensor *lastCellStateGradPtr, Tensor *inputGradSeq,
            Tensor *bwdIntermediatesPtr, LstmWeights *weightsGrad,
            const DebugNameAndId &dnai, const LstmOpts &options,
            poplin::matmul::PlanningCache *cache) {
  auto loopBwdWithWU = [&graph, &params, &options, &inputGradSeq, &cache](
                           LstmWeights &weights, LstmWeights *weightsGrad,
                           unsigned shardIndex, std::vector<Tensor> &shardState,
                           const std::vector<Tensor> &bwdInput,
                           const Tensor &seqIdx, const Tensor &fwdIntermediates,
                           Tensor &interimOut, std::vector<Tensor> &outputs,
                           program::Sequence &initProg,
                           const DebugNameAndId &dnai) {
    auto loop = Sequence({}, {dnai});
    auto &weightsInput = weights.inputWeights;
    auto &weightsOutput = weights.outputWeights;
    auto &gradLayerNext = bwdInput[0];
    Tensor inputGrad =
        shardState[0].valid() ? shardState[0].squeeze({0}) : Tensor{};
    LstmState stateGrads = {shardState[1].squeeze({0}),
                            shardState[2].squeeze({0})};
    LstmState newStateGrads;
    Tensor bwdIntermediates;
    if (inputGradSeq) {
      Tensor nextInputGrad;
      std::tie(newStateGrads, nextInputGrad, bwdIntermediates) =
          popnn::lstm::basicLstmBackwardStep(
              graph, &gradLayerNext, fwdIntermediates, stateGrads, weightsInput,
              weightsOutput, initProg, loop, options, params.cellOrder, {dnai},
              cache);
      if (inputGrad.valid()) {
        loop.add(Copy(nextInputGrad, inputGrad, false, {dnai}));
      }
    } else {
      std::tie(newStateGrads, bwdIntermediates) = basicLstmBackwardStep(
          graph, &gradLayerNext, fwdIntermediates, stateGrads, weightsOutput,
          initProg, loop, options, params.cellOrder, {dnai}, cache);
    }
    if (interimOut.valid()) {
      loop.add(Copy(bwdIntermediates, interimOut, false, {dnai}));
    }
    loop.add(Copy(newStateGrads.getAsTensor(), stateGrads.getAsTensor(), false,
                  {dnai}));
    if (weightsGrad) {
      auto &prevLayerOut = bwdInput[1];
      auto &prevStepOut = bwdInput[2];
      if (shardIndex == options.numShards - 1) {
        *weightsGrad = createWeightAccumulators(
            graph, weights, bwdIntermediates, options, {dnai, "weightsGrad"});
        zeroWeightAccumulators(graph, initProg, *weightsGrad, options,
                               {dnai, "zeroWeightAcc"});
      }
      basicLstmParamUpdate(graph, prevLayerOut, prevStepOut, bwdIntermediates,
                           *weightsGrad, loop, options,
                           {dnai, "basicLstmParamUpdate"}, cache);
    }
    return loop;
  };

  Tensor recomputedIntermediatesSeq = recomputeFwdIntermediates(
      graph, fwdStateInit, fwdIntermediatesSeq, params, options, prog,
      {dnai, "recomputeFwdIntermediates"});
  Tensor fwdIntermediates = reconstructIntermediatesFromRecomputed(
      fwdIntermediatesSeq, recomputedIntermediatesSeq, params, options);
  auto lastOutGradInit = rnn::createInitialState(
      graph, params.rnn, true, 1, options.numShards, {dnai, "lastOutGradInit"});
  if (params.outputFullSequence) {
    zero(graph, lastOutGradInit, prog, {dnai, "zeroLastOutGrad"});
  } else {
    prog.add(Copy(gradLayerNext, lastOutGradInit, false, {dnai}));
  }
  auto lastCellStateGradInit =
      rnn::createInitialState(graph, params.rnn, true, 1, options.numShards,
                              {dnai, "lastCellStateGradInit"});
  if (lastCellStateGradPtr) {
    prog.add(Copy(*lastCellStateGradPtr, lastCellStateGradInit, false,
                  {dnai, "initLastOutGrad"}));
  } else {
    zero(graph, lastCellStateGradInit, prog, {dnai, "initCellStateGrad"});
  }
  Tensor gradLayerNextRearranged;
  if (params.outputFullSequence) {
    gradLayerNextRearranged =
        rnn::createOutputTensor(graph, params.rnn, options.numShards,
                                {dnai, "gradLayerNextRearranged"});
    prog.add(Copy(gradLayerNext, gradLayerNextRearranged, false,
                  {dnai, "initGradLayerNextRearranged"}));
  }
  Tensor inputGradInit;
  boost::optional<rnn::StateSequence> inputGrad;
  if (inputGradSeq) {
    inputGradInit =
        rnn::createInitialState(graph, params.rnn, false, 1, options.numShards,
                                {dnai, "inputGradInit"});
    *inputGradSeq = rnn::createInputTensor(graph, params.rnn, options.numShards,
                                           {dnai, "inputGrad"});
    inputGrad = rnn::StateSequence{*inputGradSeq, 0};
  }
  std::vector<Tensor> bwdStateInit = {inputGradInit, lastOutGradInit,
                                      lastCellStateGradInit};
  if (bwdIntermediatesPtr) {
    *bwdIntermediatesPtr =
        rnn::createOutputTensor(graph, params.rnn, BASIC_LSTM_CELL_NUM_UNITS,
                                options.numShards, {dnai, "bwdIntermediates"})
            .reshapePartial(0, 1,
                            {params.rnn.timeSteps, BASIC_LSTM_CELL_NUM_UNITS});
    prog.add(WriteUndef(*bwdIntermediatesPtr, {dnai}));
  }
  Tensor prevLayerOut, prevStepOut;
  if (weightsGrad) {
    // make a copy of the activations so that they are sliced efficiently
    prevLayerOut = createInputTensor(graph, params.rnn, options.numShards,
                                     {dnai, "prevLayerOut"});
    prog.add(Copy(fwdInputSeq, prevLayerOut, false, {dnai}));
    auto fwdOut = (params.outputFullSequence)
                      ? fwdOutput
                      : fwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT];
    prevStepOut =
        rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdStateInit.output,
                            prog, options.numShards, {dnai, "fwdOutShifted"});
  }
  std::vector<Tensor> bwdAndWuInputs = {gradLayerNextRearranged, prevLayerOut,
                                        prevStepOut};
  using namespace std::placeholders;
  const auto shardingLoop = std::bind(loopBwdWithWU, weights, weightsGrad, _1,
                                      _2, _3, _4, _5, _6, _7, _8, _9);
  std::vector<Tensor> dummyOutputs;
  auto updatedState =
      rnn::Rnn(graph, params.rnn, true, bwdStateInit, inputGrad, bwdAndWuInputs,
               &fwdIntermediates, bwdIntermediatesPtr, dummyOutputs, prog,
               shardingLoop, options.numShards, {dnai, "updatedState"});
  if (weightsGrad) {
    *weightsGrad =
        basicLstmParamUpdateFinal(graph, weights, *weightsGrad, prog,
                                  {dnai, "basicLstmParamUpdateFinal"});
  }
  LstmState stateGrads = {updatedState[1].squeeze({0}),
                          updatedState[2].squeeze({0})};
  return stateGrads;
}

LstmState lstmBwd(Graph &graph, const LstmParams &params,
                  program::Sequence &prog, const LstmState &fwdStateInit,
                  const Tensor &fwdIntermediatesSeq, const LstmWeights &weights,
                  const Tensor &fwdInputSeq, const Tensor &fwdOutput,
                  const Tensor &gradLayerNext,
                  const Tensor *lastCellStateGradPtr, Tensor *inputGrad,
                  Tensor *bwdIntermediates,
                  const poplar::DebugContext &debugContext,
                  const OptionFlags &options_,
                  poplin::matmul::PlanningCache *planningCache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdIntermediatesSeq, weights, fwdInputSeq, fwdOutput,
              gradLayerNext, lastCellStateGradPtr, inputGrad, bwdIntermediates,
              fwdStateInit, params, options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_, params.rnn.dataType);

  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  LstmState outputs = lstmBwdImpl(
      graph, params, prog, fwdStateInit, fwdIntermediatesSeq, weights,
      fwdInputSeq, fwdOutput, gradLayerNext, lastCellStateGradPtr, inputGrad,
      bwdIntermediates, nullptr, {di}, std::move(options), planningCache);
  di.addOutputs(DI_ARGS(outputs));
  return outputs;
}

static LstmWeights
lstmWUImpl(Graph &graph, const LstmParams &params, program::Sequence &prog,
           const LstmState &fwdStateInit, const Tensor &fwdIntermediatesSeq,
           const Tensor &bwdIntermediatesSeq, const LstmWeights &weights,
           const Tensor &input, const Tensor &output,
           const DebugNameAndId &dnai, const LstmOpts &options,
           poplin::matmul::PlanningCache *planningCache) {
  LstmWeights weightGrads = createWeightAccumulators(
      graph, weights, bwdIntermediatesSeq[0], options, {dnai});
  zeroWeightAccumulators(graph, prog, weightGrads, options, {dnai});
  auto loopWU = [&graph, &options, &planningCache](
                    LstmWeights &weightGrads, unsigned shardIndex,
                    std::vector<Tensor> &shardState,
                    const std::vector<Tensor> &wuInput, const Tensor &seqIdx,
                    const Tensor &interimIn, Tensor &interimOut,
                    std::vector<Tensor> &output, program::Sequence &initProg,
                    const DebugNameAndId &dnai) {
    auto loop = Sequence({}, {dnai});
    auto &prevLayerOut = wuInput[0];
    auto &prevStepOut = wuInput[1];
    auto &bwdIntermediates = wuInput[2];
    basicLstmParamUpdate(graph, prevLayerOut, prevStepOut, bwdIntermediates,
                         weightGrads, loop, options, {dnai}, planningCache);
    return loop;
  };

  // make a copy of the activations so that they are sliced efficiently
  Tensor inputCopy = createInputTensor(graph, params.rnn, options.numShards,
                                       {dnai, "inputCopy"});
  prog.add(Copy(input, inputCopy, false, {dnai}));
  auto fwdOut = (params.outputFullSequence) ? output : fwdIntermediatesSeq;
  Tensor prevStepOut =
      rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdStateInit.output, prog,
                          options.numShards, {dnai, "fwdOutshifted"});
  std::vector<Tensor> wuInputs = {inputCopy, prevStepOut, bwdIntermediatesSeq};
  using namespace std::placeholders;
  const auto shardingLoop =
      std::bind(loopWU, weightGrads, _1, _2, _3, _4, _5, _6, _7, _8, 9);
  std::vector<Tensor> dummyOutputs;
  boost::optional<rnn::StateSequence> stateSequence{boost::none};
  auto updatedState = rnn::Rnn(graph, params.rnn, true, {}, stateSequence,
                               wuInputs, nullptr, nullptr, dummyOutputs, prog,
                               shardingLoop, options.numShards, {dnai, "rnn"});
  weightGrads =
      basicLstmParamUpdateFinal(graph, weights, weightGrads, prog, {dnai});
  return weightGrads;
}

LstmWeights lstmWU(Graph &graph, const LstmParams &params,
                   program::Sequence &prog, const LstmState &fwdStateInit,
                   const Tensor &fwdIntermediates,
                   const Tensor &bwdIntermediates, const LstmWeights &weights,
                   const Tensor &input, const Tensor &output,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_,
                   poplin::matmul::PlanningCache *planningCache) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(fwdIntermediates, bwdIntermediates,
                                         weights, input, output, fwdStateInit,
                                         params, options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_, params.rnn.dataType);

  auto outputs = lstmWUImpl(graph, params, prog, fwdStateInit, fwdIntermediates,
                            bwdIntermediates, weights, input, output, {di},
                            std::move(options), planningCache);
  di.addOutputs(DI_ARGS(outputs));
  return outputs;
}

LstmState lstmBwdWithWU(poplar::Graph &graph, const LstmParams &params,
                        poplar::program::Sequence &prog,
                        const LstmState &fwdStateInit,
                        const poplar::Tensor &fwdIntermediates,
                        const LstmWeights &weights, const poplar::Tensor &input,
                        const poplar::Tensor &output,
                        const poplar::Tensor &outputGrad,
                        const poplar::Tensor *lastCellStateGrad,
                        poplar::Tensor *inputGrad, LstmWeights &weightsGrad_,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options_,
                        poplin::matmul::PlanningCache *planningCache) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(fwdIntermediates, weights, input,
                                         output, outputGrad, lastCellStateGrad,
                                         inputGrad, weightsGrad_, fwdStateInit,
                                         params, options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_, params.rnn.dataType);

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
  LstmState stateGrads = lstmBwdImpl(
      graph, params, prog, fwdStateInit, fwdIntermediates, weights, input,
      output, outputGrad, lastCellStateGrad, inputGrad,
      interleaveWU ? nullptr : &bwdIntermediates,
      interleaveWU ? &weightsGrad_ : nullptr, {di}, options, planningCache);

  if (!interleaveWU) {
    weightsGrad_ = lstmWUImpl(
        graph, params, prog, fwdStateInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, {di}, std::move(options), planningCache);
  }

  di.addOutputs(DI_ARGS(stateGrads));
  return stateGrads;
}

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  auto weighInput = params.doInputWeightCalc;
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.

  uint64_t multsWeighInp = weighInput
                               ? static_cast<uint64_t>(inputSize) * 4 *
                                     outputSize * batchSize * sequenceSize * 2
                               : 0;
  uint64_t multsWeighOut = static_cast<uint64_t>(outputSize) * 4 * outputSize *
                           batchSize * sequenceSize * 2;

  // We ignore FLOPs for bias addition - in theory we could initialize the
  // accumulators with the biases during the matrix multiplication.
  uint64_t mulFlops =
      3 * static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  uint64_t addFlops =
      static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  return multsWeighInp + multsWeighOut + addFlops + mulFlops;
}

uint64_t getBasicLstmCellBwdFlops(const LstmParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  auto calcInputGrad = params.calcInputGradients;
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.

  uint64_t mulFlops =
      static_cast<uint64_t>(sequenceSize) * 6 * batchSize * outputSize;
  uint64_t inputGradFlops = calcInputGrad
                                ? static_cast<uint64_t>(inputSize) * 4 *
                                      outputSize * batchSize * sequenceSize * 2
                                : 0;
  uint64_t outputGradFlops = static_cast<uint64_t>(outputSize) * 4 *
                             outputSize * batchSize * sequenceSize * 2;
  return mulFlops + inputGradFlops + outputGradFlops;
}

uint64_t getBasicLstmCellWuFlops(const LstmParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];

  uint64_t weightFlops = static_cast<uint64_t>(inputSize + outputSize) * 4 *
                         outputSize * batchSize * sequenceSize * 2;
  uint64_t biasFlops =
      static_cast<uint64_t>(outputSize) * 4 * batchSize * sequenceSize * 2;
  return weightFlops + biasFlops;
}

} // namespace lstm
} // namespace popnn
