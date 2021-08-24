// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplibs_support/Tracepoint.hpp"
#include <cmath>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/PlanConstraints.hpp>
#include <poplibs_support/logging.hpp>
#include <popnn/Lstm.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Cast.hpp>

#include "MatMulInternal.hpp"
#include "RnnUtil.hpp"
#include "poplin/FullyConnected.hpp"

using namespace poplar;
using namespace poplar::program;

using namespace poplin;
using namespace popnn;
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

namespace ph = std::placeholders;

namespace { // Anonymous namespace
bool isCSNotSupported(popnn::NonLinearityType nl) {
  return (nl == popnn::NonLinearityType::SOFTMAX ||
          nl == popnn::NonLinearityType::SOFTMAX_STABLE ||
          nl == popnn::NonLinearityType::SOFTMAX_SCALED ||
          nl == popnn::NonLinearityType::HARD_SIGMOID);
}

expr::BinaryOp
fusedNonLinearityMulInPlaceExpr(NonLinearityType nonLinearityType) {
  switch (nonLinearityType) {
  case NonLinearityType::SIGMOID:
    return expr::_1 * expr::Sigmoid(expr::_2);

  case NonLinearityType::HARD_SIGMOID:
    return expr::_1 *
           expr::Max(expr::Const(0),
                     expr::Min(expr::Const(1), expr::Const(0.2f) * expr::_2 +
                                                   expr::Const(0.5f)));

  case NonLinearityType::RELU:
    return expr::_1 * expr::Max(expr::_2, expr::Const(0));

  case NonLinearityType::TANH:
    return expr::_1 * expr::Tanh(expr::_2);

  default:
    throw poputil::poplibs_error("Cannot compute expression for nonLinearity");
  }
}

void fusedNonLinearityMulInPlace(poplar::Graph &graph,
                                 popnn::NonLinearityType nonLinearityType,
                                 poplar::Tensor t1, poplar::Tensor t2,
                                 poplar::program::Sequence &prog,
                                 const DebugNameAndId &dnai) {
  switch (nonLinearityType) {
  case NonLinearityType::SIGMOID:
  case NonLinearityType::HARD_SIGMOID:
  case NonLinearityType::RELU:
  case NonLinearityType::TANH:
    mapInPlace(graph, fusedNonLinearityMulInPlaceExpr(nonLinearityType),
               {t1, t2}, prog, {dnai, "mapInPlace"});
    break;

  default: {
    auto nonlin = popnn::nonLinearity(graph, nonLinearityType, t2, prog,
                                      {dnai, "nonLinearity"});
    popops::mulInPlace(graph, t1, nonlin, prog, {dnai, "mulInPlace"});
  } break;
  }
}
} // Anonymous namespace

static void applyGateNonlinearities(Graph &graph, const Tensor &t,
                                    Sequence &prog,
                                    const std::vector<std::size_t> &cellIndices,
                                    const popnn::lstm::LstmParams &params,
                                    const DebugNameAndId &dnai) {
  auto sigmoidIn = concat({t[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]],
                           t[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]],
                           t[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]]});
  if (isCSNotSupported(params.activation) ||
      isCSNotSupported(params.recurrentActivation)) {
    nonLinearityInPlace(graph, params.recurrentActivation, sigmoidIn, prog,
                        {dnai});
    nonLinearityInPlace(graph, params.activation,
                        t[cellIndices[BASIC_LSTM_CELL_CANDIDATE]], prog,
                        {dnai});
  } else {
    auto cs = graph.addComputeSet({dnai, "OutputGate"});
    nonLinearityInPlace(graph, params.recurrentActivation, sigmoidIn, cs,
                        {dnai});
    nonLinearityInPlace(graph, params.activation,
                        t[cellIndices[BASIC_LSTM_CELL_CANDIDATE]], cs, {dnai});
    prog.add(Execute(cs, {dnai}));
  }
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
  boost::optional<std::size_t> numShards;
  boost::optional<bool> rnnCodeReuse;
  boost::optional<unsigned> rnnStepsPerWU;
  bool disableWUPartialInterleaving;
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

static OptionFlags getRnnOpts(const LstmOpts &lstmOpts) {
  OptionFlags rnnOpts;
  if (lstmOpts.rnnCodeReuse) {
    rnnOpts.set("codeReuse", std::to_string(lstmOpts.rnnCodeReuse.get()));
  }
  return rnnOpts;
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
  lstmOpts.numShards = boost::none;
  lstmOpts.rnnCodeReuse = boost::none;
  lstmOpts.disableWUPartialInterleaving = false;
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
      {"rnnCodeReuse", OptionHandler::createWithBool(lstmOpts.rnnCodeReuse)},
      {"rnnStepsPerWU",
       OptionHandler::createWithInteger(lstmOpts.rnnStepsPerWU)},
      {"disableWUPartialInterleaving",
       OptionHandler::createWithBool(lstmOpts.disableWUPartialInterleaving)},
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

static unsigned getNumFwdIntermediatesToSave(const LstmParams &params,
                                             const LstmOpts &options) {
  unsigned numIntermediates = 0;
  if (options.recomputationMode == LstmRecomputationMode::None) {
    numIntermediates += 6;
  } else if (options.recomputationMode == LstmRecomputationMode::CellAndTanh) {
    numIntermediates += 4;
  } else {
    throw poputil::poplibs_error("Unhandled recomputation type");
  }
  if (!params.outputFullSequence) {
    numIntermediates++;
  }
  return numIntermediates;
}

// Sharding is relevant for LSTM/GRU models which use significantly fewer
// tiles for storage of sequences than are available on the target. The total
// memory required to store the input and output dimensions is directly
// proportional to the LSTM sequence size. For large sequence sizes the tiles
// on which the sequences have been mapped would run out of memory, even with
// the availability of spare memory on the unmapped tiles on the same IPU.
// Sharding alleviates this problem by mapping the sequences to disjoint
// sets of tiles. The ratio of the total number of tiles on the target to the
// number of tiles that the sequences would be mapped to without sharding
// determines the maximum number of shards. However sharding involves code
// duplication and memory overheads due to additional exchanges. These memory
// usage overheads could become prohibitive when excessive sharding is applied.
// Likewise sharding also adds execution time overheads.
//
// For reasonably sized batch/feature dimensions the maximum number of shards
// is a small enough number which can be used to directly determine the number
// of shards. However this approach does not work well for smaller sized LSTM
// models. For very small input and output layer sizes and small batch sizes
// the maximum number of shards could run into the hundreds or thousands.
//
// To limit sharding when batch/feature dimensions are small, we allow operands
// to occupy up to 10% of total tile memory before sharding further. Layers
// with reasonably large batch/feature dimensions typically utilise enough tiles
// that the maximum shards calculated is small even if memory usage per-tile for
// operands is high. Hence this only really applies to the small cases.
//
// All LSTM passes - Fwd, Bwd & WU passes - must use the same number of shards.
// Hence, operand memory is calculated based on the Fwd pass since it can
// be used as a reasonable approximation for all the passes.
static std::size_t getNumShards(const Graph &graph, const LstmParams &params,
                                const LstmOpts &opt,
                                const DebugNameAndId &dnai) {
  auto target = graph.getTarget();
  auto tileMemory = target.getBytesPerTile();
  auto maxShards = params.rnn.getMaxShards(graph);
  auto inputSize = params.rnn.getInputBytesPerTile(graph);
  auto outputSize = params.rnn.getOutputBytesPerTile(graph);
  auto numIntermediates = getNumFwdIntermediatesToSave(params, opt);
  auto operandSingleIteration =
      inputSize + (outputSize * (1 + numIntermediates));
  auto operandSize = operandSingleIteration * params.rnn.maxTimeSteps;

  // Fraction of total tile memory that is nominally designated for operands
  double operandFraction = 0.1;

  double availableOperandMemory = tileMemory * operandFraction;
  std::size_t estShards = std::ceil(operandSize / availableOperandMemory);
  auto numShards = std::min(estShards, maxShards);
  if (opt.numShards) {
    if ((*opt.numShards < 1) || (*opt.numShards > maxShards)) {
      throw poputil::poplibs_error("LSTM numShards must be within "
                                   "interval [1," +
                                   std::to_string(maxShards) + "]");
    }
    numShards = *opt.numShards;
  }
  logging::popnn::debug(
      "'{}': inputSize={} outputSize={} operandSize={} numInter={} "
      "available={} maxShards={} estimated-shards={} numShards={}",
      dnai.getPathName(), inputSize, outputSize, operandSize, numIntermediates,
      availableOperandMemory, maxShards, estShards, numShards);
  return numShards;
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
    auto fcBatchSize = params.rnn.maxTimeSteps * params.rnn.batchSize;
    auto in = createMatMulInputLHS(
        graph, params.rnn.dataType, {fcBatchSize, fcInputSize},
        {fcInputSize, fcOutputSize}, {dnai}, mmOpt, cache);
    return in.reshape(
        {params.rnn.maxTimeSteps, params.rnn.batchSize, inputSize});
  } else {
    auto numShards = getNumShards(graph, params, opt, {dnai, "numShards"});
    return rnn::createInputTensor(graph, params.rnn, numShards,
                                  {dnai, "input"});
  }
}

Tensor createInput(Graph &graph, const LstmParams &params,
                   const poplar::DebugContext &debugContext,
                   const OptionFlags &options, matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  auto output = createInput(graph, params, {di},
                            parseOptions(options, params.rnn.dataType), cache);
  di.addOutput(output);
  return output;
}

Tensor createInitialOutput(Graph &graph, const LstmParams &params,
                           const poplar::DebugContext &debugContext,
                           const OptionFlags &options,
                           matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));
  auto opt = parseOptions(options, params.rnn.dataType);
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto output = rnn::createInitialState(graph, params.rnn, true, 1, numShards,
                                        {di, "initialOutput"})
                    .squeeze({0});
  di.addOutput(output);
  return output;
}

Tensor createInitialCellState(Graph &graph, const LstmParams &params,
                              const poplar::DebugContext &debugContext,
                              const OptionFlags &options,
                              matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));
  auto opt = parseOptions(options, params.rnn.dataType);
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto output = rnn::createInitialState(graph, params.rnn, true, 1, numShards,
                                        {di, "initialCellState"})
                    .squeeze({0});
  di.addOutput(output);
  return output;
}

LstmState createInitialState(Graph &graph, const LstmParams &params,
                             const poplar::DebugContext &debugContext,
                             const OptionFlags &options,
                             matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));
  auto opt = parseOptions(options, params.rnn.dataType);

  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto initialOutput = rnn::createInitialState(graph, params.rnn, true, 1,
                                               numShards, {di, "initialOutput"})
                           .squeeze({0});
  auto initialCellState =
      rnn::createInitialState(graph, params.rnn, true, 1, numShards,
                              {di, "initialCellState"})
          .squeeze({0});
  LstmState outputs = {initialOutput, initialCellState};
  di.addOutputs(DI_ARGS(outputs));
  return outputs;
}

void zeroInitialState(Graph &graph, const LstmState &state, Sequence &prog,
                      const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(state));

  zero(graph, concat(state.output, state.cellState), prog, {di});
}

std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const LstmParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options,
                    poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
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
      aShape[0] = params.rnn.maxTimeSteps * params.rnn.batchSize;
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
  POPNN_TRACEPOINT();
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
  POPNN_TRACEPOINT();
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
    const std::vector<std::size_t> &cellIndices, const LstmParams &params,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
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
                          params, {dnai});
}

static std::pair<LstmState, LstmInternalState> basicLstmCellForwardPass(
    Graph &graph, const Tensor &in, const Tensor &biases,
    const LstmState &prevState, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const LstmOpts &opt,
    bool inferenceOnly, const rnn::TimeStepState &time,
    const rnn::RnnBatchwiseFlags &batchwiseFlags, const LstmParams &params,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  const auto &prevCellState = prevState.cellState;
  const std::string baseStr = "BasicLstmCell";

  std::vector<Tensor> toConcat;
  toConcat.reserve(BASIC_LSTM_CELL_NUM_UNITS);
  assert(params.cellOrder.size() == BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    const auto unit = params.cellOrder.at(i);
    toConcat.push_back(
        graph.clone(prevCellState, {dnai, getUnitName(unit) + "Rearranged"})
            .expand({0}));
  }

  // build reverse mapping of cellOrder
  std::vector<std::size_t> cellIndices(BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i < params.cellOrder.size(); ++i) {
    cellIndices[params.cellOrder[i]] = i;
  }

  auto unitsOutput = concat(toConcat);
  lstmCellForwardPassCalcUnits(
      graph, in, biases, prevState, weightsInput, weightsOutput, prog, opt,
      inferenceOnly, unitsOutput, cellIndices, params, {dnai, baseStr}, cache);
  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]];
  auto candidate = unitsOutput[cellIndices[BASIC_LSTM_CELL_CANDIDATE]];
  auto outputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]];
  auto inputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]];
  if (batchwiseFlags.valid()) {
    Sequence maskProg;
    auto mask = batchwiseFlags.mask.expand({1});
    auto gates = concat({forgetGate, inputGate, candidate, outputGate});
    auto gateMasks = mask.broadcast(BASIC_LSTM_CELL_NUM_UNITS, 0);
    mapInPlace(graph, expr::_1 * expr::_2, {gates, gateMasks}, maskProg,
               {dnai});
    prog.add(If(time.variableSeqFlag, maskProg, Sequence(),
                {dnai, baseStr + "/maskedGates"}));
  }
  auto prod = mul(graph, concat(forgetGate, candidate),
                  concat(prevCellState, inputGate), prog,
                  {dnai, baseStr + "/{Forget + Input}Gate"});

  auto updatedCellState = prod.slice(0, forgetGate.dim(0));
  auto updatedCandidate =
      prod.slice(forgetGate.dim(0), forgetGate.dim(0) + candidate.dim(0));

  addInPlace(graph, updatedCellState, updatedCandidate, prog,
             {dnai, baseStr + "/AddCellCand"});
  auto tanhOutput = popnn::nonLinearity(
      graph, params.activation, updatedCellState, prog, {dnai, baseStr});
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
    bool inferenceOnly, const LstmParams &params, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  auto cellState = state.cellState;
  auto output = state.output;
  const std::string baseStr = "BasicLstmCell";

  std::vector<Tensor> toConcat;
  toConcat.reserve(BASIC_LSTM_CELL_NUM_UNITS);
  assert(params.cellOrder.size() == BASIC_LSTM_CELL_NUM_UNITS);
  for (unsigned i = 0; i != BASIC_LSTM_CELL_NUM_UNITS; ++i) {
    const auto unit = params.cellOrder.at(i);
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
  for (unsigned i = 0; i < params.cellOrder.size(); ++i) {
    cellIndices[params.cellOrder[i]] = i;
  }

  auto unitsOutput = concat(toConcat);

  lstmCellForwardPassCalcUnits(
      graph, in, biases, state, weightsInput, weightsOutput, prog, opt,
      inferenceOnly, unitsOutput, cellIndices, params, {dnai, baseStr}, cache);

  assert(unitsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_FORGET_GATE]];
  auto candidate = unitsOutput[cellIndices[BASIC_LSTM_CELL_CANDIDATE]];
  auto outputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_OUTPUT_GATE]];
  auto inputGate = unitsOutput[cellIndices[BASIC_LSTM_CELL_INPUT_GATE]];
  mulInPlace(graph, concat(cellState, candidate), concat(forgetGate, inputGate),
             prog, {dnai, baseStr + "/{Forget + Input}Gate"});
  addInPlace(graph, cellState, candidate, prog,
             {dnai, baseStr + "/AddCellCand"});
  fusedNonLinearityMulInPlace(graph, params.activation, outputGate, cellState,
                              prog, {dnai, baseStr + "/CalcNextOutput"});
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
                       std::vector<std::size_t> layerSizes,
                       NonLinearityType activation,
                       NonLinearityType recurrentActivation)
    : rnn(dataType, batchSize, timeSteps, layerSizes), dataType(dataType),
      batchSize(batchSize), timeSteps(timeSteps), layerSizes(layerSizes),
      activation(activation), recurrentActivation(recurrentActivation) {}

LstmParams::LstmParams(poplar::Type dataType, std::size_t batchSize,
                       std::size_t maxTimeSteps,
                       const poplar::Tensor &timeSteps,
                       std::vector<std::size_t> layerSizes,
                       NonLinearityType activation,
                       NonLinearityType recurrentActivation)
    : rnn(dataType, batchSize, maxTimeSteps, timeSteps, layerSizes),
      dataType(dataType), batchSize(batchSize), timeSteps(maxTimeSteps),
      layerSizes(layerSizes), activation(activation),
      recurrentActivation(recurrentActivation) {}

static Tensor lstmFwd(Graph &graph, const LstmParams &params,
                      program::Sequence &prog, const LstmState &fwdStateInit,
                      const LstmWeights &weights, const Tensor &prevLayerActs,
                      Tensor *intermediatesSeq, Tensor *finalCellState,
                      const poplar::DebugContext &debugContext,
                      const OptionFlags &options,
                      poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(prevLayerActs, weights, intermediatesSeq,
                            fwdStateInit, params, options, cache));

  validateParams(params);
  auto opt = parseOptions(options, params.rnn.dataType);

  Tensor weightedIn;
  if (!params.doInputWeightCalc) {
    weightedIn =
        graph.addVariable(params.rnn.dataType,
                          {params.rnn.maxTimeSteps, BASIC_LSTM_CELL_NUM_UNITS,
                           params.rnn.batchSize, params.rnn.layerSizes[1]},
                          {di, "dummyWeightedIn"});
    for (unsigned s = 0; s < params.rnn.maxTimeSteps; ++s) {
      mapTensorLinearly(graph, weightedIn[s]);
    }
  } else if (opt.preCalcWeights) {
    weightedIn =
        calcSequenceWeightedInputs(graph, prevLayerActs, weights.inputWeights,
                                   prog, opt, {di, "lstm/weightInputs"}, cache);
  }
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  std::vector<Tensor> initState = {fwdStateInit.output.expand({0}),
                                   fwdStateInit.cellState.expand({0})};
  auto shardingLoop = [&weights, &params, &opt, finalCellState,
                       &cache](Graph &graph, const rnn::TimeStepState &time,
                               const rnn::RnnBatchwiseFlags &batchwiseFlags,
                               std::vector<Tensor> &fwdState,
                               const rnn::RnnSlice &slice,
                               std::vector<Tensor> &created,
                               program::Sequence *initProg,
                               const DebugNameAndId &dnai) {
    auto loop = Sequence{{}, {dnai}};
    auto &fwdInput = slice.inputs[0];
    LstmState state = {fwdState[0].squeeze({0}), fwdState[1].squeeze({0})};
    bool useWeightedIn = !params.doInputWeightCalc || opt.preCalcWeights;
    const Tensor *inputWeightsPtr =
        useWeightedIn ? nullptr : &weights.inputWeights;
    if (batchwiseFlags.valid() || slice.interimOut.valid()) {
      LstmState newState;
      LstmInternalState internalState;
      std::tie(newState, internalState) = basicLstmCellForwardPass(
          graph, fwdInput, weights.biases, state, inputWeightsPtr,
          weights.outputWeights, loop, opt, opt.inferenceOnly, time,
          batchwiseFlags, params, {dnai}, cache);
      if (slice.interimOut.valid()) {
        auto fwdIntermediates = getFwdIntermediatesToSave(
            state, newState, internalState, opt, params);
        loop.add(Copy(fwdIntermediates, slice.interimOut, false, {dnai}));
      }
      auto newStateTensor = newState.getAsTensor();
      auto stateTensor = concat(fwdState);
      loop.add(Copy(newStateTensor, stateTensor, false, {dnai}));
    } else {
      basicLstmCellForwardPassInPlace(graph, fwdInput, weights.biases, state,
                                      inputWeightsPtr, weights.outputWeights,
                                      loop, opt, opt.inferenceOnly, params,
                                      {dnai}, cache);
    }
    return loop;
  };
  bool useWeightedIn = !params.doInputWeightCalc || opt.preCalcWeights;

  // make a copy of the activations so that they are sliced efficiently
  auto prevLayerActsCopy =
      createInput(graph, params, {di, "prevLayerActsCopy"}, opt, cache);
  prog.add(Copy(prevLayerActs, prevLayerActsCopy, false, {di}));

  auto input = useWeightedIn ? weightedIn : prevLayerActsCopy;
  auto numIntermediates =
      intermediatesSeq ? getNumFwdIntermediatesToSave(params, opt) : 0;

  // If variable time steps is used, Zero out the output as well as the
  // interemediates for the entire sequence so that the RNN time step loop
  // can be terminated at the maximum time step over the batch.
  rnn::StateSequence stateSequence;
  if (params.outputFullSequence) {
    stateSequence = rnn::StateSequence{
        rnn::createOutputTensor(graph, params.rnn, numShards, {di, "output"}),
        0};
    if (params.rnn.variableTimeSteps()) {
      popops::zero(graph, stateSequence.output, prog, {di, "zeroOutput"});
    }
    prog.add(WriteUndef(stateSequence.output, {di}));
  }
  if (intermediatesSeq) {
    *intermediatesSeq =
        rnn::createOutputTensor(graph, params.rnn, numIntermediates, numShards,
                                {di, "fwdIntermediatesSeq"})
            .reshapePartial(0, 1, {params.rnn.maxTimeSteps, numIntermediates});
    prog.add(WriteUndef(*intermediatesSeq, {di}));
    if (params.rnn.variableTimeSteps()) {
      popops::zero(graph, *intermediatesSeq, prog,
                   {di, "zeroIntermediatesSeq"});
    }
  }
  auto rnnOptions = getRnnOpts(opt);
  auto updatedState =
      rnn::Rnn(graph, params.rnn, false, initState, stateSequence, {input},
               nullptr, intermediatesSeq, {}, {}, prog, shardingLoop, numShards,
               rnnOptions, {di, "rnn"});
  if (finalCellState != nullptr) {
    *finalCellState = updatedState[1].squeeze({0});
  }
  return params.outputFullSequence ? stateSequence.output
                                   : updatedState[0].squeeze({0});
}

std::pair<Tensor, Tensor>
lstmFwd(Graph &graph, const LstmParams &params, const LstmState &fwdStateInit,
        const Tensor &prevLayerActs, const LstmWeights &weights,
        Tensor *intermediatesSeq, program::Sequence &fwdProg,
        const poplar::DebugContext &debugContext, const OptionFlags &options,
        poplin::matmul::PlanningCache *cache) {
  Tensor finalCellState;
  auto output =
      lstmFwd(graph, params, fwdProg, fwdStateInit, weights, prevLayerActs,
              intermediatesSeq, &finalCellState, debugContext, options, cache);
  return {output, finalCellState};
}

Tensor lstmFwd(Graph &graph, const LstmParams &params, program::Sequence &prog,
               const LstmState &fwdStateInit, const LstmWeights &weights,
               const Tensor &prevLayerActs, Tensor *intermediatesSeq,
               const poplar::DebugContext &debugContext,
               const OptionFlags &options,
               poplin::matmul::PlanningCache *cache) {
  return lstmFwd(graph, params, prog, fwdStateInit, weights, prevLayerActs,
                 intermediatesSeq, nullptr, debugContext, options, cache);
}

static Tensor lstmBwdRearrangeWeights(Graph &graph, const LstmParams &params,
                                      const Tensor *weightsInput,
                                      const Tensor &weightsOutput,
                                      Sequence &initProg, const LstmOpts &opt,
                                      const DebugNameAndId &dnai,
                                      matmul::PlanningCache *cache) {
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  std::vector<std::size_t> gradsShape{
      params.rnn.batchSize, params.cellOrder.size() * params.rnn.layerSizes[1]};
  Tensor weightsTransposed;
  if (weightsInput == nullptr) {
    weightsTransposed = flattenUnits(weightsOutput).transpose();
  } else {
    weightsTransposed =
        flattenUnits(concat(*weightsInput, weightsOutput, 1)).transpose();
  }

  weightsTransposed = preArrangeMatMulInputRHS(
      graph, gradsShape, weightsTransposed, initProg, dnai, mmOpt, cache);
  return weightsTransposed;
}

static std::tuple<LstmState, Tensor, Tensor>
backwardStepImpl(Graph &graph, const Tensor *gradNextLayer,
                 const Tensor &fwdIntermediates, const LstmState &stateGrad,
                 bool inputGradSupplied, const Tensor weights, Sequence &prog,
                 const LstmOpts &opt, const LstmParams &params,
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

  poplar::Tensor gradAtOTanhOutput;
  poplar::Tensor gradOutputGate;
  if (isCSNotSupported(params.activation) ||
      isCSNotSupported(params.recurrentActivation)) {
    gradAtOTanhOutput = nonLinearityInputGradient(
        graph, params.activation, actOutputTanh, gradAtOTanhInput, prog,
        {dnai, fPrefix + "/OuputTanh"});
    gradOutputGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actOutputGate, gradAtOutputGateInput,
        prog, {dnai, fPrefix + "/OutputGate"});
  } else {
    auto cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
    gradAtOTanhOutput = nonLinearityInputGradient(
        graph, params.activation, actOutputTanh, gradAtOTanhInput, cs1,
        {dnai, fPrefix + "/OuputTanh"});
    gradOutputGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actOutputGate, gradAtOutputGateInput,
        cs1, {dnai, fPrefix + "/OutputGate"});
    prog.add(Execute(cs1, {dnai}));
  }

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

  poplar::Tensor gradInputGate;
  poplar::Tensor gradCandidate;
  poplar::Tensor gradForgetGate;
  if (isCSNotSupported(params.activation) ||
      isCSNotSupported(params.recurrentActivation)) {
    gradInputGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actInputGate, gradAtInputGateInput,
        prog, {dnai, fPrefix + "/InputGate"});
    gradCandidate = nonLinearityInputGradient(graph, params.activation,
                                              actCandidate, gradAtCandTanhInput,
                                              prog, {dnai, fPrefix + "/Cand"});
    gradForgetGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actForgetGate, gradAtForgetGateInput,
        prog, {dnai, fPrefix + "/Cand"});
  } else {
    auto cs2 = graph.addComputeSet({dnai, fPrefix + "/{Input+Candidate}Gate"});
    gradInputGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actInputGate, gradAtInputGateInput,
        cs2, {dnai, fPrefix + "/InputGate"});
    gradCandidate = nonLinearityInputGradient(graph, params.activation,
                                              actCandidate, gradAtCandTanhInput,
                                              cs2, {dnai, fPrefix + "/Cand"});
    gradForgetGate = nonLinearityInputGradient(
        graph, params.recurrentActivation, actForgetGate, gradAtForgetGateInput,
        cs2, {dnai, fPrefix + "/Cand"});
    prog.add(Execute(cs2, {dnai}));
  }

  const auto gradUnits = [&] {
    std::vector<Tensor> gradUnitsT;
    for (const auto cell : params.cellOrder) {
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
  Tensor gradientIn, gradientPrevStep;
  if (inputGradSupplied) {
    auto outputSize = gradCellState.dim(1);
    auto inputSize = weights.dim(1) - outputSize;
    auto out = matMul(graph, grads, weights, prog,
                      {dnai, fPrefix + "/{Prev + Input}Grad"}, mmOpt, cache);
    out = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer, prog,
                                     {dnai, fPrefix});
    gradientIn = out.slice(0, inputSize, 1);
    gradientPrevStep = out.slice(inputSize, inputSize + outputSize, 1);
  } else {
    gradientPrevStep = matMul(graph, grads, weights, prog,
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
    const LstmState &stateGrad, bool inputGradSupplied, const Tensor &weights,
    Sequence &prog, const LstmOpts &opt, const LstmParams &params,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  return backwardStepImpl(graph, gradNextLayer, fwdIntermediates, stateGrad,
                          inputGradSupplied, weights, prog, opt, params, {dnai},
                          cache);
}

/// Add the partial weight gradients from this timestep to the accumulated
/// weight gradients. Once all the gradients have been accumulated call
/// basicLstmParamUpdateFinal() to do any final accumulation / rearrangement
/// that is required.
static void
basicLstmParamUpdate(Graph &graph, const Tensor &prevLayerActs,
                     const Tensor &prevStepActs, const Tensor &bwdIntermediates,
                     const unsigned stepSize, LstmWeights &weightGrads,
                     Sequence &prog, const LstmOpts &opt,
                     const boost::optional<PlanConstraints> &planConstraints,
                     const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  logging::popnn::debug("basicLstmParamUpdate begin {}", dnai.getPathName());
  const std::string fPrefix = "LstmDeltas";
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            (stepSize > 1) ? "INFERENCE_FWD" : "TRAINING_WU");

  // Configure matmul options with plan constraints
  if (planConstraints) {
    assert(stepSize > 1);
    std::ostringstream constraints;
    boost::property_tree::write_json(constraints, *planConstraints);
    mmOpt.set("planConstraints", constraints.str());

    // The plan constraints were created based on a given memory proportion
    // and are being applied to the following matmul. The memory proportion
    // is relaxed here in order to ensure that the same plan is used.
    mmOpt.set("availableMemoryProportion", std::to_string(1.0));
  }

  auto allWeights = concat(flattenUnits(weightGrads.inputWeights),
                           flattenUnits(weightGrads.outputWeights));
  auto activationsTr =
      concat(prevLayerActs.transpose(), prevStepActs.transpose());
  auto gradients = flattenUnits(bwdIntermediates, stepSize);
  matMulAcc(graph, allWeights, 1.0, activationsTr, gradients, prog,
            {dnai, fPrefix + "/Wi"}, mmOpt, cache);

  // Any casting that might be required to the `weightGrads` type is done
  // by the `reduceWithOutput` function.
  popops::reduceWithOutput(graph,
                           bwdIntermediates.reshapePartial(
                               0, 1, {stepSize, BASIC_LSTM_CELL_NUM_UNITS}),
                           weightGrads.biases, {0},
                           {popops::Operation::ADD, true}, prog,
                           {dnai, fPrefix + "/basicLstmParamUpdate"});
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
  weightAccs.biases =
      graph.clone(options.accumulatorsType,
                  bwdIntermediates.slice(0, BASIC_LSTM_CELL_NUM_UNITS),
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
  const auto totalBwdIntermediates = batchSize * outputSize *
                                     BASIC_LSTM_CELL_NUM_UNITS *
                                     params.rnn.maxTimeSteps;
  logging::popnn::debug("interleavedWUIsBeneficial totalTransposeParams={} "
                        "totalBwdIntermediates={}",
                        totalTransposeParams, totalBwdIntermediates);
  return totalTransposeParams <= totalBwdIntermediates;
}

// Return time step interval for weight update. Return zero if no suitable
// interval was found.
static std::pair<std::size_t, boost::optional<PlanConstraints>>
interleaveWUCadence(const Graph &graph, const LstmParams &params,
                    const LstmOpts &options,
                    poplin::matmul::PlanningCache *cache) {
  boost::optional<PlanConstraints> planConstraints{boost::none};
  auto target = graph.getTarget();
  const auto batchSize = params.rnn.batchSize;
  const auto inputSize = params.rnn.layerSizes[0];
  const auto outputSize = params.rnn.layerSizes[1];
  const auto dataType = params.rnn.dataType;
  const auto accumType = options.accumulatorsType;
  const auto bwdIntermPerStep =
      params.rnn.getOutputBytesPerTile(graph) * BASIC_LSTM_CELL_NUM_UNITS;

  unsigned stepsPerWU = 0;
  if (options.disableWUPartialInterleaving) {
    // Weight Update Partial Interleaving could be optionally disabled by the
    // user if legacy behaviour is required. In this case if interleaving is
    // found to be preferable, the weight update interval size is limited to 1.
    stepsPerWU = interleavedWUIsBeneficial(params) ? 1 : 0;
    return std::make_pair(stepsPerWU, planConstraints);
  }

  // User override if required.
  if (options.rnnStepsPerWU) {
    stepsPerWU = *options.rnnStepsPerWU;
    logging::popnn::debug("interleaveWUCadence stepsPerWUoverride={}",
                          stepsPerWU);
    return std::make_pair(stepsPerWU, planConstraints);
  }

  // Available tile memory
  auto tileMemory = target.getBytesPerTile();
  constexpr double defaultAvailMemoryProp = 0.2;
  double memoryPropLstm =
      options.availableMemoryProportion.value_or(defaultAvailMemoryProp);
  auto availableMemory = static_cast<unsigned>(tileMemory * memoryPropLstm);

  // Limit the BWD intermediates to use not more than 10% of available memory.
  auto bwdIntermMemory = static_cast<unsigned>(0.1 * availableMemory);
  auto maxSteps =
      std::min(params.rnn.maxTimeSteps, bwdIntermMemory / bwdIntermPerStep);

  if (maxSteps == 0) {
    // If Bwd intermediates takes too much memory choose between interleaving
    // and non-interleaving schemes.
    stepsPerWU = interleavedWUIsBeneficial(params) ? 1 : 0;
    return std::make_pair(stepsPerWU, planConstraints);
  }

  // Reduce available tile memory to account for Backward intermediates
  auto bwdInterm = bwdIntermPerStep * maxSteps;
  availableMemory -= bwdInterm;
  double memoryPropMatMul = 1.0 * availableMemory / tileMemory;

  OptionFlags mmOpt = {
      {"partialsType", options.partialsType.toString()},
  };
  mmOpt.set("availableMemoryProportion", std::to_string(memoryPropMatMul));
  mmOpt.set("fullyConnectedPass", "INFERENCE_FWD");

  const auto dataPathWidth =
      target.getDataPathWidth() / (target.getTypeSize(dataType) * 8);

  // Round the time steps to the data type size in order to minimise
  // rearrangement costs.
  auto numSteps = roundDown(maxSteps, dataPathWidth);

  // Get planConstraints
  planConstraints.emplace(poplin::matMulPlanConstraints(
      graph, dataType, accumType,
      {inputSize + outputSize, numSteps * batchSize},
      {numSteps * batchSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize}, mmOpt,
      cache));

  // Get inner dimension serialisation for the matmul.
  auto serialSplits = getMatMulSerialSplits(*planConstraints);
  unsigned inputChanSerialSplit = std::get<3>(serialSplits);

  if (inputChanSerialSplit < numSteps) {
    // The Weight Update time steps are along the inner dimension of the matmul.
    // Use the obtained plan to execute one weight update per input channel
    // serial split. Therefore modify the inChanSplit.serial constraint to 1.
    planConstraints->put<unsigned>("0.partition.inChanSplit.serial", 1);

    // The accumulating dimension of matrix multiplication is the product
    // of number of weight update steps and batchSize. The weight update steps
    // are estimated after removing the batchSize factor.
    stepsPerWU = numSteps / inputChanSerialSplit;

    // Round the steps per WU to a multiple of the data type size.
    stepsPerWU = roundDown(stepsPerWU, dataPathWidth);
  } else {
    planConstraints = boost::none;
    stepsPerWU = 1;
  }

  logging::popnn::debug(
      "interleaveWUCadence maxStepsPerWU={} inputChanSerialSplit={} "
      "stepsPerWU={} memProp=(lstm={}, matmul={}) ",
      maxSteps, inputChanSerialSplit, stepsPerWU, memoryPropLstm,
      memoryPropMatMul);

  return std::make_pair(stepsPerWU, planConstraints);
}

static Tensor recomputeCellAndTanhImpl(Graph &graph, const LstmParams &params,
                                       const LstmOpts &options,
                                       const LstmState &fwdStateInit,
                                       const Tensor &fwdIntermediatesSeq,
                                       program::Sequence &prog,
                                       const DebugNameAndId &dnai) {
  auto shardingLoop = [&params,
                       &options](Graph &graph, const rnn::TimeStepState &time,
                                 const rnn::RnnBatchwiseFlags &batchwiseFlags,
                                 std::vector<Tensor> &shardState,
                                 const rnn::RnnSlice &slice,
                                 std::vector<Tensor> &created,
                                 program::Sequence *initProg,
                                 const DebugNameAndId &dnai) {
    auto loop = Sequence{{}, {dnai}};
    auto prevCellState = shardState[0].squeeze({0});
    auto forgetGate = getSavedFwdIntermediate(
        slice.interimIn, params, options, LSTM_FWD_INTERMEDIATE_FORGET_GATE);
    auto candidate = getSavedFwdIntermediate(slice.interimIn, params, options,
                                             LSTM_FWD_INTERMEDIATE_CAND_TANH);
    auto outputGate = getSavedFwdIntermediate(
        slice.interimIn, params, options, LSTM_FWD_INTERMEDIATE_OUTPUT_GATE);
    auto inputGate = getSavedFwdIntermediate(slice.interimIn, params, options,
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
          popnn::nonLinearity(graph, params.activation, newCellState, loop,
                              {dnai, "TanhCellState"});
    }

    loop.add(Copy(concat(newTanhOutput.expand({0}), prevCellState.expand({0})),
                  slice.outputs[0], false, {dnai}));
    loop.add(Copy(newCellState, prevCellState, false, {dnai}));
    return loop;
  };

  auto numShards = getNumShards(graph, params, options, {dnai, "numShards"});
  std::size_t numToRecompute =
      LSTM_FWD_INTERMEDIATE_OUTPUT - LSTM_FWD_INTERMEDIATE_OUTPUT_TANH;
  std::vector<Tensor> recomputedIntermediatesSeq{
      rnn::createOutputTensor(graph, params.rnn, numToRecompute, numShards,
                              {dnai, "recomputedIntermediates"})};
  for (const auto &t : recomputedIntermediatesSeq) {
    prog.add(WriteUndef(t));
  }
  std::vector<Tensor> initState = {fwdStateInit.cellState.expand({0})};
  auto rnnOptions = getRnnOpts(options);
  rnn::Rnn(graph, params.rnn, false, initState, {}, {}, &fwdIntermediatesSeq,
           nullptr, recomputedIntermediatesSeq, {}, prog, shardingLoop,
           numShards, rnnOptions, {dnai, "rnn"});
  return recomputedIntermediatesSeq[0].reshapePartial(
      0, 1, {params.rnn.maxTimeSteps, numToRecompute});
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
            const boost::optional<unsigned> &stepsPerWU,
            const boost::optional<PlanConstraints> &wuPlanConstraints,
            const DebugNameAndId &dnai, const LstmOpts &options,
            poplin::matmul::PlanningCache *cache) {
  auto numShards = getNumShards(graph, params, options, {dnai, "numShards"});
  auto weightsRearranged = lstmBwdRearrangeWeights(
      graph, params, inputGradSeq ? &weights.inputWeights : nullptr,
      weights.outputWeights, prog, options, {dnai, "/PreArrangeWeights"},
      cache);
  auto loopBwdWithWU = [&params, &options, &inputGradSeq, &cache,
                        &weightsRearranged, &stepsPerWU](
                           LstmWeights &weights, LstmWeights *weightsGrad,
                           Graph &graph, const rnn::TimeStepState &time,
                           const rnn::RnnBatchwiseFlags &batchwiseFlags,
                           std::vector<Tensor> &shardState,
                           const rnn::RnnSlice &slice,
                           std::vector<Tensor> &created,
                           program::Sequence *initProg,
                           const DebugNameAndId &dnai) {
    const auto &fwdIntermediates = slice.interimIn;
    const Tensor *gradLayerNextThisStepPtr =
        slice.inputs[0].valid() ? &slice.inputs[0] : nullptr;
    Tensor inputGrad =
        shardState[0].valid() ? shardState[0].squeeze({0}) : Tensor{};
    Tensor bwdIntermediates;
    auto backwardStep = [&](const LstmState &stateGrads,
                            program::Sequence &prog) {
      LstmState nextStepGrads;
      if (inputGradSeq) {
        Tensor nextInputGrad;
        std::tie(nextStepGrads, nextInputGrad, bwdIntermediates) =
            popnn::lstm::basicLstmBackwardStep(
                graph, gradLayerNextThisStepPtr, fwdIntermediates, stateGrads,
                true, weightsRearranged, prog, options, params, {dnai}, cache);
        if (inputGrad.valid()) {
          prog.add(Copy(nextInputGrad, inputGrad, false, {dnai}));
        }
      } else {
        std::tie(nextStepGrads, std::ignore, bwdIntermediates) =
            basicLstmBackwardStep(
                graph, gradLayerNextThisStepPtr, fwdIntermediates, stateGrads,
                false, weightsRearranged, prog, options, params, {dnai}, cache);
      }
      if (slice.interimOut.valid()) {
        prog.add(Copy(bwdIntermediates, slice.interimOut, false, {dnai}));
      }
      return nextStepGrads;
    };

    Sequence prog;
    LstmState stateGrads = {shardState[1].squeeze({0}),
                            shardState[2].squeeze({0})};
    auto newStateGrads = backwardStep(stateGrads, prog);

    auto updateStateGrad = Copy(newStateGrads.getAsTensor(),
                                stateGrads.getAsTensor(), false, {dnai});
    if (!params.outputFullSequence && batchwiseFlags.valid()) {
      // update output gradient state if the batchwise time steps is within
      // the specified range for that batch. Do not update the state if the
      // time steps exceeds the range.
      Sequence updateStateGradBatch;
      auto mask = batchwiseFlags.mask.expand({0, 1});
      auto maskInv = batchwiseFlags.inverse.expand({0, 1});
      auto newStateGradsTensor = newStateGrads.getAsTensor();
      auto stateGradsTensor = stateGrads.getAsTensor();
      mapInPlace(graph, expr::_3 * expr::_2 + expr::_4 * expr::_1,
                 {stateGradsTensor, newStateGradsTensor, mask, maskInv},
                 updateStateGradBatch, {dnai, "selectState"});
      prog.add(If(time.variableSeqFlag, updateStateGradBatch, updateStateGrad));
    } else {
      prog.add(updateStateGrad);
    }
    return prog;
  };
  auto updateWU = [&params, &options, &wuPlanConstraints, &stepsPerWU,
                   &cache](LstmWeights &weights, LstmWeights *weightsGrad,
                           Graph &graph, const rnn::RnnSlice &slice,
                           unsigned stepsPerGather, program::Sequence *initProg,
                           const DebugNameAndId &dnai) {
    auto update = Sequence{{}, {dnai}};
    auto &prevLayerOut = slice.inputs[0];
    auto &prevStepOut = slice.inputs[1];
    auto &bwdIntermediates = slice.interimIn;
    if (initProg != nullptr) {
      *weightsGrad = createWeightAccumulators(graph, weights, bwdIntermediates,
                                              options, {dnai, "weightsGrad"});
      zeroWeightAccumulators(graph, *initProg, *weightsGrad, options,
                             {dnai, "zeroWeightAcc"});
    }

    // The constraints obtained from the planning of the weight update step
    // interval cadence is applicable in general except potentially
    // for the remainder if it involves fewer weight update time steps.
    boost::optional<PlanConstraints> matmulPlanConstraints(boost::none);
    if ((stepsPerGather > 1) && (*stepsPerWU == stepsPerGather) &&
        wuPlanConstraints) {
      matmulPlanConstraints = wuPlanConstraints;
    }
    basicLstmParamUpdate(graph, prevLayerOut, prevStepOut, bwdIntermediates,
                         stepsPerGather, *weightsGrad, update, options,
                         matmulPlanConstraints, {dnai, "basicLstmParamUpdate"},
                         cache);
    return update;
  };

  Tensor recomputedIntermediatesSeq = recomputeFwdIntermediates(
      graph, fwdStateInit, fwdIntermediatesSeq, params, options, prog,
      {dnai, "recomputeFwdIntermediates"});

  // If variable time steps is used, The `fwdIntermediates` are expected to be
  // zeros for time steps beyond the time step value for each batch element.
  // In addition zero out `inputGradSeq` for the entire sequence so that the
  // RNN time step loop can be terminated at the maximum time step over the
  // batch.
  Tensor fwdIntermediates = reconstructIntermediatesFromRecomputed(
      fwdIntermediatesSeq, recomputedIntermediatesSeq, params, options);
  auto lastOutGradInit = rnn::createInitialState(
      graph, params.rnn, true, 1, numShards, {dnai, "lastOutGradInit"});
  if (params.outputFullSequence) {
    zero(graph, lastOutGradInit, prog, {dnai, "zeroLastOutGrad"});
  } else {
    prog.add(Copy(gradLayerNext, lastOutGradInit, false, {dnai}));
  }
  auto lastCellStateGradInit = rnn::createInitialState(
      graph, params.rnn, true, 1, numShards, {dnai, "lastCellStateGradInit"});
  if (lastCellStateGradPtr) {
    prog.add(Copy(*lastCellStateGradPtr, lastCellStateGradInit, false,
                  {dnai, "initLastOutGrad"}));
  } else {
    zero(graph, lastCellStateGradInit, prog, {dnai, "initCellStateGrad"});
  }
  Tensor gradLayerNextRearranged;
  if (params.outputFullSequence) {
    gradLayerNextRearranged = rnn::createOutputTensor(
        graph, params.rnn, numShards, {dnai, "gradLayerNextRearranged"});
    prog.add(Copy(gradLayerNext, gradLayerNextRearranged, false,
                  {dnai, "initGradLayerNextRearranged"}));
  }
  Tensor inputGradInit;
  rnn::StateSequence inputGrad;
  if (inputGradSeq) {
    inputGradInit = rnn::createInitialState(graph, params.rnn, false, 1,
                                            numShards, {dnai, "inputGradInit"});
    *inputGradSeq = rnn::createInputTensor(graph, params.rnn, numShards,
                                           {dnai, "inputGrad"});
    if (params.rnn.variableTimeSteps()) {
      zero(graph, *inputGradSeq, prog, {dnai, "zeroInputGrad"});
    }
    inputGrad = rnn::StateSequence{*inputGradSeq, 0};
    prog.add(WriteUndef(inputGradInit, {dnai}));
    prog.add(WriteUndef(*inputGradSeq, {dnai}));
  }
  std::vector<Tensor> bwdStateInit = {inputGradInit, lastOutGradInit,
                                      lastCellStateGradInit};
  if (bwdIntermediatesPtr) {
    *bwdIntermediatesPtr =
        rnn::createOutputTensor(graph, params.rnn, BASIC_LSTM_CELL_NUM_UNITS,
                                numShards, {dnai, "bwdIntermediates"})
            .reshapePartial(
                0, 1, {params.rnn.maxTimeSteps, BASIC_LSTM_CELL_NUM_UNITS});
    prog.add(WriteUndef(*bwdIntermediatesPtr, {dnai}));
  }
  Tensor prevLayerOut, prevStepOut;
  if (weightsGrad) {
    // make a copy of the activations so that they are sliced efficiently
    prevLayerOut =
        createInputTensor(graph, params.rnn, numShards, {dnai, "prevLayerOut"});
    prog.add(Copy(fwdInputSeq, prevLayerOut, false, {dnai}));
    auto fwdOut =
        (params.outputFullSequence)
            ? fwdOutput
            : fwdIntermediates.dimRoll(1)[LSTM_FWD_INTERMEDIATE_OUTPUT];
    prevStepOut =
        rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdStateInit.output,
                            prog, numShards, {dnai, "fwdOutShifted"});
  }
  using namespace std::placeholders;
  const auto shardingLoop =
      std::bind(loopBwdWithWU, weights, weightsGrad, ph::_1, ph::_2, ph::_3,
                ph::_4, ph::_5, ph::_6, ph::_7, ph::_8);
  auto rnnOptions = getRnnOpts(options);
  std::vector<Tensor> bwdInputs = {gradLayerNextRearranged};
  std::vector<Tensor> updatedState;
  if (weightsGrad) {
    assert(stepsPerWU);
    const auto shardingUpdate = std::bind(
        updateWU, weights, weightsGrad, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5);
    std::vector<Tensor> wuInputs = {prevLayerOut, prevStepOut};
    updatedState = rnn::Rnn(
        graph, params.rnn, bwdStateInit, inputGrad, bwdInputs, fwdIntermediates,
        BASIC_LSTM_CELL_NUM_UNITS, prog, shardingLoop, wuInputs, shardingUpdate,
        numShards, *stepsPerWU, rnnOptions, {dnai, "updatedState"});
  } else {
    updatedState =
        rnn::Rnn(graph, params.rnn, true, bwdStateInit, inputGrad, bwdInputs,
                 &fwdIntermediates, bwdIntermediatesPtr, {}, {}, prog,
                 shardingLoop, numShards, rnnOptions, {dnai, "updatedState"});
  }
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
  POPNN_TRACEPOINT();
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

  LstmState outputs =
      lstmBwdImpl(graph, params, prog, fwdStateInit, fwdIntermediatesSeq,
                  weights, fwdInputSeq, fwdOutput, gradLayerNext,
                  lastCellStateGradPtr, inputGrad, bwdIntermediates, nullptr,
                  {}, {}, {di}, std::move(options), planningCache);
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
  auto loopWU = [&options, &planningCache](
                    LstmWeights &weightGrads, Graph &graph,
                    const rnn::TimeStepState &time,
                    const rnn::RnnBatchwiseFlags &batchwiseFlags,
                    std::vector<Tensor> &shardState, const rnn::RnnSlice &slice,
                    std::vector<Tensor> &created, program::Sequence *initProg,
                    const DebugNameAndId &dnai) {
    auto loop = Sequence{{}, {dnai}};
    auto &prevLayerOut = slice.inputs[0];
    auto &prevStepOut = slice.inputs[1];
    auto &bwdIntermediates = slice.inputs[2];
    basicLstmParamUpdate(graph, prevLayerOut, prevStepOut, bwdIntermediates, 1,
                         weightGrads, loop, options, {}, {dnai}, planningCache);
    return loop;
  };

  // make a copy of the activations so that they are sliced efficiently
  auto numShards = getNumShards(graph, params, options, {dnai, "numShards"});
  Tensor inputCopy =
      createInputTensor(graph, params.rnn, numShards, {dnai, "inputCopy"});
  prog.add(Copy(input, inputCopy, false, {dnai}));
  auto fwdOut =
      (params.outputFullSequence)
          ? output
          : fwdIntermediatesSeq.dimRoll(1)[LSTM_FWD_INTERMEDIATE_OUTPUT];
  Tensor prevStepOut =
      rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdStateInit.output, prog,
                          numShards, {dnai, "fwdOutshifted"});

  std::vector<Tensor> wuInputs = {inputCopy, prevStepOut, bwdIntermediatesSeq};
  using namespace std::placeholders;
  const auto shardingLoop =
      std::bind(loopWU, weightGrads, ph::_1, ph::_2, ph::_3, ph::_4, ph::_5,
                ph::_6, ph::_7, ph::_8);
  auto rnnOptions = getRnnOpts(options);
  auto updatedState =
      rnn::Rnn(graph, params.rnn, true, {}, {}, wuInputs, nullptr, nullptr, {},
               {}, prog, shardingLoop, numShards, rnnOptions, {dnai, "rnn"});
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
  POPNN_TRACEPOINT();
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
  POPNN_TRACEPOINT();
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

  auto [wuCadence, wuPlanConstraints] =
      interleaveWUCadence(graph, params, options, planningCache);
  bool minimiseMemoryUsage = (wuCadence == 0) ? true : false;

  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  LstmState stateGrads =
      lstmBwdImpl(graph, params, prog, fwdStateInit, fwdIntermediates, weights,
                  input, output, outputGrad, lastCellStateGrad, inputGrad,
                  minimiseMemoryUsage ? &bwdIntermediates : nullptr,
                  minimiseMemoryUsage ? nullptr : &weightsGrad_, wuCadence,
                  wuPlanConstraints, {di}, options, planningCache);

  if (minimiseMemoryUsage) {
    weightsGrad_ = lstmWUImpl(
        graph, params, prog, fwdStateInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, {di}, std::move(options), planningCache);
  }

  di.addOutputs(DI_ARGS(stateGrads));
  return stateGrads;
}

uint64_t getBasicLstmCellFwdFlops(const LstmParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.maxTimeSteps;
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
  auto sequenceSize = params.rnn.maxTimeSteps;
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
  auto sequenceSize = params.rnn.maxTimeSteps;
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
