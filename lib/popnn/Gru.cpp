// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poplibs_support/Algorithm.hpp>
#include <popnn/Gru.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Cast.hpp>

#include "RnnUtil.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/DebugInfo.hpp"

using namespace poplar;
using namespace poplar::program;

using namespace poplibs_support;
using namespace poplin;
using namespace popnn::Rnn;
using namespace popops;
using namespace popops::expr;
using namespace poputil;

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const BasicGruCellUnit &t) {
  switch (t) {
  case BASIC_GRU_CELL_RESET_GATE:
    return poplar::ProfileValue("BASIC_GRU_CELL_RESET_GATE");
  case BASIC_GRU_CELL_UPDATE_GATE:
    return poplar::ProfileValue("BASIC_GRU_CELL_UPDATE_GATE");
  case BASIC_GRU_CELL_CANDIDATE:
    return poplar::ProfileValue("BASIC_GRU_CELL_CANDIDATE");
  case BASIC_GRU_CELL_NUM_UNITS:
    return poplar::ProfileValue("BASIC_GRU_CELL_NUM_UNITS");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}

template <>
poplar::ProfileValue toProfileValue(const popnn::gru::GruWeights &t) {
  poplar::ProfileValue::Map v;
  v.insert({"inputWeights", toProfileValue(t.inputWeights)});
  v.insert({"outputWeights", toProfileValue(t.outputWeights)});
  v.insert({"biases", toProfileValue(t.biases)});
  return v;
}

template <>
poplar::ProfileValue toProfileValue(const popnn::gru::GruParams &t) {
  poplar::ProfileValue::Map v;
  v.insert({"rnn", toProfileValue(t.rnn)});
  v.insert({"outputFullSequence", toProfileValue(t.outputFullSequence)});
  v.insert({"calcInputGradients", toProfileValue(t.calcInputGradients)});

  v.insert({"cellOrder", toProfileValue(t.cellOrder)});
  // std::vector<BasicGruCellUnit> cellOrder = getDefaultBasicGruCellOrder();
  v.insert({"resetAfter", toProfileValue(t.resetAfter)});
  return v;
}
} // namespace poputil

// Utility macro to print out tensor
// debug_tensor is only defined when "DEBUG_TENSOR" is defined

//#define DEBUG_TENSOR

#ifdef DEBUG_TENSOR
#define debug_tensor(prog, msg, tensor) prog.add(PrintTensor(msg, tensor))
#else
#define debug_tensor(prog, msg, tensor)
#endif

// Tensor elements maintained in forward state. The number of elements is a
// function of the amount of recomputation done in the backward pass
enum FwdIntermediates {
  // Saved unless doing full recomputation
  GRU_FWD_INTERMEDIATE_RESET_GATE,
  GRU_FWD_INTERMEDIATE_UPDATE_GATE,
  GRU_FWD_INTERMEDIATE_CANDIDATE,

  // Saved if `outputFullSequence` is not set i.e. outputs aren't already
  // saved as part of the forward pass output.
  GRU_FWD_INTERMEDIATE_OUTPUT,

  // h_prev x candidate recurrant weights + candidate recurrant bias
  // Used to calculate reset gate gradient when resetAfter=true
  GRU_FWD_INTERMEDIATE_CANDIDATE_RECURRANT
};

struct GruInternalState {
  Tensor resetGate;
  Tensor updateGate;
  Tensor candidate;

  // h_prev x candidate recurrant weights + candidate recurrant bias
  // Used to calculate reset gate gradient when resetAfter=true
  Tensor candidateRecurrant;
};

namespace { // Anonymous namespace
bool isCSNotSupported(popnn::NonLinearityType nl) {
  return (nl == popnn::NonLinearityType::SOFTMAX ||
          nl == popnn::NonLinearityType::SOFTMAX_STABLE ||
          nl == popnn::NonLinearityType::SOFTMAX_SCALED ||
          nl == popnn::NonLinearityType::HARD_SIGMOID);
}
} // Anonymous namespace

// Computes the output before nonlinearities to all the units are applied
static Tensor basicGruUnitsNlInput(Graph &graph, Tensor prevAct,
                                   Tensor prevOutput, size_t num_unit,
                                   Tensor weightsInput, Tensor weightsOutput,
                                   Sequence &prog, OptionFlags &mmOpt,
                                   matmul::PlanningCache *cache,
                                   const DebugNameAndId &dnai) {
  assert(weightsInput.dim(0) == num_unit);
  assert(weightsOutput.dim(0) == num_unit);
  auto weights = concat(weightsInput, weightsOutput, 1);
  return unflattenUnits(matMul(graph, concat(prevAct, prevOutput, 1),
                               flattenUnits(weights), prog, {dnai, "Weight"},
                               mmOpt, cache),
                        num_unit);
}

namespace popnn {
namespace gru {

GruParams::GruParams(poplar::Type dataType, std::size_t batchSize,
                     std::size_t timeSteps, std::vector<std::size_t> layerSizes,
                     NonLinearityType activation,
                     NonLinearityType recurrentActivation)
    : rnn(dataType, batchSize, timeSteps, layerSizes), dataType(dataType),
      batchSize(batchSize), timeSteps(timeSteps), layerSizes(layerSizes),
      activation(activation), recurrentActivation(recurrentActivation) {}

GruParams::GruParams(const GruParams &other) = default;

struct GruOpts {
  bool inferenceOnly;
  poplar::Type partialsType;
  boost::optional<double> availableMemoryProportion;
  boost::optional<std::size_t> numShards;
  boost::optional<bool> rnnCodeReuse;
};

std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                    {"float", poplar::FLOAT}};

static OptionFlags getMMOpts(const GruOpts &gruOpts) {
  OptionFlags mmOpts = {
      {"partialsType", gruOpts.partialsType.toString()},
  };
  if (gruOpts.availableMemoryProportion) {
    mmOpts.set("availableMemoryProportion",
               std::to_string(gruOpts.availableMemoryProportion.get()));
  }
  return mmOpts;
}

static OptionFlags getRnnOpts(const GruOpts &gruOpts) {
  OptionFlags rnnOpts;
  if (gruOpts.rnnCodeReuse) {
    rnnOpts.set("codeReuse", std::to_string(gruOpts.rnnCodeReuse.get()));
  }
  return rnnOpts;
}

static GruOpts parseOptions(const OptionFlags &options) {
  GruOpts gruOpts;
  gruOpts.inferenceOnly = true;
  gruOpts.partialsType = poplar::FLOAT;
  gruOpts.numShards = boost::none;
  gruOpts.rnnCodeReuse = boost::none;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec gruSpec{
      {"inferenceOnly", OptionHandler::createWithBool(gruOpts.inferenceOnly)},
      {"partialsType",
       OptionHandler::createWithEnum(gruOpts.partialsType, partialsTypeMap)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(gruOpts.availableMemoryProportion)},
      {"numShards", OptionHandler::createWithInteger(gruOpts.numShards)},
      {"rnnCodeReuse", OptionHandler::createWithBool(gruOpts.rnnCodeReuse)},
  };
  for (const auto &entry : options) {
    gruSpec.parse(entry.first, entry.second);
  }
  return gruOpts;
}

static void validateParams(const GruParams &params) {
  if (params.rnn.layerSizes.size() != 2) {
    throw poplibs_error("Invalid GRU params (layerSize != 2)");
  }
}

const std::vector<BasicGruCellUnit> getDefaultBasicGruCellOrder() {
  return {BASIC_GRU_CELL_RESET_GATE, BASIC_GRU_CELL_UPDATE_GATE,
          BASIC_GRU_CELL_CANDIDATE};
}

static unsigned getNumFwdIntermediatesToSave(const GruParams &params) {
  int numberToConcat = 3;
  if (!params.outputFullSequence)
    numberToConcat += 1;
  if (params.resetAfter)
    numberToConcat += 1;
  return numberToConcat;
}

static Tensor getFwdIntermediatesToSave(const Tensor &newOutput,
                                        const GruInternalState &internalState,
                                        const GruParams &params) {
  std::vector<Tensor> intermediatesToConcat;
  auto numberToConcat = getNumFwdIntermediatesToSave(params);
  intermediatesToConcat.reserve(numberToConcat);

  intermediatesToConcat.push_back(internalState.resetGate.expand({0}));
  intermediatesToConcat.push_back(internalState.updateGate.expand({0}));
  intermediatesToConcat.push_back(internalState.candidate.expand({0}));
  if (!params.outputFullSequence) {
    intermediatesToConcat.push_back(newOutput.expand({0}));
  }
  if (params.resetAfter) {
    intermediatesToConcat.push_back(
        internalState.candidateRecurrant.expand({0}));
  }
  Tensor intermediates = concat(intermediatesToConcat);
  return intermediates;
}

// Sharding is relevant for LSTM/GRU models which use significantly fewer
// tiles for storage of sequences than are available on the target. The total
// memory required to store the input and output dimensions is directly
// proportional to the GRU sequence size. For large sequence sizes the tiles
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
// of shards. However this approach does not work well for smaller sized GRU
// models. For very small input and output layer sizes and small batch sizes
// the maximum number of shards could run into the hundreds or thousands.
//
// To limit sharding when batch/feature dimensions are small, we allow operands
// to occupy up to 10% of total tile memory before sharding further. Layers
// with reasonably large batch/feature dimensions typically utilise enough tiles
// that the maximum shards calculated is small even if memory usage per-tile for
// operands is high. Hence this only really applies to the small cases.
//
// All GRU passes - Fwd, Bwd & WU passes - must use the same number of shards.
// Hence, operand memory is calculated based on the Fwd pass since it can
// be used as a reasonable approximation for all the passes.
static std::size_t getNumShards(const Graph &graph, const GruParams &params,
                                const GruOpts &opt,
                                const DebugNameAndId &dnai) {
  auto target = graph.getTarget();
  auto tileMemory = target.getBytesPerTile();
  auto maxShards = params.rnn.getMaxShards(graph);
  auto inputSize = params.rnn.getInputBytesPerTile(graph);
  auto outputSize = params.rnn.getOutputBytesPerTile(graph);
  auto numIntermediates = getNumFwdIntermediatesToSave(params);
  auto operandSingleIteration =
      inputSize + (outputSize * (1 + numIntermediates));
  auto operandSize = operandSingleIteration * params.rnn.timeSteps;

  // Fraction of total tile memory that is nominally designated for operands
  double operandFraction = 0.1;

  double availableOperandMemory = tileMemory * operandFraction;
  std::size_t estShards = std::ceil(operandSize / availableOperandMemory);
  auto numShards = std::min(estShards, maxShards);
  if (opt.numShards) {
    if ((*opt.numShards < 1) || (*opt.numShards > maxShards)) {
      throw poputil::poplibs_error("GRU numShards must be within "
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

Tensor createInput(Graph &graph, const GruParams &params,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options,
                   poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, options, planningCache));
  const auto opt = parseOptions(options);
  validateParams(params);
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto output =
      rnn::createInputTensor(graph, params.rnn, numShards, {di, "input"});
  di.addOutput(output);
  return output;
}

Tensor createInitialState(Graph &graph, const GruParams &params,
                          const poplar::DebugContext &debugContext,
                          const OptionFlags &options,
                          matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));
  const auto opt = parseOptions(options);
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto output = rnn::createInitialState(graph, params.rnn, true, 1, numShards,
                                        {di, "initialOutput"})
                    .squeeze({0});
  di.addOutput(output);
  return output;
}

void zeroInitialState(Graph &graph, const Tensor &init_output, Sequence &prog,
                      const DebugNameAndId &dnai) {
  zero(graph, init_output, prog, {dnai});
}

// the outermost dimension of the weights and bias tensors is the gate that
// those weights are associated with.
constexpr auto gateDim = 0u;

// rearrange the gate dimension to reflect the order provided by the user.
static poplar::Tensor
toCellOrder(poplar::Tensor tensor,
            const std::vector<BasicGruCellUnit> &cellOrder) {
  assert(tensor.shape().at(0) == BASIC_GRU_CELL_NUM_UNITS);

  std::vector<poplar::Tensor> rearranged;
  rearranged.reserve(BASIC_GRU_CELL_NUM_UNITS);

  for (const auto &gate : cellOrder) {
    const auto idx = static_cast<std::size_t>(gate);
    rearranged.push_back(tensor.slice(idx, idx + 1, gateDim));
  }

  return concat(std::move(rearranged));
}

static GruWeights toCellOrder(GruWeights weights,
                              const std::vector<BasicGruCellUnit> &cellOrder) {
  weights.inputWeights =
      toCellOrder(std::move(weights.inputWeights), cellOrder);
  weights.outputWeights =
      toCellOrder(std::move(weights.outputWeights), cellOrder);
  weights.biases = toCellOrder(std::move(weights.biases), cellOrder);

  return weights;
}

// rearrange the gate dimension back from the user configured order to the
// internal order (which is the order that the gates appear in the enum).
static poplar::Tensor
fromCellOrder(poplar::Tensor tensor,
              const std::vector<BasicGruCellUnit> &cellOrder) {
  assert(tensor.shape().at(0) == BASIC_GRU_CELL_NUM_UNITS);

  std::vector<poplar::Tensor> rearranged;
  rearranged.resize(BASIC_GRU_CELL_NUM_UNITS);

  for (unsigned i = 0; i < rearranged.size(); ++i) {
    const auto idx = static_cast<std::size_t>(cellOrder.at(i));
    rearranged[idx] = tensor.slice(i, i + 1, gateDim);
  }

  return concat(std::move(rearranged));
}

static GruWeights
fromCellOrder(GruWeights weights,
              const std::vector<BasicGruCellUnit> &cellOrder) {
  weights.inputWeights =
      fromCellOrder(std::move(weights.inputWeights), cellOrder);
  weights.outputWeights =
      fromCellOrder(std::move(weights.outputWeights), cellOrder);
  weights.biases = fromCellOrder(std::move(weights.biases), cellOrder);

  return weights;
}

std::pair<poplar::Tensor, poplar::Tensor>
createWeightsKernel(poplar::Graph &graph, const GruParams &params,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options,
                    poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  validateParams(params);
  auto opt = parseOptions(options);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];

  poplar::Tensor inputWeights, outputWeights;

  if (params.resetAfter) {
    auto weights_in = createMatMulInputRHS(
        graph, params.rnn.dataType, {params.rnn.batchSize, inputSize},
        {inputSize, BASIC_GRU_CELL_NUM_UNITS * outputSize}, {di, "weights"},
        mmOpt, cache);
    auto weights_out = createMatMulInputRHS(
        graph, params.rnn.dataType, {params.rnn.batchSize, outputSize},
        {outputSize, BASIC_GRU_CELL_NUM_UNITS * outputSize}, {di, "weights"},
        mmOpt, cache);

    inputWeights = unflattenUnits(weights_in, BASIC_GRU_CELL_NUM_UNITS);
    outputWeights = unflattenUnits(weights_out, BASIC_GRU_CELL_NUM_UNITS);
  } else {
    auto weights_ru =
        createMatMulInputRHS(graph, params.rnn.dataType,
                             {params.rnn.batchSize, inputSize + outputSize},
                             {inputSize + outputSize, 2 * outputSize},
                             {di, "weights"}, mmOpt, cache);
    poplar::Tensor inputWeights_ru =
        unflattenUnits(weights_ru.slice(0, inputSize), 2);
    poplar::Tensor outputWeights_ru =
        unflattenUnits(weights_ru.slice(inputSize, inputSize + outputSize), 2);

    auto weights_c = createMatMulInputRHS(
        graph, params.rnn.dataType,
        {params.rnn.batchSize, inputSize + outputSize},
        {inputSize + outputSize, outputSize}, {di, "weights"}, mmOpt, cache);
    poplar::Tensor inputWeights_c =
        unflattenUnits(weights_c.slice(0, inputSize), 1);
    poplar::Tensor outputWeights_c =
        unflattenUnits(weights_c.slice(inputSize, inputSize + outputSize), 1);

    inputWeights = concat(inputWeights_ru, inputWeights_c);
    outputWeights = concat(outputWeights_ru, outputWeights_c);
  }
  // rearrange the outermost dimension according to the cellOrder parameter.
  std::pair<poplar::Tensor, poplar::Tensor> outputs = {
      toCellOrder(std::move(inputWeights), params.cellOrder),
      toCellOrder(std::move(outputWeights), params.cellOrder)};
  di.addOutputs({{"inputWeights", toProfileValue(outputs.first)},
                 {"outputWeights", toProfileValue(outputs.second)}});
  return outputs;
}

/** Create the weights biases.
 */
poplar::Tensor
createWeightsBiases(poplar::Graph &graph, const GruParams &params,
                    const poplar::DebugContext &debugContext,
                    const OptionFlags &options,
                    poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, options, planningCache));

  validateParams(params);
  auto outputSize = params.rnn.layerSizes[1];
  Tensor biases;
  if (params.resetAfter) {
    biases = graph.addVariable(params.rnn.dataType,
                               {BASIC_GRU_CELL_NUM_UNITS, 2, outputSize},
                               {di, "biases"});
  } else {
    biases = graph.addVariable(params.rnn.dataType,
                               {BASIC_GRU_CELL_NUM_UNITS, outputSize},
                               {di, "biases"});
  }
  mapTensorLinearly(graph, biases);
  auto output = toCellOrder(biases, params.cellOrder);
  di.addOutput(output);
  return output;
}

GruWeights createWeights(Graph &graph, const GruParams &params,
                         const poplar::DebugContext &debugContext,
                         const OptionFlags &options,
                         poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  GruWeights gruWeights;
  std::tie(gruWeights.inputWeights, gruWeights.outputWeights) =
      createWeightsKernel(graph, params, {di}, options, cache);
  gruWeights.biases = createWeightsBiases(graph, params, {di}, options, cache);
  di.addOutputs(DI_ARGS(gruWeights));
  return gruWeights;
}

Tensor createAttention(Graph &graph, const GruParams &params,
                       const poplar::DebugContext &debugContext,
                       const OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options));
  validateParams(params);
  auto opt = parseOptions(options);
  auto numShards = getNumShards(graph, params, opt, {di, "numShards"});
  auto output = rnn::createRecurrentTensor(graph, params.rnn, 1, numShards,
                                           {di, "attention"})
                    .squeeze({2})
                    .transpose();
  di.addOutput(output);
  return output;
}

static Tensor rearrangeUnitsOutputFwd(Graph &graph, int num_unit,
                                      Tensor outputUnits,
                                      Tensor outputUnitsRearranged,
                                      Sequence &prog,
                                      const DebugNameAndId &dnai) {
  const auto outputGrouping =
      detectInnermostGrouping(graph, outputUnitsRearranged);
  // Typically the matrix multiplication result is laid out in memory such
  // that innermost dimension is groups batch elements. Try to rearrange the
  // result so the innermost dimension of the underlying memory is groups of the
  // specified number of outputs.
  return unflattenUnits(
      tryGroupedPartialTranspose(graph, flattenUnits(outputUnits),
                                 outputGrouping, prog, {dnai}),
      num_unit);
}

static void gruCellForwardPassCalcUnits(
    Graph &graph, bool forCandidate, const Tensor &in, const Tensor &prevOutput,
    const Tensor &biases, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const GruOpts &opt,
    const Tensor &unitsOutputRearranged, const GruParams &params,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  const unsigned outputSize = prevOutput.dim(1);
  const unsigned batchSize = prevOutput.dim(0);

  // if this call is for candiate, only one unit need processed
  // otherwise, reset and update units need be processed.
  unsigned numUnit = 2;
  if (forCandidate)
    numUnit = 1;

  if (weightsInput) {
#ifndef NDEBUG
    const unsigned inputSize = in.dim(1);
#endif
    // BASIC_GRU_CELL_CANDIDATE is not used to calculate units
    assert(weightsInput->dim(0) == numUnit);
    assert(weightsInput->dim(1) == inputSize);
    assert(weightsInput->dim(2) == outputSize);
  }
  assert(weightsOutput.dim(0) == numUnit);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

  const auto dType = in.elementType();

  auto bBiases =
      graph.addVariable(dType, {0, batchSize, outputSize}, {dnai, "bbiases"});
  if (forCandidate) {
    auto unitBias =
        biases[2].broadcast(batchSize, 0).reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  } else {
    for (unsigned u = 0; u != numUnit; ++u) {
      auto unitBias =
          biases[u].broadcast(batchSize, 0).reshape({batchSize, outputSize});
      bBiases = append(bBiases, unitBias);
    }
  }

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  Tensor unitsOutput = basicGruUnitsNlInput(
      graph, in, prevOutput, numUnit, *weightsInput, weightsOutput, prog, mmOpt,
      cache, {dnai, "ProcessUnits"});

  // Rearrange the output of the matrix multiplication so each output unit
  // arranged the same as the cell state. This avoids the rearrangement
  // during the subsequent binary operations.
  {
    auto out = rearrangeUnitsOutputFwd(graph, numUnit, unitsOutput,
                                       unitsOutputRearranged, prog, {dnai});
    prog.add(Copy(out, unitsOutputRearranged, false, {dnai}));
  }

  for (unsigned u = 0; u != numUnit; ++u) {
    graph.setTileMapping(biases[u],
                         graph.getTileMapping(unitsOutputRearranged[u][0]));
  }
  addInPlace(graph, unitsOutputRearranged, bBiases, prog, {dnai, "AddBias"});

  // Apply non linear function
  if (isCSNotSupported(params.activation) ||
      isCSNotSupported(params.recurrentActivation)) {
    if (forCandidate) {
      nonLinearityInPlace(graph, params.activation, unitsOutputRearranged, prog,
                          {dnai, "Candidate Tanh"});
    } else {
      nonLinearityInPlace(graph, params.recurrentActivation,
                          unitsOutputRearranged, prog,
                          {dnai, "update/reset sigmod"});
    }
  } else {
    auto cs = graph.addComputeSet({dnai, "non-linear"});
    if (forCandidate) {
      nonLinearityInPlace(graph, params.activation, unitsOutputRearranged, cs,
                          {dnai, "Candidate Tanh"});
    } else {
      nonLinearityInPlace(graph, params.recurrentActivation,
                          unitsOutputRearranged, cs,
                          {dnai, "update/reset sigmod"});
    }
    prog.add(Execute(cs, {dnai}));
  }
}

static void gruCellForwardPassCalcUnitsResetAfter(
    Graph &graph, bool forCandidate, const Tensor &in, const Tensor &prevOutput,
    const Tensor &biases, const Tensor *weightsInput,
    const Tensor &weightsOutput, Tensor *candidateRecurrant, Sequence &prog,
    const GruOpts &opt, const Tensor &unitsOutputRearranged,
    const GruParams &params, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  const unsigned outputSize = prevOutput.dim(1);
  const unsigned batchSize = prevOutput.dim(0);

  if (weightsInput) {
#ifndef NDEBUG
    const unsigned inputSize = in.dim(1);
#endif
    // BASIC_GRU_CELL_CANDIDATE is not used to calculate units
    assert(weightsInput->dim(0) == BASIC_GRU_CELL_NUM_UNITS);
    assert(weightsInput->dim(1) == inputSize);
    assert(weightsInput->dim(2) == outputSize);
  }
  assert(weightsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

  const auto dType = in.elementType();

  auto bBiases =
      graph.addVariable(dType, {0, batchSize, outputSize}, {dnai, "bbiases"});
  auto bRecurrantBiases = graph.addVariable(dType, {0, batchSize, outputSize},
                                            {dnai, "brecurrentbiases"});
  for (unsigned u = 0; u != BASIC_GRU_CELL_NUM_UNITS; ++u) {
    auto unitBias =
        biases[u][0].broadcast(batchSize, 0).reshape({batchSize, outputSize});
    auto unitRecurrantBias =
        biases[u][1].broadcast(batchSize, 0).reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
    bRecurrantBiases = append(bRecurrantBiases, unitRecurrantBias);
  }

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");

  auto unitsOutput =
      unflattenUnits(matMul(graph, in, flattenUnits(*weightsInput), prog,
                            {dnai, "Weights"}, mmOpt, cache),
                     BASIC_GRU_CELL_NUM_UNITS);
  auto recurrantUnitsOutput =
      unflattenUnits(matMul(graph, prevOutput, flattenUnits(weightsOutput),
                            prog, {dnai, "Weights"}, mmOpt, cache),
                     BASIC_GRU_CELL_NUM_UNITS);
  auto unitsOutputAll = concat(unitsOutput, recurrantUnitsOutput, 2);

  // Rearrange the outputs of the matrix multiplications so each output unit
  // Rearrange the output of the matrix multiplication so each output unit
  // arranged the same as the cell state. This avoids the rearrangement
  // during the subsequent binary operations.
  Tensor recurrentUnitsOutputRearranged;
  {
    auto out = rearrangeUnitsOutputFwd(
        graph, BASIC_GRU_CELL_NUM_UNITS, unitsOutputAll,
        unitsOutputRearranged.broadcast(2, 0), prog, {dnai});
    auto len = unitsOutputRearranged.dim(2);
    prog.add(Copy(out.slice(0, len, 2), unitsOutputRearranged, false, {dnai}));
    recurrentUnitsOutputRearranged = out.slice(len, len * 2, 2);
  }

  auto unitsOutputRearranged_r = unitsOutputRearranged.slice(0, 1);
  auto unitsOutputRearranged_ru = unitsOutputRearranged.slice(0, 2);
  auto unitsOutputRearranged_c = unitsOutputRearranged.slice(2, 3);
  auto recurrentUnitsOutputRearranged_ru =
      recurrentUnitsOutputRearranged.slice(0, 2);
  auto recurrentUnitsOutputRearranged_c =
      recurrentUnitsOutputRearranged.slice(2, 3);

  // Bias tile mapping
  for (unsigned u = 0; u != BASIC_GRU_CELL_NUM_UNITS; ++u) {
    graph.setTileMapping(biases[u][0],
                         graph.getTileMapping(unitsOutputRearranged[u][0]));
    graph.setTileMapping(
        biases[u][1],
        graph.getTileMapping(recurrentUnitsOutputRearranged[u][0]));
  }

  // Add biases
  {
    auto bBiasesAll = concat(bBiases, bRecurrantBiases);
    auto unitsOutputRearrangedAll =
        concat(unitsOutputRearranged, recurrentUnitsOutputRearranged);
    addInPlace(graph, unitsOutputRearrangedAll, bBiasesAll, prog,
               {dnai, "AddBias"});
  }

  if (candidateRecurrant) {
    *candidateRecurrant = graph.clone(
        recurrentUnitsOutputRearranged[BASIC_GRU_CELL_CANDIDATE], {dnai});
    prog.add(Copy(recurrentUnitsOutputRearranged[BASIC_GRU_CELL_CANDIDATE],
                  *candidateRecurrant, false, {dnai}));
  }

  // Add recurrent component for Reset/Update
  addInPlace(graph, unitsOutputRearranged_ru, recurrentUnitsOutputRearranged_ru,
             prog, {dnai, "Weights"});

  // Apply non-linearity for Reset/Update
  nonLinearityInPlace(graph, params.recurrentActivation,
                      unitsOutputRearranged_ru, prog,
                      {dnai, "update/reset sigmod"});

  // Apply reset gate to candidate recurrent component
  mulInPlace(graph, recurrentUnitsOutputRearranged_c, unitsOutputRearranged_r,
             prog, {dnai, "ch * r"});

  // Add recurrent component for Candidate
  addInPlace(graph, unitsOutputRearranged_c, recurrentUnitsOutputRearranged_c,
             prog, {dnai, "Weights"});

  // Apply non linear function to Candidate
  nonLinearityInPlace(graph, params.activation, unitsOutputRearranged_c, prog,
                      {dnai, "Candidate Tanh"});
}

static std::pair<Tensor, GruInternalState>
basicGruCellForwardPass(Graph &graph, const Tensor &in, const Tensor &biases,
                        const Tensor &prevOutput, const Tensor *weightsInput,
                        const Tensor &weightsOutput,
                        const boost::optional<const Tensor &> &attScoreOpt,
                        Sequence &prog, const GruOpts &opt,
                        const GruParams &params, const DebugNameAndId &dnai,
                        matmul::PlanningCache *cache) {
  debug_tensor(prog, "fwd h_prev", prevOutput);
  debug_tensor(prog, "fwd input", in);

  const std::string baseStr = "BasicGruCell";

  Tensor resetGate = graph.clone(prevOutput, {dnai, "Update Gate Rearranged"});
  Tensor updateGate = graph.clone(prevOutput, {dnai, "Reset Gate Rearranged"});
  Tensor candidate = graph.clone(prevOutput, {dnai, "candidate Rearranged"});

  Tensor candidateRecurrant;

  std::vector<Tensor> toConcat;
  toConcat.reserve(params.resetAfter ? BASIC_GRU_CELL_NUM_UNITS
                                     : BASIC_GRU_CELL_NUM_UNITS - 1);

  toConcat.push_back(resetGate.expand({0}));
  toConcat.push_back(updateGate.expand({0}));
  if (params.resetAfter)
    toConcat.push_back(candidate.expand({0}));

  auto unitsOutput = concat(toConcat);

  if (params.resetAfter) {
    gruCellForwardPassCalcUnitsResetAfter(
        graph, false, in, prevOutput, biases, weightsInput, weightsOutput,
        &candidateRecurrant, prog, opt, unitsOutput, params, {dnai, baseStr},
        cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS);
  } else {
    const Tensor weightsInput2 = weightsInput->slice(0, 2);
    const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
    const Tensor biases2 = biases.slice(0, 2);
    gruCellForwardPassCalcUnits(graph, false, in, prevOutput, biases2,
                                &weightsInput2, weightsOutput2, prog, opt,
                                unitsOutput, params, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS - 1);
  }

  resetGate = unitsOutput[BASIC_GRU_CELL_RESET_GATE];
  updateGate = unitsOutput[BASIC_GRU_CELL_UPDATE_GATE];

  Tensor resetGateOut;

  if (params.resetAfter) {
    resetGateOut = resetGate;
    candidate = unitsOutput[BASIC_GRU_CELL_CANDIDATE];
  } else {
    resetGateOut = graph.clone(resetGate, {dnai, "resetGateOut"});
    prog.add(Copy(resetGate, resetGateOut, false, {dnai}));

    const Tensor weightsInput3 = weightsInput->slice(2, 3);
    const Tensor weightsOutput3 = weightsOutput.slice(2, 3);
    mulInPlace(graph, resetGate, prevOutput, prog,
               {dnai, baseStr + "resetGate * prevOutput"});
    Tensor candidateExpand = candidate.expand({0});
    gruCellForwardPassCalcUnits(
        graph, true, in, resetGate, biases, &weightsInput3, weightsOutput3,
        prog, opt, candidateExpand, params, {dnai, baseStr}, cache);
    candidate = candidateExpand[0];
  }

  if (attScoreOpt) {
    mapInPlace(graph, _1 - (_2 * _1), {updateGate, *attScoreOpt}, prog,
               {dnai, baseStr + "/UpdateScores"});
  }

  Tensor newOutput =
      map(graph, _1 + _2 * (_3 - _1), {candidate, updateGate, prevOutput}, prog,
          {dnai, baseStr + "/CalcNextOutput"});

  GruInternalState internalState = {resetGateOut, updateGate, candidate,
                                    candidateRecurrant};

  debug_tensor(prog, "fwd resetGate", resetGateOut);
  debug_tensor(prog, "fwd updateGate", updateGate);
  debug_tensor(prog, "fwd candidate", candidate);
  debug_tensor(prog, "fwd output", newOutput);

  return {newOutput, internalState};
}

static void basicGruCellForwardPassInPlace(
    Graph &graph, const Tensor &in, const Tensor &biases, const Tensor &output,
    const Tensor *weightsInput, const Tensor &weightsOutput,
    const boost::optional<const Tensor &> &attScoreOpt, Sequence &prog,
    const GruOpts &opt, const GruParams &params, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache, bool resetAfter) {
  debug_tensor(prog, "fwd h_prev", output);
  debug_tensor(prog, "fwd input", in);
  const std::string baseStr = "BasicGruCellInPlace";

  Tensor resetGate = graph.clone(output, {dnai, "Update Gate Rearranged"});
  Tensor updateGate = graph.clone(output, {dnai, "Reset Gate Rearranged"});
  Tensor candidate = graph.clone(output, {dnai, "candidate Rearranged"});

  int numToConcat = params.resetAfter ? BASIC_GRU_CELL_NUM_UNITS
                                      : BASIC_GRU_CELL_NUM_UNITS - 1;
  std::vector<Tensor> toConcat;
  toConcat.reserve(numToConcat);

  toConcat.push_back(resetGate.expand({0}));
  toConcat.push_back(updateGate.expand({0}));
  if (params.resetAfter)
    toConcat.push_back(candidate.expand({0}));

  auto unitsOutput = concat(toConcat);

  if (params.resetAfter) {
    gruCellForwardPassCalcUnitsResetAfter(
        graph, false, in, output, biases, weightsInput, weightsOutput, nullptr,
        prog, opt, unitsOutput, params, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS);
  } else {
    const Tensor weightsInput2 = weightsInput->slice(0, 2);
    const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
    gruCellForwardPassCalcUnits(graph, false, in, output, biases,
                                &weightsInput2, weightsOutput2, prog, opt,
                                unitsOutput, params, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS - 1);
  }

  updateGate = unitsOutput[BASIC_GRU_CELL_UPDATE_GATE];
  resetGate = unitsOutput[BASIC_GRU_CELL_RESET_GATE];

  if (attScoreOpt) {
    logging::popnn::info("updateGate u {} score {}", updateGate.shape(),
                         (*attScoreOpt).shape());
    mapInPlace(graph, _1 - (_2 * _1), {updateGate, *attScoreOpt}, prog,
               {dnai, baseStr + "/UpdateScores"});
  }

  debug_tensor(prog, "fwd resetGate", resetGate);
  debug_tensor(prog, "fwd updateGate", updateGate);

  if (params.resetAfter) {
    candidate = unitsOutput[BASIC_GRU_CELL_CANDIDATE];
  } else {
    const Tensor weightsInput3 = weightsInput->slice(2, 3);
    const Tensor weightsOutput3 = weightsOutput.slice(2, 3);
    mulInPlace(graph, resetGate, output, prog,
               {dnai, baseStr + "resetGate * output"});
    Tensor candidateExpand = candidate.expand({0});
    gruCellForwardPassCalcUnits(
        graph, true, in, resetGate, biases, &weightsInput3, weightsOutput3,
        prog, opt, candidateExpand, params, {dnai, baseStr}, cache);
    candidate = candidateExpand[0];
  }

  debug_tensor(prog, "fwd candidate", candidate);

  mapInPlace(graph, _3 + _2 * (_1 - _3), {output, updateGate, candidate}, prog,
             {dnai, "CalcNextOutput"});

  debug_tensor(prog, "fwd output", output);
}

Tensor gruFwdImpl(Graph &graph, const GruParams &params,
                  const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                  const GruWeights &weights_, Tensor *intermediatesSeq,
                  const boost::optional<const Tensor &> &attScoresOpt,
                  const boost::optional<const Tensor &> &realTimeStepsOpt,
                  program::Sequence &fwdProg, const DebugNameAndId &dnai,
                  const OptionFlags &options,
                  poplin::matmul::PlanningCache *cache) {
  logging::popnn::info("gruFwdImpl(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, dnai.getPathName());

  validateParams(params);
  auto opt = parseOptions(options);

  auto weights = fromCellOrder(weights_, params.cellOrder);
  auto numShards = getNumShards(graph, params, opt, {dnai, "numShards"});

  auto loopFwd =
      [&params, &weights, &opt, &cache, &realTimeStepsOpt,
       numShards](const Tensor &seqLen, Graph &graph, const Tensor &shardSeqIdx,
                  const Tensor &seqIdx, std::vector<Tensor> &fwdState,
                  const std::vector<Tensor> &inputs, const Tensor &interimIn,
                  Tensor &interimOut, std::vector<Tensor> &outputs,
                  std::vector<Tensor> &created, program::Sequence *initProg,
                  const DebugNameAndId &dnai) {
        auto loop = Sequence{{}, {dnai}};
        debug_tensor(loop, "fwdLoop:", seqIdx);
        auto &fwdInput = inputs[0];
        boost::optional<const Tensor &> sliceAttScoresOpt(boost::none);
        if (inputs[1].valid()) {
          sliceAttScoresOpt = inputs[1];
        }

        Tensor seqIdxAbsolute = seqIdx;
        if (realTimeStepsOpt && (numShards > 1)) {
          seqIdxAbsolute =
              add(graph, shardSeqIdx, seqIdx, loop, {dnai, "sequenceIndex"});
        }

        const Tensor *inputWeightsPtr = &weights.inputWeights;
        auto sliceOutput = fwdState[0].squeeze({0});
        if (interimOut.valid()) {
          auto [newOutput, internalState] = basicGruCellForwardPass(
              graph, fwdInput, weights.biases, sliceOutput, inputWeightsPtr,
              weights.outputWeights, sliceAttScoresOpt, loop, opt, params,
              {dnai}, cache);
          if (realTimeStepsOpt) {
            auto mask = gt(graph, seqLen, seqIdxAbsolute, loop, {dnai});
            mask = cast(graph, mask, newOutput.elementType(), loop,
                        {dnai, "realTimeMask"});
            mask = mask.expand({1}).broadcast(newOutput.dim(1), 1);
            mapInPlace(graph, _1 * _2, {newOutput, mask}, loop, {dnai});
            mapInPlace(graph, _1 * _2, {internalState.resetGate, mask}, loop,
                       {dnai});
            mapInPlace(graph, _1 * _2, {internalState.updateGate, mask}, loop,
                       {dnai});
            mapInPlace(graph, _1 * _2, {internalState.candidate, mask}, loop,
                       {dnai});
          }
          auto fwdIntermediates =
              getFwdIntermediatesToSave(newOutput, internalState, params);
          loop.add(Copy(fwdIntermediates, interimOut, false, {dnai}));
          loop.add(Copy(newOutput, fwdState[0], false, {dnai}));
        } else {
          basicGruCellForwardPassInPlace(
              graph, fwdInput, weights.biases, sliceOutput, inputWeightsPtr,
              weights.outputWeights, sliceAttScoresOpt, loop, opt, params,
              {dnai}, cache, params.resetAfter);
          if (realTimeStepsOpt) {
            auto mask = gt(graph, seqLen, seqIdxAbsolute, loop, {dnai});
            mask = cast(graph, mask, sliceOutput.elementType(), loop,
                        {dnai, "realTimeMask"});
            mask = mask.expand({1}).broadcast(sliceOutput.dim(1), 1);
            mapInPlace(graph, _1 * _2, {sliceOutput, mask}, loop, {dnai});
          }
        }
        return loop;
      };

  Tensor seqLen;
  if (realTimeStepsOpt) {
    seqLen = cast(graph, *realTimeStepsOpt, UNSIGNED_INT, fwdProg,
                  {dnai, "realTimeStep"});
  }

  Tensor attScores;
  if (attScoresOpt) {
    attScores = (*attScoresOpt).transpose().expand({(*attScoresOpt).rank()});
  }

  debug_tensor(fwdProg, "fwd weightsInput", weights.inputWeights);
  debug_tensor(fwdProg, "fwd weightsOutput", weights.outputWeights);
  debug_tensor(fwdProg, "fwd bias", weights.biases);

  // make a copy of the activations so that they are sliced efficiently
  auto prevLayerActsCopy =
      createInput(graph, params, {dnai, "prevLayerActsCopy"}, options, cache);
  fwdProg.add(Copy(prevLayerActs, prevLayerActsCopy, false, {dnai}));

  std::vector<Tensor> inputs = {prevLayerActsCopy, attScores};
  std::vector<Tensor> initState = {fwdOutputInit.expand({0})};
  rnn::StateSequence stateSequence;
  if (params.outputFullSequence) {
    stateSequence = rnn::StateSequence{
        rnn::createOutputTensor(graph, params.rnn, numShards, {dnai, "output"}),
        0};
  }
  if (intermediatesSeq) {
    auto numIntermediates = getNumFwdIntermediatesToSave(params);
    *intermediatesSeq =
        rnn::createOutputTensor(graph, params.rnn, numIntermediates, numShards,
                                {dnai, "intermediatesSeq"})
            .reshapePartial(0, 1, {params.rnn.timeSteps, numIntermediates});
    fwdProg.add(WriteUndef(*intermediatesSeq, {dnai}));
  }
  std::vector<Tensor> dummyOutputs;
  std::vector<Tensor> dummyCreated;
  const auto shardingLoop = std::bind(
      loopFwd, seqLen, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
      std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
      std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
  auto rnnOptions = getRnnOpts(opt);
  auto updatedState =
      rnn::Rnn(graph, params.rnn, false, initState, stateSequence, inputs,
               nullptr, intermediatesSeq, dummyOutputs, dummyCreated, fwdProg,
               shardingLoop, numShards, rnnOptions, {dnai, "rnn"});
  return params.outputFullSequence ? stateSequence.output
                                   : updatedState[0].squeeze({0});
}

Tensor gruFwd(Graph &graph, const GruParams &params,
              const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
              const GruWeights &weights_, Tensor *intermediatesSeq,
              program::Sequence &fwdProg,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {

  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, prevLayerActs, weights_,
                            intermediatesSeq, params, options, cache));

  boost::optional<const Tensor &> attScoresOpt(boost::none);
  boost::optional<const Tensor &> realTimeStepsOpt(boost::none);

  auto output = gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs,
                           weights_, intermediatesSeq, attScoresOpt,
                           realTimeStepsOpt, fwdProg, {di}, options, cache);
  di.addOutput(output);
  return output;
}

Tensor gruFwd(Graph &graph, const GruParams &params,
              const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
              const Tensor &realTimeSteps, const GruWeights &weights_,
              Tensor *intermediatesSeq, program::Sequence &fwdProg,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(fwdOutputInit, prevLayerActs,
                                         realTimeSteps, weights_, params,
                                         intermediatesSeq, options, cache));

  if (!params.outputFullSequence) {
    throw poplibs_error(std::string("The outputFullSequence should be true ") +
                        "if realTimeSteps given");
  }

  boost::optional<const Tensor &> attScoresOpt(boost::none);
  boost::optional<const Tensor &> realTimeStepsOpt(realTimeSteps);

  auto output = gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs,
                           weights_, intermediatesSeq, attScoresOpt,
                           realTimeStepsOpt, fwdProg, {di}, options, cache);
  di.addOutput(output);
  return output;
}

Tensor auGruFwd(Graph &graph, const GruParams &params,
                const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                const GruWeights &weights_, Tensor *intermediatesSeq,
                const Tensor &attScores, program::Sequence &fwdProg,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options,
                poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, prevLayerActs, attScores, weights_,
                            params, intermediatesSeq, options, cache));

  boost::optional<const Tensor &> attScoresOpt(attScores);
  boost::optional<const Tensor &> realTimeStepsOpt(boost::none);

  auto output = gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs,
                           weights_, intermediatesSeq, attScoresOpt,
                           realTimeStepsOpt, fwdProg, {di}, options, cache);
  di.addOutput(output);
  return output;
}

Tensor auGruFwd(Graph &graph, const GruParams &params,
                const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                const Tensor &realTimeSteps, const GruWeights &weights_,
                Tensor *intermediatesSeq, const Tensor &attScores,
                program::Sequence &fwdProg,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options,
                poplin::matmul::PlanningCache *cache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, prevLayerActs, realTimeSteps, attScores, weights_,
              params, intermediatesSeq, options, cache));

  if (!params.outputFullSequence) {
    throw poplibs_error(std::string("The outputFullSequence should be true ") +
                        "if realTimeSteps given");
  }

  boost::optional<const Tensor &> attScoresOpt(attScores);
  boost::optional<const Tensor &> realTimeStepsOpt(realTimeSteps);

  auto output = gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs,
                           weights_, intermediatesSeq, attScoresOpt,
                           realTimeStepsOpt, fwdProg, {di}, options, cache);
  di.addOutput(output);
  return output;
}

static Tensor gruBackwardRearrangeWeights(
    Graph &graph, const GruParams &params, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &initProg, const GruOpts &opt,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  Tensor w_ru, w_c;
  if (weightsInput == nullptr) {
    w_ru = concat(weightsOutput[0], weightsOutput[1], 1).transpose();
    w_c = weightsOutput[2].transpose();
  } else {
    w_ru = concat(concat((*weightsInput)[0], weightsOutput[0], 0),
                  concat((*weightsInput)[1], weightsOutput[1], 0), 1)
               .transpose();
    w_c = concat((*weightsInput)[2], weightsOutput[2], 0).transpose();
  }
  std::vector<std::size_t> d_c_shape{params.rnn.batchSize,
                                     params.rnn.layerSizes[1]};
  w_c = preArrangeMatMulInputRHS(graph, d_c_shape, w_c, initProg,
                                 {dnai, "/PreArrangeWeights C "}, mmOpt, cache);
  std::vector<std::size_t> d_ru_shape{params.rnn.batchSize,
                                      2 * params.rnn.layerSizes[1]};
  w_ru =
      preArrangeMatMulInputRHS(graph, d_ru_shape, w_ru, initProg,
                               {dnai, "/PreArrangeWeights RU"}, mmOpt, cache);
  auto weights = concat(w_ru, w_c, 0);
  return weights;
}

Tensor gruBackwardRearrangeWeightsResetAfter(
    Graph &graph, const GruParams &params, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &initProg, const GruOpts &opt,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  Tensor w_out, w_in;
  if (weightsInput == nullptr) {
    w_out = concat({weightsOutput[BASIC_GRU_CELL_RESET_GATE],
                    weightsOutput[BASIC_GRU_CELL_UPDATE_GATE],
                    weightsOutput[BASIC_GRU_CELL_CANDIDATE]},
                   1)
                .transpose();
  } else {
    w_out = concat({weightsOutput[BASIC_GRU_CELL_RESET_GATE],
                    weightsOutput[BASIC_GRU_CELL_UPDATE_GATE],
                    weightsOutput[BASIC_GRU_CELL_CANDIDATE]},
                   1)
                .transpose();
    w_in = concat({(*weightsInput)[BASIC_GRU_CELL_RESET_GATE],
                   (*weightsInput)[BASIC_GRU_CELL_UPDATE_GATE],
                   (*weightsInput)[BASIC_GRU_CELL_CANDIDATE]},
                  1)
               .transpose();
  }

  std::vector<std::size_t> d_r_d_u_d_c_out_shape{params.rnn.batchSize,
                                                 3 * params.rnn.layerSizes[1]};
  w_out = preArrangeMatMulInputRHS(graph, d_r_d_u_d_c_out_shape, w_out,
                                   initProg, {dnai, "/PreArrangeOutputWeights"},
                                   mmOpt, cache);
  if (weightsInput) {
    std::vector<std::size_t> d_r_d_u_d_c_in_shape{params.rnn.batchSize,
                                                  3 * params.rnn.layerSizes[1]};
    w_in = preArrangeMatMulInputRHS(graph, d_r_d_u_d_c_in_shape, w_in, initProg,
                                    {dnai, "/PreArrangeInputWeights"}, mmOpt,
                                    cache);
  }
  auto weights = weightsInput ? concat(w_in, w_out, 1) : w_out;
  return weights;
}

static std::tuple<Tensor, Tensor, Tensor> backwardStepImpl(
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const Tensor &prevStepOut, const Tensor &outputGrad, const Tensor weights,
    bool weightsIncludeInputs, const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoresOpt,
    const boost::optional<Tensor &> &attScoresGradsOpt, Sequence &prog,
    const GruOpts &opt, const GruParams &params, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  const std::string fPrefix = "GruBwdOneStep";
  auto outputGroupingIntoLayer = detectInnermostGrouping(graph, outputGrad);
  Tensor d_h = graph.clone(outputGrad, {dnai});
  debug_tensor(prog, "bwd outGrad", outputGrad);
  if (gradNextLayer)
    debug_tensor(prog, "bwd gradNextLayer", *gradNextLayer);
  prog.add(Copy(outputGrad, d_h, false, {dnai}));
  if (gradNextLayer)
    d_h = popops::add(graph, d_h, *gradNextLayer, prog,
                      {dnai, fPrefix + "/AddActGrads"});

  auto u = fwdIntermediates[GRU_FWD_INTERMEDIATE_UPDATE_GATE];
  auto r = fwdIntermediates[GRU_FWD_INTERMEDIATE_RESET_GATE];
  auto c = fwdIntermediates[GRU_FWD_INTERMEDIATE_CANDIDATE];
  auto h_prev = prevStepOut;

  auto one_matrix =
      graph.addConstant(outputGrad.elementType(), outputGrad.shape(), 1,
                        {dnai, fPrefix + "/one_matrix"});
  graph.setTileMapping(one_matrix, graph.getTileMapping(u));
  auto var_one_matrix =
      graph.addVariable(outputGrad.elementType(), outputGrad.shape(),
                        {dnai, fPrefix + "/var_one_matrix"});
  graph.setTileMapping(var_one_matrix, graph.getTileMapping(u));
  prog.add(Copy(one_matrix, var_one_matrix, false, {dnai}));

  debug_tensor(prog, "bwd d_h", d_h);
  debug_tensor(prog, "bwd r", r);
  debug_tensor(prog, "bwd u", u);
  debug_tensor(prog, "bwd c", c);
  debug_tensor(prog, "bwd h_prev", h_prev);

  // u_com = 1 - u
  Tensor u_com =
      sub(graph, var_one_matrix, u, prog, {dnai, fPrefix + "/1-updateGate"});
  // h_prev_c = h_prev - c
  auto h_prev_c =
      sub(graph, h_prev, c, prog, {dnai, fPrefix + "/preOutput-candidate"});
  // (1-u) * d_h, (h_prev - c) * d_h
  auto t = mul(graph, concat({u_com, h_prev_c}), d_h.broadcast(2, 0), prog,
               {dnai, fPrefix + "/MulOGate"});
  auto gradAtCandidateInput = t.slice(0, outputGrad.dim(0));
  auto gradAtUpdateGateInput =
      t.slice(outputGrad.dim(0), 2 * outputGrad.dim(0));

  debug_tensor(prog, "bwd outputGrad", d_h);
  debug_tensor(prog, "bwd h_prev_c", h_prev_c);

  ComputeSet cs1;
  Tensor d_c;

  bool usingSoftMax = isCSNotSupported(params.activation) ||
                      isCSNotSupported(params.recurrentActivation);

  if (usingSoftMax) {
    d_c = nonLinearityInputGradient(graph, params.activation, c,
                                    gradAtCandidateInput, prog,
                                    {dnai, fPrefix + "/OutputTanh"});
  } else {
    cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
    d_c = nonLinearityInputGradient(graph, params.activation, c,
                                    gradAtCandidateInput, cs1,
                                    {dnai, fPrefix + "/OutputTanh"});
  }

  Tensor d_u;
  if (attScoresOpt) {
    auto u0 = map(graph, _1 / (Const(1) - _2), {u, *attScoresOpt}, prog,
                  {dnai, fPrefix + "/UpdateScores"});

    Tensor temp = map(graph, -_1 * _2, {gradAtUpdateGateInput, u0}, prog,
                      {dnai, fPrefix + "/AttentionsGrad"});
    *attScoresGradsOpt =
        reduce(graph, temp, {1}, {popops::Operation::ADD}, prog, {dnai});

    mapInPlace(graph, _1 - (_1 * _2), {gradAtUpdateGateInput, *attScoresOpt},
               prog, {dnai, fPrefix + "/UpdateGateGrad"});

    if (usingSoftMax) {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u0,
                                      gradAtUpdateGateInput, prog,
                                      {dnai, fPrefix + "/OutputGate"});
    } else {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u0,
                                      gradAtUpdateGateInput, cs1,
                                      {dnai, fPrefix + "/OutputGate"});
    }
  } else {
    if (usingSoftMax) {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u,
                                      gradAtUpdateGateInput, prog,
                                      {dnai, fPrefix + "/OutputGate"});
    } else {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u,
                                      gradAtUpdateGateInput, cs1,
                                      {dnai, fPrefix + "/OutputGate"});
    }
  }

  if (!usingSoftMax)
    prog.add(Execute(cs1, {dnai}));

  if (maskOpt) {
    auto mask = cast(graph, *maskOpt, c.elementType(), prog, {dnai});
    mask = mask.squeeze({0}).expand({1}).broadcast(c.dim(1), 1);
    mapInPlace(graph, _1 * _2, {d_c, mask}, prog, {dnai});
  }

  debug_tensor(prog, "bwd d_c", d_c);
  debug_tensor(prog, "bwd d_u", d_u);

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  int outputSize = outputGrad.dim(1);
  Tensor w_ru = weights.slice(0, 2 * outputSize);
  Tensor w_c = weights.slice(2 * outputSize, 3 * outputSize);

  Tensor d_x2, d_hr, d_x2_hr;
  if (weightsIncludeInputs) {
    int inputSize = w_c.dim(1) - outputSize;
    auto out = matMul(graph, d_c, w_c, prog,
                      {dnai, fPrefix + "/d_x2_d_h_prevr"}, mmOpt, cache);
    d_x2_hr = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer,
                                         prog, {dnai, fPrefix});
    debug_tensor(prog, "bwd d_x2_h_prevr", d_x2_hr);
    d_x2 = d_x2_hr.slice(0, inputSize, 1);
    d_hr = d_x2_hr.slice(inputSize, inputSize + outputSize, 1);
  } else {
    auto out = matMul(graph, d_c, w_c, prog, {dnai, fPrefix + "/PrevStepGrad"},
                      mmOpt, cache);
    d_hr = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer, prog,
                                      {dnai, fPrefix});
  }

  Tensor d_r;
  {
    auto t = mul(graph, d_hr, h_prev, prog, {dnai, fPrefix + "/d_hr * h_prev"});
    d_r = nonLinearityInputGradient(graph, params.recurrentActivation, r, t,
                                    prog, {dnai, fPrefix + "/t * r * (1-r)"});
  }

  Tensor d_r_d_u = concat(d_r, d_u, 1);
  auto out = matMul(graph, d_r_d_u, w_ru, prog,
                    {dnai, fPrefix + "/d_x1_d_h_prev1 X w_ru"}, mmOpt, cache);
  Tensor d_x1_d_hprev1 = tryGroupedPartialTranspose(
      graph, out, outputGroupingIntoLayer, prog, {dnai, fPrefix});
  debug_tensor(prog, "bwd d_x1_d_hprev1", d_x1_d_hprev1);

  Tensor d_x;
  if (weightsIncludeInputs) {
    int inputSize = w_c.dim(1) - outputSize;
    d_x = add(graph, d_x1_d_hprev1.slice(0, inputSize, 1), d_x2, prog,
              {dnai, fPrefix + "/dx"});
  }

  Tensor d_hprev1 = d_x1_d_hprev1.slice(d_x1_d_hprev1.dim(1) - outputSize,
                                        d_x1_d_hprev1.dim(1), 1);
  Tensor d_h_prev =
      map(graph, ((_1 * _2) + (_3 * _4)) + _5, {d_hr, r, d_h, u, d_hprev1},
          prog, {dnai, fPrefix + "/d_h_prev"});

  debug_tensor(prog, "bwd d_h_prev", d_h_prev);
  debug_tensor(prog, "bwd d_x", d_x);
  debug_tensor(prog, "bwd d_r", d_r);
  debug_tensor(prog, "bwd d_u", d_u);
  debug_tensor(prog, "bwd d_c", d_c);
  return std::make_tuple(
      d_h_prev, d_x,
      concat({d_r.expand({0}), d_u.expand({0}), d_c.expand({0})}));
}

static std::tuple<Tensor, Tensor, Tensor> backwardStepImplResetAfter(
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const Tensor &prevStepOut, const Tensor &outputGrad, const Tensor weights,
    bool weightsIncludeInputs, const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoresOpt,
    const boost::optional<Tensor &> &attScoresGradsOpt, Sequence &prog,
    const GruOpts &opt, const GruParams &params, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  const std::string fPrefix = "GruBwdOneStep";
  auto outputGroupingIntoLayer = detectInnermostGrouping(graph, outputGrad);
  Tensor d_h = graph.clone(outputGrad, {dnai});
  debug_tensor(prog, "bwd outGrad", outputGrad);
  if (gradNextLayer)
    debug_tensor(prog, "bwd gradNextLayer", *gradNextLayer);
  prog.add(Copy(outputGrad, d_h, false, {dnai}));
  if (gradNextLayer)
    d_h = popops::add(graph, d_h, *gradNextLayer, prog,
                      {dnai, fPrefix + "/AddActGrads"});

  auto u = fwdIntermediates[GRU_FWD_INTERMEDIATE_UPDATE_GATE];
  auto r = fwdIntermediates[GRU_FWD_INTERMEDIATE_RESET_GATE];
  auto c = fwdIntermediates[GRU_FWD_INTERMEDIATE_CANDIDATE];

  unsigned c_recurrent_index = GRU_FWD_INTERMEDIATE_CANDIDATE_RECURRANT;
  if (params.outputFullSequence) {
    c_recurrent_index -= 1;
  }

  auto c_recurrant = fwdIntermediates[c_recurrent_index];
  auto h_prev = prevStepOut;

  auto one_matrix =
      graph.addConstant(outputGrad.elementType(), outputGrad.shape(), 1,
                        {dnai, fPrefix + "/one_matrix"});
  graph.setTileMapping(one_matrix, graph.getTileMapping(u));
  auto var_one_matrix =
      graph.addVariable(outputGrad.elementType(), outputGrad.shape(),
                        {dnai, fPrefix + "/var_one_matrix"});
  graph.setTileMapping(var_one_matrix, graph.getTileMapping(u));
  prog.add(Copy(one_matrix, var_one_matrix, false, {dnai}));

  debug_tensor(prog, "bwd d_h", d_h);
  debug_tensor(prog, "bwd r", r);
  debug_tensor(prog, "bwd u", u);
  debug_tensor(prog, "bwd c", c);
  debug_tensor(prog, "bwd h_prev", h_prev);

  // u_com = 1 - u
  Tensor u_com =
      sub(graph, var_one_matrix, u, prog, {dnai, fPrefix + "/1-updateGate"});
  // h_prev_c = h_prev - c
  auto h_prev_c =
      sub(graph, h_prev, c, prog, {dnai, fPrefix + "/preOutput-candidate"});
  // (1-u) * d_h, (h_prev - c) * d_h
  auto t = mul(graph, concat({u_com, h_prev_c}), d_h.broadcast(2, 0), prog,
               {dnai, fPrefix + "/MulOGate"});
  auto gradAtCandidateInput = t.slice(0, outputGrad.dim(0));
  auto gradAtUpdateGateInput =
      t.slice(outputGrad.dim(0), 2 * outputGrad.dim(0));

  debug_tensor(prog, "bwd outputGrad", d_h);
  debug_tensor(prog, "bwd h_prev_c", h_prev_c);

  ComputeSet cs1;
  Tensor d_c;

  bool usingSoftMax = isCSNotSupported(params.activation) ||
                      isCSNotSupported(params.recurrentActivation);

  if (usingSoftMax) {
    auto d_c = nonLinearityInputGradient(graph, params.activation, c,
                                         gradAtCandidateInput, prog,
                                         {dnai, fPrefix + "/OutputTanh"});
  } else {
    cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
    d_c = nonLinearityInputGradient(graph, params.activation, c,
                                    gradAtCandidateInput, cs1,
                                    {dnai, fPrefix + "/OutputTanh"});
  }
  Tensor d_u;

  if (attScoresOpt) {
    auto u0 = map(graph, _1 / (Const(1) - _2), {u, *attScoresOpt}, prog,
                  {dnai, fPrefix + "/UpdateScores"});

    Tensor temp = map(graph, -_1 * _2, {gradAtUpdateGateInput, u0}, prog,
                      {dnai, fPrefix + "/AttentionsGrad"});
    *attScoresGradsOpt =
        reduce(graph, temp, {1}, {popops::Operation::ADD}, prog);

    mapInPlace(graph, _1 - (_1 * _2), {gradAtUpdateGateInput, *attScoresOpt},
               prog, {dnai, fPrefix + "/UpdateGateGrad"});

    if (usingSoftMax) {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u0,
                                      gradAtUpdateGateInput, prog,
                                      {dnai, fPrefix + "/OutputGate"});
    } else {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u0,
                                      gradAtUpdateGateInput, cs1,
                                      {dnai, fPrefix + "/OutputGate"});
    }
  } else {
    if (usingSoftMax) {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u,
                                      gradAtUpdateGateInput, prog,
                                      {dnai, fPrefix + "/OutputGate"});
    } else {
      d_u = nonLinearityInputGradient(graph, params.recurrentActivation, u,
                                      gradAtUpdateGateInput, cs1,
                                      {dnai, fPrefix + "/OutputGate"});
    }
  }

  if (!usingSoftMax)
    prog.add(Execute(cs1, {dnai}));

  if (maskOpt) {
    auto mask = cast(graph, *maskOpt, c.elementType(), prog);
    mask = mask.squeeze({0}).expand({1}).broadcast(c.dim(1), 1);
    mapInPlace(graph, _1 * _2, {d_c, mask}, prog);
  }

  auto gradAtResetGateInput =
      mul(graph, d_c, c_recurrant, prog, {dnai, fPrefix + "/d_c * h_prev2"});
  auto d_r = nonLinearityInputGradient(graph, params.recurrentActivation, r,
                                       gradAtResetGateInput, prog);

  debug_tensor(prog, "bwd d_c", d_c);
  debug_tensor(prog, "bwd d_u", d_u);
  debug_tensor(prog, "bwd d_r", d_r);

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  auto d_cr = mul(graph, d_c, r, prog, {dnai, fPrefix + "/d_c * r"});
  auto d_r_d_u_d_c_out = concat({d_r, d_u, d_cr}, 1);

  auto outputSize = d_r.dim(1);
  auto inputSize = weights.dim(1) - outputSize;
  Tensor d_h_prev;
  {
    auto weightsOut = weights.slice(inputSize, weights.dim(1), 1);
    auto out = matMul(graph, d_r_d_u_d_c_out, weightsOut, prog,
                      {dnai, fPrefix + "/PrevStepGrad"}, mmOpt, cache);
    d_h_prev = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer,
                                          prog, fPrefix);
  }

  Tensor d_x;
  if (weightsIncludeInputs) {
    Tensor d_r_d_u_d_c_in = concat({d_r, d_u, d_c}, 1);
    auto weightsIn = weights.slice(0, inputSize, 1);
    auto out = matMul(graph, d_r_d_u_d_c_in, weightsIn, prog,
                      {dnai, fPrefix + "/PrevStepGrad"}, mmOpt, cache);
    d_x = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer, prog,
                                     fPrefix);
  }

  d_h_prev = map(graph, (_1 * _2) + _3, {d_h, u, d_h_prev}, prog,
                 {dnai, fPrefix + "/d_h_prev"});

  debug_tensor(prog, "bwd d_h_prev", d_h_prev);
  debug_tensor(prog, "bwd d_x", d_x);
  debug_tensor(prog, "bwd d_r", d_r);
  debug_tensor(prog, "bwd d_u", d_u);
  debug_tensor(prog, "bwd d_c", d_c);
  return std::make_tuple(
      d_h_prev, d_x,
      concat({d_r.expand({0}), d_u.expand({0}), d_c.expand({0})}));
}

std::tuple<Tensor, Tensor, Tensor>
basicGruBackwardStep(Graph &graph, const GruParams &params,
                     const Tensor *gradNextLayer,
                     const Tensor &fwdIntermediates, const Tensor &prevStepOut,
                     const Tensor &outGrad, const Tensor &weights,
                     const boost::optional<const Tensor &> &maskOpt,
                     const boost::optional<const Tensor &> &attScoreOpt,
                     const boost::optional<Tensor &> &attScoresGradsOpt,
                     Sequence &prog, const GruOpts &opt,
                     const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  bool weightsIncludeInputs =
      (weights.dim(1) == inputSize + outputSize) ? true : false;
  if (params.resetAfter) {
    return backwardStepImplResetAfter(
        graph, gradNextLayer, fwdIntermediates, prevStepOut, outGrad, weights,
        weightsIncludeInputs, maskOpt, attScoreOpt, attScoresGradsOpt, prog,
        opt, params, {dnai}, cache);
  } else {
    return backwardStepImpl(graph, gradNextLayer, fwdIntermediates, prevStepOut,
                            outGrad, weights, weightsIncludeInputs, maskOpt,
                            attScoreOpt, attScoresGradsOpt, prog, opt, params,
                            {dnai}, cache);
  }
}

/// Add the partial weight gradients from this timestep to the accumulated
/// weight gradients. Once all the gradients have been accumulated call
/// basicGruParamUpdateFinal() to do any final accumulation / rearrangement
/// that is required.
static void basicGruParamUpdate(Graph &graph, const Tensor &prevLayerActs,
                                const Tensor &prevStepActs,
                                const Tensor &fwdIntermediates,
                                const Tensor &bwdIntermediates,
                                const GruWeights &weightGrads, Sequence &prog,
                                const GruOpts &opt, const DebugNameAndId &dnai,
                                matmul::PlanningCache *cache, bool resetAfter) {
  const std::string fPrefix = "GruDeltas";
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_WU");

  GruWeights weightGrads2;
  weightGrads2.inputWeights = weightGrads.inputWeights.slice(0, 2);
  weightGrads2.outputWeights = weightGrads.outputWeights.slice(0, 2);
  GruWeights weightGrads3;
  weightGrads3.inputWeights = weightGrads.inputWeights.slice(2, 3);
  weightGrads3.outputWeights = weightGrads.outputWeights.slice(2, 3);

  Tensor h_prev = prevStepActs;
  Tensor x = prevLayerActs;
  Tensor x_h_prev = concat(x, h_prev, 1);
  Tensor r = fwdIntermediates[BASIC_GRU_CELL_RESET_GATE];
  debug_tensor(prog, "wu x", x);
  debug_tensor(prog, "wu h_prev", h_prev);
  debug_tensor(prog, "wu r", r);
  debug_tensor(prog, "wu fwdIntermediates", fwdIntermediates);

  Tensor d_r = bwdIntermediates[BASIC_GRU_CELL_RESET_GATE];
  Tensor d_u = bwdIntermediates[BASIC_GRU_CELL_UPDATE_GATE];
  Tensor d_c = bwdIntermediates[BASIC_GRU_CELL_CANDIDATE];

  if (resetAfter) {
    Tensor d_cr = mul(graph, d_c, r, prog, {dnai, fPrefix + "/d_c * r"});
    matMulAcc(graph, flattenUnits(weightGrads.inputWeights), 1.0, x.transpose(),
              flattenUnits(
                  concat({d_r.expand({0}), d_u.expand({0}), d_c.expand({0})})),
              prog, {dnai, fPrefix + "/dw for input weights"}, mmOpt, cache);
    matMulAcc(graph, flattenUnits(weightGrads.outputWeights), 1.0,
              h_prev.transpose(),
              flattenUnits(
                  concat({d_r.expand({0}), d_u.expand({0}), d_cr.expand({0})})),
              prog, {dnai, fPrefix + "/dw for output weights"}, mmOpt, cache);
    debug_tensor(prog, "wu d_cr", d_cr);

    auto biasGrads = concat({d_r, d_r, d_u, d_u, d_c, d_cr})
                         .reshape(weightGrads.biases.shape());
    // We defer the reduction across the batch to later.
    popops::addInPlace(graph, weightGrads.biases, biasGrads, prog,
                       {dnai, fPrefix + "/Bias"});
  } else {
    matMulAcc(graph,
              concat(flattenUnits(weightGrads2.inputWeights),
                     flattenUnits(weightGrads2.outputWeights)),
              1.0, x_h_prev.transpose(),
              flattenUnits(concat(d_r.expand({0}), d_u.expand({0}))), prog,
              {dnai, fPrefix + "/dw for reset and update weight"}, mmOpt,
              cache);

    Tensor h_prevr =
        mul(graph, h_prev, r, prog, {dnai, fPrefix + "/h_prev * r"});
    Tensor x_h_prevr = concat(x, h_prevr, 1);
    matMulAcc(graph,
              concat(flattenUnits(weightGrads3.inputWeights),
                     flattenUnits(weightGrads3.outputWeights)),
              1.0, x_h_prevr.transpose(), d_c, prog,
              {dnai, fPrefix + "/dw for candidate weight"}, mmOpt, cache);
    debug_tensor(prog, "wu x_h_prevr", x_h_prevr);

    // We defer the reduction across the batch to later.
    popops::addInPlace(graph, weightGrads.biases, bwdIntermediates, prog,
                       {dnai, fPrefix + "/Bias"});
  }
  debug_tensor(prog, "wu d_c", d_c);
  debug_tensor(prog, "wu inputWeightsGrad", weightGrads.inputWeights);
  debug_tensor(prog, "wu outputWeightsGrad", weightGrads.outputWeights);
}

static GruWeights
basicGruParamUpdateFinal(Graph &graph, const GruWeights &weights,
                         const GruWeights &weightGrads, Sequence &prog,
                         const DebugNameAndId &dnai, bool resetAfter) {
  // The accumulated bias gradients still has a batch axis that we must
  // accumulate over - do this now.
  auto biasGrad = graph.clone(weights.biases, {dnai, "biasGrad"});
  unsigned long reduceDimension = resetAfter ? 2 : 1;
  popops::reduceWithOutput(graph, weightGrads.biases, biasGrad,
                           {reduceDimension}, {popops::Operation::ADD}, prog,
                           {dnai, "FinalBiasReduction"});
  auto finalWeightGrads = weightGrads;
  finalWeightGrads.biases = biasGrad;
  return finalWeightGrads;
}

/// Create variables used to accumulate gradients of the weights in the
/// backward pass.
static GruWeights
createWeightAccumulators(Graph &graph, const GruWeights &weights,
                         const Tensor &bwdIntermediates, const GruOpts &options,
                         const poplar::DebugNameAndId &dnai, bool resetAfter) {
  GruWeights weightAccs;
  // inputWeights and outputWeights are slices of the one variable. Clone
  // them together as it results in a less complex tensor expression.
  auto concatenated = concat(flattenUnits(weights.inputWeights),
                             flattenUnits(weights.outputWeights));
  auto weightsDeltaAcc = graph.clone(concatenated, {dnai, "weightsDeltaAcc"});
  const auto inputSize = weights.inputWeights.dim(1);
  const auto outputSize = weights.outputWeights.dim(1);
  weightAccs.inputWeights =
      unflattenUnits(weightsDeltaAcc.slice(0, inputSize), 3);
  weightAccs.outputWeights = unflattenUnits(
      weightsDeltaAcc.slice(inputSize, inputSize + outputSize), 3);
  // We delay reducing across the batch until after we have accumulated
  // gradients from each timestep and therefore the bias accumlator still has
  // a batch axis. This amortizes the cost of reducing over the batch which
  // otherwise can be significant.
  if (resetAfter) {
    weightAccs.biases =
        concat({graph.clone(bwdIntermediates, {dnai, "bwdIntermediatesAcc"})
                    .expand({1}),
                graph.clone(bwdIntermediates, {dnai, "bwdIntermediatesAcc"})
                    .expand({1})},
               1);
  } else {
    weightAccs.biases =
        graph.clone(bwdIntermediates, {dnai, "bwdIntermediatesAcc"});
  }
  return weightAccs;
}

static void zeroWeightAccumulators(Graph &graph, program::Sequence &prog,
                                   const GruWeights &weightsAcc,
                                   const DebugNameAndId &dnai) {
  popops::zero(
      graph,
      concat({weightsAcc.inputWeights.flatten(),
              weightsAcc.outputWeights.flatten(), weightsAcc.biases.flatten()}),
      prog, {dnai, "zeroWeightAccumulators"});
}

// Perform an GRU backward pass.
// Optionally return the intermediates from the backward pass (sequence
// cell unit gradients), or calculate weight gradients directly during
// this pass interleaved with the backward pass.
static Tensor gruBwdImpl(
    Graph &graph, const GruParams &params, program::Sequence &prog,
    const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
    const GruWeights &weights, const Tensor &fwdInputSeq,
    const Tensor &fwdOutput, const Tensor &gradLayerNext, Tensor *inputGradSeq,
    Tensor *bwdIntermediatesPtr, GruWeights *weightsGrad,
    const boost::optional<const Tensor &> &realTimeStepsOpt,
    const boost::optional<const Tensor &> &attScoresOpt, Tensor *attScoresGrads,
    const DebugNameAndId &dnai, const poplar::OptionFlags &options_,
    poplin::matmul::PlanningCache *cache) {
  logging::popnn::info("gruBwdImpl(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, dnai.getPathName());

  auto options = parseOptions(options_);
  auto numShards = getNumShards(graph, params, options, {dnai, "numShards"});
  debug_tensor(prog, "bwd fwdIntermediatesSeq", fwdIntermediatesSeq);

  // Rearrange weights
  const Tensor *weightsInput = (inputGradSeq) ? &weights.inputWeights : nullptr;
  auto &weightsOutput = weights.outputWeights;
  Tensor weightsRearranged;
  if (params.resetAfter) {
    weightsRearranged = gruBackwardRearrangeWeightsResetAfter(
        graph, params, weightsInput, weightsOutput, prog, options,
        {dnai, "weightsRearranged"}, cache);
  } else {
    weightsRearranged = gruBackwardRearrangeWeights(
        graph, params, weightsInput, weightsOutput, prog, options,
        {dnai, "weightsRearranged"}, cache);
  }

  auto loopBwdWithWU =
      [&params, &options, &inputGradSeq, &cache, numShards, &weightsRearranged](
          GruWeights &weights, GruWeights *weightsGrad, const Tensor &seqLen,
          Graph &graph, const Tensor &shardSeqIdx, const Tensor &seqIdx,
          std::vector<Tensor> &shardState, const std::vector<Tensor> &bwdInput,
          const Tensor &fwdIntermediates, Tensor &interimOut,
          std::vector<Tensor> &outputs, std::vector<Tensor> &created,
          program::Sequence *initProg, const DebugNameAndId &dnai) {
        auto loop = Sequence{{}, {dnai}};
        const Tensor *gradLayerNextThisStepPtr =
            bwdInput[0].valid() ? &bwdInput[0] : nullptr;
        auto &prevLayerOut = bwdInput[1];
        auto &prevStepOut = bwdInput[2];
        auto &attScores = bwdInput[3];
        auto &attScoresGrads = created[0];
        Tensor inputGrad =
            shardState[0].valid() ? shardState[0].squeeze({0}) : Tensor{};
        Tensor lastOutGrad = shardState[1].squeeze({0});
        bool realTimeStepsFlag = seqLen.valid();
        boost::optional<const Tensor &> maskOpt;
        Tensor maskTensor;
        if (realTimeStepsFlag && params.outputFullSequence) {
          auto seqIdxAbsolute = (numShards > 1)
                                    ? add(graph, shardSeqIdx, seqIdx, loop,
                                          {dnai, "sequenceIndex"})
                                    : seqIdx;
          maskTensor =
              lt(graph, seqIdxAbsolute.expand({0}), seqLen, loop, {dnai});
          maskOpt = maskTensor;
        }

        boost::optional<const Tensor &> sliceAttScoresOpt(boost::none);
        boost::optional<Tensor &> sliceAttScoresGradsOpt(boost::none);
        if (attScores.valid()) {
          sliceAttScoresOpt = attScores;
          sliceAttScoresGradsOpt = attScoresGrads;
        }
        Tensor newOutGrad;
        Tensor bwdIntermediates;
        Tensor nextInputGrad;
        std::tie(newOutGrad, nextInputGrad, bwdIntermediates) =
            popnn::gru::basicGruBackwardStep(
                graph, params, gradLayerNextThisStepPtr, fwdIntermediates,
                prevStepOut, lastOutGrad, weightsRearranged, maskOpt,
                sliceAttScoresOpt, sliceAttScoresGradsOpt, loop, options,
                {dnai}, cache);
        if (inputGradSeq && inputGrad.valid()) {
          loop.add(Copy(nextInputGrad, inputGrad, false, {dnai}));
        }
        if (interimOut.valid()) {
          loop.add(Copy(bwdIntermediates, interimOut, false, {dnai}));
        }
        loop.add(Copy(newOutGrad, lastOutGrad, false, {dnai}));

        if (weightsGrad) {
          if (initProg != nullptr) {
            *weightsGrad =
                createWeightAccumulators(graph, weights, bwdIntermediates,
                                         options, {dnai}, params.resetAfter);
            zeroWeightAccumulators(graph, *initProg, *weightsGrad, {dnai});
          }
          basicGruParamUpdate(graph, prevLayerOut, prevStepOut,
                              fwdIntermediates, bwdIntermediates, *weightsGrad,
                              loop, options, {dnai}, cache, params.resetAfter);
        }
        return loop;
      };

  Tensor seqLen;
  if (realTimeStepsOpt) {
    seqLen = cast(graph, *realTimeStepsOpt, UNSIGNED_INT, prog,
                  {dnai, "realTimeStep"});
  }

  auto lastOutGradInit = rnn::createInitialState(
      graph, params.rnn, true, 1, numShards, {dnai, "lastOutGradInit"});
  Tensor gradLayerNextRearranged;
  if (params.outputFullSequence) {
    gradLayerNextRearranged = rnn::createOutputTensor(
        graph, params.rnn, numShards, {dnai, "gradLayerNextRearranged"});
    prog.add(Copy(gradLayerNext, gradLayerNextRearranged, false,
                  {dnai, "initGradLayerNextRearranged"}));
    zero(graph, lastOutGradInit, prog, {dnai, "zeroLastOutGrad"});
  } else {
    prog.add(Copy(gradLayerNext, lastOutGradInit, false, {dnai}));
  }
  Tensor attScores;
  std::vector<Tensor> attOutputs;
  if (attScoresOpt) {
    attScores = (*attScoresOpt).transpose().expand({(*attScoresOpt).rank()});
    *attScoresGrads =
        createAttention(graph, params, {dnai, "attScoresGrads"}, options_);
    attOutputs.push_back(attScoresGrads->transpose());
  }

  Tensor inputGradInit;
  rnn::StateSequence inputGrad;
  if (inputGradSeq) {
    inputGradInit = rnn::createInitialState(graph, params.rnn, false, 1,
                                            numShards, {dnai, "inputGradInit"});
    *inputGradSeq = rnn::createInputTensor(graph, params.rnn, numShards,
                                           {dnai, "inputGrad"});
    inputGrad = rnn::StateSequence{*inputGradSeq, 0};
  }
  if (bwdIntermediatesPtr) {
    *bwdIntermediatesPtr =
        rnn::createOutputTensor(graph, params.rnn, BASIC_GRU_CELL_NUM_UNITS,
                                numShards, {dnai, "bwdIntermediate"})
            .reshapePartial(0, 1,
                            {params.rnn.timeSteps, BASIC_GRU_CELL_NUM_UNITS});
    prog.add(WriteUndef(*bwdIntermediatesPtr, {dnai}));
  }
  Tensor prevLayerOut, prevStepOut;
  if (weightsGrad) {
    // make a copy of the activations so that they are sliced efficiently
    prevLayerOut =
        createInputTensor(graph, params.rnn, numShards, {dnai, "prevLayerOut"});
    prog.add(Copy(fwdInputSeq, prevLayerOut, false, {dnai}));
  }
  auto fwdOut =
      (params.outputFullSequence)
          ? fwdOutput
          : fwdIntermediatesSeq.dimRoll(1)[GRU_FWD_INTERMEDIATE_OUTPUT];
  prevStepOut = rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdOutputInit,
                                    prog, numShards, {dnai, "fwdOutshifted"});
  std::vector<Tensor> bwdStateInit = {inputGradInit, lastOutGradInit};
  std::vector<Tensor> bwdAndWuInputs = {gradLayerNextRearranged, prevLayerOut,
                                        prevStepOut, attScores};
  const auto shardingLoop = std::bind(
      loopBwdWithWU, weights, weightsGrad, seqLen, std::placeholders::_1,
      std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
      std::placeholders::_5, std::placeholders::_6, std::placeholders::_7,
      std::placeholders::_8, std::placeholders::_9, std::placeholders::_10,
      std::placeholders::_11);
  std::vector<Tensor> dummyOutputs;
  auto rnnOptions = getRnnOpts(options);
  auto updatedState = rnn::Rnn(
      graph, params.rnn, true, bwdStateInit, inputGrad, bwdAndWuInputs,
      &fwdIntermediatesSeq, bwdIntermediatesPtr, dummyOutputs, attOutputs, prog,
      shardingLoop, numShards, rnnOptions, {dnai, "updatedState"});
  if (weightsGrad) {
    *weightsGrad = basicGruParamUpdateFinal(graph, weights, *weightsGrad, prog,
                                            {dnai}, params.resetAfter);
  }
  return updatedState[1].squeeze({0});
}

Tensor gruBwd(Graph &graph, const GruParams &params, program::Sequence &prog,
              const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
              const GruWeights &weights_, const Tensor &fwdInputSeq,
              const Tensor &fwdOutput, const Tensor &gradLayerNext,
              Tensor *inputGrad, Tensor *bwdIntermediates,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options_,
              poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              fwdOutput, gradLayerNext, params, inputGrad, bwdIntermediates,
              options_, planningCache));

  validateParams(params);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);

  boost::optional<const Tensor &> realTimeStepsOpt(boost::none);
  boost::optional<const Tensor &> attScoresOpt(boost::none);
  auto output =
      gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                 weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                 bwdIntermediates, nullptr, realTimeStepsOpt, attScoresOpt,
                 nullptr, {di}, std::move(options_), planningCache);
  di.addOutput(output);
  return output;
}

Tensor gruBwd(Graph &graph, const GruParams &params, program::Sequence &prog,
              const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
              const GruWeights &weights_, const Tensor &fwdInputSeq,
              const Tensor &realTimeSteps, const Tensor &fwdOutput,
              const Tensor &gradLayerNext, Tensor *inputGrad,
              Tensor *bwdIntermediates,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options_,
              poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              realTimeSteps, fwdOutput, gradLayerNext, params, inputGrad,
              bwdIntermediates, options_, planningCache));

  validateParams(params);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);

  boost::optional<const Tensor &> realTimeStepsOpt(realTimeSteps);
  boost::optional<const Tensor &> attScoresOpt(boost::none);
  auto output =
      gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                 weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                 bwdIntermediates, nullptr, realTimeStepsOpt, attScoresOpt,
                 nullptr, {di}, std::move(options_), planningCache);
  di.addOutput(output);
  return output;
}

Tensor auGruBwd(Graph &graph, const GruParams &params, program::Sequence &prog,
                const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
                const GruWeights &weights_, const Tensor &fwdInputSeq,
                const Tensor &fwdOutput, const Tensor &gradLayerNext,
                Tensor *inputGrad, Tensor *bwdIntermediates,
                const Tensor &attentions, Tensor *attentionsGrad,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options_,
                poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              fwdOutput, gradLayerNext, inputGrad, bwdIntermediates, attentions,
              attentionsGrad, params, options_, planningCache));

  validateParams(params);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);

  boost::optional<const Tensor &> timeSteps;
  boost::optional<const Tensor &> attScores(attentions);
  return gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                    weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                    bwdIntermediates, nullptr, timeSteps, attScores,
                    attentionsGrad, {di}, std::move(options_), planningCache);
}

Tensor auGruBwd(Graph &graph, const GruParams &params, program::Sequence &prog,
                const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
                const GruWeights &weights_, const Tensor &fwdInputSeq,
                const Tensor &realTimeSteps, const Tensor &fwdOutput,
                const Tensor &gradLayerNext, Tensor *inputGrad,
                Tensor *bwdIntermediates, const Tensor &attentions,
                Tensor *attentionsGrad,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options_,
                poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              realTimeSteps, fwdOutput, gradLayerNext, inputGrad,
              bwdIntermediates, attentions, attentionsGrad, params, options_,
              planningCache));
  validateParams(params);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);

  boost::optional<const Tensor &> timeSteps(realTimeSteps);
  boost::optional<const Tensor &> attScores(attentions);
  auto output =
      gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                 weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                 bwdIntermediates, nullptr, timeSteps, attScores,
                 attentionsGrad, {di}, std::move(options_), planningCache);
  di.addOutput(output);
  return output;
}

static GruWeights
gruWUImpl(Graph &graph, const GruParams &params, program::Sequence &prog,
          const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
          const Tensor &bwdIntermediatesSeq, const GruWeights &weights,
          const Tensor &input, const Tensor &output, const DebugNameAndId &dnai,
          const GruOpts &options,
          poplin::matmul::PlanningCache *planningCache) {
  auto loopWU = [&params, &options, &planningCache](
                    GruWeights &weightGrads, Graph &graph,
                    const Tensor &shardSeqIdx, const Tensor &seqIdx,
                    std::vector<Tensor> &shardState,
                    const std::vector<Tensor> &wuInput,
                    const Tensor &fwdIntermediates, Tensor &interimOut,
                    std::vector<Tensor> &outputs, std::vector<Tensor> &created,
                    program::Sequence *initProg, const DebugNameAndId &dnai) {
    auto loop = Sequence{{}, {dnai}};
    auto &prevLayerOut = wuInput[0];
    auto &prevStepOut = wuInput[1];
    auto &bwdIntermediates = wuInput[2];
    basicGruParamUpdate(graph, prevLayerOut, prevStepOut, fwdIntermediates,
                        bwdIntermediates, weightGrads, loop, options, {dnai},
                        planningCache, params.resetAfter);
    return loop;
  };

  GruWeights weightGrads =
      createWeightAccumulators(graph, weights, bwdIntermediatesSeq[0], options,
                               {dnai}, params.resetAfter);
  zeroWeightAccumulators(graph, prog, weightGrads, {dnai});

  // make a copy of the activations so that they are sliced efficiently
  auto numShards = getNumShards(graph, params, options, {dnai, "numShards"});
  Tensor inputCopy =
      createInputTensor(graph, params.rnn, numShards, {dnai, "inputCopy"});
  prog.add(Copy(input, inputCopy, false, {dnai}));
  auto fwdOut =
      (params.outputFullSequence)
          ? output
          : fwdIntermediatesSeq.dimRoll(1)[GRU_FWD_INTERMEDIATE_OUTPUT];
  Tensor prevStepOut =
      rnn::shiftRnnTensor(graph, params.rnn, fwdOut, fwdOutputInit, prog,
                          numShards, {dnai, "fwdOutshifted"});
  std::vector<Tensor> wuInputs = {inputCopy, prevStepOut, bwdIntermediatesSeq};
  const auto shardingLoop = std::bind(
      loopWU, weightGrads, std::placeholders::_1, std::placeholders::_2,
      std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
      std::placeholders::_6, std::placeholders::_7, std::placeholders::_8,
      std::placeholders::_9, std::placeholders::_10, std::placeholders::_11);
  std::vector<Tensor> dummyOutputs;
  std::vector<Tensor> dummyCreated;
  auto rnnOptions = getRnnOpts(options);
  auto updatedState =
      rnn::Rnn(graph, params.rnn, true, {}, {}, wuInputs, &fwdIntermediatesSeq,
               nullptr, dummyOutputs, dummyCreated, prog, shardingLoop,
               numShards, rnnOptions, {dnai, "rnn"});
  weightGrads = basicGruParamUpdateFinal(graph, weights, weightGrads, prog,
                                         {dnai}, params.resetAfter);
  return weightGrads;
}

GruWeights gruWU(Graph &graph, const GruParams &params, program::Sequence &prog,
                 const Tensor &fwdOutputInit, const Tensor &fwdIntermediates,
                 const Tensor &bwdIntermediates, const GruWeights &weights_,
                 const Tensor &input, const Tensor &output,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options_,
                 poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, bwdIntermediates, weights_,
              input, output, params, options_, planningCache));

  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());
  validateParams(params);
  auto options = parseOptions(options_);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  auto grads = gruWUImpl(graph, params, prog, fwdOutputInit, fwdIntermediates,
                         bwdIntermediates, weights, input, output, {di},
                         std::move(options), planningCache);
  auto outputs = toCellOrder(std::move(grads), params.cellOrder);
  di.addOutputs(DI_ARGS(outputs));
  return outputs;
}

GruWeights augruWU(Graph &graph, const GruParams &params,
                   program::Sequence &prog, const Tensor &fwdOutputInit,
                   const Tensor &fwdIntermediates,
                   const Tensor &bwdIntermediates, const GruWeights &weights_,
                   const Tensor &input, const Tensor &output,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_,
                   poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, bwdIntermediates, weights_,
              input, output, params, options_, planningCache));
  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());

  validateParams(params);
  auto options = parseOptions(options_);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  auto grads = gruWUImpl(graph, params, prog, fwdOutputInit, fwdIntermediates,
                         bwdIntermediates, weights, input, output, {di},
                         std::move(options), planningCache);
  auto outputs = toCellOrder(std::move(grads), params.cellOrder);
  di.addOutputs(DI_ARGS(outputs));

  return outputs;
}

// Is it beneficial memory-wise to interleave weight update with
// backwards pass.
static bool interleavedWUIsBeneficial(const GruParams &params) {
  const auto batchSize = params.rnn.batchSize;
  const auto inputSize = params.rnn.layerSizes[0];
  const auto outputSize = params.rnn.layerSizes[1];
  // Total elements needed for transposed weights.
  const auto totalTransposeParams =
      (inputSize + outputSize) * outputSize * BASIC_GRU_CELL_NUM_UNITS;
  // Total elements needed for unit gradients for weight update if
  // not interleaved with backpropagation.
  const auto totalBwdIntermediates =
      batchSize * outputSize * BASIC_GRU_CELL_NUM_UNITS * params.rnn.timeSteps;
  return totalTransposeParams <= totalBwdIntermediates;
}

Tensor
gruBwdWithWU(poplar::Graph &graph, const GruParams &params,
             poplar::program::Sequence &prog, const Tensor &fwdOutputInit,
             const poplar::Tensor &fwdIntermediates, const GruWeights &weights_,
             const poplar::Tensor &input, const poplar::Tensor &output,
             const poplar::Tensor &outputGrad, poplar::Tensor *inputGrad,
             GruWeights &weightsGrad_, const poplar::DebugContext &debugContext,
             const poplar::OptionFlags &options_,
             poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input,
                            output, outputGrad, inputGrad, weightsGrad_, params,
                            options_, planningCache));
  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());
  validateParams(params);
  auto options = parseOptions(options_);

  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);
  GruWeights weightsGrad;

  bool interleaveWU = interleavedWUIsBeneficial(params);
  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  boost::optional<const Tensor &> realTimeSteps;
  boost::optional<const Tensor &> attScores;

  Tensor outGrads = gruBwdImpl(
      graph, params, prog, fwdOutputInit, fwdIntermediates, weights, input,
      output, outputGrad, inputGrad, interleaveWU ? nullptr : &bwdIntermediates,
      interleaveWU ? &weightsGrad : nullptr, realTimeSteps, attScores, nullptr,
      {di}, options_, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(graph, params, prog, fwdOutputInit,
                            fwdIntermediates, bwdIntermediates, weights, input,
                            output, {di}, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
  di.addOutput(outGrads);
  return outGrads;
}

Tensor
gruBwdWithWU(poplar::Graph &graph, const GruParams &params,
             poplar::program::Sequence &prog, const Tensor &fwdOutputInit,
             const poplar::Tensor &fwdIntermediates, const GruWeights &weights_,
             const poplar::Tensor &input, const poplar::Tensor &realTimeSteps,
             const poplar::Tensor &output, const poplar::Tensor &outputGrad,
             poplar::Tensor *inputGrad, GruWeights &weightsGrad_,
             const poplar::DebugContext &debugContext,
             const poplar::OptionFlags &options_,
             poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input,
                            realTimeSteps, output, outputGrad, inputGrad,
                            weightsGrad_, params, options_, planningCache));

  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());
  validateParams(params);
  auto options = parseOptions(options_);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);
  GruWeights weightsGrad;

  bool interleaveWU = interleavedWUIsBeneficial(params);
  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  boost::optional<const Tensor &> timeSteps(realTimeSteps);
  boost::optional<const Tensor &> attScores;
  Tensor outGrads = gruBwdImpl(
      graph, params, prog, fwdOutputInit, fwdIntermediates, weights, input,
      output, outputGrad, inputGrad, interleaveWU ? nullptr : &bwdIntermediates,
      interleaveWU ? &weightsGrad : nullptr, timeSteps, attScores, nullptr,
      {di}, options_, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(graph, params, prog, fwdOutputInit,
                            fwdIntermediates, bwdIntermediates, weights, input,
                            output, {di}, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
  di.addOutput(outGrads);
  return outGrads;
}

Tensor
auGruBwdWithWU(poplar::Graph &graph, const GruParams &params,
               poplar::program::Sequence &prog, const Tensor &fwdOutputInit,
               const poplar::Tensor &fwdIntermediates,
               const GruWeights &weights_, const poplar::Tensor &input,
               const poplar::Tensor &output, const poplar::Tensor &outputGrad,
               poplar::Tensor *inputGrad, GruWeights &weightsGrad_,
               const poplar::Tensor &attentions, poplar::Tensor *attentionsGrad,
               const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options_,
               poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input, output,
              outputGrad, inputGrad, weightsGrad_, attentions, attentionsGrad,
              params, options_, planningCache));

  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());
  validateParams(params);
  auto options = parseOptions(options_);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);
  GruWeights weightsGrad;

  bool interleaveWU = interleavedWUIsBeneficial(params);
  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  boost::optional<const Tensor &> timeSteps;
  boost::optional<const Tensor &> attScores(attentions);
  Tensor outGrads = gruBwdImpl(
      graph, params, prog, fwdOutputInit, fwdIntermediates, weights, input,
      output, outputGrad, inputGrad, interleaveWU ? nullptr : &bwdIntermediates,
      interleaveWU ? &weightsGrad : nullptr, timeSteps, attScores,
      attentionsGrad, {di}, options_, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(graph, params, prog, fwdOutputInit,
                            fwdIntermediates, bwdIntermediates, weights, input,
                            output, {di}, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
  di.addOutput(outGrads);
  return outGrads;
}

Tensor
auGruBwdWithWU(poplar::Graph &graph, const GruParams &params,
               poplar::program::Sequence &prog, const Tensor &fwdOutputInit,
               const poplar::Tensor &fwdIntermediates,
               const GruWeights &weights_, const poplar::Tensor &input,
               const poplar::Tensor &realTimeSteps,
               const poplar::Tensor &output, const poplar::Tensor &outputGrad,
               poplar::Tensor *inputGrad, GruWeights &weightsGrad_,
               const poplar::Tensor &attentions, poplar::Tensor *attentionsGrad,
               const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options_,
               poplin::matmul::PlanningCache *planningCache) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input, realTimeSteps,
              output, outputGrad, inputGrad, weightsGrad_, attentions,
              attentionsGrad, params, options_, planningCache));

  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.rnn.timeSteps, params.rnn.batchSize,
                       params.rnn.layerSizes, debugContext.getPathName());
  validateParams(params);
  auto options = parseOptions(options_);
  if (bool(inputGrad) != params.calcInputGradients) {
    throw poplibs_error(std::string("The inputGradSeq argument should be ") +
                        (inputGrad ? "non null" : "null") +
                        " if and only if params.calcInputGradients is " +
                        (inputGrad ? "true" : "false"));
  }

  auto weights = fromCellOrder(weights_, params.cellOrder);
  GruWeights weightsGrad;

  bool interleaveWU = interleavedWUIsBeneficial(params);
  Tensor bwdIntermediates;

  // Perform the backward pass. If interleaving the weight update with the
  // backward pass is beneficial, directly calculate the weight gradients
  // during the backward pass. Otherwise, save backward intermediates and
  // calculate weight deltas below.
  boost::optional<const Tensor &> timeSteps(realTimeSteps);
  boost::optional<const Tensor &> attScores(attentions);
  Tensor outGrads = gruBwdImpl(
      graph, params, prog, fwdOutputInit, fwdIntermediates, weights, input,
      output, outputGrad, inputGrad, interleaveWU ? nullptr : &bwdIntermediates,
      interleaveWU ? &weightsGrad : nullptr, timeSteps, attScores,
      attentionsGrad, {di}, options_, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(graph, params, prog, fwdOutputInit,
                            fwdIntermediates, bwdIntermediates, weights, input,
                            output, {di}, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
  di.addOutput(outGrads);
  return outGrads;
}

uint64_t getBasicGruCellFwdFlops(const GruParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.
  uint64_t multsWeighInp = static_cast<uint64_t>(inputSize) * 3 * outputSize *
                           batchSize * sequenceSize * 2;

  uint64_t multsWeighOut = static_cast<uint64_t>(outputSize) * 3 * outputSize *
                           batchSize * sequenceSize * 2;

  // We ignore FLOPs for bias addition - in theory we could initialize the
  // accumulators with the biases during the matrix multipliciation.
  uint64_t mulFlops =
      3 * static_cast<uint64_t>(sequenceSize) * batchSize * outputSize;
  uint64_t addFlops =
      static_cast<uint64_t>(sequenceSize) * batchSize * outputSize * 2;
  return multsWeighInp + multsWeighOut + addFlops + mulFlops;
}

uint64_t getBasicGruCellBwdFlops(const GruParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];
  auto calcInputGrad = params.calcInputGradients;
  // Note we ignore FLOPs for non linearities - this is consistent with how
  // FLOPs are reported for other operations.
  uint64_t addFlops =
      static_cast<uint64_t>(sequenceSize) * 5 * batchSize * outputSize;

  uint64_t mulFlops =
      static_cast<uint64_t>(sequenceSize) * 3 * batchSize * outputSize;
  uint64_t inputGradFlops = calcInputGrad
                                ? static_cast<uint64_t>(inputSize) * 3 *
                                      outputSize * batchSize * sequenceSize * 2
                                : 0;
  uint64_t outputGradFlops = static_cast<uint64_t>(outputSize) * 6 *
                             outputSize * batchSize * sequenceSize * 2;
  return addFlops + mulFlops + inputGradFlops + outputGradFlops;
}

uint64_t getBasicGruCellWuFlops(const GruParams &params) {
  auto batchSize = params.rnn.batchSize;
  auto sequenceSize = params.rnn.timeSteps;
  auto inputSize = params.rnn.layerSizes[0];
  auto outputSize = params.rnn.layerSizes[1];

  uint64_t weightFlops = static_cast<uint64_t>(inputSize + outputSize) * 3 *
                         outputSize * batchSize * sequenceSize * 2;
  uint64_t biasFlops =
      static_cast<uint64_t>(outputSize) * 3 * batchSize * sequenceSize * 2;
  return weightFlops + biasFlops;
}

} // namespace gru
} // namespace popnn
