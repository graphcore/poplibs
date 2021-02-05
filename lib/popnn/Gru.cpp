// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popnn/Gru.hpp>
#include <popops/Cast.hpp>

#include "RnnUtil.hpp"
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
  poplar::Type dataType;
  v.insert({"batchSize", toProfileValue(t.batchSize)});
  v.insert({"timeSteps", toProfileValue(t.timeSteps)});
  v.insert({"layerSizes", toProfileValue(t.layerSizes)});
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
                     std::size_t timeSteps, std::vector<std::size_t> layerSizes)
    : dataType(std::move(dataType)), batchSize(batchSize), timeSteps(timeSteps),
      layerSizes(std::move(layerSizes)) {}

GruParams::GruParams(const GruParams &other) = default;

struct GruOpts {
  bool inferenceOnly;
  poplar::Type partialsType;
  boost::optional<double> availableMemoryProportion;
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

static GruOpts parseOptions(const OptionFlags &options) {
  GruOpts gruOpts;
  gruOpts.inferenceOnly = true;
  gruOpts.partialsType = poplar::FLOAT;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec gruSpec{
      {"inferenceOnly", OptionHandler::createWithBool(gruOpts.inferenceOnly)},
      {"partialsType",
       OptionHandler::createWithEnum(gruOpts.partialsType, partialsTypeMap)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(gruOpts.availableMemoryProportion)},
  };
  for (const auto &entry : options) {
    gruSpec.parse(entry.first, entry.second);
  }
  return gruOpts;
}

static void validateParams(const GruParams &params) {
  if (params.layerSizes.size() != 2) {
    throw poplibs_error("Invalid GRU params (layerSize != 2)");
  }
}

const std::vector<BasicGruCellUnit> getDefaultBasicGruCellOrder() {
  return {BASIC_GRU_CELL_RESET_GATE, BASIC_GRU_CELL_UPDATE_GATE,
          BASIC_GRU_CELL_CANDIDATE};
}

static Tensor createOutputTensor(Graph &graph, const GruParams &params,
                                 unsigned sequenceLength,
                                 const poplar::DebugNameAndId &dnai) {
  const auto outputSize = params.layerSizes[1];
  const auto batchSize = params.batchSize;
  const auto outputGrouping = gcd(16UL, outputSize);
  const auto numGroups = (outputSize * batchSize) / outputGrouping;
  auto output =
      createDynamicSliceTensor(graph, params.dataType, sequenceLength,
                               numGroups, outputGrouping, {dnai})
          .reshapePartial(1, 2, {outputSize / outputGrouping, batchSize})
          .dimRoll(1, 2)
          .flatten(2, 4);
  return output;
}

Tensor createInput(Graph &graph, const GruParams &params,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options,
                   poplin::matmul::PlanningCache *planningCache) {

  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, options, planningCache));

  validateParams(params);

  auto inputSize = params.layerSizes[0];
  const auto batchSize = params.batchSize;
  const auto inputGrouping = gcd(16UL, inputSize);
  const auto numInputGroups = (inputSize * batchSize) / inputGrouping;
  auto in = createDynamicSliceTensor(graph, params.dataType, params.timeSteps,
                                     numInputGroups, inputGrouping, {di});
  auto output = in.reshapePartial(1, 2, {inputSize / inputGrouping, batchSize})
                    .dimRoll(1, 2)
                    .flatten(2, 4);

  di.addOutput(output);
  return output;
}

Tensor createInitialState(Graph &graph, const GruParams &params,
                          const poplar::DebugContext &debugContext,
                          const OptionFlags &options,
                          matmul::PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  auto output =
      createOutputTensor(graph, params, 1, {di, "initialOutput"}).squeeze({0});
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
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  validateParams(params);
  auto opt = parseOptions(options);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];

  poplar::Tensor inputWeights, outputWeights;

  if (params.resetAfter) {
    auto weights_in = createMatMulInputRHS(
        graph, params.dataType, {params.batchSize, inputSize},
        {inputSize, BASIC_GRU_CELL_NUM_UNITS * outputSize}, {di, "weights"},
        mmOpt, cache);
    auto weights_out = createMatMulInputRHS(
        graph, params.dataType, {params.batchSize, outputSize},
        {outputSize, BASIC_GRU_CELL_NUM_UNITS * outputSize}, {di, "weights"},
        mmOpt, cache);

    inputWeights = unflattenUnits(weights_in, BASIC_GRU_CELL_NUM_UNITS);
    outputWeights = unflattenUnits(weights_out, BASIC_GRU_CELL_NUM_UNITS);
  } else {
    auto weights_ru = createMatMulInputRHS(
        graph, params.dataType, {params.batchSize, inputSize + outputSize},
        {inputSize + outputSize, 2 * outputSize}, {di, "weights"}, mmOpt,
        cache);
    poplar::Tensor inputWeights_ru =
        unflattenUnits(weights_ru.slice(0, inputSize), 2);
    poplar::Tensor outputWeights_ru =
        unflattenUnits(weights_ru.slice(inputSize, inputSize + outputSize), 2);

    auto weights_c = createMatMulInputRHS(
        graph, params.dataType, {params.batchSize, inputSize + outputSize},
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
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(params, options, planningCache));

  validateParams(params);
  auto outputSize = params.layerSizes[1];
  Tensor biases;
  if (params.resetAfter) {
    biases = graph.addVariable(params.dataType,
                               {BASIC_GRU_CELL_NUM_UNITS, 2, outputSize},
                               {di, "biases"});
  } else {
    biases = graph.addVariable(params.dataType,
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
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, options, cache));

  GruWeights gruWeights;
  std::tie(gruWeights.inputWeights, gruWeights.outputWeights) =
      createWeightsKernel(graph, params, {di}, options, cache);
  gruWeights.biases = createWeightsBiases(graph, params, {di}, options, cache);
  di.addOutputs(DI_ARGS(gruWeights));
  return gruWeights;
}

Tensor createAttention(Graph &graph, const GruParams &params,
                       const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params));

  validateParams(params);

  auto output = createSliceableTensor(graph, params.dataType,
                                      {params.batchSize, params.timeSteps}, {1},
                                      {1}, 0, {di});
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
    const Tensor &unitsOutputRearranged, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
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
  auto cs = graph.addComputeSet({dnai, "non-linear"});

  if (forCandidate) {
    nonLinearityInPlace(graph, popnn::NonLinearityType::TANH,
                        unitsOutputRearranged, cs, {dnai, "Candidate Tanh"});
  } else {
    nonLinearityInPlace(graph, popnn::NonLinearityType::SIGMOID,
                        unitsOutputRearranged, cs,
                        {dnai, "update/reset sigmod"});
  }
  prog.add(Execute(cs, {dnai}));
}

static void gruCellForwardPassCalcUnitsResetAfter(
    Graph &graph, bool forCandidate, const Tensor &in, const Tensor &prevOutput,
    const Tensor &biases, const Tensor *weightsInput,
    const Tensor &weightsOutput, Tensor *candidateRecurrant, Sequence &prog,
    const GruOpts &opt, const Tensor &unitsOutputRearranged,
    const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
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
  nonLinearityInPlace(graph, popnn::NonLinearityType::SIGMOID,
                      unitsOutputRearranged_ru, prog,
                      {dnai, "update/reset sigmod"});

  // Apply reset gate to candidate recurrent component
  mulInPlace(graph, recurrentUnitsOutputRearranged_c, unitsOutputRearranged_r,
             prog, {dnai, "ch * r"});

  // Add recurrent component for Candidate
  addInPlace(graph, unitsOutputRearranged_c, recurrentUnitsOutputRearranged_c,
             prog, {dnai, "Weights"});

  // Apply non linear function to Candidate
  nonLinearityInPlace(graph, popnn::NonLinearityType::TANH,
                      unitsOutputRearranged_c, prog, {dnai, "Candidate Tanh"});
}

static std::pair<Tensor, GruInternalState>
basicGruCellForwardPass(Graph &graph, const Tensor &in, const Tensor &biases,
                        const Tensor &prevOutput, const Tensor *weightsInput,
                        const Tensor &weightsOutput,
                        const boost::optional<const Tensor &> &attScoreOpt,
                        Sequence &prog, const GruOpts &opt,
                        const DebugNameAndId &dnai,
                        matmul::PlanningCache *cache, bool resetAfter) {
  debug_tensor(prog, "fwd h_prev", prevOutput);
  debug_tensor(prog, "fwd input", in);

  const std::string baseStr = "BasicGruCell";

  Tensor resetGate = graph.clone(prevOutput, {dnai, "Update Gate Rearranged"});
  Tensor updateGate = graph.clone(prevOutput, {dnai, "Reset Gate Rearranged"});
  Tensor candidate = graph.clone(prevOutput, {dnai, "candidate Rearranged"});

  Tensor candidateRecurrant;

  std::vector<Tensor> toConcat;
  toConcat.reserve(resetAfter ? BASIC_GRU_CELL_NUM_UNITS
                              : BASIC_GRU_CELL_NUM_UNITS - 1);

  toConcat.push_back(resetGate.expand({0}));
  toConcat.push_back(updateGate.expand({0}));
  if (resetAfter)
    toConcat.push_back(candidate.expand({0}));

  auto unitsOutput = concat(toConcat);

  if (resetAfter) {
    gruCellForwardPassCalcUnitsResetAfter(
        graph, false, in, prevOutput, biases, weightsInput, weightsOutput,
        &candidateRecurrant, prog, opt, unitsOutput, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS);
  } else {
    const Tensor weightsInput2 = weightsInput->slice(0, 2);
    const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
    const Tensor biases2 = biases.slice(0, 2);
    gruCellForwardPassCalcUnits(graph, false, in, prevOutput, biases2,
                                &weightsInput2, weightsOutput2, prog, opt,
                                unitsOutput, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS - 1);
  }

  resetGate = unitsOutput[BASIC_GRU_CELL_RESET_GATE];
  updateGate = unitsOutput[BASIC_GRU_CELL_UPDATE_GATE];

  Tensor resetGateOut;

  if (resetAfter) {
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
    gruCellForwardPassCalcUnits(graph, true, in, resetGate, biases,
                                &weightsInput3, weightsOutput3, prog, opt,
                                candidateExpand, {dnai, baseStr}, cache);
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
    const GruOpts &opt, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache, bool resetAfter) {
  logging::popnn::info("basicGruCellForwardPassInPlace {}", in.shape());

  debug_tensor(prog, "fwd h_prev", output);
  debug_tensor(prog, "fwd input", in);
  const std::string baseStr = "BasicGruCellInPlace";

  Tensor resetGate = graph.clone(output, {dnai, "Update Gate Rearranged"});
  Tensor updateGate = graph.clone(output, {dnai, "Reset Gate Rearranged"});
  Tensor candidate = graph.clone(output, {dnai, "candidate Rearranged"});

  int numToConcat =
      resetAfter ? BASIC_GRU_CELL_NUM_UNITS : BASIC_GRU_CELL_NUM_UNITS - 1;
  std::vector<Tensor> toConcat;
  toConcat.reserve(numToConcat);

  toConcat.push_back(resetGate.expand({0}));
  toConcat.push_back(updateGate.expand({0}));
  if (resetAfter)
    toConcat.push_back(candidate.expand({0}));

  auto unitsOutput = concat(toConcat);

  if (resetAfter) {
    gruCellForwardPassCalcUnitsResetAfter(
        graph, false, in, output, biases, weightsInput, weightsOutput, nullptr,
        prog, opt, unitsOutput, {dnai, baseStr}, cache);
    assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS);
  } else {
    const Tensor weightsInput2 = weightsInput->slice(0, 2);
    const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
    gruCellForwardPassCalcUnits(graph, false, in, output, biases,
                                &weightsInput2, weightsOutput2, prog, opt,
                                unitsOutput, {dnai, baseStr}, cache);
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

  if (resetAfter) {
    candidate = unitsOutput[BASIC_GRU_CELL_CANDIDATE];
  } else {
    const Tensor weightsInput3 = weightsInput->slice(2, 3);
    const Tensor weightsOutput3 = weightsOutput.slice(2, 3);
    mulInPlace(graph, resetGate, output, prog,
               {dnai, baseStr + "resetGate * output"});
    Tensor candidateExpand = candidate.expand({0});
    gruCellForwardPassCalcUnits(graph, true, in, resetGate, biases,
                                &weightsInput3, weightsOutput3, prog, opt,
                                candidateExpand, {dnai, baseStr}, cache);
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
                       params.timeSteps, params.batchSize, params.layerSizes,
                       dnai.getPathName());

  validateParams(params);
  auto opt = parseOptions(options);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  Tensor output = duplicate(graph, fwdOutputInit, fwdProg, {dnai, "fwdOutput"});

  // loop counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "seqIdx"});
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, fwdProg, {dnai, "initSeqIdx"});

  unsigned seqSize = prevLayerActs.dim(0);
  // make a copy of the activations so that they are sliced efficiently
  auto prevLayerActsCopy =
      createInput(graph, params, {dnai, "prevLayerActsCopy"}, options, cache);
  fwdProg.add(Copy(prevLayerActs, prevLayerActsCopy, false, {dnai}));

  Tensor mask;
  Tensor seqLen;
  if (realTimeStepsOpt) {
    seqLen = cast(graph, *realTimeStepsOpt, UNSIGNED_INT, fwdProg);
  }

  // core loop
  auto loop = Sequence();

  Tensor attScores;
  boost::optional<const Tensor &> sliceAttScoresOpt(boost::none);
  if (attScoresOpt) {
    attScores = popops::dynamicSlice(
        graph, (*attScoresOpt).transpose().expand({(*attScoresOpt).rank()}),
        seqIdx, {0}, {1}, loop, {dnai, "auGruAttScores"})[0];
    sliceAttScoresOpt = attScores;
  }

  Tensor fwdInput = popops::dynamicSlice(graph, prevLayerActsCopy, seqIdx, {0},
                                         {1}, loop, {dnai, "gru"})[0];
  const Tensor *inputWeightsPtr = &weights.inputWeights;

  debug_tensor(fwdProg, "fwd weightsInput", weights.inputWeights);
  debug_tensor(fwdProg, "fwd weightsOutput", weights.outputWeights);
  debug_tensor(fwdProg, "fwd bias", weights.biases);
  debug_tensor(loop, "fwd Loop:", seqIdx);
  if (intermediatesSeq) {
    Tensor newOutput;
    GruInternalState internalState;
    std::tie(newOutput, internalState) = basicGruCellForwardPass(
        graph, fwdInput, weights.biases, output, inputWeightsPtr,
        weights.outputWeights, sliceAttScoresOpt, loop, opt, {dnai}, cache,
        params.resetAfter);
    if (realTimeStepsOpt) {
      mask = gt(graph, seqLen, seqIdx, loop);
      mask = cast(graph, mask, newOutput.elementType(), loop);
      mask = mask.expand({1}).broadcast(newOutput.dim(1), 1);
      mapInPlace(graph, _1 * _2, {newOutput, mask}, loop, {dnai});
      mapInPlace(graph, _1 * _2, {internalState.resetGate, mask}, loop, {dnai});
      mapInPlace(graph, _1 * _2, {internalState.updateGate, mask}, loop,
                 {dnai});
      mapInPlace(graph, _1 * _2, {internalState.candidate, mask}, loop, {dnai});
    }

    std::vector<Tensor> intermediatesToConcat;

    int numberToConcat = 3;
    if (!params.outputFullSequence)
      numberToConcat += 1;
    if (params.resetAfter)
      numberToConcat += 1;
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

    const auto numIntermediates = intermediates.dim(0);
    *intermediatesSeq =
        createOutputTensor(graph, params, seqSize * numIntermediates,
                           {dnai, "fwdIntermediatesSeq"})
            .reshapePartial(0, 1, {seqSize, numIntermediates});

    auto intermediatesRearranged = createOutputTensor(
        graph, params, numIntermediates, {dnai, "fwdIntermediatesRearranged"});
    loop.add(Copy(intermediates, intermediatesRearranged, false, {dnai}));
    fwdProg.add(WriteUndef(*intermediatesSeq, {dnai}));
    popops::dynamicUpdate(graph, *intermediatesSeq,
                          intermediatesRearranged.expand({0}), seqIdx, {0}, {1},
                          loop, {dnai, "gruUpdateIntermediates"});

    graph.setTileMapping(output, graph.getTileMapping(newOutput));
    loop.add(Copy(newOutput, output, false, {dnai}));
  } else {
    basicGruCellForwardPassInPlace(graph, fwdInput, weights.biases, output,
                                   inputWeightsPtr, weights.outputWeights,
                                   sliceAttScoresOpt, loop, opt, {dnai}, cache,
                                   params.resetAfter);

    if (realTimeStepsOpt) {
      mask = gt(graph, seqLen, seqIdx, loop);
      mask = cast(graph, mask, output.elementType(), loop);
      mask = mask.expand({1}).broadcast(output.dim(1), 1);
      mapInPlace(graph, _1 * _2, {output, mask}, loop, {dnai});
    }
  }

  Tensor outputSeq;
  if (params.outputFullSequence) {
    outputSeq = createOutputTensor(graph, params, seqSize, {dnai, "Output"});
    fwdProg.add(WriteUndef(outputSeq, {dnai}));
    popops::dynamicUpdate(graph, outputSeq, output.expand({0}), seqIdx, {0},
                          {1}, loop, {dnai, "updateOutputSeq"});
  }

  addInPlace(graph, seqIdx, one, loop, {dnai, "seqIdxIncr"});

  fwdProg.add(Repeat(seqSize, loop, {dnai}));

  return params.outputFullSequence ? outputSeq : output;
}

Tensor gruFwd(Graph &graph, const GruParams &params,
              const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
              const GruWeights &weights_, Tensor *intermediatesSeq,
              program::Sequence &fwdProg,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {

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

static std::tuple<Tensor, Tensor, Tensor>
backwardStepImpl(Graph &graph, const Tensor *gradNextLayer,
                 const Tensor &fwdIntermediates, const Tensor &prevStepOut,
                 const Tensor &outputGrad, const Tensor *weightsInput,
                 const Tensor &weightsOutput,
                 const boost::optional<const Tensor &> &maskOpt,
                 const boost::optional<const Tensor &> &attScoresOpt,
                 const boost::optional<Tensor &> &attScoresGradsOpt,
                 Sequence &initProg, Sequence &prog, const GruOpts &opt,
                 const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
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

  auto cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
  auto d_c = nonLinearityInputGradient(graph, NonLinearityType::TANH, c,
                                       gradAtCandidateInput, cs1,
                                       {dnai, fPrefix + "/OutputTanh"});

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

    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u0,
                                    gradAtUpdateGateInput, cs1,
                                    {dnai, fPrefix + "/OutputGate"});
  } else {
    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u,
                                    gradAtUpdateGateInput, cs1,
                                    {dnai, fPrefix + "/OutputGate"});
  }
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

  Tensor w_ru, w_c;
  if (weightsInput == nullptr) {
    w_ru =
        flattenUnits(concat(weightsOutput[0], weightsOutput[1], 1)).transpose();
    w_c = flattenUnits(weightsOutput[2]).transpose();
  } else {
    w_ru = concat(concat((*weightsInput)[0], weightsOutput[0], 0),
                  concat((*weightsInput)[1], weightsOutput[1], 0), 1)
               .transpose();
    w_c = concat((*weightsInput)[2], weightsOutput[2], 0).transpose();
  }
  w_c = preArrangeMatMulInputRHS(graph, d_c.shape(), w_c, initProg,
                                 {dnai, fPrefix + "/PreArrangeWeights C "},
                                 mmOpt, cache);

  Tensor d_x2, d_hr, d_x2_hr;
  int inputSize = weightsInput->dim(1);
  int outputSize = weightsOutput.dim(1);
  if (weightsInput) {
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
    d_r = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, r, t,
                                    prog, {dnai, fPrefix + "/t * r * (1-r)"});
  }

  Tensor d_r_d_u = concat(d_r, d_u, 1);
  w_ru = preArrangeMatMulInputRHS(graph, d_r_d_u.shape(), w_ru, initProg,
                                  {dnai, fPrefix + "/PreArrangeWeights RU"},
                                  mmOpt, cache);
  auto out = matMul(graph, d_r_d_u, w_ru, prog,
                    {dnai, fPrefix + "/d_x1_d_h_prev1 X w_ru"}, mmOpt, cache);
  Tensor d_x1_d_hprev1 = tryGroupedPartialTranspose(
      graph, out, outputGroupingIntoLayer, prog, {dnai, fPrefix});
  debug_tensor(prog, "bwd d_x1_d_hprev1", d_x1_d_hprev1);

  Tensor d_x;
  if (weightsInput) {
    d_x = add(graph, d_x1_d_hprev1.slice(0, inputSize, 1), d_x2, prog,
              {dnai, fPrefix + "/dx"});
  }

  Tensor d_hprev1 = d_x1_d_hprev1.slice(inputSize, inputSize + outputSize, 1);
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
    const Tensor &prevStepOut, const Tensor &outputGrad,
    const Tensor *weightsInput, const Tensor &weightsOutput,
    const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoresOpt,
    const boost::optional<Tensor &> &attScoresGradsOpt, Sequence &initProg,
    Sequence &prog, const GruOpts &opt, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache, bool outputFullSequence) {
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
  if (outputFullSequence) {
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

  auto cs1 = graph.addComputeSet({dnai, fPrefix + "/OutputGate"});
  auto d_c = nonLinearityInputGradient(graph, NonLinearityType::TANH, c,
                                       gradAtCandidateInput, cs1,
                                       {dnai, fPrefix + "/OutputTanh"});
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

    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u0,
                                    gradAtUpdateGateInput, cs1,
                                    {dnai, fPrefix + "/OutputGate"});
  } else {
    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u,
                                    gradAtUpdateGateInput, cs1,
                                    {dnai, fPrefix + "/OutputGate"});
  }
  prog.add(Execute(cs1, {dnai}));

  if (maskOpt) {
    auto mask = cast(graph, *maskOpt, c.elementType(), prog);
    mask = mask.squeeze({0}).expand({1}).broadcast(c.dim(1), 1);
    mapInPlace(graph, _1 * _2, {d_c, mask}, prog);
  }

  auto gradAtResetGateInput =
      mul(graph, d_c, c_recurrant, prog, {dnai, fPrefix + "/d_c * h_prev2"});
  auto d_r = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, r,
                                       gradAtResetGateInput, prog);

  debug_tensor(prog, "bwd d_c", d_c);
  debug_tensor(prog, "bwd d_u", d_u);
  debug_tensor(prog, "bwd d_r", d_r);

  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass", "TRAINING_BWD");
  mmOpt.set("inputRHSIsPreArranged", "true");

  Tensor w_out, w_in;
  if (weightsInput == nullptr) {
    w_out = flattenUnits(concat({weightsOutput[BASIC_GRU_CELL_RESET_GATE],
                                 weightsOutput[BASIC_GRU_CELL_UPDATE_GATE],
                                 weightsOutput[BASIC_GRU_CELL_CANDIDATE]},
                                1))
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

  auto d_cr = mul(graph, d_c, r, prog, {dnai, fPrefix + "/d_c * r"});
  auto d_r_d_u_d_c_out = concat({d_r, d_u, d_cr}, 1);

  Tensor d_h_prev;
  {
    w_out = preArrangeMatMulInputRHS(
        graph, d_r_d_u_d_c_out.shape(), w_out, initProg,
        {dnai, fPrefix + "/PreArrangeOutputWeights"}, mmOpt, cache);
    auto out = matMul(graph, d_r_d_u_d_c_out, w_out, prog,
                      {dnai, fPrefix + "/PrevStepGrad"}, mmOpt, cache);
    d_h_prev = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer,
                                          prog, fPrefix);
  }

  Tensor d_x;
  if (weightsInput) {
    Tensor d_r_d_u_d_c_in = concat({d_r, d_u, d_c}, 1);
    w_in = preArrangeMatMulInputRHS(
        graph, d_r_d_u_d_c_in.shape(), w_in, initProg,
        {dnai, fPrefix + "/PreArrangeInputWeights"}, mmOpt, cache);
    auto out = matMul(graph, d_r_d_u_d_c_in, w_in, prog,
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

std::tuple<Tensor, Tensor, Tensor> basicGruBackwardStep(
    Graph &graph, const GruParams &params, const Tensor *gradNextLayer,
    const Tensor &fwdIntermediates, const Tensor &prevStepOut,
    const Tensor &outGrad, const Tensor &weightsInput,
    const Tensor &weightsOutput, const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoreOpt,
    const boost::optional<Tensor &> &attScoresGradsOpt, Sequence &initProg,
    Sequence &prog, const GruOpts &opt, const DebugNameAndId &dnai,
    matmul::PlanningCache *cache) {
  if (params.resetAfter) {
    return backwardStepImplResetAfter(
        graph, gradNextLayer, fwdIntermediates, prevStepOut, outGrad,
        &weightsInput, weightsOutput, maskOpt, attScoreOpt, attScoresGradsOpt,
        initProg, prog, opt, {dnai}, cache, params.outputFullSequence);
  } else {
    return backwardStepImpl(graph, gradNextLayer, fwdIntermediates, prevStepOut,
                            outGrad, &weightsInput, weightsOutput, maskOpt,
                            attScoreOpt, attScoresGradsOpt, initProg, prog, opt,
                            {dnai}, cache);
  }
}

std::pair<Tensor, Tensor>
basicGruBackwardStep(Graph &graph, const GruParams &params,
                     const Tensor *gradNextLayer,
                     const Tensor &fwdIntermediates, const Tensor &prevStepOut,
                     const Tensor &outGrad, const Tensor &weightsOutput,
                     const boost::optional<const Tensor &> &maskOpt,
                     const boost::optional<const Tensor &> &attScoreOpt,
                     const boost::optional<Tensor &> &attScoresGradsOpt,
                     Sequence &initProg, Sequence &prog, const GruOpts &opt,
                     const DebugNameAndId &dnai, matmul::PlanningCache *cache) {
  Tensor prevStateGrad;
  Tensor bwdIntermediates;

  if (params.resetAfter) {
    std::tie(prevStateGrad, std::ignore, bwdIntermediates) =
        backwardStepImplResetAfter(
            graph, gradNextLayer, fwdIntermediates, prevStepOut, outGrad,
            nullptr, weightsOutput, maskOpt, attScoreOpt, attScoresGradsOpt,
            initProg, prog, opt, {dnai}, cache, params.outputFullSequence);
  } else {
    std::tie(prevStateGrad, std::ignore, bwdIntermediates) =
        backwardStepImpl(graph, gradNextLayer, fwdIntermediates, prevStepOut,
                         outGrad, nullptr, weightsOutput, maskOpt, attScoreOpt,
                         attScoresGradsOpt, initProg, prog, opt, {dnai}, cache);
  }
  return std::make_pair(prevStateGrad, bwdIntermediates);
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
static Tensor
gruBwdImpl(Graph &graph, const GruParams &params, program::Sequence &prog,
           const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
           const GruWeights &weights, const Tensor &fwdInputSeq,
           const Tensor &fwdOutput, const Tensor &gradLayerNext,
           Tensor *inputGradSeq, Tensor *bwdIntermediatesPtr,
           GruWeights *weightsGrad,
           const boost::optional<const Tensor &> &realTimeStepsOpt,
           const boost::optional<const Tensor &> &attScoresOpt,
           Tensor *attScoresGrads, const DebugNameAndId &dnai,
           const GruOpts &options, poplin::matmul::PlanningCache *cache) {
  logging::popnn::info("gruBwdImpl(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       dnai.getPathName());

  debug_tensor(prog, "bwd fwdIntermediatesSeq", fwdIntermediatesSeq);

  unsigned seqSize = params.timeSteps;

  Tensor fwdOutputNew;
  if (params.outputFullSequence) {
    fwdOutputNew =
        concat(fwdOutputInit.expand({0}), fwdOutput.slice(0, seqSize - 1));
  } else {
    fwdOutputNew = fwdOutputInit.expand({0});
    for (unsigned s = 0; s < seqSize - 1; s++)
      fwdOutputNew = concat(
          fwdOutputNew,
          fwdIntermediatesSeq[s][GRU_FWD_INTERMEDIATE_OUTPUT].expand({0}));
  }

  auto &weightsInput = weights.inputWeights;
  auto &weightsOutput = weights.outputWeights;

  // sequence down-counter
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "seqIdx"});
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);

  Tensor seqLen;
  if (realTimeStepsOpt) {
    seqLen = cast(graph, *realTimeStepsOpt, UNSIGNED_INT, prog, {dnai});
  }

  Tensor start = graph.addConstant(UNSIGNED_INT, {1}, seqSize, {dnai, "start"});
  graph.setTileMapping(start, 0);

  prog.add(Copy(start, seqIdx, false, {dnai}));
  subInPlace(graph, seqIdx, one, prog, {dnai});

  auto lastOutGrad = createOutputTensor(graph, params, 1, {dnai, "outGrad"})[0];

  Tensor gradLayerNextRearranged;
  if (params.outputFullSequence) {
    gradLayerNextRearranged = createOutputTensor(
        graph, params, seqSize, {dnai, "gradLayerNextRearranged"});
    prog.add(Copy(gradLayerNext, gradLayerNextRearranged, false, {dnai}));
    zero(graph, lastOutGrad, prog, {dnai, "initLastOutGrad"});
  } else {
    prog.add(Copy(gradLayerNext, lastOutGrad, false, {dnai}));
  }

  auto sliceIntermediates = Sequence({}, {dnai});

  Tensor fwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1},
                   sliceIntermediates, {dnai, "getFwdIntermediates"})
          .squeeze({0});

  Tensor prevStepOut =
      dynamicSlice(graph, fwdOutputNew, seqIdx, {0}, {1}, sliceIntermediates,
                   {dnai, "getPrevStepOut"})
          .squeeze({0});

  Tensor d_a_t;
  Tensor attScores;
  boost::optional<const Tensor &> maskOpt;
  boost::optional<const Tensor &> sliceAttScoresOpt;
  boost::optional<Tensor &> sliceAttScoresGradsOpt;
  if (attScoresOpt) {
    attScores = dynamicSlice(
        graph, (*attScoresOpt).transpose().expand({(*attScoresOpt).rank()}),
        seqIdx, {0}, {1}, sliceIntermediates, {dnai, "attScores"})[0];
    sliceAttScoresOpt = attScores;

    *attScoresGrads = createAttention(graph, params, {dnai, "attScoresGrads"});
    d_a_t = createAttention(graph, params, {dnai, "attScoresGrads/t"})
                .transpose()[0];
    sliceAttScoresGradsOpt = d_a_t;
  }

  prog.add(sliceIntermediates);

  auto loop = Sequence({}, {dnai});
  auto bwdLoopBody = Sequence({}, {dnai});
  auto wuLoopBody = Sequence({}, {dnai});
  {
    Tensor newOutGrad;
    Tensor maskTensor;
    Tensor bwdIntermediates;
    Tensor gradLayerNextThisStep;
    Tensor *gradLayerNextThisStepPtr = nullptr;
    if (params.outputFullSequence) {
      gradLayerNextThisStep =
          dynamicSlice(graph, gradLayerNextRearranged, seqIdx, {0}, {1},
                       bwdLoopBody, {dnai, "gradLayerNext"})
              .squeeze({0});
      gradLayerNextThisStepPtr = &gradLayerNextThisStep;
    }

    if (realTimeStepsOpt && params.outputFullSequence) {
      maskTensor = lt(graph, seqIdx.expand({0}), seqLen, bwdLoopBody, {dnai});
      maskOpt = maskTensor;
    }

    if (inputGradSeq) {
      Tensor inputGrad;
      std::tie(newOutGrad, inputGrad, bwdIntermediates) =
          popnn::gru::basicGruBackwardStep(
              graph, params, gradLayerNextThisStepPtr, fwdIntermediates,
              prevStepOut, lastOutGrad, weightsInput, weightsOutput, maskOpt,
              sliceAttScoresOpt, sliceAttScoresGradsOpt, prog, bwdLoopBody,
              options, {dnai}, cache);
      *inputGradSeq = createInput(graph, params, {dnai, "inputGradSeq"});

      GruParams tmp_params(params);
      tmp_params.timeSteps = 1;
      auto inputGradRearranged =
          createInput(graph, tmp_params, {dnai, "inputGradSeq"})[0];

      bwdLoopBody.add(Copy(inputGrad, inputGradRearranged, false, {dnai}));
      prog.add(WriteUndef(*inputGradSeq, {dnai}));
      dynamicUpdate(graph, *inputGradSeq, inputGradRearranged.expand({0}),
                    seqIdx, {0}, {1}, bwdLoopBody, {dnai, "gradLayerPrev"});
    } else {
      std::tie(newOutGrad, bwdIntermediates) = basicGruBackwardStep(
          graph, params, gradLayerNextThisStepPtr, fwdIntermediates,
          prevStepOut, lastOutGrad, weightsOutput, maskOpt, sliceAttScoresOpt,
          sliceAttScoresGradsOpt, prog, bwdLoopBody, options, {dnai}, cache);
    }

    if (attScoresOpt) {
      auto d_a_tRearranged =
          createAttention(graph, params, {dnai, "attentionGrad/rearrangement"})
              .transpose()[0];

      bwdLoopBody.add(Copy(d_a_t, d_a_tRearranged, false, {dnai}));
      dynamicUpdate(
          graph, attScoresGrads->transpose().expand({(*attScoresOpt).rank()}),
          d_a_tRearranged.expand({1}).expand({0}), seqIdx, {0}, {1},
          bwdLoopBody, {dnai, "gruAttGrad"});
      debug_tensor(bwdLoopBody, "bwd attGrad", (*attScoresGrads));
    }

    // If bwdIntermediatesPtr is given, create a sequence containing gradients
    // for each cell unit in each step.
    if (bwdIntermediatesPtr) {
      *bwdIntermediatesPtr =
          createOutputTensor(graph, params, seqSize * BASIC_GRU_CELL_NUM_UNITS,
                             {dnai, "bwdIntermediates"})
              .reshapePartial(0, 1, {seqSize, BASIC_GRU_CELL_NUM_UNITS});
      auto bwdIntermediatesRearranged =
          createOutputTensor(graph, params, BASIC_GRU_CELL_NUM_UNITS,
                             {dnai, "bwdIntermediatesRearranged"});
      bwdLoopBody.add(
          Copy(bwdIntermediates, bwdIntermediatesRearranged, false, {dnai}));
      prog.add(WriteUndef(*bwdIntermediatesPtr, {dnai}));
      dynamicUpdate(graph, *bwdIntermediatesPtr,
                    bwdIntermediatesRearranged.expand({0}), seqIdx, {0}, {1},
                    bwdLoopBody, {dnai, "bwdIntermediates"});
    }
    Tensor prevLayerOut;
    if (weightsGrad) {
      // make a copy of the activations so that they are sliced efficiently
      auto fwdInputSeqCopy =
          createInput(graph, params, {dnai, "fwdInputSeqCopy"}, {}, cache);
      prog.add(Copy(fwdInputSeq, fwdInputSeqCopy, false, {dnai}));
      prevLayerOut = dynamicSlice(graph, fwdInputSeqCopy, seqIdx, {0}, {1},
                                  bwdLoopBody, {dnai, "prevLayerActsBwd"})
                         .squeeze({0});
    }
    bwdLoopBody.add(Copy(newOutGrad, lastOutGrad, false, {dnai}));
    subInPlace(graph, seqIdx, one, bwdLoopBody, {dnai, "seqIdxDecr"});
    debug_tensor(loop, "bwd Loop ", seqIdx);
    loop.add(bwdLoopBody);

    if (weightsGrad) {
      *weightsGrad = createWeightAccumulators(
          graph, weights, bwdIntermediates, options, {dnai}, params.resetAfter);
      zeroWeightAccumulators(graph, prog, *weightsGrad, {dnai});

      basicGruParamUpdate(graph, prevLayerOut, prevStepOut, fwdIntermediates,
                          bwdIntermediates, *weightsGrad, wuLoopBody, options,
                          {dnai}, cache, params.resetAfter);
    }
    loop.add(wuLoopBody);
    // Go to next step
    loop.add(sliceIntermediates);
  }

  prog.add(Repeat(seqSize - 1, loop, {dnai}));

  debug_tensor(prog, "bwd Loop ", seqIdx);
  prog.add(bwdLoopBody);
  if (weightsGrad) {
    prog.add(wuLoopBody);
    *weightsGrad = basicGruParamUpdateFinal(graph, weights, *weightsGrad, prog,
                                            {dnai}, params.resetAfter);
  }

  return lastOutGrad;
}

Tensor gruBwd(Graph &graph, const GruParams &params, program::Sequence &prog,
              const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
              const GruWeights &weights_, const Tensor &fwdInputSeq,
              const Tensor &fwdOutput, const Tensor &gradLayerNext,
              Tensor *inputGrad, Tensor *bwdIntermediates,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options_,
              poplin::matmul::PlanningCache *planningCache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              fwdOutput, gradLayerNext, params, inputGrad, bwdIntermediates,
              options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_);
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
                 nullptr, {di}, std::move(options), planningCache);
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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              realTimeSteps, fwdOutput, gradLayerNext, params, inputGrad,
              bwdIntermediates, options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_);
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
                 nullptr, {di}, std::move(options), planningCache);
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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              fwdOutput, gradLayerNext, inputGrad, bwdIntermediates, attentions,
              attentionsGrad, params, options_, planningCache));

  validateParams(params);
  auto options = parseOptions(options_);
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
                    attentionsGrad, {di}, std::move(options), planningCache);
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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediatesSeq, weights_, fwdInputSeq,
              realTimeSteps, fwdOutput, gradLayerNext, inputGrad,
              bwdIntermediates, attentions, attentionsGrad, params, options_,
              planningCache));
  validateParams(params);
  auto options = parseOptions(options_);
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
                 attentionsGrad, {di}, std::move(options), planningCache);
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
  Tensor fwdOutputNew;
  if (params.outputFullSequence) {
    fwdOutputNew = concat(fwdOutputInit.expand({0}),
                          output.slice(0, params.timeSteps - 1));
  } else {
    fwdOutputNew = fwdOutputInit.expand({0});
    for (unsigned s = 0; s < params.timeSteps - 1; s++)
      fwdOutputNew = concat(
          fwdOutputNew,
          fwdIntermediatesSeq[s][GRU_FWD_INTERMEDIATE_OUTPUT].expand({0}));
  }

  GruWeights weightGrads =
      createWeightAccumulators(graph, weights, bwdIntermediatesSeq[0], options,
                               {dnai}, params.resetAfter);
  zeroWeightAccumulators(graph, prog, weightGrads, {dnai});

  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, {dnai, "seqIdx"});
  auto start = graph.addConstant(UNSIGNED_INT, {1}, params.timeSteps - 1,
                                 {dnai, "start"});
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, {dnai, "one"});
  graph.setTileMapping(start, 0);
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx, false, {dnai}));

  auto sliceLoopBody = Sequence({}, {dnai});
  Tensor prevStepOut = dynamicSlice(graph, fwdOutputNew, seqIdx, {0}, {1},
                                    sliceLoopBody, {dnai, "getPrevStepOut"})
                           .squeeze({0});
  Tensor fwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1}, sliceLoopBody,
                   {dnai, "getFwdIntermediates"})
          .squeeze({0});

  auto loop = Sequence({}, {dnai});
  auto wuLoopBody = Sequence({}, {dnai});
  {
    // Dynamic slice required state per-step
    // make a copy of the activations so that they are sliced efficiently
    auto inputCopy = createInput(graph, params, {dnai, "inputCopy"}, {});
    prog.add(Copy(input, inputCopy, false, {dnai}));
    auto prevLayerOut = dynamicSlice(graph, inputCopy, seqIdx, {0}, {1},
                                     sliceLoopBody, {dnai, "prevLayerActsWu"})
                            .squeeze({0});
    auto bwdIntermediates =
        dynamicSlice(graph, bwdIntermediatesSeq, seqIdx, {0}, {1},
                     sliceLoopBody, {dnai, "getBwdIntermediates"})
            .squeeze({0});
    subInPlace(graph, seqIdx, one, sliceLoopBody, {dnai, "seqIdxDecr"});
    loop.add(sliceLoopBody);

    basicGruParamUpdate(graph, prevLayerOut, prevStepOut, fwdIntermediates,
                        bwdIntermediates, weightGrads, wuLoopBody, options,
                        {dnai}, planningCache, params.resetAfter);
    loop.add(wuLoopBody);
  }
  prog.add(Repeat(params.timeSteps - 1, loop, {dnai}));
  prog.add(sliceLoopBody);
  prog.add(wuLoopBody);

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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, bwdIntermediates, weights_,
              input, output, params, options_, planningCache));

  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());
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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, bwdIntermediates, weights_,
              input, output, params, options_, planningCache));
  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());

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
  const auto batchSize = params.batchSize;
  const auto inputSize = params.layerSizes[0];
  const auto outputSize = params.layerSizes[1];
  // Total elements needed for transposed weights.
  const auto totalTransposeParams =
      (inputSize + outputSize) * outputSize * BASIC_GRU_CELL_NUM_UNITS;
  // Total elements needed for unit gradients for weight update if
  // not interleaved with backpropagation.
  const auto totalBwdIntermediates =
      batchSize * outputSize * BASIC_GRU_CELL_NUM_UNITS * params.timeSteps;
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
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input,
                            output, outputGrad, inputGrad, weightsGrad_, params,
                            options_, planningCache));
  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());
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
      {di}, options, planningCache);

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
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input,
                            realTimeSteps, output, outputGrad, inputGrad,
                            weightsGrad_, params, options_, planningCache));

  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());
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
      {di}, options, planningCache);

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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input, output,
              outputGrad, inputGrad, weightsGrad_, attentions, attentionsGrad,
              params, options_, planningCache));

  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());
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
      attentionsGrad, {di}, options, planningCache);

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
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(fwdOutputInit, fwdIntermediates, weights_, input, realTimeSteps,
              output, outputGrad, inputGrad, weightsGrad_, attentions,
              attentionsGrad, params, options_, planningCache));

  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugContext.getPathName());
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
      attentionsGrad, {di}, options, planningCache);

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
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
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
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];
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
  auto batchSize = params.batchSize;
  auto sequenceSize = params.timeSteps;
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];

  uint64_t weightFlops = static_cast<uint64_t>(inputSize + outputSize) * 3 *
                         outputSize * batchSize * sequenceSize * 2;
  uint64_t biasFlops =
      static_cast<uint64_t>(outputSize) * 3 * batchSize * sequenceSize * 2;
  return weightFlops + biasFlops;
}

} // namespace gru
} // namespace popnn
