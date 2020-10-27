// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popnn/Gru.hpp>
#include <popops/Cast.hpp>

#include "RnnUtil.hpp"
#include "poplibs_support/logging.hpp"

using namespace poplar;
using namespace poplar::program;

using namespace poplibs_support;
using namespace poplin;
using namespace popnn::Rnn;
using namespace popops;
using namespace popops::expr;
using namespace poputil;

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
  GRU_FWD_INTERMEDIATE_OUTPUT
};

struct GruInternalState {
  Tensor resetGate;
  Tensor updateGate;
  Tensor candidate;

  Tensor getAsTensor() const {
    return concat(
        {resetGate.expand({0}), updateGate.expand({0}), candidate.expand({0})});
  }
};

// Computes the output before nonlinearities to all the units are applied
static Tensor basicGruUnitsNlInput(Graph &graph, Tensor prevAct,
                                   Tensor prevOutput, size_t num_unit,
                                   Tensor weightsInput, Tensor weightsOutput,
                                   Sequence &prog, OptionFlags &mmOpt,
                                   matmul::PlanningCache *cache,
                                   const std::string &debugStr) {
  assert(weightsInput.dim(0) == num_unit);
  assert(weightsOutput.dim(0) == num_unit);
  auto weights = concat(weightsInput, weightsOutput, 1);
  return unflattenUnits(matMul(graph, concat(prevAct, prevOutput, 1),
                               flattenUnits(weights), prog,
                               debugStr + "/Weight", mmOpt, cache),
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
                                 const std::string &name) {
  const auto outputSize = params.layerSizes[1];
  const auto batchSize = params.batchSize;
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

Tensor createInput(Graph &graph, const GruParams &params,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options,
                   poplin::matmul::PlanningCache *planningCache) {
  const auto name = debugContext.getPathName();
  validateParams(params);

  auto inputSize = params.layerSizes[0];
  const auto batchSize = params.batchSize;
  const auto inputGrouping = gcd(16UL, inputSize);
  const auto numInputGroups = (inputSize * batchSize) / inputGrouping;
  auto in = createDynamicSliceTensor(graph, params.dataType, params.timeSteps,
                                     numInputGroups, inputGrouping, name);
  return in.reshapePartial(1, 2, {inputSize / inputGrouping, batchSize})
      .dimRoll(1, 2)
      .flatten(2, 4);
}

Tensor createInitialState(Graph &graph, const GruParams &params,
                          const poplar::DebugContext &debugContext,
                          const OptionFlags &options,
                          matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  return createOutputTensor(graph, params, 1, "/initialOutput").squeeze({0});
}

void zeroInitialState(Graph &graph, const Tensor &init_output, Sequence &prog,
                      const std::string &debugPrefix) {
  zero(graph, init_output, prog, debugPrefix);
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
  const auto name = debugContext.getPathName();
  validateParams(params);
  auto opt = parseOptions(options);
  auto mmOpt = getMMOpts(opt);
  mmOpt.set("fullyConnectedPass",
            opt.inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];

  auto weights_ru = createMatMulInputRHS(
      graph, params.dataType, {params.batchSize, inputSize + outputSize},
      {inputSize + outputSize, 2 * outputSize}, name + "/weights", mmOpt,
      cache);
  poplar::Tensor inputWeights_ru =
      unflattenUnits(weights_ru.slice(0, inputSize), 2);
  poplar::Tensor outputWeights_ru =
      unflattenUnits(weights_ru.slice(inputSize, inputSize + outputSize), 2);

  auto weights_c = createMatMulInputRHS(
      graph, params.dataType, {params.batchSize, inputSize + outputSize},
      {inputSize + outputSize, outputSize}, name + "/weights", mmOpt, cache);
  poplar::Tensor inputWeights_c =
      unflattenUnits(weights_c.slice(0, inputSize), 1);
  poplar::Tensor outputWeights_c =
      unflattenUnits(weights_c.slice(inputSize, inputSize + outputSize), 1);
  poplar::Tensor inputWeights = concat(inputWeights_ru, inputWeights_c);
  poplar::Tensor outputWeights = concat(outputWeights_ru, outputWeights_c);
  // rearrange the outermost dimension according to the cellOrder parameter.
  return {toCellOrder(std::move(inputWeights), params.cellOrder),
          toCellOrder(std::move(outputWeights), params.cellOrder)};
}

/** Create the weights biases.
 */
poplar::Tensor createWeightsBiases(poplar::Graph &graph,
                                   const GruParams &params,
                                   const poplar::DebugContext &debugContext,
                                   const OptionFlags &,
                                   poplin::matmul::PlanningCache *) {
  const auto name = debugContext.getPathName();
  validateParams(params);
  auto outputSize = params.layerSizes[1];
  auto biases =
      graph.addVariable(params.dataType, {BASIC_GRU_CELL_NUM_UNITS, outputSize},
                        name + "/biases");
  mapTensorLinearly(graph, biases);
  return toCellOrder(biases, params.cellOrder);
}

GruWeights createWeights(Graph &graph, const GruParams &params,
                         const poplar::DebugContext &debugContext,
                         const OptionFlags &options,
                         poplin::matmul::PlanningCache *cache) {
  const auto name = debugContext.getPathName();
  GruWeights GruWeights;
  std::tie(GruWeights.inputWeights, GruWeights.outputWeights) =
      createWeightsKernel(graph, params, name, options, cache);
  GruWeights.biases = createWeightsBiases(graph, params, name, options, cache);
  return GruWeights;
}

Tensor createAttention(Graph &graph, const GruParams &params,
                       const poplar::DebugContext &debugContext) {
  const auto name = debugContext.getPathName();
  validateParams(params);

  return createSliceableTensor(graph, params.dataType,
                               {params.batchSize, params.timeSteps}, {1}, {1},
                               0, name);
}

static void rearrangeUnitsOutputFwd(Graph &graph, int num_unit,
                                    Tensor outputUnits,
                                    Tensor outputUnitsRearranged,
                                    Sequence &prog,
                                    const std::string &debugPrefix) {
  const auto outputGrouping =
      detectInnermostGrouping(graph, outputUnitsRearranged);
  // Typically the matrix multiplication result is laid out in memory such
  // that innermost dimension is groups batch elements. Try to rearrange the
  // result so the innermost dimension of the underlying memory is groups of the
  // specified number of outputs.
  outputUnits = unflattenUnits(
      tryGroupedPartialTranspose(graph, flattenUnits(outputUnits),
                                 outputGrouping, prog, debugPrefix),
      num_unit);
  prog.add(Copy(outputUnits, outputUnitsRearranged));
}

static void gruCellForwardPassCalcUnits(
    Graph &graph, bool forCandidate, const Tensor &in, const Tensor &prevOutput,
    const Tensor &biases, const Tensor *weightsInput,
    const Tensor &weightsOutput, Sequence &prog, const GruOpts &opt,
    const Tensor &unitsOutputRearranged, const std::string &baseStr,
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
      graph.addVariable(dType, {0, batchSize, outputSize}, "bbiases");
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
      cache, baseStr + "/ProcessUnits");

  // Rearrange the output of the matrix multiplication so each output unit
  // arranged the same as the cell state. This avoids the rearrangement
  // during the subsequent binary operations.
  rearrangeUnitsOutputFwd(graph, numUnit, unitsOutput, unitsOutputRearranged,
                          prog, baseStr);

  for (unsigned u = 0; u != numUnit; ++u) {
    graph.setTileMapping(biases[u],
                         graph.getTileMapping(unitsOutputRearranged[u][0]));
  }
  addInPlace(graph, unitsOutputRearranged, bBiases, prog, baseStr + "/AddBias");

  // Apply non linear function
  auto cs = graph.addComputeSet(baseStr + "/non-linear");

  if (forCandidate) {
    nonLinearityInPlace(graph, popnn::NonLinearityType::TANH,
                        unitsOutputRearranged, cs, baseStr + "Candidate Tanh");
  } else {
    nonLinearityInPlace(graph, popnn::NonLinearityType::SIGMOID,
                        unitsOutputRearranged, cs,
                        baseStr + "update/reset sigmod");
  }
  prog.add(Execute(cs));
}

static std::pair<Tensor, GruInternalState>
basicGruCellForwardPass(Graph &graph, const Tensor &in, const Tensor &biases,
                        const Tensor &prevOutput, const Tensor *weightsInput,
                        const Tensor &weightsOutput,
                        const boost::optional<const Tensor &> &attScoreOpt,
                        Sequence &prog, const GruOpts &opt,
                        const std::string &debugPrefix,
                        matmul::PlanningCache *cache) {
  debug_tensor(prog, "fwd h_prev", prevOutput);
  debug_tensor(prog, "fwd input", in);

  const std::string baseStr = debugPrefix + "/BasicGruCell";
  std::vector<Tensor> toConcat;
  toConcat.reserve(2);
  toConcat.push_back(
      graph.clone(prevOutput, debugPrefix + "/" + "Update Gate Rearranged")
          .expand({0}));
  toConcat.push_back(
      graph.clone(prevOutput, debugPrefix + "/" + "Reset Gate Rearranged")
          .expand({0}));
  auto unitsOutput = concat(toConcat);
  const Tensor weightsInput2 = weightsInput->slice(0, 2);
  const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
  const Tensor biases2 = biases.slice(0, 2);
  gruCellForwardPassCalcUnits(graph, false, in, prevOutput, biases2,
                              &weightsInput2, weightsOutput2, prog, opt,
                              unitsOutput, baseStr, cache);
  assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS - 1);
  auto resetGate = unitsOutput[BASIC_GRU_CELL_RESET_GATE];
  Tensor resetGateOut =
      graph.clone(resetGate, debugPrefix + "/" + "resetGateOut");
  prog.add(Copy(resetGate, resetGateOut));
  auto updateGate = unitsOutput[BASIC_GRU_CELL_UPDATE_GATE];

  Tensor candidate =
      graph.clone(prevOutput, debugPrefix + "/" + "candidate Rearranged");
  const Tensor weightsInput3 = weightsInput->slice(2, 3);
  const Tensor weightsOutput3 = weightsOutput.slice(2, 3);
  mulInPlace(graph, resetGate, prevOutput, prog,
             baseStr + "resetGate * prevOutput");
  Tensor candidateExpand = candidate.expand({0});
  gruCellForwardPassCalcUnits(graph, true, in, resetGate, biases,
                              &weightsInput3, weightsOutput3, prog, opt,
                              candidateExpand, baseStr, cache);
  candidate = candidateExpand[0];

  if (attScoreOpt) {
    mapInPlace(graph, _1 - (_2 * _1), {updateGate, *attScoreOpt}, prog,
               baseStr + "/UpdateScores");
  }

  Tensor newOutput =
      map(graph, _1 + _2 * (_3 - _1), {candidate, updateGate, prevOutput}, prog,
          baseStr + "/CalcNextOutput");

  GruInternalState internalState = {resetGateOut, updateGate, candidate};

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
    const GruOpts &opt, const std::string &debugPrefix,
    matmul::PlanningCache *cache) {
  logging::popnn::info("basicGruCellForwardPassInPlace {}", in.shape());

  debug_tensor(prog, "fwd h_prev", output);
  debug_tensor(prog, "fwd input", in);
  const std::string baseStr = debugPrefix + "/BasicGruCellInPlace";

  std::vector<Tensor> toConcat;
  toConcat.push_back(
      graph.clone(output, debugPrefix + "/" + "Update Gate Rearranged")
          .expand({0}));
  toConcat.push_back(
      graph.clone(output, debugPrefix + "/" + "Reset Gate Rearranged")
          .expand({0}));
  auto unitsOutput = concat(toConcat);
  const Tensor weightsInput2 = weightsInput->slice(0, 2);
  const Tensor weightsOutput2 = weightsOutput.slice(0, 2);
  gruCellForwardPassCalcUnits(graph, false, in, output, biases, &weightsInput2,
                              weightsOutput2, prog, opt, unitsOutput, baseStr,
                              cache);
  assert(unitsOutput.dim(0) == BASIC_GRU_CELL_NUM_UNITS - 1);
  auto updateGate = unitsOutput[BASIC_GRU_CELL_UPDATE_GATE];
  auto resetGate = unitsOutput[BASIC_GRU_CELL_RESET_GATE];

  if (attScoreOpt) {
    logging::popnn::info("updateGate u {} score {}", updateGate.shape(),
                         (*attScoreOpt).shape());
    mapInPlace(graph, _1 - (_2 * _1), {updateGate, *attScoreOpt}, prog,
               baseStr + "/UpdateScores");
  }

  debug_tensor(prog, "fwd resetGate", resetGate);
  debug_tensor(prog, "fwd updateGate", updateGate);

  Tensor candidate =
      graph.clone(output, debugPrefix + "/" + "candidate Rearranged");
  const Tensor weightsInput3 = weightsInput->slice(2, 3);
  const Tensor weightsOutput3 = weightsOutput.slice(2, 3);
  mulInPlace(graph, resetGate, output, prog, baseStr + "resetGate * output");
  Tensor candidateExpand = candidate.expand({0});
  gruCellForwardPassCalcUnits(graph, true, in, resetGate, biases,
                              &weightsInput3, weightsOutput3, prog, opt,
                              candidateExpand, baseStr, cache);
  candidate = candidateExpand[0];
  debug_tensor(prog, "fwd candidate", candidate);

  mapInPlace(graph, _3 + _2 * (_1 - _3), {output, updateGate, candidate}, prog,
             baseStr + "/CalcNextOutput");

  debug_tensor(prog, "fwd output", output);
}

Tensor gruFwdImpl(Graph &graph, const GruParams &params,
                  const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                  const GruWeights &weights_, Tensor *intermediatesSeq,
                  const boost::optional<const Tensor &> &attScoresOpt,
                  const boost::optional<const Tensor &> &realTimeStepsOpt,
                  program::Sequence &fwdProg, const std::string &debugPrefix,
                  const OptionFlags &options,
                  poplin::matmul::PlanningCache *cache) {
  logging::popnn::info("gruFwdImpl(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);

  validateParams(params);
  auto opt = parseOptions(options);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  Tensor output =
      duplicate(graph, fwdOutputInit, fwdProg, debugPrefix + "/fwdOutput");

  // loop counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  popops::zero(graph, seqIdx, fwdProg, debugPrefix + "/initSeqIdx");

  unsigned seqSize = prevLayerActs.dim(0);
  // make a copy of the activations so that they are sliced efficiently
  auto prevLayerActsCopy = createInput(
      graph, params, debugPrefix + "/prevLayerActsCopy", options, cache);
  fwdProg.add(Copy(prevLayerActs, prevLayerActsCopy));

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
        seqIdx, {0}, {1}, loop, debugPrefix + "/auGruAttScores")[0];
    sliceAttScoresOpt = attScores;
  }

  Tensor fwdInput = popops::dynamicSlice(graph, prevLayerActsCopy, seqIdx, {0},
                                         {1}, loop, debugPrefix + "/gru")[0];
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
        weights.outputWeights, sliceAttScoresOpt, loop, opt, debugPrefix,
        cache);

    if (realTimeStepsOpt) {
      mask = gt(graph, seqLen, seqIdx, loop);
      mask = cast(graph, mask, newOutput.elementType(), loop);
      mask = mask.expand({1}).broadcast(newOutput.dim(1), 1);
      mapInPlace(graph, _1 * _2, {newOutput, mask}, loop);
      mapInPlace(graph, _1 * _2, {internalState.resetGate, mask}, loop);
      mapInPlace(graph, _1 * _2, {internalState.updateGate, mask}, loop);
      mapInPlace(graph, _1 * _2, {internalState.candidate, mask}, loop);
    }

    Tensor intermediates;
    if (params.outputFullSequence)
      intermediates = concat({internalState.resetGate.expand({0}),
                              internalState.updateGate.expand({0}),
                              internalState.candidate.expand({0})});
    else
      intermediates =
          concat({internalState.resetGate.expand({0}),
                  internalState.updateGate.expand({0}),
                  internalState.candidate.expand({0}), newOutput.expand({0})});

    const auto numIntermediates = intermediates.dim(0);
    *intermediatesSeq =
        createOutputTensor(graph, params, seqSize * numIntermediates,
                           debugPrefix + "/fwdIntermediatesSeq")
            .reshapePartial(0, 1, {seqSize, numIntermediates});

    auto intermediatesRearranged =
        createOutputTensor(graph, params, numIntermediates,
                           debugPrefix + "/fwdIntermediatesRearranged");
    loop.add(Copy(intermediates, intermediatesRearranged));
    fwdProg.add(WriteUndef(*intermediatesSeq));
    popops::dynamicUpdate(graph, *intermediatesSeq,
                          intermediatesRearranged.expand({0}), seqIdx, {0}, {1},
                          loop, debugPrefix + "/gruUpdateIntermediates");

    graph.setTileMapping(output, graph.getTileMapping(newOutput));
    loop.add(Copy(newOutput, output));
  } else {
    basicGruCellForwardPassInPlace(graph, fwdInput, weights.biases, output,
                                   inputWeightsPtr, weights.outputWeights,
                                   sliceAttScoresOpt, loop, opt, debugPrefix,
                                   cache);

    if (realTimeStepsOpt) {
      mask = gt(graph, seqLen, seqIdx, loop);
      mask = cast(graph, mask, output.elementType(), loop);
      mask = mask.expand({1}).broadcast(output.dim(1), 1);
      mapInPlace(graph, _1 * _2, {output, mask}, loop);
    }
  }

  Tensor outputSeq;
  if (params.outputFullSequence) {
    outputSeq =
        createOutputTensor(graph, params, seqSize, debugPrefix + "/Output");
    fwdProg.add(WriteUndef(outputSeq));
    popops::dynamicUpdate(graph, outputSeq, output.expand({0}), seqIdx, {0},
                          {1}, loop, debugPrefix + "/updateOutputSeq");
  }

  addInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxIncr");

  fwdProg.add(Repeat(seqSize, loop));

  return params.outputFullSequence ? outputSeq : output;
}

Tensor gruFwd(Graph &graph, const GruParams &params,
              const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
              const GruWeights &weights_, Tensor *intermediatesSeq,
              program::Sequence &fwdProg,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  boost::optional<const Tensor &> attScoresOpt(boost::none);
  boost::optional<const Tensor &> realTimeStepsOpt(boost::none);

  return gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs, weights_,
                    intermediatesSeq, attScoresOpt, realTimeStepsOpt, fwdProg,
                    debugPrefix, options, cache);
}

Tensor gruFwd(Graph &graph, const GruParams &params,
              const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
              const Tensor &realTimeSteps, const GruWeights &weights_,
              Tensor *intermediatesSeq, program::Sequence &fwdProg,
              const poplar::DebugContext &debugContext,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  if (!params.outputFullSequence) {
    throw poplibs_error(std::string("The outputFullSequence should be true ") +
                        "if realTimeSteps given");
  }

  boost::optional<const Tensor &> attScoresOpt(boost::none);
  boost::optional<const Tensor &> realTimeStepsOpt(realTimeSteps);

  return gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs, weights_,
                    intermediatesSeq, attScoresOpt, realTimeStepsOpt, fwdProg,
                    debugPrefix, options, cache);
}

Tensor auGruFwd(Graph &graph, const GruParams &params,
                const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                const GruWeights &weights_, Tensor *intermediatesSeq,
                const Tensor &attScores, program::Sequence &fwdProg,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options,
                poplin::matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  boost::optional<const Tensor &> attScoresOpt(attScores);
  boost::optional<const Tensor &> realTimeStepsOpt(boost::none);

  return gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs, weights_,
                    intermediatesSeq, attScoresOpt, realTimeStepsOpt, fwdProg,
                    debugPrefix, options, cache);
}

Tensor auGruFwd(Graph &graph, const GruParams &params,
                const Tensor &fwdOutputInit, const Tensor &prevLayerActs,
                const Tensor &realTimeSteps, const GruWeights &weights_,
                Tensor *intermediatesSeq, const Tensor &attScores,
                program::Sequence &fwdProg,
                const poplar::DebugContext &debugContext,
                const OptionFlags &options,
                poplin::matmul::PlanningCache *cache) {
  const auto debugPrefix = debugContext.getPathName();
  if (!params.outputFullSequence) {
    throw poplibs_error(std::string("The outputFullSequence should be true ") +
                        "if realTimeSteps given");
  }

  boost::optional<const Tensor &> attScoresOpt(attScores);
  boost::optional<const Tensor &> realTimeStepsOpt(realTimeSteps);

  return gruFwdImpl(graph, params, fwdOutputInit, prevLayerActs, weights_,
                    intermediatesSeq, attScoresOpt, realTimeStepsOpt, fwdProg,
                    debugPrefix, options, cache);
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
                 const std::string &debugPrefix, matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/GruBwdOneStep";
  auto outputGroupingIntoLayer = detectInnermostGrouping(graph, outputGrad);
  Tensor d_h = graph.clone(outputGrad);
  debug_tensor(prog, "bwd outGrad", outputGrad);
  if (gradNextLayer)
    debug_tensor(prog, "bwd gradNextLayer", *gradNextLayer);
  prog.add(Copy(outputGrad, d_h));
  if (gradNextLayer)
    d_h =
        popops::add(graph, d_h, *gradNextLayer, prog, fPrefix + "/AddActGrads");

  auto u = fwdIntermediates[GRU_FWD_INTERMEDIATE_UPDATE_GATE];
  auto r = fwdIntermediates[GRU_FWD_INTERMEDIATE_RESET_GATE];
  auto c = fwdIntermediates[GRU_FWD_INTERMEDIATE_CANDIDATE];
  auto h_prev = prevStepOut;

  auto one_matrix = graph.addConstant(
      outputGrad.elementType(), outputGrad.shape(), 1, fPrefix + "/one_matrix");
  graph.setTileMapping(one_matrix, graph.getTileMapping(u));
  auto var_one_matrix =
      graph.addVariable(outputGrad.elementType(), outputGrad.shape(),
                        fPrefix + "/var_one_matrix");
  graph.setTileMapping(var_one_matrix, graph.getTileMapping(u));
  prog.add(Copy(one_matrix, var_one_matrix));

  debug_tensor(prog, "bwd d_h", d_h);
  debug_tensor(prog, "bwd r", r);
  debug_tensor(prog, "bwd u", u);
  debug_tensor(prog, "bwd c", c);
  debug_tensor(prog, "bwd h_prev", h_prev);

  // u_com = 1 - u
  Tensor u_com = sub(graph, var_one_matrix, u, prog, fPrefix + "/1-updateGate");
  // h_prev_c = h_prev - c
  auto h_prev_c = sub(graph, h_prev, c, prog, fPrefix + "/preOutput-candidate");
  // (1-u) * d_h, (h_prev - c) * d_h
  auto t = mul(graph, concat({u_com, h_prev_c}), d_h.broadcast(2, 0), prog,
               fPrefix + "/MulOGate");
  auto gradAtCandidateInput = t.slice(0, outputGrad.dim(0));
  auto gradAtUpdateGateInput =
      t.slice(outputGrad.dim(0), 2 * outputGrad.dim(0));

  debug_tensor(prog, "bwd outputGrad", d_h);
  debug_tensor(prog, "bwd h_prev_c", h_prev_c);

  auto cs1 = graph.addComputeSet(fPrefix + "/OutputGate");
  auto d_c = nonLinearityInputGradient(graph, NonLinearityType::TANH, c,
                                       gradAtCandidateInput, cs1,
                                       fPrefix + "/OutputTanh");

  Tensor d_u;
  if (attScoresOpt) {
    auto u0 = map(graph, _1 / (Const(1) - _2), {u, *attScoresOpt}, prog,
                  fPrefix + "/UpdateScores");

    Tensor temp = map(graph, -_1 * _2, {gradAtUpdateGateInput, u0}, prog,
                      fPrefix + "/AttentionsGrad");
    *attScoresGradsOpt =
        reduce(graph, temp, {1}, {popops::Operation::ADD}, prog);

    mapInPlace(graph, _1 - (_1 * _2), {gradAtUpdateGateInput, *attScoresOpt},
               prog, fPrefix + "/UpdateGateGrad");

    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u0,
                                    gradAtUpdateGateInput, cs1,
                                    fPrefix + "/OutputGate");
  } else {
    d_u = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, u,
                                    gradAtUpdateGateInput, cs1,
                                    fPrefix + "/OutputGate");
  }
  prog.add(Execute(cs1));

  if (maskOpt) {
    auto mask = cast(graph, *maskOpt, c.elementType(), prog);
    mask = mask.squeeze({0}).expand({1}).broadcast(c.dim(1), 1);
    mapInPlace(graph, _1 * _2, {d_c, mask}, prog);
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
  w_c =
      preArrangeMatMulInputRHS(graph, d_c.shape(), w_c, initProg,
                               fPrefix + "/PreArrangeWeights C", mmOpt, cache);

  Tensor d_x2, d_hr, d_x2_hr;
  int inputSize = weightsInput->dim(1);
  int outputSize = weightsOutput.dim(1);
  if (weightsInput) {
    auto out = matMul(graph, d_c, w_c, prog, fPrefix + "/d_x2_d_h_prevr", mmOpt,
                      cache);
    d_x2_hr = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer,
                                         prog, fPrefix);
    debug_tensor(prog, "bwd d_x2_h_prevr", d_x2_hr);
    d_x2 = d_x2_hr.slice(0, inputSize, 1);
    d_hr = d_x2_hr.slice(inputSize, inputSize + outputSize, 1);
  } else {
    auto out =
        matMul(graph, d_c, w_c, prog, fPrefix + "/PrevStepGrad", mmOpt, cache);
    d_hr = tryGroupedPartialTranspose(graph, out, outputGroupingIntoLayer, prog,
                                      fPrefix);
  }

  Tensor d_r;
  {
    auto t = mul(graph, d_hr, h_prev, prog, fPrefix + "/d_hr * h_prev");
    d_r = nonLinearityInputGradient(graph, NonLinearityType::SIGMOID, r, t,
                                    prog, fPrefix + "/t * r * (1-r)");
  }

  Tensor d_r_d_u = concat(d_r, d_u, 1);
  w_ru =
      preArrangeMatMulInputRHS(graph, d_r_d_u.shape(), w_ru, initProg,
                               fPrefix + "/PreArrangeWeights RU", mmOpt, cache);
  auto out = matMul(graph, d_r_d_u, w_ru, prog,
                    fPrefix + "/d_x1_d_h_prev1 X w_ru", mmOpt, cache);
  Tensor d_x1_d_hprev1 = tryGroupedPartialTranspose(
      graph, out, outputGroupingIntoLayer, prog, fPrefix);
  debug_tensor(prog, "bwd d_x1_d_hprev1", d_x1_d_hprev1);

  Tensor d_x;
  if (weightsInput) {
    d_x = add(graph, d_x1_d_hprev1.slice(0, inputSize, 1), d_x2, prog,
              fPrefix + "/dx");
  }

  Tensor d_hprev1 = d_x1_d_hprev1.slice(inputSize, inputSize + outputSize, 1);
  Tensor d_h_prev =
      map(graph, ((_1 * _2) + (_3 * _4)) + _5, {d_hr, r, d_h, u, d_hprev1},
          prog, fPrefix + "/d_h_prev");

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
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const Tensor &prevStepOut, const Tensor &outGrad,
    const Tensor &weightsInput, const Tensor &weightsOutput,
    const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoreOpt,
    const boost::optional<Tensor &> &attScoresGradsOpt, Sequence &initProg,
    Sequence &prog, const GruOpts &opt, const std::string &debugPrefix,
    matmul::PlanningCache *cache) {
  return backwardStepImpl(graph, gradNextLayer, fwdIntermediates, prevStepOut,
                          outGrad, &weightsInput, weightsOutput, maskOpt,
                          attScoreOpt, attScoresGradsOpt, initProg, prog, opt,
                          debugPrefix, cache);
}

std::pair<Tensor, Tensor> basicGruBackwardStep(
    Graph &graph, const Tensor *gradNextLayer, const Tensor &fwdIntermediates,
    const Tensor &prevStepOut, const Tensor &outGrad,
    const Tensor &weightsOutput, const boost::optional<const Tensor &> &maskOpt,
    const boost::optional<const Tensor &> &attScoreOpt,
    const boost::optional<Tensor &> &attScoreGradsOpt, Sequence &initProg,
    Sequence &prog, const GruOpts &opt, const std::string &debugPrefix,
    matmul::PlanningCache *cache) {
  Tensor prevStateGrad;
  Tensor bwdIntermediates;
  std::tie(prevStateGrad, std::ignore, bwdIntermediates) = backwardStepImpl(
      graph, gradNextLayer, fwdIntermediates, prevStepOut, outGrad, nullptr,
      weightsOutput, maskOpt, attScoreOpt, attScoreGradsOpt, initProg, prog,
      opt, debugPrefix, cache);
  return std::make_pair(prevStateGrad, bwdIntermediates);
}

/// Add the partial weight gradients from this timestep to the accumulated
/// weight gradients. Once all the gradients have been accumulated call
/// basicGruParamUpdateFinal() to do any final accumulation / rearrangement
/// that is required.
static void basicGruParamUpdate(
    Graph &graph, const Tensor &prevLayerActs, const Tensor &prevStepActs,
    const Tensor &fwdIntermediates, const Tensor &bwdIntermediates,
    const GruWeights &weightGrads, Sequence &prog, const GruOpts &opt,
    const std::string &debugPrefix, matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/GruDeltas";
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
  Tensor h_prevr = mul(graph, h_prev, r, prog, fPrefix + "/h_prev * r");
  Tensor x_h_prevr = concat(x, h_prevr, 1);
  debug_tensor(prog, "wu x", x);
  debug_tensor(prog, "wu h_prev", h_prev);
  debug_tensor(prog, "wu r", r);
  debug_tensor(prog, "wu fwdIntermediates", fwdIntermediates);

  Tensor d_r = bwdIntermediates[BASIC_GRU_CELL_RESET_GATE];
  Tensor d_u = bwdIntermediates[BASIC_GRU_CELL_UPDATE_GATE];
  Tensor d_c = bwdIntermediates[BASIC_GRU_CELL_CANDIDATE];

  matMulAcc(graph,
            concat(flattenUnits(weightGrads2.inputWeights),
                   flattenUnits(weightGrads2.outputWeights)),
            1.0, x_h_prev.transpose(),
            flattenUnits(concat(d_r.expand({0}), d_u.expand({0}))), prog,
            fPrefix + "/dw for reset and update weight", mmOpt, cache);

  matMulAcc(graph,
            concat(flattenUnits(weightGrads3.inputWeights),
                   flattenUnits(weightGrads3.outputWeights)),
            1.0, x_h_prevr.transpose(), d_c, prog,
            fPrefix + "/dw for candidate weight", mmOpt, cache);
  debug_tensor(prog, "wu x_h_prevr", x_h_prevr);
  debug_tensor(prog, "wu d_c", d_c);
  debug_tensor(prog, "wu inputWeightsGrad", weightGrads.inputWeights);
  debug_tensor(prog, "wu outputWeightsGrad", weightGrads.outputWeights);

  // We defer the reduction across the batch to later.
  popops::addInPlace(graph, weightGrads.biases, bwdIntermediates, prog,
                     fPrefix + "/Bias");
}

static GruWeights basicGruParamUpdateFinal(Graph &graph,
                                           const GruWeights &weights,
                                           const GruWeights &weightGrads,
                                           Sequence &prog,
                                           const std::string &debugPrefix) {
  // The accumulated bias gradients still has a batch axis that we must
  // accumulate over - do this now.
  auto biasGrad = graph.clone(weights.biases, debugPrefix + "/biasGrad");
  popops::reduceWithOutput(graph, weightGrads.biases, biasGrad, {1},
                           {popops::Operation::ADD}, prog,
                           debugPrefix + "/FinalBiasReduction");
  auto finalWeightGrads = weightGrads;
  finalWeightGrads.biases = biasGrad;
  return finalWeightGrads;
}

/// Create variables used to accumulate gradients of the weights in the
/// backward pass.
static GruWeights createWeightAccumulators(Graph &graph,
                                           const GruWeights &weights,
                                           const Tensor &bwdIntermediates,
                                           const GruOpts &options,
                                           const std::string &debugPrefix) {
  GruWeights weightAccs;
  // inputWeights and outputWeights are slices of the one variable. Clone
  // them together as it results in a less complex tensor expression.
  auto concatenated = concat(flattenUnits(weights.inputWeights),
                             flattenUnits(weights.outputWeights));
  auto weightsDeltaAcc =
      graph.clone(concatenated, debugPrefix + "/weightsDeltaAcc");
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
  weightAccs.biases =
      graph.clone(bwdIntermediates, debugPrefix + "/bwdIntermediatesAcc");
  return weightAccs;
}

static void zeroWeightAccumulators(Graph &graph, program::Sequence &prog,
                                   const GruWeights &weightsAcc,
                                   const std::string &debugPrefix) {
  popops::zero(
      graph,
      concat({weightsAcc.inputWeights.flatten(),
              weightsAcc.outputWeights.flatten(), weightsAcc.biases.flatten()}),
      prog, debugPrefix + "/zeroWeightAccumulators");
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
           Tensor *attScoresGrads, const std::string &debugPrefix,
           const GruOpts &options, poplin::matmul::PlanningCache *cache) {
  logging::popnn::info("gruBwdImpl(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);

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
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);

  Tensor seqLen;
  if (realTimeStepsOpt) {
    seqLen = cast(graph, *realTimeStepsOpt, UNSIGNED_INT, prog);
  }

  Tensor start =
      graph.addConstant(UNSIGNED_INT, {1}, seqSize, debugPrefix + "/start");
  graph.setTileMapping(start, 0);

  prog.add(Copy(start, seqIdx));
  subInPlace(graph, seqIdx, one, prog);

  auto lastOutGrad =
      createOutputTensor(graph, params, 1, debugPrefix + "/outGrad")[0];

  Tensor gradLayerNextRearranged;
  if (params.outputFullSequence) {
    gradLayerNextRearranged = createOutputTensor(
        graph, params, seqSize, debugPrefix + "/gradLayerNextRearranged");
    prog.add(Copy(gradLayerNext, gradLayerNextRearranged));
    zero(graph, lastOutGrad, prog, debugPrefix + "/initLastOutGrad");
  } else {
    prog.add(Copy(gradLayerNext, lastOutGrad));
  }

  auto sliceIntermediates = Sequence();

  Tensor fwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1},
                   sliceIntermediates, debugPrefix + "/getFwdIntermediates")
          .squeeze({0});

  Tensor prevStepOut =
      dynamicSlice(graph, fwdOutputNew, seqIdx, {0}, {1}, sliceIntermediates,
                   debugPrefix + "/getPrevStepOut")
          .squeeze({0});

  Tensor d_a_t;
  Tensor attScores;
  boost::optional<const Tensor &> maskOpt;
  boost::optional<const Tensor &> sliceAttScoresOpt;
  boost::optional<Tensor &> sliceAttScoresGradsOpt;
  if (attScoresOpt) {
    attScores = dynamicSlice(
        graph, (*attScoresOpt).transpose().expand({(*attScoresOpt).rank()}),
        seqIdx, {0}, {1}, sliceIntermediates, debugPrefix + "/attScores")[0];
    sliceAttScoresOpt = attScores;

    *attScoresGrads =
        createAttention(graph, params, debugPrefix + "/attScoresGrads");
    d_a_t = createAttention(graph, params, debugPrefix + "/attScoresGrads/t")
                .transpose()[0];
    sliceAttScoresGradsOpt = d_a_t;
  }

  prog.add(sliceIntermediates);

  auto loop = Sequence();
  auto bwdLoopBody = Sequence();
  auto wuLoopBody = Sequence();
  {
    Tensor newOutGrad;
    Tensor maskTensor;
    Tensor bwdIntermediates;
    Tensor gradLayerNextThisStep;
    Tensor *gradLayerNextThisStepPtr = nullptr;
    if (params.outputFullSequence) {
      gradLayerNextThisStep =
          dynamicSlice(graph, gradLayerNextRearranged, seqIdx, {0}, {1},
                       bwdLoopBody, debugPrefix + "/gradLayerNext")
              .squeeze({0});
      gradLayerNextThisStepPtr = &gradLayerNextThisStep;
    }

    if (realTimeStepsOpt && params.outputFullSequence) {
      maskTensor =
          lt(graph, seqIdx.expand({0}), seqLen, bwdLoopBody, debugPrefix);
      maskOpt = maskTensor;
    }

    if (inputGradSeq) {
      Tensor inputGrad;
      std::tie(newOutGrad, inputGrad, bwdIntermediates) =
          popnn::gru::basicGruBackwardStep(
              graph, gradLayerNextThisStepPtr, fwdIntermediates, prevStepOut,
              lastOutGrad, weightsInput, weightsOutput, maskOpt,
              sliceAttScoresOpt, sliceAttScoresGradsOpt, prog, bwdLoopBody,
              options, debugPrefix, cache);
      *inputGradSeq = createInput(graph, params, debugPrefix + "/inputGradSeq");

      GruParams tmp_params(params);
      tmp_params.timeSteps = 1;
      auto inputGradRearranged =
          createInput(graph, tmp_params, debugPrefix + "/inputGradSeq")[0];

      bwdLoopBody.add(Copy(inputGrad, inputGradRearranged));
      prog.add(WriteUndef(*inputGradSeq));
      dynamicUpdate(graph, *inputGradSeq, inputGradRearranged.expand({0}),
                    seqIdx, {0}, {1}, bwdLoopBody,
                    debugPrefix + "/gradLayerPrev");
    } else {
      std::tie(newOutGrad, bwdIntermediates) = basicGruBackwardStep(
          graph, gradLayerNextThisStepPtr, fwdIntermediates, prevStepOut,
          lastOutGrad, weightsOutput, maskOpt, sliceAttScoresOpt,
          sliceAttScoresGradsOpt, prog, bwdLoopBody, options, debugPrefix,
          cache);
    }

    if (attScoresOpt) {
      auto d_a_tRearranged =
          createAttention(graph, params,
                          debugPrefix + "/attentionGrad/rearrangement")
              .transpose()[0];

      bwdLoopBody.add(Copy(d_a_t, d_a_tRearranged));
      dynamicUpdate(
          graph, attScoresGrads->transpose().expand({(*attScoresOpt).rank()}),
          d_a_tRearranged.expand({1}).expand({0}), seqIdx, {0}, {1},
          bwdLoopBody, debugPrefix + "/gruAttGrad");
      debug_tensor(bwdLoopBody, "bwd attGrad", (*attScoresGrads));
    }

    // If bwdIntermediatesPtr is given, create a sequence containing gradients
    // for each cell unit in each step.
    if (bwdIntermediatesPtr) {
      *bwdIntermediatesPtr =
          createOutputTensor(graph, params, seqSize * BASIC_GRU_CELL_NUM_UNITS,
                             debugPrefix + "/bwdIntermediates")
              .reshapePartial(0, 1, {seqSize, BASIC_GRU_CELL_NUM_UNITS});
      auto bwdIntermediatesRearranged =
          createOutputTensor(graph, params, BASIC_GRU_CELL_NUM_UNITS,
                             debugPrefix + "/bwdIntermediatesRearranged");
      bwdLoopBody.add(Copy(bwdIntermediates, bwdIntermediatesRearranged));
      prog.add(WriteUndef(*bwdIntermediatesPtr));
      dynamicUpdate(graph, *bwdIntermediatesPtr,
                    bwdIntermediatesRearranged.expand({0}), seqIdx, {0}, {1},
                    bwdLoopBody, debugPrefix + "/bwdIntermediates");
    }
    Tensor prevLayerOut;
    if (weightsGrad) {
      // make a copy of the activations so that they are sliced efficiently
      auto fwdInputSeqCopy = createInput(
          graph, params, debugPrefix + "/fwdInputSeqCopy", {}, cache);
      prog.add(Copy(fwdInputSeq, fwdInputSeqCopy));
      prevLayerOut =
          dynamicSlice(graph, fwdInputSeqCopy, seqIdx, {0}, {1}, bwdLoopBody,
                       debugPrefix + "/prevLayerActsBwd")
              .squeeze({0});
    }
    bwdLoopBody.add(Copy(newOutGrad, lastOutGrad));
    subInPlace(graph, seqIdx, one, bwdLoopBody, debugPrefix + "/seqIdxDecr");
    debug_tensor(loop, "bwd Loop ", seqIdx);
    loop.add(bwdLoopBody);

    if (weightsGrad) {
      *weightsGrad = createWeightAccumulators(graph, weights, bwdIntermediates,
                                              options, debugPrefix);
      zeroWeightAccumulators(graph, prog, *weightsGrad, debugPrefix);

      basicGruParamUpdate(graph, prevLayerOut, prevStepOut, fwdIntermediates,
                          bwdIntermediates, *weightsGrad, wuLoopBody, options,
                          debugPrefix, cache);
    }
    loop.add(wuLoopBody);
    // Go to next step
    loop.add(sliceIntermediates);
  }

  prog.add(Repeat(seqSize - 1, loop));

  debug_tensor(prog, "bwd Loop ", seqIdx);
  prog.add(bwdLoopBody);
  if (weightsGrad) {
    prog.add(wuLoopBody);
    *weightsGrad = basicGruParamUpdateFinal(graph, weights, *weightsGrad, prog,
                                            debugPrefix);
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
  const auto debugPrefix = debugContext.getPathName();
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
  return gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                    weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                    bwdIntermediates, nullptr, realTimeStepsOpt, attScoresOpt,
                    nullptr, debugPrefix, std::move(options), planningCache);
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
  const auto debugPrefix = debugContext.getPathName();
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
  return gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                    weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                    bwdIntermediates, nullptr, realTimeStepsOpt, attScoresOpt,
                    nullptr, debugPrefix, std::move(options), planningCache);
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
  const auto debugPrefix = debugContext.getPathName();
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
                    attentionsGrad, debugPrefix, std::move(options),
                    planningCache);
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
  const auto debugPrefix = debugContext.getPathName();
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
  return gruBwdImpl(graph, params, prog, fwdOutputInit, fwdIntermediatesSeq,
                    weights, fwdInputSeq, fwdOutput, gradLayerNext, inputGrad,
                    bwdIntermediates, nullptr, timeSteps, attScores,
                    attentionsGrad, debugPrefix, std::move(options),
                    planningCache);
}

static GruWeights
gruWUImpl(Graph &graph, const GruParams &params, program::Sequence &prog,
          const Tensor &fwdOutputInit, const Tensor &fwdIntermediatesSeq,
          const Tensor &bwdIntermediatesSeq, const GruWeights &weights,
          const Tensor &input, const Tensor &output,
          const std::string &debugPrefix, const GruOpts &options,
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

  GruWeights weightGrads = createWeightAccumulators(
      graph, weights, bwdIntermediatesSeq[0], options, debugPrefix);
  zeroWeightAccumulators(graph, prog, weightGrads, debugPrefix);

  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start = graph.addConstant(UNSIGNED_INT, {1}, params.timeSteps - 1,
                                 debugPrefix + "/start");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(start, 0);
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  auto sliceLoopBody = Sequence();
  Tensor prevStepOut =
      dynamicSlice(graph, fwdOutputNew, seqIdx, {0}, {1}, sliceLoopBody,
                   debugPrefix + "/getPrevStepOut")
          .squeeze({0});
  Tensor fwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1}, sliceLoopBody,
                   debugPrefix + "/getFwdIntermediates")
          .squeeze({0});

  auto loop = Sequence();
  auto wuLoopBody = Sequence();
  {
    // Dynamic slice required state per-step
    // make a copy of the activations so that they are sliced efficiently
    auto inputCopy = createInput(graph, params, debugPrefix + "/inputCopy", {});
    prog.add(Copy(input, inputCopy));
    auto prevLayerOut =
        dynamicSlice(graph, inputCopy, seqIdx, {0}, {1}, sliceLoopBody,
                     debugPrefix + "/prevLayerActsWu")
            .squeeze({0});
    auto bwdIntermediates =
        dynamicSlice(graph, bwdIntermediatesSeq, seqIdx, {0}, {1},
                     sliceLoopBody, debugPrefix + "/getBwdIntermediates")
            .squeeze({0});
    subInPlace(graph, seqIdx, one, sliceLoopBody, debugPrefix + "/seqIdxDecr");
    loop.add(sliceLoopBody);

    basicGruParamUpdate(graph, prevLayerOut, prevStepOut, fwdIntermediates,
                        bwdIntermediates, weightGrads, wuLoopBody, options,
                        debugPrefix, planningCache);
    loop.add(wuLoopBody);
  }
  prog.add(Repeat(params.timeSteps - 1, loop));
  prog.add(sliceLoopBody);
  prog.add(wuLoopBody);

  weightGrads =
      basicGruParamUpdateFinal(graph, weights, weightGrads, prog, debugPrefix);

  return weightGrads;
}

GruWeights gruWU(Graph &graph, const GruParams &params, program::Sequence &prog,
                 const Tensor &fwdOutputInit, const Tensor &fwdIntermediates,
                 const Tensor &bwdIntermediates, const GruWeights &weights_,
                 const Tensor &input, const Tensor &output,
                 const poplar::DebugContext &debugContext,
                 const poplar::OptionFlags &options_,
                 poplin::matmul::PlanningCache *planningCache) {
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);
  validateParams(params);
  auto options = parseOptions(options_);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  auto grads = gruWUImpl(graph, params, prog, fwdOutputInit, fwdIntermediates,
                         bwdIntermediates, weights, input, output, debugPrefix,
                         std::move(options), planningCache);
  return toCellOrder(std::move(grads), params.cellOrder);
}

GruWeights augruWU(Graph &graph, const GruParams &params,
                   program::Sequence &prog, const Tensor &fwdOutputInit,
                   const Tensor &fwdIntermediates,
                   const Tensor &bwdIntermediates, const GruWeights &weights_,
                   const Tensor &input, const Tensor &output,
                   const poplar::DebugContext &debugContext,
                   const poplar::OptionFlags &options_,
                   poplin::matmul::PlanningCache *planningCache) {
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("gruWU(steps={}, batch {} x layers {}, name{}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);

  validateParams(params);
  auto options = parseOptions(options_);

  auto weights = fromCellOrder(weights_, params.cellOrder);

  auto grads = gruWUImpl(graph, params, prog, fwdOutputInit, fwdIntermediates,
                         bwdIntermediates, weights, input, output, debugPrefix,
                         std::move(options), planningCache);
  return toCellOrder(std::move(grads), params.cellOrder);
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
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);
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
      debugPrefix, options, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(
        graph, params, prog, fwdOutputInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, debugPrefix, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
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
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("gruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);
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
      debugPrefix, options, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(
        graph, params, prog, fwdOutputInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, debugPrefix, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
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
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);
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
      attentionsGrad, debugPrefix, options, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(
        graph, params, prog, fwdOutputInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, debugPrefix, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
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
  const auto debugPrefix = debugContext.getPathName();
  logging::popnn::info("auGruBwdWithWU(steps={}, batch {} x layers {}, name {}",
                       params.timeSteps, params.batchSize, params.layerSizes,
                       debugPrefix);
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
      attentionsGrad, debugPrefix, options, planningCache);

  if (!interleaveWU) {
    weightsGrad = gruWUImpl(
        graph, params, prog, fwdOutputInit, fwdIntermediates, bwdIntermediates,
        weights, input, output, debugPrefix, std::move(options), planningCache);
  }

  weightsGrad_ = toCellOrder(weightsGrad, params.cellOrder);
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
