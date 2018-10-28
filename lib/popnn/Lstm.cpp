#include <popnn/Lstm.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <poputil/VertexTemplates.hpp>
#include <popops/ElementWise.hpp>
#include <poplin/Convolution.hpp>
#include <poplin/ConvUtil.hpp>
#include <popops/Zero.hpp>
#include <poputil/Util.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/gcd.hpp"
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
enum FwdIntermediates {
  LSTM_FWD_INTERMEDIATE_PREV_CELL_STATE,
  LSTM_FWD_INTERMEDIATE_FORGET_GATE,
  LSTM_FWD_INTERMEDIATE_INPUT_GATE,
  LSTM_FWD_INTERMEDIATE_CAND_TANH,
  LSTM_FWD_INTERMEDIATE_OUTPUT_GATE,
  LSTM_FWD_INTERMEDIATE_OUTPUT_TANH,
  LSTM_FWD_INTERMEDIATE_OUTPUT,
  LSTM_NUM_FWD_INTERMEDIATES
};

// Tensor elements maintained in backward state. The number of elements is a
// function of the amount of recomputation done in the weight update pass
enum BwdStateTensorElems {
  LSTM_BWD_STATE_GRAD_CELL_STATE = 0,
  LSTM_BWD_STATE_GRAD_ACT_GRAD,
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

static void
applyGateNonlinearities(Graph &graph,
                        const Tensor &t,
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
  auto weights = concat(weightsInput, weightsOutput, 1);
  return
      unflattenUnits(matMul(graph, concat(prevAct, prevOutput, 1),
                            flattenUnits(weights), prog,
                            debugStr + "/Weigh", mmOpt, cache));
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

/// Create a tensor with dimensions [sequenceLength, numGrains, grainSize]
/// that satisfies the following properties:
/// - Grains are never split across tiles.
/// - The tile mapping and layout is identical for each sub-tensor in the
///   sequence.
/// - The elements on a tile form a single contigous region where the
///   sequenceLength the outer dimension.
/// These properties make the tensor well suited for use with dynamic
/// slice / dynamic update
static Tensor
createDynamicSliceTensor(Graph &graph,
                         poplar::Type dataType,
                         unsigned sequenceLength,
                         unsigned numGrains, unsigned grainSize,
                         const std::string &name) {
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto grainsPerTile = (numGrains + numTiles - 1) / numTiles;
  const auto numUsedTiles =
      (numGrains + grainsPerTile - 1) / grainsPerTile;
  const auto grainsOnLastTile =
      numGrains - (numUsedTiles - 1) * grainsPerTile;
  auto tExcludingLast =
    graph.addVariable(dataType, {numUsedTiles - 1, sequenceLength,
                                 grainsPerTile, grainSize},
                      name);
  auto tLast =
    graph.addVariable(dataType, {sequenceLength, grainsOnLastTile, grainSize},
                      name);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    unsigned usedTileIndex = tile * numUsedTiles / numTiles;
    if (usedTileIndex != (tile + 1) * numUsedTiles / numTiles) {
      if (usedTileIndex + 1 == numUsedTiles) {
        graph.setTileMapping(tLast, tile);
      } else {
        graph.setTileMapping(tExcludingLast[usedTileIndex], tile);
      }
    }
  }
  return concat(
    tExcludingLast.dimRoll(0, 1).flatten(1, 3),
    tLast,
    1
  );
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
  OptionFlags mmOpt{
    { "partialsType", opt.partialsType.toString() },
    { "fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                                "TRAINING_FWD" }
  };

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

LstmState createInitialState(Graph &graph, const LstmParams &params,
                             const std::string &debugPrefix,
                             const OptionFlags &options,
                             matmul::PlanningCache *cache) {
  validateParams(params);
  auto initialOutput =
      createOutputTensor(graph, params, 1, debugPrefix + "/initialOutput")
          .squeeze({0});
  auto initialCellState = graph.clone(initialOutput,
                                      debugPrefix + "/initialCellState");
  return {initialOutput, initialCellState};
}

void zeroInitialState(Graph &graph, const LstmState &state,
                      Sequence &prog, const std::string &debugPrefix) {
  zero(graph, concat(state.output, state.cellState), prog, debugPrefix);
}

LstmWeights
createWeights(Graph &graph, const LstmParams &params,
              const std::string &name,
              const OptionFlags &options,
              poplin::matmul::PlanningCache *cache) {
  validateParams(params);
  auto opt = parseOptions(options);

  LstmWeights lstmWeights;
  OptionFlags mmOpt{
    { "partialsType", opt.partialsType.toString() },
    { "fullyConnectedPass", opt.inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" }
  };
  auto inputSize = params.layerSizes[0];
  auto outputSize = params.layerSizes[1];

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
      lstmWeights.inputWeights = unflattenUnits(weightsInput);
    }
    auto weightsOutput =
        createMatMulInputRHS(graph, params.dataType,
                             {params.batchSize, outputSize},
                             {outputSize,
                              BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             name + "/weightsOut",
                             mmOpt, cache);
    lstmWeights.outputWeights = unflattenUnits(weightsOutput);
  } else {
    auto weights =
        createMatMulInputRHS(graph, params.dataType,
                             {params.batchSize, inputSize + outputSize},
                             {inputSize + outputSize,
                              BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             name + "/weights",
                             mmOpt, cache);
    lstmWeights.inputWeights = unflattenUnits(weights.slice(0, inputSize));
    lstmWeights.outputWeights =
        unflattenUnits(weights.slice(inputSize, inputSize + outputSize));
  }

  auto biases = graph.addVariable(params.dataType,
                                  {BASIC_LSTM_CELL_NUM_UNITS, outputSize},
                                  "biases");
  mapTensorLinearly(graph, biases);
  lstmWeights.biases = biases;
  return lstmWeights;
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

// Typically the matrix multiplication result is laid out in memory such
// that innermost dimension is groups batch elements. Try to rearrange the
// result so the innermost dimension of the underlying memory is groups of the
// specified number of outputs.
static Tensor matMulOutputPartialRearrange(Graph &graph, Tensor outputUnits,
                                           unsigned outputGrouping,
                                           Sequence &prog,
                                           const std::string &debugPrefix) {
  if (outputGrouping == 1)
    return outputUnits;
  auto output = flattenUnits(outputUnits);
  const auto batchGrouping =
      detectChannelGrouping(output.transpose());
  if (batchGrouping == 1)
    return outputUnits;
  const auto batchSize = output.dim(0);
  const auto outputSize = output.dim(1);
  auto groupedOutput =
      output.reshape({batchSize / batchGrouping, batchGrouping,
                      outputSize / outputGrouping, outputGrouping})
            .dimShuffle({0, 2, 3, 1});
  auto cs = graph.addComputeSet(debugPrefix + "/MatMulOutPartialTranspose");
  auto outputPartialTranspose = partialTranspose(graph, groupedOutput, cs);
  prog.add(Execute(cs));
  auto partiallyRearrangedOutput =
      outputPartialTranspose.dimShuffle({0, 2, 1, 3})
                            .reshape({batchSize, outputSize});
  return unflattenUnits(partiallyRearrangedOutput);
}

static void rearrangeMatMulOutput(Graph &graph, Tensor outputUnits,
                                  Tensor outputUnitsRearranged,
                                  Sequence &prog,
                                  const std::string &debugPrefix) {
  const auto outputGrouping = detectChannelGrouping(outputUnitsRearranged);
  // Try to rearrange the innermost dimension of the underlying storage to
  // improve the efficiency of the subsequent copy.
  outputUnits = matMulOutputPartialRearrange(graph, outputUnits,
                                             outputGrouping, prog,
                                             debugPrefix);
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
                             const Type &partialsType,
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
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" :
                                            "TRAINING_FWD" }
  };

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
  rearrangeMatMulOutput(graph, unitsOutput, unitsOutputRearranged,
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
                         const Type &partialsType,
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
                               weightsOutput, prog, partialsType,
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
                                const Type &partialsType,
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
                               weightsOutput, prog, partialsType,
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

static Tensor getFwdIntermediates(const LstmState &state,
                                  const LstmState &newState,
                                  const LstmInternalState &internalState,
                                  const LstmParams &params) {
  Tensor intermediates = concat(state.cellState.expand({0}),
                                internalState.getAsTensor());
  if (!params.outputFullSequence) {
    // TODO: It may be cheaper to save the previous output rather than
    // the output for the current step here for the backward pass so that
    // when we aren't saving the full output sequence we can avoid
    // unrolling the last step in the backward pass.
    intermediates = concat(intermediates, newState.output.expand({0}));
  }
  return intermediates;
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
            loop, opt.partialsType, opt.inferenceOnly, debugPrefix,
            cache);
    auto intermediates =
      getFwdIntermediates(state, newState, internalState, params);
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
        loop, opt.partialsType, opt.inferenceOnly, debugPrefix,
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
                 const Type &partialsType,
                 const std::string &debugPrefix,
                 matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmBwd";
  auto outputGrad = stateGrad.output;
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

  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", "TRAINING_BWD" },
    { "inputRHSIsPreArranged", "true"}
  };

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
    gradientIn = out.slice(0, inputSize, 1);
    gradientPrevStep = out.slice(inputSize, inputSize + outputSize, 1);
  } else {
    gradientPrevStep =
      matMul(graph, grads, weightsTransposed, prog, fPrefix + "/PrevStepGrad",
             mmOpt, cache);
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
                      const Type &partialsType,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  return
    backwardStepImpl(graph, gradNextLayer, fwdIntermediates,
                     stateGrad, &weightsInput, weightsOutput, initProg, prog,
                     partialsType, debugPrefix, cache);
}

std::pair<LstmState, Tensor>
basicLstmBackwardStep(Graph &graph,
                      const Tensor *gradNextLayer,
                      const Tensor &fwdIntermediates,
                      const LstmState &stateGrad,
                      const Tensor &weightsOutput,
                      Sequence &initProg,
                      Sequence &prog,
                      const Type &partialsType,
                      const std::string &debugPrefix,
                      matmul::PlanningCache *cache) {
  LstmState prevStateGrad;
  Tensor bwdIntermediates;
  std::tie(prevStateGrad, std::ignore, bwdIntermediates) =
    backwardStepImpl(graph, gradNextLayer, fwdIntermediates,
                     stateGrad, nullptr, weightsOutput, initProg, prog,
                     partialsType, debugPrefix, cache);
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
                     const Type &partialsType,
                     const std::string &debugPrefix,
                     matmul::PlanningCache *cache) {
  const auto fPrefix = debugPrefix + "/LstmDeltas";
  OptionFlags mmOpt{
    { "partialsType", partialsType.toString() },
    { "fullyConnectedPass", "TRAINING_WU" }
  };
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
                           {popops::Operation::ADD, 1.0f, true},
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
                         const std::string &debugPrefix) {
  LstmWeights weightAccs;
  weightAccs.inputWeights =
    graph.clone(weights.inputWeights, debugPrefix + "/inputWeightsDeltaAcc");
  weightAccs.outputWeights =
    graph.clone(weights.outputWeights, debugPrefix + "/outputWeightsDeltaAcc");
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
                       const LstmWeights &weightsAcc) {
  popops::zero(graph,
               concat({weightsAcc.inputWeights.flatten(),
                       weightsAcc.outputWeights.flatten(),
                       weightsAcc.biases.flatten()}),
               prog);
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
  auto start = graph.addConstant(UNSIGNED_INT, {1}, seqSize - 1);
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  const auto batchSize = params.batchSize;

  Tensor gradLayerNextRearranged =
    createOutputTensor(graph, params, seqSize, "/gradLayerNextRearranged");
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
  auto fwdIntermediates =
    dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1},
                 sliceIntermediates,
                 debugPrefix + "/getFwdIntermediates").squeeze({0});
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
          options.partialsType, debugPrefix, cache);
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
                                prog, bwdLoopBody, options.partialsType,
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
                                              debugPrefix);
      zeroWeightAccumulators(graph, prog, *weightsGrad);

      basicLstmParamUpdate(
        graph, prevLayerOut, prevStepOut, bwdIntermediates,
        *weightsGrad, wuLoopBody, options.partialsType, debugPrefix, cache);
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
    throw poplib_error(std::string("The inputGradSeq argument should be ") +
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
    createWeightAccumulators(graph, weights, bwdIntermediatesSeq[0],
                             debugPrefix);
  zeroWeightAccumulators(graph, prog, weightGrads);

  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start = graph.addConstant(UNSIGNED_INT, {1}, params.timeSteps - 1);
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  auto sliceOutput = Sequence();
  Tensor prevStepOut;
  if (params.outputFullSequence) {
    prevStepOut =
      dynamicSlice(graph, output, seqIdx, {0}, {1}, sliceOutput,
                   debugPrefix + "/getPrevStepOut").squeeze({0});
  } else {
    auto prevFwdIntermediates =
      dynamicSlice(graph, fwdIntermediatesSeq, seqIdx, {0}, {1}, sliceOutput,
                   debugPrefix + "/getFwdIntermediates").squeeze({0});
    prevStepOut = prevFwdIntermediates[LSTM_FWD_INTERMEDIATE_OUTPUT];
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
      weightGrads, wuLoopBody, options.partialsType,
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
    throw poplib_error(std::string("The inputGradSeq argument should be ") +
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
