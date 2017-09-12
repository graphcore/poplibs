#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <popstd/HadamardProduct.hpp>
#include <popstd/VertexTemplates.hpp>
#include <popnn/Lstm.hpp>
#include <popstd/Operations.hpp>
#include <popconv/Convolution.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;

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
applyGateNonlinearities(Graph &graph, Tensor &t, Sequence &prog,
                        const std::string &debugStr) {
  auto sigmoidIn = concat({t[BASIC_LSTM_CELL_INPUT_GATE],
                           t[BASIC_LSTM_CELL_FORGET_GATE],
                           t[BASIC_LSTM_CELL_OUTPUT_GATE]});
  nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
               sigmoidIn, prog, debugStr);
  nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_TANH,
               t[BASIC_LSTM_CELL_CANDIDATE], prog, debugStr);
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

// Computes the output before nonlinearities to all the units are applies
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
                       const std::string &debugStr) {
  assert(gatesOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  auto forgetGate = gatesOutput[BASIC_LSTM_CELL_FORGET_GATE];
  auto candidate = gatesOutput[BASIC_LSTM_CELL_CANDIDATE];
  auto outputGate = gatesOutput[BASIC_LSTM_CELL_OUTPUT_GATE];
  auto inputGate = gatesOutput[BASIC_LSTM_CELL_INPUT_GATE];
  const auto dType = cellState.elementType();
  addTo(graph, gatesOutput, bBiases, 1.0, prog, debugStr + "/AddBias");
  applyGateNonlinearities(graph, gatesOutput, prog, debugStr);
  hadamardProduct(graph, concat(cellState, candidate),
                  concat(forgetGate, inputGate), prog,
                  debugStr + "/HadamardProd/{Forget + Input}Gate");
  addTo(graph, cellState, candidate, 1.0, prog, debugStr + "/AddCellCand");
  auto output = tanh(graph, cellState, prog, debugStr);
  hadamardProduct(graph, output, outputGate, prog,
                  debugStr + "/HadamardProd/OutputGate");
  return output;
}

namespace popnn {
namespace lstm {

Tensor createInput(Graph &graph, const std::string &dType,
                   unsigned sequenceSize, unsigned batchSize,
                   unsigned inputSize, unsigned outputSize,
                   const std::string &name) {
  auto fcOutputSize = BASIC_LSTM_CELL_NUM_UNITS * outputSize;
  auto fcInputSize = inputSize;
  auto fcBatchSize = sequenceSize * batchSize;
  auto in = createMatMulInputLHS(graph, dType,
                                 {fcBatchSize, fcInputSize},
                                 {fcInputSize, fcOutputSize},
                                 name);
  return in.reshape({sequenceSize, batchSize, inputSize});
}

Tensor createWeightsInput(Graph &graph, const std::string &dType,
                          const std::string &partialsType,
                          const Tensor &prevAct, unsigned outputSize,
                          bool preweights) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  auto seqSize = prevAct.dim(0);
  auto batchSize = prevAct.dim(1);
  auto inputSize = prevAct.dim(2);
  std::vector<std::size_t> aShape(2);
  aShape[0] = preweights ? seqSize * batchSize : batchSize;
  aShape[1] = inputSize;

  auto weightsInput =
      createMatMulInputRHS(graph, dType,
                           aShape,
                           {inputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                           "weightsIn",
                           mmOpt);
  return unflattenUnits(weightsInput);
}

Tensor createWeightsOutput(Graph &graph, const std::string &dType,
                           const std::string &partialsType,
                           const Tensor &cellState,
                           unsigned outputSize) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  const auto batchSize = cellState.dim(0);
  auto weightsOutput =
      createMatMulInputRHS(graph, dType,
                           {batchSize, outputSize},
                           {outputSize, BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                           "weightsOut",
                           mmOpt);
  return unflattenUnits(weightsOutput);
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

Tensor
calcSequenceWeightedInputs(Graph &graph,
                           const Tensor &in_,
                           const Tensor &weightsInput_,
                           program::Sequence &prog,
                           const std::string partialsTypeStr,
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


Tensor basicLstmCellForwardPassWeightedInputs(Graph &graph,
                                              const Tensor &weightedIn,
                                              const Tensor &biases,
                                              const Tensor &prevOutput,
                                              const Tensor &weightsOutput,
                                              const Tensor &cellState,
                                              Sequence &prog,
                                              const std::string partialsTypeStr,
                                              const std::string &debugPrefix) {

  const unsigned sequenceSize = weightedIn.dim(1);
  const unsigned batchSize = weightedIn.dim(2);
  const unsigned outputSize = cellState.dim(1);

  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

  const auto dType = weightedIn.elementType();

  auto bBiases = graph.addTensor(dType, {0, batchSize, outputSize}, "bbiases");
  for (unsigned u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto unitBias = biases[u].broadcast(batchSize, 0)
                             .reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.cache = &cache;

  Tensor actOut = graph.addTensor(dType, {0, batchSize, outputSize}, "actOut");

  auto weightedInReshaped = weightedIn.dimShuffle({1, 0, 2, 3});
  for (auto s = 0U; s != sequenceSize; ++s) {
    const std::string baseStr = debugPrefix
                                + "/BasicLstmCell/"
                                + std::to_string(s);

    auto prevOutputThisStep = s == 0 ? prevOutput : actOut[s-1];

    auto unitsOutput =
        basicLstmUnitsNlInputPreWeighted(graph,
                                         weightedInReshaped[s],
                                         prevOutputThisStep,
                                         weightsOutput,
                                         prog, mmOpt,
                                         baseStr + "/ProcessUnits");
    if (s == 0) {
      for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
        graph.setTileMapping(biases[u],
                             graph.getTileMapping(unitsOutput[u][0]));
      }
    }
    auto output = basicLstmComputeOutput(graph, unitsOutput, cellState, bBiases,
                                         prog, baseStr);
    actOut = append(actOut, output);
  }
  return actOut;
}

Tensor basicLstmCellForwardPass(Graph &graph,
                                const Tensor &in, const Tensor &biases,
                                const Tensor &prevOutput,
                                const Tensor &weightsInput,
                                const Tensor &weightsOutput,
                                const Tensor &cellState,
                                Sequence &prog,
                                const std::string partialsTypeStr,
                                const std::string &debugPrefix) {

  const unsigned sequenceSize = in.dim(0);
  const unsigned batchSize = in.dim(1);
  const unsigned outputSize = cellState.dim(1);
#ifndef NDEBUG
  const unsigned inputSize = in.dim(2);
#endif
  assert(weightsInput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsInput.dim(1) == inputSize);
  assert(weightsInput.dim(2) == outputSize);

  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

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
  mmOpt.cache = &cache;

  Tensor actOut = graph.addTensor(dType, {0, batchSize, outputSize}, "actOut");
  /* create and map temporary tensors */
  for (auto s = 0U; s != sequenceSize; ++s) {
    const std::string baseStr = debugPrefix
                                + "/BasicLstmCell/"
                                + std::to_string(s);

    auto prevOutputThisStep = s == 0 ? prevOutput : actOut[s-1];

    auto unitsOutput =
        basicLstmUnitsNlInput(graph, in[s],
                              prevOutputThisStep,
                              weightsInput,
                              weightsOutput,
                              prog, mmOpt,
                              baseStr + "/ProcessUnits");
    if (s == 0) {
      for (auto u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
        graph.setTileMapping(biases[u],
                             graph.getTileMapping(unitsOutput[u][0]));
      }
    }
    auto output = basicLstmComputeOutput(graph, unitsOutput, cellState, bBiases,
                                         prog, baseStr);
    actOut = append(actOut, output);
  }
  return actOut;
}
}
} // namespace lstm
