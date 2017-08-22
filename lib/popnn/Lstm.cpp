#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <popstd/HadamardProduct.hpp>
#include <popstd/VertexTemplates.hpp>
#include <popnn/Lstm.hpp>
#include <popconv/Convolution.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;



static Tensor processBasicLstmUnit(Graph &graph,
                                   Tensor prevAct,
                                   Tensor bBiases,
                                   Tensor prevOutput,
                                   Tensor weightsInput,
                                   Tensor weightsOutput,
                                   Sequence &prog,
                                   MatMulOptions &mmOpt,
                                   popnn::NonLinearityType nonLinearityType,
                                   const std::string debugStr) {

  auto prodInp =
          matMul(graph, prevAct, weightsInput,
                 prog, debugStr + "/WeighInput", mmOpt);
  auto output =
          matMul(graph, prevOutput, weightsOutput,
                 prog, debugStr + "/WeighOutput", mmOpt);

  addTo(graph, output, prodInp, 1.0, prog);
  addTo(graph, output, bBiases, 1.0, prog);
  nonLinearity(graph, nonLinearityType, output, prog, debugStr);
  return output;
}

static Tensor
processBasicLstmUnitPreWeighted(Graph &graph,
                                Tensor weightedIn,
                                Tensor bBiases,
                                Tensor prevOutput,
                                Tensor weightsOutput,
                                Sequence &prog,
                                MatMulOptions &mmOpt,
                                popnn::NonLinearityType nonLinearityType,
                                const std::string debugStr) {
  auto output =
          matMul(graph, prevOutput, weightsOutput,
                 prog, debugStr + "/WeighOutput", mmOpt);

  addTo(graph, output, weightedIn, 1.0, prog);
  addTo(graph, output, bBiases, 1.0, prog);
  nonLinearity(graph, nonLinearityType, output, prog, debugStr);
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
  mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  mmOpt.partialsType = partialsType;
  auto seqSize = prevAct.dim(0);
  auto batchSize = prevAct.dim(1);
  auto inputSize = prevAct.dim(2);
  Tensor weightsInput;
  if (preweights) {
    weightsInput =
        createMatMulInputRHS(graph, dType,
                             {seqSize * batchSize, inputSize},
                             {inputSize,
                              BASIC_LSTM_CELL_NUM_UNITS * outputSize},
                             "weightsIn",
                             mmOpt)
        .reshape({inputSize, BASIC_LSTM_CELL_NUM_UNITS, outputSize})
        .dimShuffle({1, 0, 2});
  } else {
    Tensor weightsInput = graph.addTensor(dType,
                                          {0, inputSize, outputSize},
                                          "");
    for (auto u = 0U; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
      auto wInp =
          createMatMulInputRHS(graph, dType,
                               {batchSize, inputSize},
                               {inputSize, outputSize},
                               "weightsInput" + std::to_string(u), mmOpt);
      weightsInput = append(weightsInput, wInp);
    }
  }
  return weightsInput;
}

Tensor createWeightsOutput(Graph &graph, const std::string &dType,
                           const std::string &partialsType,
                           const Tensor &cellState,
                           unsigned outputSize) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  Tensor weightsOutput = graph.addTensor(dType,
                                         {0, outputSize, outputSize},
                                         "");
  const auto batchSize = cellState.dim(0);
  for (auto u = 0U; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    auto wOut = createMatMulInputRHS(graph, dType,
                                     {batchSize, outputSize},
                                     {outputSize, outputSize},
                                     "weightsOutput" + std::to_string(u),
                                     mmOpt);
    weightsOutput = append(weightsOutput, wOut);
  }
  return weightsOutput;
}

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize) {
  uint64_t multsWeighInp =
      static_cast<uint64_t>(inputSize) * outputSize * batchSize * sequenceSize;
  uint64_t multsWeighOut =
      static_cast<uint64_t>(outputSize) * outputSize * batchSize * sequenceSize;

  uint64_t addsWeighInp  =
      static_cast<uint64_t>(inputSize - 1) * outputSize * batchSize
                                           * sequenceSize;
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
  mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  auto sequenceSize = in_.dim(0);
  auto batchSize = in_.dim(1);
  auto inputSize = in_.dim(2);
  auto in = in_.reshape({sequenceSize * batchSize, inputSize});
  auto outputSize = weightsInput_.dim(2);
  auto weightsInput =
      weightsInput_.dimShuffle({1, 0, 2})
                   .reshape({inputSize,
                             BASIC_LSTM_CELL_NUM_UNITS * outputSize});

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
    mapTensorLinearly(graph, biases[u]);

    auto unitBias = biases[u].broadcast(batchSize, 0)
                                       .reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  mmOpt.cache = &cache;

  Tensor actOut = graph.addTensor(dType, {0, batchSize, outputSize}, "actOut");

  for (auto s = 0U; s != sequenceSize; ++s) {
    const std::string baseStr = debugPrefix
                                + "/BasicLstmCell/"
                                + std::to_string(s);

    auto prevOutputThisStep = s == 0 ? prevOutput : actOut[s-1];

    /* Forget gate */
    auto forgetGate =
    processBasicLstmUnitPreWeighted(
                         graph,
                         weightedIn[BASIC_LSTM_CELL_FORGET_GATE][s],
                         bBiases[BASIC_LSTM_CELL_FORGET_GATE],
                         prevOutputThisStep,
                         weightsOutput[BASIC_LSTM_CELL_FORGET_GATE],
                         prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/ForgetGate");

    auto inputGate =
    processBasicLstmUnitPreWeighted(
                         graph,
                         weightedIn[BASIC_LSTM_CELL_INPUT_GATE][s],
                         bBiases[BASIC_LSTM_CELL_INPUT_GATE],
                         prevOutputThisStep,
                         weightsOutput[BASIC_LSTM_CELL_INPUT_GATE],
                         prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/InputGate");

    auto outputGate =
    processBasicLstmUnitPreWeighted(
                         graph,
                         weightedIn[BASIC_LSTM_CELL_OUTPUT_GATE][s],
                         bBiases[BASIC_LSTM_CELL_OUTPUT_GATE],
                         prevOutputThisStep,
                         weightsOutput[BASIC_LSTM_CELL_OUTPUT_GATE],
                         prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/OutputGate");

    auto candidate =
    processBasicLstmUnitPreWeighted(
                         graph,
                         weightedIn[BASIC_LSTM_CELL_CANDIDATE][s],
                         bBiases[BASIC_LSTM_CELL_CANDIDATE],
                         prevOutputThisStep,
                         weightsOutput[BASIC_LSTM_CELL_CANDIDATE],
                         prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_TANH,
                         baseStr + "/Candidate");

    hadamardProduct(graph, cellState, forgetGate, prog,
                    baseStr + "/HadamardProd/ForgetGate");
    hadamardProduct(graph, candidate, inputGate, prog,
                    baseStr + "/HadamardProd/InputGate");
    addTo(graph, cellState, candidate, 1.0, prog);

    auto output = graph.addTensor(dType, {batchSize, outputSize},
                                  "output");
    mapTensorLinearly(graph, output);

    prog.add(Copy(cellState, output));
    nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_TANH,
                 output, prog);
    hadamardProduct(graph, output, outputGate, prog,
                    baseStr + "/HadamardProd/OutputGate");
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
    mapTensorLinearly(graph, biases[u]);

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

    /* Forget gate */
    auto forgetGate =
        processBasicLstmUnit(graph, in[s],
                             bBiases[BASIC_LSTM_CELL_FORGET_GATE],
                             prevOutputThisStep,
                             weightsInput[BASIC_LSTM_CELL_FORGET_GATE],
                             weightsOutput[BASIC_LSTM_CELL_FORGET_GATE],
                             prog, mmOpt,
                             popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                             baseStr + "/ForgetGate");

    auto inputGate =
        processBasicLstmUnit(graph, in[s],
                             bBiases[BASIC_LSTM_CELL_INPUT_GATE],
                             prevOutputThisStep,
                             weightsInput[BASIC_LSTM_CELL_INPUT_GATE],
                             weightsOutput[BASIC_LSTM_CELL_INPUT_GATE],
                             prog, mmOpt,
                             popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                             baseStr + "/InputGate");

    auto outputGate =
        processBasicLstmUnit(graph, in[s],
                             bBiases[BASIC_LSTM_CELL_OUTPUT_GATE],
                             prevOutputThisStep,
                             weightsInput[BASIC_LSTM_CELL_OUTPUT_GATE],
                             weightsOutput[BASIC_LSTM_CELL_OUTPUT_GATE],
                             prog, mmOpt,
                             popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                             baseStr + "/OutputGate");
    auto candidate =
        processBasicLstmUnit(graph, in[s],
                             bBiases[BASIC_LSTM_CELL_CANDIDATE],
                             prevOutputThisStep,
                             weightsInput[BASIC_LSTM_CELL_CANDIDATE],
                             weightsOutput[BASIC_LSTM_CELL_CANDIDATE],
                             prog, mmOpt,
                             popnn::NonLinearityType::NON_LINEARITY_TANH,
                             baseStr + "/Candidate");

    hadamardProduct(graph, cellState, forgetGate, prog,
                    baseStr + "/HadamardProd/ForgetGate");
    hadamardProduct(graph, candidate, inputGate, prog,
                    baseStr + "/HadamardProd/InputGate");
    addTo(graph, cellState, candidate, 1.0, prog);

    auto output = graph.addTensor(dType, {batchSize, outputSize},
                                  "output");
    mapTensorLinearly(graph, output);

    prog.add(Copy(cellState, output));
    nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_TANH,
                 output, prog);
    hadamardProduct(graph, output, outputGate, prog,
                    baseStr + "/HadamardProd/OutputGate");
    actOut = append(actOut, output);
  }
  return actOut;
}
}
} // namespace lstm
