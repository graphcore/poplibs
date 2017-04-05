#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/ActivationMapping.hpp>
#include <popnn/NonLinearity.hpp>
#include <popstd/VertexTemplates.hpp>
#include <popnn/Lstm.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;



static void processBasicLstmUnit(Graph &graph,
                            Tensor prevAct,
                            Tensor bBiases,
                            Tensor prevOutput,
                            Tensor weightsInput,
                            Tensor weightsOutput,
                            Tensor output,
                            Sequence &prog,
                            MatMulOptions &mmOpt,
                            popnn::NonLinearityType nonLinearityType,
                            const std::string debugStr) {

  auto prodInp =
          matMul(graph, prevAct,
                 weightsInput.transpose(),
                 prog, debugStr + "/WeighInput", mmOpt);
  auto prodOut =
          matMul(graph, prevOutput,
                 weightsOutput.transpose(),
                 prog, debugStr + "/WeighOutput", mmOpt);

  prog.add(Copy(prodOut, output));

  addTo(graph, output, prodInp, 1.0, prog);
  addTo(graph, output, bBiases, 1.0, prog);
  nonLinearity(graph, nonLinearityType, output, prog, debugStr);
}


static void hadamardProduct(Graph &graph,
                            Tensor input,
                            Tensor inputOutput,
                            Sequence &prog,
                            const std::string &debugPrefix) {
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  const auto numWorkers = graph.getDevice().getDeviceInfo().numWorkerContexts;
  const auto dType = graph.getTensorElementType(input);
  const auto dataPathWidth =  graph.getDevice().getDeviceInfo().dataPathWidth;

  const auto batchSize = input.dim(0);

  ComputeSet hadamardCs = graph.addComputeSet(debugPrefix + "/HadamardProd");

  for (auto b = 0U; b != batchSize; ++b) {
    auto mapping = computeActivationsMapping(graph, input[b], b, batchSize);
    for (auto tile = 0U; tile != numTiles; ++tile) {
      const auto tileActivationsBegin = mapping[tile];
      const auto tileActivationsEnd = mapping[tile + 1];
      auto numActivations = tileActivationsEnd - tileActivationsBegin;

      if (!numActivations)
        continue;

      /* divide amongst workers */
      unsigned activationsPerWorker = (numActivations + numWorkers - 1)
                                      / numWorkers;
      for (unsigned worker = 0; worker != numWorkers && numActivations;
            ++worker) {
        auto actsThisWorker = numActivations >= activationsPerWorker ?
                                activationsPerWorker : numActivations;
        auto sliceBegin = tileActivationsBegin + worker * activationsPerWorker;
        auto sliceEnd = sliceBegin + actsThisWorker;
        auto in = input[b].slice({sliceBegin, sliceEnd});
        auto inOut = inputOutput[b].slice({sliceBegin, sliceEnd});

        auto v = graph.addVertex(hadamardCs,
                                 templateVertex("popnn::HadamardProduct",
                                                dType),
                                 {
                                  {"in", in},
                                  {"inOut", inOut}
                                 });
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setTileMapping(v, tile);
        numActivations -= actsThisWorker;
      }
    }
  }
  if (!graph.getComputeSet(hadamardCs).empty()) {
    prog.add(Execute(hadamardCs));
  }
}

namespace popnn {
namespace lstm {

uint64_t getBasicLstmCellFwdFlops(unsigned sequenceSize, unsigned batchSize,
                                  unsigned inputSize, unsigned outputSize) {
  uint64_t multsWeighInp = inputSize * outputSize * batchSize * sequenceSize;
  uint64_t multsWeighOut = outputSize * outputSize * batchSize * sequenceSize;
  uint64_t addsWeighInp  = (inputSize - 1) * outputSize * batchSize
                                          * sequenceSize;
  uint64_t addsWeighOut  = (outputSize - 1) * outputSize * batchSize
                                          * sequenceSize;
  uint64_t hadamardProd = 3 * sequenceSize * batchSize * outputSize;
  uint64_t cellStateAdd = sequenceSize * batchSize * outputSize;

  return 4 * (multsWeighInp + multsWeighOut + addsWeighInp + addsWeighOut)
         + hadamardProd + cellStateAdd;
}


Tensor basicLstmCellForwardPass(Graph  &graph,
                                Tensor in, Tensor biases,
                                Tensor prevOutput, Tensor weightsInput,
                                Tensor weightsOutput, Tensor cellState,
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
  assert(weightsInput.dim(1) == outputSize);
  assert(weightsInput.dim(2) == inputSize);

  assert(weightsOutput.dim(0) == BASIC_LSTM_CELL_NUM_UNITS);
  assert(weightsOutput.dim(1) == outputSize);
  assert(weightsOutput.dim(2) == outputSize);

  const auto dType = graph.getTensorElementType(in);

  auto bBiases = graph.addTensor(dType, {0, batchSize, outputSize}, "bbiases");

  for (unsigned u = 0; u != BASIC_LSTM_CELL_NUM_UNITS; ++u) {
    mapTensor(graph, biases[u]);

    auto unitBias = biases[u].broadcast(batchSize, 0)
                                       .reshape({batchSize, outputSize});
    bBiases = append(bBiases, unitBias);
  }

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = false;
  mmOpt.cache = &cache;

  Tensor actOut = graph.addTensor(dType, {0, batchSize, outputSize}, "actOut");

  /* create and map temporary tensors */
  auto forgetGate = graph.addTensor(dType, {batchSize, outputSize},
                                    "forgetGate");
  mapActivations(graph, forgetGate);

  auto inputGate = graph.addTensor(dType, {batchSize, outputSize}, "inputGate");
  mapActivations(graph, inputGate);

  auto candidate = graph.addTensor(dType, {batchSize, outputSize}, "candidate");
  mapActivations(graph, candidate);

  auto outputGate = graph.addTensor(dType, {batchSize, outputSize},
                                    "outputGate");
  mapActivations(graph, outputGate);

  for (auto s = 0U; s != sequenceSize; ++s) {
    const std::string baseStr = debugPrefix
                                + "/BasicLstmCell/"
                                + std::to_string(s);

    auto prevOutputThisStep = s == 0 ? prevOutput : actOut[s-1];

    /* Forget gate */
    processBasicLstmUnit(graph, in[s],
                         bBiases[BASIC_LSTM_CELL_FORGET_GATE],
                         prevOutputThisStep,
                         weightsInput[BASIC_LSTM_CELL_FORGET_GATE],
                         weightsOutput[BASIC_LSTM_CELL_FORGET_GATE],
                         forgetGate, prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/ForgetGate");

    processBasicLstmUnit(graph, in[s],
                         bBiases[BASIC_LSTM_CELL_INPUT_GATE],
                         prevOutputThisStep,
                         weightsInput[BASIC_LSTM_CELL_INPUT_GATE],
                         weightsOutput[BASIC_LSTM_CELL_INPUT_GATE],
                         inputGate, prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/InputGate");

    processBasicLstmUnit(graph, in[s],
                         bBiases[BASIC_LSTM_CELL_OUTPUT_GATE],
                         prevOutputThisStep,
                         weightsInput[BASIC_LSTM_CELL_OUTPUT_GATE],
                         weightsOutput[BASIC_LSTM_CELL_OUTPUT_GATE],
                         outputGate, prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_SIGMOID,
                         baseStr + "/OutputGate");

    processBasicLstmUnit(graph, in[s],
                         bBiases[BASIC_LSTM_CELL_CANDIDATE],
                         prevOutputThisStep,
                         weightsInput[BASIC_LSTM_CELL_CANDIDATE],
                         weightsOutput[BASIC_LSTM_CELL_CANDIDATE],
                         candidate, prog, mmOpt,
                         popnn::NonLinearityType::NON_LINEARITY_TANH,
                         baseStr + "/Candidate");

    hadamardProduct(graph, forgetGate, cellState, prog,
                    baseStr + "/HadamardProd/ForgetGate");
    hadamardProduct(graph, inputGate, candidate, prog,
                    baseStr + "/HadamardProd/InputGate");
    addTo(graph, cellState, candidate, 1.0, prog);

    auto output = graph.addTensor(dType, {batchSize, outputSize},
                                  "output");
    mapActivations(graph, output);

    prog.add(Copy(cellState, output));
    nonLinearity(graph, popnn::NonLinearityType::NON_LINEARITY_TANH,
                 output, prog);
    hadamardProduct(graph, outputGate, output, prog,
                    baseStr + "/HadamardProd/OutputGate");
    actOut = append(actOut, output);
  }
  return actOut;
}
}
} // namespace lstm
