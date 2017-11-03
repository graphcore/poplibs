#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/Recurrent.hpp>
#include <popnn/NonLinearity.hpp>
#include <popreduce/Reduce.hpp>
#include <popstd/Zero.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;
using namespace popreduce;

// flatten the first two dimension of a 3D tensor to a 2D tensor used in matMul
static Tensor flattenSeqDims(const Tensor &t) {
  return t.reshape({t.dim(0) * t.dim(1), t.dim(2)});
}

// converts a 2D tensor to a 3D by splitting out dim(0) to {sequenceSize,
// batchSize}
static Tensor unflattenSeqDims(const Tensor &t, unsigned sequenceSize) {
  return t.reshape({sequenceSize, t.dim(0) / sequenceSize, t.dim(1)});
}

namespace popnn {
namespace rnn {

uint64_t getFwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool weightInput) {
  const uint64_t numMultsFeedFwd = static_cast<uint64_t>(inputSize) * outputSize
                                  * batchSize * sequenceSize;
  const uint64_t numAddsFeedFwd = static_cast<uint64_t>(outputSize)  *
                                  (inputSize - 1)  * batchSize * sequenceSize;

  const uint64_t numMultsIterate = static_cast<uint64_t>(outputSize) *
                                   outputSize * batchSize * sequenceSize;
  const uint64_t numAddsIterate = static_cast<uint64_t>(outputSize) *
                                  (outputSize - 1) * batchSize * sequenceSize
                                  + 2 * sequenceSize * outputSize;
  auto totalFlops = numMultsIterate + numAddsIterate;
  if (weightInput) {
    totalFlops += numMultsFeedFwd + numAddsFeedFwd;
  }

  return totalFlops;
}

uint64_t getBwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool calcInputGrad) {
  // non-linearity not included

  const uint64_t numMultsInput = static_cast<uint64_t>(inputSize) * outputSize
                                 * batchSize * sequenceSize;
  const uint64_t numAddsInput = static_cast<uint64_t>(outputSize)  *
                                (inputSize - 1)  * batchSize * sequenceSize;
  const uint64_t numMultsSum = static_cast<uint64_t>(outputSize) *
                               outputSize * batchSize * sequenceSize;
  const uint64_t numAddsSum = static_cast<uint64_t>(outputSize) *
                              (outputSize - 1) * batchSize * sequenceSize;
  return (numMultsInput + numAddsInput) * calcInputGrad + numMultsSum
         + numAddsSum;
}

uint64_t getWuFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize) {
  const uint64_t numMultsInput = static_cast<uint64_t>(inputSize) * outputSize
                                 * batchSize * sequenceSize;
  const uint64_t numMultsSum = static_cast<uint64_t>(outputSize) *
                               outputSize * batchSize * sequenceSize;
  const uint64_t numAddsInput = (static_cast<uint64_t>(batchSize) * sequenceSize
                               - 1) * outputSize * inputSize;
  const uint64_t numAddsSum = (static_cast<uint64_t>(batchSize) * sequenceSize
                               - 1) * outputSize * outputSize;
  const uint64_t bias = (static_cast<uint64_t>(batchSize) * sequenceSize
                        - 1) * outputSize;
  const uint64_t update = static_cast<uint64_t>(outputSize) * inputSize +
                          outputSize * outputSize + outputSize;

  return numMultsInput + numAddsInput + numMultsSum + numAddsSum + bias
         + update;
}


Tensor createFwdState(Graph &graph, const std::string &dType,
                      unsigned batchSize, unsigned outputSize,
                      Sequence &prog,
                      bool initState,
                      bool /* inferenceOnly */, const std::string &name) {
  auto state = createMatMulInputLHS(graph, dType,
                                    {batchSize, outputSize},
                                    {outputSize, outputSize},
                                    name + "/FwdState");
  if (initState) {
    popstd::zero(graph, state, prog, name);
  }
  return state;
}

Tensor getOutputFromFwdState(const Tensor &fwdState) {
  // In future we may need to keep a hidden state object rather than a tensor
  return fwdState;
}

Tensor createBwdState(Graph &graph,
                      const std::string &dType,
                      unsigned batchSize,
                      unsigned outputSize,
                      Sequence &prog,
                      const std::string &name) {
  auto state = createMatMulInputLHS(graph, dType,
                                    {batchSize, outputSize},
                                    {outputSize, outputSize},
                                    name + "/BwdState");
  popstd::zero(graph, state, prog, name + "/BwdState");
  return state;
}

Tensor createInput(Graph &graph,
                   unsigned sequenceSize,
                   unsigned batchSize,
                   unsigned inputSize,
                   unsigned outputSize,
                   const std::string &dType,
                   const std::string &partialsType,
                   bool inferenceOnly,
                   const std::string &name) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  std::vector<Tensor> input;
  input.emplace_back(createMatMulInputLHS(graph, dType,
                                 {batchSize, inputSize},
                                 {inputSize, outputSize},
                                 name, mmOpt).expand({0}));
  for (unsigned s = 1; s != sequenceSize; ++s) {
    input.emplace_back(graph.clone(input[0],
                                   name + "/" + std::to_string(s)));
  }
  return concat(input);
}


poplar::Tensor
createWeightsInput(Graph &graph,
                   unsigned /* sequenceSize */,
                   unsigned batchSize,
                   unsigned inputSize,
                   unsigned outputSize,
                   const std::string &dType,
                   const std::string &partialsType,
                   bool inferenceOnly,
                   const std::string &namePrefix) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  return
      createMatMulInputRHS(graph, dType,
                           {batchSize, inputSize},
                           {inputSize,  outputSize},
                           "weightsInput",
                           mmOpt);
}

poplar::Tensor
createWeightsFeedback(Graph &graph,
                      unsigned batchSize,
                      unsigned outputSize,
                      const std::string &dType,
                      const std::string &partialsType,
                      bool inferenceOnly,
                      const std::string &namePrefix) {
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsType;
  if (!inferenceOnly) {
    mmOpt.fullyConnectedPass = FullyConnectedPass::FWD;
  }
  return
      createMatMulInputRHS(graph, dType,
                           {batchSize, outputSize},
                           {outputSize,  outputSize},
                           "weightsRecurrent",
                           mmOpt);
}

Tensor forwardWeightInput(Graph &graph, const Tensor &actIn,
                          const Tensor &weights,
                          Sequence &prog,
                          const std::string partialsTypeStr,
                          const std::string &debugPrefix) {
  const unsigned sequenceSize = actIn.dim(0);
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.cache = &cache;

  return unflattenSeqDims(matMul(graph, flattenSeqDims(actIn), weights, prog,
                                 debugPrefix + "/RnnFwd/FeedFwd", mmOpt),
                          sequenceSize);
}

Tensor forwardIterate(Graph  &graph,
                      const Tensor &feedFwdIn,
                      const Tensor &initState,
                      const Tensor &weightsFeedback,
                      const Tensor &biases,
                      Sequence &prog,
                      popnn::NonLinearityType nonLinearityType,
                      const std::string partialsTypeStr,
                      const std::string &debugPrefix) {
  const unsigned sequenceSize = feedFwdIn.dim(0);
  const unsigned batchSize = feedFwdIn.dim(1);
  const unsigned outputSize = feedFwdIn.dim(2);

  auto bBiases = biases.broadcast(batchSize, 0)
                       .reshape({batchSize, outputSize});

  auto actOut = graph.addTensor(feedFwdIn.elementType(),
                                {0, batchSize, outputSize},
                                "actOut");
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.cache = &cache;

  for (unsigned s = 0U; s != sequenceSize; ++s) {
    const auto dbgStr = debugPrefix + "/RnnFwd/Feedback/"+ std::to_string(s);
    Tensor yP = s == 0 ? initState : actOut[s - 1];
    auto prod = matMul(graph, yP, weightsFeedback, prog, dbgStr,
                       mmOpt);
    addTo(graph, prod, feedFwdIn[s], 1.0, prog, dbgStr + "/Sum");

    /* Add broadcast bias */
    addTo(graph, prod, bBiases, 1.0, prog, dbgStr + "/Bias");

    nonLinearity(graph, nonLinearityType, prod, prog, dbgStr);

    actOut = append(actOut, prod);
  }
  return actOut;
}

std::pair<Tensor, Tensor>
backwardGradientStepImpl(Graph &graph,
                     const Tensor &gradientOut,
                     const Tensor &bwdState,
                     const Tensor &actOut,
                     const Tensor *weightsInput,
                     const Tensor *weightsFeedback,
                     Sequence &prog,
                     popnn::NonLinearityType nonLinearityType,
                     const std::string &partialsTypeStr,
                     const std::string &debugPrefix
                     ) {
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.cache = &cache;
  mmOpt.fullyConnectedPass = FullyConnectedPass::BWD;

  auto t = matMul(graph, bwdState, weightsFeedback->transpose(), prog,
                  debugPrefix + "/RnnBwd/Fb", mmOpt);
  addTo(graph, t, gradientOut, prog, debugPrefix + "/RnnBwd/AddOutGrad");

  auto newBwdState =
      nonLinearityInputGradient(graph, nonLinearityType, actOut, t, prog,
                                debugPrefix + "/RnnBwd");
  Tensor gradientAtInput;
  if (weightsInput != nullptr) {
    gradientAtInput =
      matMul(graph, newBwdState, weightsInput->transpose(), prog,
             debugPrefix + "/RnnBwd/Ff");

  }
  return std::make_pair(gradientAtInput, newBwdState);
}

std::pair<Tensor, Tensor>
backwardGradientStep(Graph &graph,
                     const Tensor &gradientOut,
                     const Tensor &bwdState,
                     const Tensor &actOut,
                     const Tensor &weightsInput,
                     const Tensor &weightsFeedback,
                     Sequence &prog,
                     popnn::NonLinearityType nonLinearityType,
                     const std::string &partialsTypeStr,
                     const std::string &debugPrefix
                     ) {
  return backwardGradientStepImpl(graph,
                                  gradientOut,
                                  bwdState,
                                  actOut,
                                  &weightsInput,
                                  &weightsFeedback,
                                  prog,
                                  nonLinearityType,
                                  partialsTypeStr,
                                  debugPrefix);
}

Tensor
backwardGradientStep(Graph &graph,
                     const Tensor &gradientOut,
                     const Tensor &bwdState,
                     const Tensor &actOut,
                     const Tensor &weightsFeedback,
                     Sequence &prog,
                     popnn::NonLinearityType nonLinearityType,
                     const std::string &partialsTypeStr,
                     const std::string &debugPrefix
                     ) {
  Tensor gradAtInput, state;

  std::tie(gradAtInput, state) =
    backwardGradientStepImpl(graph,
                             gradientOut,
                             bwdState,
                             actOut,
                             nullptr,
                             &weightsFeedback,
                             prog,
                             nonLinearityType,
                             partialsTypeStr,
                             debugPrefix);
  return state;
}

void paramDeltaUpdate(Graph &graph,
                      const Tensor &bwdState,
                      const Tensor &actIn,
                      const Tensor &prevOut,
                      Tensor &weightsInputDeltasAcc,
                      Tensor &weightsFeedbackDeltasAcc,
                      Tensor &biasDeltasAcc,
                      Sequence &prog,
                      const std::string &partialsTypeStr,
                      const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/RnDeltas";
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.cache = &cache;
  mmOpt.fullyConnectedPass = FullyConnectedPass::WU;
  const bool combineMatMul =  false;

  if (combineMatMul) {
    matMulAcc(graph, concat(weightsInputDeltasAcc, weightsFeedbackDeltasAcc),
              1.0,
              concat(actIn.transpose(), prevOut.transpose()), bwdState,
              prog, fnPrefix + "/Wi+Wfb", mmOpt);
  } else {
    matMulAcc(graph, weightsInputDeltasAcc, 1.0, actIn.transpose(), bwdState,
              prog, fnPrefix + "/Wi", mmOpt);
    matMulAcc(graph, weightsFeedbackDeltasAcc, 1.0, prevOut.transpose(),
              bwdState, prog, fnPrefix + "/Wfb", mmOpt);
  }
  auto r = reduce(graph, bwdState, {0}, popreduce::Operation::ADD, prog,
                  fnPrefix);
  addTo(graph, biasDeltasAcc, r, prog, fnPrefix);
}

} // namespace rnn
} // namespace popnn
