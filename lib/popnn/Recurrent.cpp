#include <cstdint>
#include <poplibs_support/logging.hpp>
#include <poplin/MatMul.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDefUtil.hpp>
#include <popnn/Recurrent.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace poputil;
using namespace popops;

namespace logging = poplibs_support::logging;

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
  const uint64_t numMultsFeedFwd =
      static_cast<uint64_t>(inputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numAddsFeedFwd = static_cast<uint64_t>(outputSize) *
                                  (inputSize - 1) * batchSize * sequenceSize;

  const uint64_t numMultsIterate =
      static_cast<uint64_t>(outputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numAddsIterate = static_cast<uint64_t>(outputSize) *
                                      (outputSize - 1) * batchSize *
                                      sequenceSize +
                                  2 * sequenceSize * outputSize;
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

  const uint64_t numMultsInput =
      static_cast<uint64_t>(inputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numAddsInput = static_cast<uint64_t>(outputSize) *
                                (inputSize - 1) * batchSize * sequenceSize;
  const uint64_t numMultsSum =
      static_cast<uint64_t>(outputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numAddsSum = static_cast<uint64_t>(outputSize) *
                              (outputSize - 1) * batchSize * sequenceSize;
  return (numMultsInput + numAddsInput) * calcInputGrad + numMultsSum +
         numAddsSum;
}

uint64_t getWuFlops(unsigned sequenceSize, unsigned batchSize,
                    unsigned inputSize, unsigned outputSize) {
  const uint64_t numMultsInput =
      static_cast<uint64_t>(inputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numMultsSum =
      static_cast<uint64_t>(outputSize) * outputSize * batchSize * sequenceSize;
  const uint64_t numAddsInput =
      (static_cast<uint64_t>(batchSize) * sequenceSize - 1) * outputSize *
      inputSize;
  const uint64_t numAddsSum =
      (static_cast<uint64_t>(batchSize) * sequenceSize - 1) * outputSize *
      outputSize;
  const uint64_t bias =
      (static_cast<uint64_t>(batchSize) * sequenceSize - 1) * outputSize;
  const uint64_t update = static_cast<uint64_t>(outputSize) * inputSize +
                          outputSize * outputSize + outputSize;

  return numMultsInput + numAddsInput + numMultsSum + numAddsSum + bias +
         update;
}

Tensor createFwdState(Graph &graph, const Type &dType, unsigned batchSize,
                      unsigned outputSize, Sequence &prog, bool initState,
                      bool /* inferenceOnly */, const std::string &name) {
  auto state =
      createMatMulInputLHS(graph, dType, {batchSize, outputSize},
                           {outputSize, outputSize}, name + "/FwdState");
  if (initState) {
    popops::zero(graph, state, prog, name);
  }
  return state;
}

Tensor getOutputFromFwdState(const Tensor &fwdState) {
  // In future we may need to keep a hidden state object rather than a tensor
  return fwdState;
}

Tensor createBwdState(Graph &graph, const Type &dType, unsigned batchSize,
                      unsigned outputSize, Sequence &prog,
                      const std::string &name) {
  auto state =
      createMatMulInputLHS(graph, dType, {batchSize, outputSize},
                           {outputSize, outputSize}, name + "/BwdState");
  popops::zero(graph, state, prog, name + "/BwdState");
  return state;
}

Tensor createInput(Graph &graph, unsigned sequenceSize, unsigned batchSize,
                   unsigned inputSize, unsigned outputSize, const Type &dType,
                   const Type &partialsType, bool inferenceOnly,
                   const std::string &name) {
  OptionFlags mmOpt{
      {"partialsType", partialsType.toString()},
      {"fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD"}};
  std::vector<Tensor> input;
  input.emplace_back(createMatMulInputLHS(graph, dType, {batchSize, inputSize},
                                          {inputSize, outputSize}, name, mmOpt)
                         .expand({0}));
  for (unsigned s = 1; s != sequenceSize; ++s) {
    input.emplace_back(graph.clone(input[0], name + "/" + std::to_string(s)));
  }
  return concat(input);
}

poplar::Tensor createWeightsInput(Graph &graph, unsigned /* sequenceSize */,
                                  unsigned batchSize, unsigned inputSize,
                                  unsigned outputSize, const Type &dType,
                                  const Type &partialsType, bool inferenceOnly,
                                  const std::string &namePrefix) {
  OptionFlags mmOpt{
      {"partialsType", partialsType.toString()},
      {"fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD"}};
  return createMatMulInputRHS(graph, dType, {batchSize, inputSize},
                              {inputSize, outputSize}, "weightsInput", mmOpt);
}

poplar::Tensor createWeightsFeedback(Graph &graph, unsigned batchSize,
                                     unsigned outputSize, const Type &dType,
                                     const Type &partialsType,
                                     bool inferenceOnly,
                                     const std::string &namePrefix) {
  OptionFlags mmOpt{
      {"partialsType", partialsType.toString()},
      {"fullyConnectedPass", inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD"}};
  return createMatMulInputRHS(graph, dType, {batchSize, outputSize},
                              {outputSize, outputSize}, "weightsRecurrent",
                              mmOpt);
}

Tensor forwardWeightInput(Graph &graph, const Tensor &actIn,
                          const Tensor &weights, Sequence &prog,
                          const Type &partialsType,
                          const std::string &debugPrefix) {
  const unsigned sequenceSize = actIn.dim(0);
  matmul::PlanningCache cache;
  OptionFlags mmOpt{{"partialsType", partialsType.toString()}};

  return unflattenSeqDims(matMul(graph, flattenSeqDims(actIn), weights, prog,
                                 debugPrefix + "/RnnFwd/FeedFwd", mmOpt,
                                 &cache),
                          sequenceSize);
}

Tensor forwardIterate(Graph &graph, const Tensor &feedFwdIn,
                      const Tensor &initState, const Tensor &weightsFeedback,
                      const Tensor &biases, Sequence &prog,
                      popnn::NonLinearityType nonLinearityType,
                      const Type &partialsType,
                      const std::string &debugPrefix) {
  const unsigned sequenceSize = feedFwdIn.dim(0);
  const unsigned batchSize = feedFwdIn.dim(1);
  const unsigned outputSize = feedFwdIn.dim(2);

  auto bBiases =
      biases.broadcast(batchSize, 0).reshape({batchSize, outputSize});

  auto actOut = graph.addVariable(feedFwdIn.elementType(),
                                  {0, batchSize, outputSize}, "actOut");
  matmul::PlanningCache cache;
  OptionFlags mmOpt{{"partialsType", partialsType.toString()}};

  for (unsigned s = 0U; s != sequenceSize; ++s) {
    const auto dbgStr = debugPrefix + "/RnnFwd/Feedback/" + std::to_string(s);
    Tensor yP = s == 0 ? initState : actOut[s - 1];
    auto prod = matMul(graph, yP, weightsFeedback, prog, dbgStr, mmOpt, &cache);
    addInPlace(graph, prod, feedFwdIn[s], prog, dbgStr + "/Sum");

    /* Add broadcast bias */
    addInPlace(graph, prod, bBiases, prog, dbgStr + "/Bias");

    nonLinearityInPlace(graph, nonLinearityType, prod, prog, dbgStr);

    actOut = append(actOut, prod);
  }
  return actOut;
}

static std::string maybeShape(const poplar::Tensor *t) {
  if (t) {
    std::stringstream ss;
    printContainer(t->shape(), ss);
    return ss.str();
  } else {
    return "-";
  }
}

poplar::Tensor rnnFwdSequence(
    poplar::Graph &graph, poplar::program::Sequence &prog,
    const poplar::Tensor &fwdStateInit, const poplar::Tensor *weightedIn,
    const poplar::Tensor &biases, const poplar::Tensor &feedFwdWeights,
    const poplar::Tensor &feedbackWeights, const poplar::Tensor &prevLayerActs,
    const popnn::NonLinearityType &nonLinearityType,
    const poplar::Type &partialsType, const std::string &debugPrefix) {
  logging::info("rnnFwdSequence fwdStateInit={}, weightedIn={}, biases={}, "
                "feedFwdWeights={} feedbackWeights={}, prevLayerActs={}, "
                "nonLinearityType={}, type={}, name={}",
                fwdStateInit.shape(), maybeShape(weightedIn), biases.shape(),
                feedFwdWeights.shape(), feedbackWeights.shape(),
                prevLayerActs.shape(), nonLinearityType, partialsType,
                debugPrefix);

  auto seqSize = prevLayerActs.dim(0);
  auto stateShape = fwdStateInit.shape();
  stateShape.insert(stateShape.begin(), seqSize);
  const auto dType = prevLayerActs.elementType();
  auto fwdState =
      graph.addVariable(dType, stateShape, debugPrefix + "/fwdState");
  // loop counter
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);

  popops::zero(graph, seqIdx, prog, debugPrefix + "/seqIdx");

  // state for current layer, start from initialiser
  Tensor thisState = poputil::duplicate(graph, fwdStateInit, prog);

  // core rnn loop
  auto loop = Sequence();
  {
    // input is either sliced from weightedIn, or
    // sliced from prevLayerActs then weighted
    Tensor feedFwdOutput;
    if (weightedIn) {
      feedFwdOutput = popops::dynamicSlice(graph, *weightedIn, seqIdx, {0}, {1},
                                           loop, debugPrefix + "/rnnInput");
    } else {
      Tensor cellInputS =
          popops::dynamicSlice(graph, prevLayerActs, seqIdx, {0}, {1}, loop,
                               debugPrefix + "/rnnInput");
      feedFwdOutput = popnn::rnn::forwardWeightInput(
          graph, cellInputS, feedFwdWeights, loop, partialsType, debugPrefix);
    }

    Tensor newState = popnn::rnn::forwardIterate(
        graph, feedFwdOutput, thisState, feedbackWeights, biases, loop,
        nonLinearityType, partialsType, debugPrefix);
    // all output sequence elements take the same mapping so will only
    // require on-tile copies
    for (unsigned i = 0; i != seqSize; ++i) {
      graph.setTileMapping(fwdState[i], graph.getTileMapping(newState[0]));
    }
    loop.add(Copy(newState, thisState));

    popops::dynamicUpdate(graph, fwdState, newState, seqIdx, {0}, {1}, loop,
                          debugPrefix + "/rnnUpdateState");
    addInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxIncr");
  }
  prog.add(Repeat(seqSize, loop));
  return fwdState;
};

std::pair<Tensor, Tensor> backwardGradientStepImpl(
    Graph &graph, const Tensor &gradientOut, const Tensor &bwdState,
    const Tensor &actOut, const Tensor *weightsInput,
    const Tensor *weightsFeedback, Sequence &prog,
    popnn::NonLinearityType nonLinearityType, const Type &partialsType,
    const std::string &debugPrefix) {
  matmul::PlanningCache cache;
  OptionFlags mmOpt{{"partialsType", partialsType.toString()},
                    {"fullyConnectedPass", "TRAINING_BWD"}};

  auto t = matMul(graph, bwdState, weightsFeedback->transpose(), prog,
                  debugPrefix + "/RnnBwd/Fb", mmOpt, &cache);
  addInPlace(graph, t, gradientOut, prog, debugPrefix + "/RnnBwd/AddOutGrad");

  auto newBwdState = nonLinearityInputGradient(
      graph, nonLinearityType, actOut, t, prog, debugPrefix + "/RnnBwd");
  Tensor gradientAtInput;
  if (weightsInput != nullptr) {
    gradientAtInput = matMul(graph, newBwdState, weightsInput->transpose(),
                             prog, debugPrefix + "/RnnBwd/Ff");
  }
  return std::make_pair(gradientAtInput, newBwdState);
}

std::pair<Tensor, Tensor>
backwardGradientStep(Graph &graph, const Tensor &gradientOut,
                     const Tensor &bwdState, const Tensor &actOut,
                     const Tensor &weightsInput, const Tensor &weightsFeedback,
                     Sequence &prog, popnn::NonLinearityType nonLinearityType,
                     const Type &partialsType, const std::string &debugPrefix) {
  return backwardGradientStepImpl(graph, gradientOut, bwdState, actOut,
                                  &weightsInput, &weightsFeedback, prog,
                                  nonLinearityType, partialsType, debugPrefix);
}

Tensor backwardGradientStep(Graph &graph, const Tensor &gradientOut,
                            const Tensor &bwdState, const Tensor &actOut,
                            const Tensor &weightsFeedback, Sequence &prog,
                            popnn::NonLinearityType nonLinearityType,
                            const Type &partialsType,
                            const std::string &debugPrefix) {
  Tensor gradAtInput, state;

  std::tie(gradAtInput, state) = backwardGradientStepImpl(
      graph, gradientOut, bwdState, actOut, nullptr, &weightsFeedback, prog,
      nonLinearityType, partialsType, debugPrefix);
  return state;
}

void paramDeltaUpdate(Graph &graph, const Tensor &bwdState, const Tensor &actIn,
                      const Tensor &prevOut, Tensor &weightsInputDeltasAcc,
                      Tensor &weightsFeedbackDeltasAcc, Tensor &biasDeltasAcc,
                      Sequence &prog, const Type &partialsType,
                      const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/RnDeltas";
  matmul::PlanningCache cache;
  OptionFlags mmOpt{{"partialsType", partialsType.toString()},
                    {"fullyConnectedPass", "TRAINING_WU"}};
  const bool combineMatMul = false;

  if (combineMatMul) {
    matMulAcc(graph, concat(weightsInputDeltasAcc, weightsFeedbackDeltasAcc),
              1.0, concat(actIn.transpose(), prevOut.transpose()), bwdState,
              prog, fnPrefix + "/Wi+Wfb", mmOpt, &cache);
  } else {
    matMulAcc(graph, weightsInputDeltasAcc, 1.0, actIn.transpose(), bwdState,
              prog, fnPrefix + "/Wi", mmOpt, &cache);
    matMulAcc(graph, weightsFeedbackDeltasAcc, 1.0, prevOut.transpose(),
              bwdState, prog, fnPrefix + "/Wfb", mmOpt, &cache);
  }
  auto r = reduce(graph, bwdState, {0}, popops::Operation::ADD, prog, fnPrefix);
  addInPlace(graph, biasDeltasAcc, r, prog, fnPrefix);
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor, poplar::Tensor>
rnnBwdSequence(poplar::Graph &graph, bool doWU, bool ignoreInputGradientCalc,
               poplar::program::Sequence &prog,
               const poplar::Tensor &fwdStateInit,
               const poplar::Tensor &fwdState, const poplar::Tensor &biases,
               const poplar::Tensor &feedFwdWeights,
               const poplar::Tensor &feedbackWeights,
               const poplar::Tensor &outGradient, const poplar::Tensor &actIn,
               const popnn::NonLinearityType &nonLinearityType,
               const poplar::Type &partialsType,
               const std::string &debugPrefix) {
  logging::info("rnnFwdSequence fwdStateInit={}, fwdState={}, biases={}, "
                "feedFwdWeights={} feedbackWeights={}, outGradient={}, "
                "actIn={}, nonLinearityType={}, type={}, name={}",
                fwdStateInit.shape(), fwdState.shape(), biases.shape(),
                feedFwdWeights.shape(), feedbackWeights.shape(),
                outGradient.shape(), actIn.shape(), nonLinearityType,
                partialsType, debugPrefix);

  Tensor feedFwdWeightsDeltaAcc, feedbackWeightsDeltaAcc, biasesDeltaAcc;
  if (doWU) {
    feedFwdWeightsDeltaAcc = graph.clone(feedFwdWeights);
    feedbackWeightsDeltaAcc = graph.clone(feedbackWeights);
    biasesDeltaAcc = graph.clone(biases);
    // zero all tensors updated in the BPTT
    zero(graph, feedFwdWeightsDeltaAcc, prog,
         debugPrefix + "/ZeroFeedFwdWeightsDeltasAcc");
    zero(graph, feedbackWeightsDeltaAcc, prog,
         debugPrefix + "/ZeroFeedbackWeightsDeltasAcc");
    zero(graph, biasesDeltaAcc, prog, debugPrefix + "/ZeroBiasesDeltasAcc");
  }
  const auto dType = actIn.elementType();
  const auto seqSize = actIn.dim(0);
  const auto batchSize = actIn.dim(1);
  const auto inputSize = actIn.dim(2);
  const auto outputSize = feedbackWeights.dim(1);

  Tensor bwdState = popnn::rnn::createBwdState(graph, dType, batchSize,
                                               outputSize, prog, debugPrefix);
  auto actOut = popnn::rnn::getOutputFromFwdState(fwdStateInit);
  auto seqIdx = graph.addVariable(UNSIGNED_INT, {1}, debugPrefix + "/seqIdx");
  auto start =
      graph.addConstant(UNSIGNED_INT, {1}, seqSize - 1, debugPrefix + "/start");
  auto one = graph.addConstant(UNSIGNED_INT, {1}, 1, debugPrefix + "/one");
  graph.setTileMapping(start, 0);
  graph.setTileMapping(one, 0);
  graph.setTileMapping(seqIdx, 0);
  prog.add(Copy(start, seqIdx));

  // state for gradient backprop
  std::vector<std::size_t> gradientShape = {
      seqSize, batchSize, !ignoreInputGradientCalc ? inputSize : outputSize};
  auto prevLayerGradsVec =
      graph.addVariable(dType, gradientShape, debugPrefix + "/gradient");
  Tensor fwdStateS = graph.clone(fwdStateInit, debugPrefix + "/fwdStateS");
  Tensor fwdStateOffset =
      concat(fwdStateInit.expand({0}), fwdState.slice(0, fwdState.dim(0) - 1));
  graph.setTileMapping(fwdStateOffset[0],
                       graph.getTileMapping(fwdStateOffset[1]));
  auto loop = Sequence();
  {
    Tensor gradientInputThisStep;
    Tensor outGradS =
        popops::dynamicSlice(graph, outGradient.dimShuffle({1, 0, 2}), seqIdx,
                             {0}, {1}, loop, debugPrefix + "/OutGradient")
            .squeeze({0});
    Tensor prevOutput;
    if (!ignoreInputGradientCalc || doWU) {
      prevOutput = popnn::rnn::getOutputFromFwdState(fwdStateS);
    }
    if (!ignoreInputGradientCalc) {
      std::tie(gradientInputThisStep, bwdState) =
          popnn::rnn::backwardGradientStep(
              graph, outGradS, bwdState, prevOutput, feedFwdWeights,
              feedbackWeights, loop, nonLinearityType, partialsType,
              debugPrefix);
      gradientInputThisStep = gradientInputThisStep.expand({0});
      for (unsigned s = 0; s != seqSize; ++s)
        graph.setTileMapping(prevLayerGradsVec[s],
                             graph.getTileMapping(gradientInputThisStep));
      popops::dynamicUpdate(graph, prevLayerGradsVec, gradientInputThisStep,
                            seqIdx, {0}, {1}, loop,
                            debugPrefix + "/rnnUpdateGradient");

    } else {
      bwdState = popnn::rnn::backwardGradientStep(
          graph, outGradS, bwdState, prevOutput, feedbackWeights, loop,
          nonLinearityType, partialsType, debugPrefix);
      for (unsigned s = 0; s != seqSize; ++s) {
        mapTensorLinearly(graph, prevLayerGradsVec.slice(s, s + 1));
      }
    }
    Tensor fwdStateSM1 =
        popops::dynamicSlice(graph, fwdStateOffset, seqIdx, {0}, {1}, loop,
                             debugPrefix + "/fwdState")
            .squeeze({0});
    loop.add(Copy(fwdStateSM1, fwdStateS));
    if (doWU) {
      Tensor actInS = popops::dynamicSlice(graph, actIn, seqIdx, {0}, {1}, loop,
                                           debugPrefix + "/rnnActIn")
                          .squeeze({0});
      // Note that the delta update uses the fwdState[s-1], or fwdInit
      // when s == -1.
      popnn::rnn::paramDeltaUpdate(graph, bwdState, actInS, fwdStateSM1,
                                   feedFwdWeightsDeltaAcc,
                                   feedbackWeightsDeltaAcc, biasesDeltaAcc,
                                   loop, partialsType, debugPrefix);
    }
    subInPlace(graph, seqIdx, one, loop, debugPrefix + "/seqIdxDecr");
  }
  prog.add(Repeat(seqSize, loop));
  return std::tie(prevLayerGradsVec, feedFwdWeightsDeltaAcc,
                  feedbackWeightsDeltaAcc, biasesDeltaAcc);
};

} // namespace rnn
} // namespace popnn
