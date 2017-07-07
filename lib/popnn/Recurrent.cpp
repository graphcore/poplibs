#include <poplin/MatMul.hpp>
#include <popstd/Add.hpp>
#include <popstd/TileMapping.hpp>
#include <popnn/Recurrent.hpp>
#include <popnn/NonLinearity.hpp>
#include <cstdint>

using namespace poplar;
using namespace poplar::program;
using namespace poplin;
using namespace popstd;

namespace popnn {
namespace rnn {

uint64_t getFwdFlops(unsigned sequenceSize, unsigned batchSize,
                     unsigned inputSize, unsigned outputSize,
                     bool weightInput) {
  const auto numMultsFeedFwd = inputSize * outputSize * batchSize
                               * sequenceSize;
  const auto numAddsFeedFwd = (inputSize - 1) * outputSize * batchSize
                               * sequenceSize;

  const auto numMultsIterate = outputSize * outputSize * batchSize
                               * sequenceSize;
  const auto numAddsIterate = (outputSize - 1) * outputSize * batchSize
                               * sequenceSize
                               + 2 * sequenceSize * outputSize;
  auto totalFlops = numMultsIterate + numAddsIterate;
  if (weightInput){
    totalFlops += numMultsFeedFwd + numAddsFeedFwd;
  }

  return totalFlops;
}

Tensor forwardWeightInput(Graph &graph, Tensor actIn, Tensor weights,
                          Sequence &prog,
                          const std::string partialsTypeStr,
                          const std::string &debugPrefix) {
  unsigned sequenceSize = actIn.dim(0);
  unsigned batchSize = actIn.dim(1);
  unsigned outputSize = weights.dim(0);

  Tensor feedFwdOutput =
      graph.addTensor(actIn.elementType(),
                      {0, batchSize, outputSize},
                      "feedFwdOutput");
  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = false;
  mmOpt.cache = &cache;

  for (auto s = 0U; s != sequenceSize; ++s) {
    const auto dbgStr = debugPrefix + "/rnn/FeedFwd/" + std::to_string(s);
     auto prod = matMul(graph, actIn[s], weights.transpose(), prog, dbgStr,
                       mmOpt);
     feedFwdOutput = append(feedFwdOutput, prod);
  }
  return feedFwdOutput;
}

Tensor forwardIterate(Graph  &graph,
                      Tensor feedFwdIn,
                      Tensor initState,
                      Tensor feedbackWeights,
                      Tensor biases,
                      Sequence &prog,
                      popnn::NonLinearityType nonLinearityType,
                      const std::string partialsTypeStr,
                      const std::string &debugPrefix) {
  unsigned sequenceSize = feedFwdIn.dim(0);
  unsigned batchSize = feedFwdIn.dim(1);
  unsigned outputSize = feedFwdIn.dim(2);

  auto bBiases = biases.broadcast(batchSize, 0)
                       .reshape({batchSize, outputSize});

  auto actOut = graph.addTensor(feedFwdIn.elementType(),
                                {0, batchSize, outputSize},
                                "actOut");

  PlanningCache cache;
  MatMulOptions mmOpt;
  mmOpt.partialsType = partialsTypeStr;
  mmOpt.leftHandArgUsedInTranspose = false;
  mmOpt.cache = &cache;

  for (unsigned s = 0U; s != sequenceSize; ++s) {
    const auto dbgStr = debugPrefix + "/rnn/Feedback/"+ std::to_string(s);
    Tensor yP = s == 0 ? initState : actOut[s - 1];
    auto prod = matMul(graph, yP, feedbackWeights.transpose(), prog, dbgStr,
                       mmOpt);
    addTo(graph, prod, feedFwdIn[s], 1.0, prog);

    /* Add broadcast bias */
    addTo(graph, prod, bBiases, 1.0, prog);

    nonLinearity(graph, nonLinearityType, prod, prog, dbgStr);

    actOut = append(actOut, prod);
  }
  return actOut;
}
} // namespace rnn
} // namespace popnn
