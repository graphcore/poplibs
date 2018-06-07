#include "popnn/Loss.hpp"

#include "poplar/Graph.hpp"
#include "popops/Encoding.hpp"
#include "popops/Reduce.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popnn {

namespace {

// Per-element of input probabilities, on the tile these are mapped to
// compute the gradient and the contribution to loss.
Tensor onTileTransform(Graph &graph,
                       const Tensor &probabilities,
                       const Tensor &expected,
                       const Tensor &deltas,
                       const Type &fpType,
                       const std::string &vertexClassTemplate,
                       Sequence &prog,
                       const std::string &debugPrefix = "") {
  const auto &target = graph.getTarget();
  const auto vertexClass =
    templateVertex(vertexClassTemplate, fpType);
  auto transformCS = graph.addComputeSet(debugPrefix + "/on_tile_transform");
  const auto batchSize = probabilities.dim(0);
  const auto perBatch = probabilities.numElements() / batchSize;
  auto transformed = graph.addVariable(fpType,
                                       {batchSize, perBatch},
                                       debugPrefix + "/Transformed");
  auto mapping = graph.getTileMapping(probabilities);
  for (std::size_t tile = 0; tile < mapping.size(); ++tile) {
    const auto &tileMapping = mapping[tile];
    const auto grainSize = target.getVectorWidth(fpType);
    auto contiguousRegions =
      graph.getSortedContiguousRegions(probabilities, tileMapping);
    // Split delta and transformed calculation between workers on
    // each tile if it's advantageous
    // Optimisation: Focus point for memory usage - more vertices = more
    // memory. Number of vertices needed depends on the spread of
    // probabilities over tiles, and grain size for split among
    // workers. Currently inputs are gathered via on-tile copy before
    // this transform. Were this gather particularly expensive specialised
    // single/multiple/2D transforms could be added but doesn't currently
    // seem worth it.
    auto workerRegions =
      splitRegionsBetweenWorkers(target, contiguousRegions,
                                 grainSize, grainSize * 2);

    for (const auto &vertexRegions : workerRegions) {
      auto vertexTransformed =
        concat(transformed.flatten().slices(vertexRegions));
      auto transformV =
        graph.addVertex(transformCS, vertexClass, {
          {"probs", concat(probabilities.flatten().slices(vertexRegions))},
          {"expected", concat(expected.flatten().slices(vertexRegions))},
          {"deltas", concat(deltas.flatten().slices(vertexRegions))},
          {"transformed", vertexTransformed}
        });
      graph.setTileMapping(vertexTransformed, tile);
      graph.setTileMapping(transformV, tile);
    }
  }
  prog.add(Execute(transformCS));
  return transformed;
}

Program calcLossSumSquared(Graph &graph,
                           const Tensor &activations,
                           const Tensor &expected,
                           const Tensor &loss,
                           const Tensor &deltas,
                           const Type &activationType,
                           const Type &expectedType,
                           const std::string &debugPrefix) {
  const auto layerPrefix = debugPrefix + "/LossSumSquared";
  const auto batchSize = activations.dim(0);
  const auto perBatch = activations.numElements() / batchSize;

  Sequence prog;
  auto oneHot =
    popops::encodeOneHot(graph, activationType,
                         expected, perBatch, prog,
                         layerPrefix);

  // Compute loss partials and deltas
  auto transformed = onTileTransform(graph,
                                     activations,
                                     oneHot, deltas,
                                     activationType,
                                     "popnn::LossSumSquaredTransform",
                                     prog, layerPrefix);
  // Reduce values in each batch
  popops::reduceWithOutput(graph, transformed, loss,
                           {1}, popops::Operation::ADD, prog,
                           layerPrefix + "/reduce_loss");
  return prog;
}

Program calcLossSoftmaxCrossEntropy(Graph &graph,
                                    const Tensor &activations,
                                    const Tensor &expected,
                                    const Tensor &loss,
                                    const Tensor &deltas,
                                    const Type &activationType,
                                    const Type &expectedType,
                                    const std::string &debugPrefix) {
  const auto layerPrefix = debugPrefix + "/LossSoftmaxCrossEntropy";
  const auto &target = graph.getTarget();
  const auto batchSize = activations.dim(0);
  const auto perBatch = activations.numElements() / batchSize;

  // Optimisation: Focus point for cycles and memory
  // At each broadcast/calculate/reduce step a choice can be  made about
  // how many tiles to spread elements over.
  // The fewer tiles to broadcast to/reduce from, the less memory is
  // used by vertices overall as well as by exchange code (generally).
  // However fewer tiles means more time spent calculating. We need
  // decent estimations of compute/exchange cycles needed for the different
  // steps to balance this.
  Sequence prog;
  auto oneHot =
    popops::encodeOneHot(graph, activationType,
                         expected, perBatch, prog,
                         layerPrefix);

  auto activationsCopy =
    graph.clone(activations.reshape({batchSize, perBatch}),
                layerPrefix + "/ActivationsPreprocessed");
  prog.add(Copy(activations, activationsCopy));

  // Optimisation: While this method is very good for numerical stability in
  // the sum of exponentials, the extra reduce/broadcast is bad news for
  // performance and memory (exchange code/vertices).
  // Any other methods that would give acceptable stability? Is it possible
  // to know ahead of time what the range of inputs will be?
  //
  // Subtract max value for purposes of numerical stability
  // when normalising exponentials: e^(x - max(X)).
  auto maxAct =
    popops::reduce(graph, activationsCopy,
                   {1}, popops::Operation::MAX,
                   prog,
                   layerPrefix + "/shift_range_for_stability")
    .reshape({batchSize, 1});

  broadcastToMatch(maxAct, activationsCopy.shape());
  popops::subInPlace(graph, activationsCopy, maxAct, prog,
                     layerPrefix + "/shift_range_for_stability");

  popops::expInPlace(graph, activationsCopy, prog,
                     layerPrefix + "/exponent");

  // Sum exponentials to perform normalisation below
  auto sumExp =
    popops::reduce(graph, activationsCopy,
                   {1}, popops::Operation::ADD,
                   prog,
                   layerPrefix + "/normalize")
    .reshape({batchSize, 1});

  // Normalise exponentials
  broadcastToMatch(sumExp, activationsCopy.shape());
  popops::divInPlace(graph, activationsCopy, sumExp, prog,
                     layerPrefix + "/normalize");

  // Finally compute loss partials and deltas
  auto transformed = onTileTransform(graph,
                                     activationsCopy,
                                     oneHot, deltas,
                                     activationType,
                                     "popnn::LossSoftmaxTransform",
                                     prog, layerPrefix);

  // Sum loss partials for final result
  popops::reduceWithOutput(graph, transformed, loss,
                           {1}, popops::Operation::ADD,
                           prog,
                           layerPrefix + "/reduce_loss");
  return prog;
}


} // end anonymous namespace

Program
calcLoss(Graph &graph,
         const Tensor& activations,
         const Tensor& expected,
         const Tensor& loss,
         const Tensor& deltas,
         const Type& activationType,
         const Type& expectedType,
         LossType lossType,
         const std::string &debugPrefix) {
  switch (lossType) {
    case LossType::SUM_SQUARED_LOSS:
      return calcLossSumSquared(graph,
                                activations,
                                expected,
                                loss,
                                deltas,
                                activationType,
                                expectedType,
                                debugPrefix);
    case LossType::SOFTMAX_CROSS_ENTROPY_LOSS:
      return calcLossSoftmaxCrossEntropy(graph,
                                         activations,
                                         expected,
                                         loss,
                                         deltas,
                                         activationType,
                                         expectedType,
                                         debugPrefix);
    default:
      throw poplib_error("Unknown loss type requested in calcLoss");
      break;
  }
  return Program{};
}

Program
calcAccuracy(Graph &graph,
             const Tensor &activations,
             const Tensor &expected,
             const Tensor &numCorrect,
             const Type &activationType,
             const Type &expectedType,
             const std::string &debugPrefix) {
  const auto layerPrefix = debugPrefix + "/Accuracy";

  // Normalize shape of numCorrect
  auto flatNumCorrect = numCorrect.flatten();
  if (flatNumCorrect.dim(0) != 1) {
    throw poplib_error("numCorrect must be scalar or single element tensor");
  }
  const auto batchSize = activations.dim(0);
  if (expected.shape().size() > 1) {
    throw poplib_error("expected must be a 1-dimensional tensor");
  }
  if (expected.dim(0) != batchSize) {
    throw poplib_error("expected tensor must be of length equal the number of "
                       "batches given in activations tensor");
  }
  // TODO: More distributed implementation. This implementation performs the
  // search for the max on a single tile with a single vertex.
  //
  // This operation will be similar to a reduction
  // in the sense that we seek the max over the activations in each batch.
  // However, we seek the index of the max element within each batch.
  // For this reason this cannot simply use reduction, it requires the index
  // to be supplied along with the activations.
  //
  // Suggested implementation involves two types of vertices seeking the index
  // of the max activation. 'MaxClass' taking a list of activations, and
  // outputting the max value/index. 'MaxClassSparse' taking a list of
  // activations and associated labels (indices), and outputting the max
  // value/index.
  //
  // First stage: 'MaxClass' split across vertices/tiles given contiguous
  // regions of the input activations. An offset applied to the max class
  // returned to give the actual index of the max element for that region.
  // (If needed) N reduction stages: 'MaxClassSparse' for max values/indices
  // given by the first stage to reach a single max value/index for each
  // batch.
  //
  // Finally check if the max index for each batch is equal 'expected' and
  // sum to give 'numCorrect'.
  auto cs = graph.addComputeSet(layerPrefix);
  auto v = graph.addVertex(cs, templateVertex("popnn::CalcAccuracy",
                                              activationType,
                                              expectedType),
                           {{"activations", activations},
                            {"labels", expected},
                            {"numCorrect", flatNumCorrect[0]}});
  graph.setTileMapping(v, 0);
  return Execute(cs);
}

Program
calcLoss(Graph &graph,
         const Tensor &activations,
         const Tensor &expected,
         const Tensor &loss,
         const Tensor &deltas,
         const Tensor &numCorrect,
         const Type &activationType,
         const Type &expectedType,
         LossType lossType,
         const std::string &debugPrefix) {
  Sequence prog(
    calcLoss(graph, activations, expected, loss, deltas,
             activationType, expectedType, lossType, debugPrefix),
    calcAccuracy(graph, activations, expected, numCorrect,
                 activationType, expectedType, debugPrefix)
  );
  return prog;
}

} // end namespace popnn
