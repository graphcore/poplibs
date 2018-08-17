#include "popnn/Loss.hpp"

#include "poplar/Graph.hpp"
#include "popops/Encoding.hpp"
#include "popops/Reduce.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"

#include <boost/optional.hpp>
#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popnn {

namespace {

// Unsigned integer version of log2 rounded up
// Single-line constexpr form added to allow compile-time calculation.
// Could be nicer if using multi-line constexpr function (needs C++14).
constexpr static unsigned ceilLog2Aux(unsigned n) {
  return (n ? 1 + ceilLog2Aux(n >> 1) : 0);
}
// Check if power of 2 and then call to count up to most significant bit
constexpr static unsigned ceilLog2(unsigned n) {
  return ((n & (n - 1)) ? 1 : 0) + ceilLog2Aux(n >> 1);
}

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

    // The maximum size of the region is 2^12 - 1
    auto workerRegions =
      splitRegionsBetweenWorkers(target, contiguousRegions, grainSize, 0xFFF);

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
      graph.setInitialValue(transformV["size"],
                            vertexTransformed.numElements());
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

  Sequence prog;

  auto oneHot = graph.clone(activationType, activations,
                            layerPrefix + "/OneHotEncoded");
  popops::encodeOneHot(graph, expected, oneHot, prog, layerPrefix);

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
  const auto batchSize = activations.dim(0);
  const auto actsPerBatch = activations.numElements() / batchSize;

  // Optimisation: Focus point for cycles and memory
  // At each broadcast/calculate/reduce step a choice can be  made about
  // how many tiles to spread elements over.
  // The fewer tiles to broadcast to/reduce from, the less memory is
  // used by vertices overall as well as by exchange code (generally).
  // However fewer tiles means more time spent calculating. We need
  // decent estimations of compute/exchange cycles needed for the different
  // steps to balance this.
  Sequence prog;

  auto oneHot = graph.clone(activationType, activations,
                            layerPrefix + "/OneHotEncoded");
  popops::encodeOneHot(graph, expected, oneHot, prog, layerPrefix);

  auto activationsCopy =
    graph.clone(activations.reshape({batchSize, actsPerBatch}),
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

  // Find out which tile `numCorrect` sits on
  auto numCorrectMapping = graph.getTileMapping(numCorrect);
  boost::optional<unsigned> numCorrectTile;
  for (const auto &tileMapping : numCorrectMapping) {
    if (!tileMapping.empty()) {
      assert(tileMapping.size() == 1 &&
             tileMapping[0].size() == 1);
      numCorrectTile = tileMapping[0].begin();
    }
  }
  assert(numCorrectTile);

  const auto &target = graph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto maxValueGrainSize =
    std::max<std::size_t>(1, target.getAtomicStoreGranularity() /
                             target.getTypeSize(activationType));
  const auto reduceGatherVertexClass =
    templateVertex("popnn::ReduceMaxClassGather",
                   activationType, expectedType);
  const auto reduceSparseVertexClass =
    templateVertex("popnn::ReduceMaxClassSparse",
                   activationType, expectedType);
  const auto calcAccuracyVertexClass =
    templateVertex("popnn::CalcAccuracy", expectedType);

  const auto numWorkers = target.getNumWorkerContexts();
  std::vector<ComputeSet> reductionCS;
  Tensor lastValuePartials = activations;
  Tensor lastIndexPartials;
  auto lastBatchPartials = lastValuePartials.numElements() / batchSize;

  std::size_t reduceIndex = 0;
  unsigned nextTile = 0;
  while (lastBatchPartials > 1) {

    // These numbers are good enough to handle existing number of activations
    // and batch size for current hardware. The advent of an enormous number
    // of batches or activations would need a bit more thought.
    std::size_t partialsFactor = 16;
    if (lastBatchPartials <= partialsFactor * 2) {
      partialsFactor = partialsFactor * 2;
    }
    const auto batchPartials =
      (lastBatchPartials + partialsFactor - 1) / partialsFactor;

    bool isFirstReduce = (reduceIndex == 0);
    bool isLastReduce = (batchPartials == 1);
    const auto vertexClass =
      isFirstReduce ? reduceGatherVertexClass
                    : reduceSparseVertexClass;
    reductionCS.push_back(
      graph.addComputeSet(layerPrefix + "/ReduceMaxClass[" +
                          std::to_string(reduceIndex) + "]"));
    const auto &cs = reductionCS.back();
    auto valuePartials =
      graph.addVariable(activationType, {batchSize, batchPartials},
                        layerPrefix + "/maxValuePartials[" +
                        std::to_string(reduceIndex) + "]");
    auto indexPartials =
      graph.addVariable(expectedType, {batchSize, batchPartials},
                        layerPrefix + "/maxIndexPartials[" +
                        std::to_string(reduceIndex) + "]");

    for (std::size_t b = 0; b < batchSize; ++b) {
      std::size_t batchOffset = 0;
      std::size_t partialsIndex = 0;
      while (batchOffset != lastBatchPartials) {
        // If this is the last reduction, put the reduction on the tile where
        // the final accuracy will be calculated.
        const auto tile = isLastReduce ? *numCorrectTile : nextTile;
        const auto v =  graph.addVertex(cs, vertexClass);
        if (isFirstReduce) {
          // This first reduction uses a supervisor vertex, so try and give it
          // a grain of splits per-worker each time.
          const auto supervisorPartials = partialsFactor * numWorkers *
                                          maxValueGrainSize;
          const auto partialsThisSplit =
            std::min(lastBatchPartials - batchOffset,
                     supervisorPartials);
          const auto divisorLog2 = ceilLog2(partialsFactor);
          const auto divisor = (1u << divisorLog2);
          const auto nOutputs = (partialsThisSplit + divisor - 1) / divisor;
          auto splitValuePartials =
            lastValuePartials[b].slice(batchOffset,
                                       batchOffset + partialsThisSplit);
          auto splitMaxValue =
            valuePartials[b].slice(partialsIndex,
                                   partialsIndex + nOutputs);
          auto splitMaxIndex =
            indexPartials[b].slice(partialsIndex,
                                   partialsIndex + nOutputs);
          graph.connect(v["activations"], splitValuePartials);
          graph.setInitialValue(v["index"], batchOffset);
          graph.connect(v["maxValue"], splitMaxValue);
          graph.connect(v["maxIndex"], splitMaxIndex);
          graph.setInitialValue(v["size"], partialsThisSplit);
          graph.setInitialValue(v["divisorLog2"], divisorLog2);
          graph.setTileMapping(splitMaxValue, tile);
          graph.setTileMapping(splitMaxIndex, tile);
          partialsIndex += nOutputs;
          batchOffset += partialsThisSplit;
        } else {
          const auto partialsThisSplit =
            std::min(lastBatchPartials - batchOffset, partialsFactor);
          auto splitValuePartials =
            lastValuePartials[b].slice(batchOffset,
                                       batchOffset + partialsThisSplit);
          auto splitIndexPartials =
            lastIndexPartials[b].slice(batchOffset,
                                       batchOffset + partialsThisSplit);
          graph.connect(v["activations"], splitValuePartials);
          graph.connect(v["labels"], splitIndexPartials);
          graph.connect(v["maxValue"], valuePartials[b][partialsIndex]);
          graph.connect(v["maxIndex"], indexPartials[b][partialsIndex]);
          graph.setTileMapping(valuePartials[b][partialsIndex], tile);
          graph.setTileMapping(indexPartials[b][partialsIndex], tile);
          ++partialsIndex;
          batchOffset += partialsThisSplit;
        }
        graph.setTileMapping(v, tile);
        nextTile = (nextTile + 1) % tilesPerIPU;
        assert(batchOffset <= lastBatchPartials);
      }
    }

    lastValuePartials = valuePartials;
    lastIndexPartials = indexPartials;
    lastBatchPartials = batchPartials;
    ++reduceIndex;
  }

  // Special case for if there happens to be just 1 act per batch (really only
  // occurs in tests).
  if (reduceIndex == 0) {
    lastIndexPartials = graph.addVariable(expectedType, {batchSize, 1},
                                          layerPrefix + "/maxIndexPartials");
    graph.setTileMapping(lastIndexPartials, *numCorrectTile);
    for (std::size_t b = 0; b < batchSize; ++b) {
      graph.setInitialValue(lastIndexPartials[b][0], 0);
    }
  }

  // This would ideally be calculated with a popops::eq followed by a
  // popops::reduceWithOutput. At the moment popops::eq outputs bool
  // so this requires some ugly casting. For now this last step is its
  // own vertex. Doesn't particularly matter while batch size is generally
  // so small.
  const auto calcAccuracyCS =
    graph.addComputeSet(layerPrefix + "/CalcAccuracy");
  auto v = graph.addVertex(calcAccuracyCS, calcAccuracyVertexClass,
                           {{"maxPerBatch", lastIndexPartials.flatten()},
                            {"expected", expected},
                            {"numCorrect", flatNumCorrect[0]}});
  graph.setTileMapping(v, *numCorrectTile);

  // Add all the reductions and final accuracy.
  Sequence prog;
  for (const auto &cs : reductionCS) {
    prog.add(Execute(cs));
  }
  prog.add(Execute(calcAccuracyCS));
  return prog;
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
