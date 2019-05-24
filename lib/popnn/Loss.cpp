#include "popnn/Loss.hpp"

#include "poplar/Graph.hpp"
#include "popops/Encoding.hpp"
#include "popops/Reduce.hpp"
#include "popops/ElementWise.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poplibs_support/Algorithm.hpp"

#include <boost/optional.hpp>
#include <cassert>
#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popnn {

namespace {

// Per-element of model outputs, on the tile these are mapped to
// compute the gradient and the contribution to loss.
Tensor onTileTransform(Graph &graph,
                       const Tensor &modelOutputs,
                       const Tensor &expected,
                       const Tensor &deltas,
                       const std::string &vertexClassTemplate,
                       Sequence &prog,
                       const std::string &debugPrefix = "") {
  const auto &target = graph.getTarget();
  const auto &dType = modelOutputs.elementType();
  const auto vertexClass = templateVertex(vertexClassTemplate, dType);
  auto transformCS = graph.addComputeSet(debugPrefix + "/on_tile_transform");
  const auto batchSize = modelOutputs.dim(0);
  const auto perBatch = modelOutputs.numElements() / batchSize;
  auto transformed = graph.addVariable(dType, {batchSize, perBatch},
                                       debugPrefix + "/Transformed");
  auto mapping = graph.getTileMapping(modelOutputs);
  for (std::size_t tile = 0; tile < mapping.size(); ++tile) {
    const auto &tileMapping = mapping[tile];
    const auto grainSize = target.getVectorWidth(dType);
    auto contiguousRegions =
      graph.getSortedContiguousRegions(modelOutputs, tileMapping);
    // Split delta and transformed calculation between workers on
    // each tile if it's advantageous
    // Optimisation: Focus point for memory usage - more vertices = more
    // memory. Number of vertices needed depends on the spread of
    // modelOutputs over tiles, and grain size for split among
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
      auto vertexDeltas = concat(deltas.flatten().slices(vertexRegions));
      auto transformV =
        graph.addVertex(transformCS, vertexClass, {
          {"probs", concat(modelOutputs.flatten().slices(vertexRegions))},
          {"expected", concat(expected.flatten().slices(vertexRegions))},
          {"deltas", vertexDeltas},
          {"transformed", vertexTransformed}
        });
      graph.setInitialValue(transformV["size"],
                            vertexTransformed.numElements());
      graph.setTileMapping(vertexTransformed, tile);
      graph.setTileMapping(vertexDeltas, tile);
      graph.setTileMapping(transformV, tile);
    }
  }
  prog.add(Execute(transformCS));
  return transformed;
}

} // end anonymous namespace

Program
calcLoss(Graph &graph,
         const Tensor& modelOutputs,
         const Tensor& expected,
         const Tensor& loss,
         const Tensor& deltas,
         LossType lossType,
         const std::string &debugPrefix) {
  std::string layerPrefix = debugPrefix;
  std::string transformVertexClass;
  switch (lossType) {
    case LossType::SUM_SQUARED_LOSS:
      layerPrefix += "/LossSumSquared";
      transformVertexClass = "popnn::LossSumSquaredTransform";
      break;
    case LossType::CROSS_ENTROPY_LOSS:
      layerPrefix += "/LossCrossEntropy";
      transformVertexClass = "popnn::LossCrossEntropyTransform";
      break;
    default:
      throw poplibs_error("Unknown loss type requested in calcLoss");
      break;
  }
  Sequence prog;
  auto oneHot = graph.clone(modelOutputs.elementType(), modelOutputs,
                            layerPrefix + "/OneHotEncoded");
  popops::encodeOneHot(graph, expected, oneHot, prog, layerPrefix);

  // Compute loss partials and deltas
  auto transformed = onTileTransform(graph,
                                     modelOutputs,
                                     oneHot, deltas,
                                     transformVertexClass,
                                     prog, layerPrefix);
  // Reduce values in each batch
  popops::reduceWithOutput(graph, transformed, loss,
                           {1}, popops::Operation::ADD, prog,
                           layerPrefix + "/reduce_loss");
  return prog;
}

static Tensor argMinOrMax(Graph &graph, const Tensor &input,
                          const Type &argminType, Sequence &prog,
                          unsigned numCorrectTile,
                          const std::string &debugPrefix, bool max = true) {

  const std::string lowerCase = max ? "max" : "min";
  const std::string capitalized = max ? "Max" : "Min";

  const auto layerPrefix = debugPrefix + "/arg" + lowerCase;
  const auto &target = graph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto batchSize = input.dim(0);

  const auto reduceGatherVertexClass =
      templateVertex("popnn::Reduce" + capitalized + "ClassGather",
                     input.elementType(), argminType);
  const auto reduceSparseVertexClass =
      templateVertex("popnn::Reduce" + capitalized + "ClassSparse", argminType);

  const auto numWorkers = target.getNumWorkerContexts();
  std::vector<ComputeSet> reductionCS;
  Tensor lastValuePartials = input;
  Tensor lastIndexPartials;
  auto lastBatchPartials = lastValuePartials.numElements() / batchSize;

  std::size_t reduceIndex = 0;
  unsigned nextTile = 0;
  while (lastBatchPartials > 1) {

    // These numbers are good enough to handle existing number of modelOutputs
    // and batch size for current hardware. The advent of an enormous number
    // of batches or modelOutputs would need a bit more thought.
    std::size_t partialsFactor = 32;
    const auto batchPartials =
        (lastBatchPartials + partialsFactor - 1) / partialsFactor;

    bool isFirstReduce = (reduceIndex == 0);
    bool isLastReduce = (batchPartials == 1);
    const auto vertexClass =
        isFirstReduce ? reduceGatherVertexClass : reduceSparseVertexClass;
    reductionCS.push_back(
        graph.addComputeSet(layerPrefix + "/Reduce" + capitalized + "Class[" +
                            std::to_string(reduceIndex) + "]"));
    const auto &cs = reductionCS.back();
    // Partial values are always 32-bit floating point. We don't need to
    // perform any arithmetic on these values so the precision is equivalent
    // to the original. This allows the half supervisor reduction vertex to
    // operate on a single split per-worker as it no longer has to avoid
    // sub-word writes. Memory cost is tiny here as there are very few
    // of these partials.
    auto valuePartials =
        graph.addVariable(FLOAT, {batchSize, batchPartials},
                          layerPrefix + "/" + lowerCase + "ValuePartials[" +
                              std::to_string(reduceIndex) + "]");
    auto indexPartials =
        graph.addVariable(argminType, {batchSize, batchPartials},
                          layerPrefix + "/" + lowerCase + "IndexPartials[" +
                              std::to_string(reduceIndex) + "]");

    for (std::size_t b = 0; b < batchSize; ++b) {
      std::size_t batchOffset = 0;
      std::size_t partialsIndex = 0;
      while (batchOffset != lastBatchPartials) {
        // If this is the last reduction, put the reduction on the tile where
        // the final accuracy will be calculated.
        const auto tile = isLastReduce ? numCorrectTile : nextTile;
        const auto v = graph.addVertex(cs, vertexClass);
        if (isFirstReduce) {
          // This first reduction uses a supervisor vertex, so try and give it
          // a grain of splits per-worker each time.
          const auto supervisorPartials = partialsFactor * numWorkers;
          const auto partialsThisSplit =
              std::min(lastBatchPartials - batchOffset, supervisorPartials);
          const auto divisorLog2 = poplibs_support::ceilLog2(partialsFactor);
          const auto divisor = (1u << divisorLog2);
          const auto nOutputs = (partialsThisSplit + divisor - 1) / divisor;
          auto splitValuePartials = lastValuePartials[b].slice(
              batchOffset, batchOffset + partialsThisSplit);
          auto splitMinValue =
              valuePartials[b].slice(partialsIndex, partialsIndex + nOutputs);
          auto splitMinIndex =
              indexPartials[b].slice(partialsIndex, partialsIndex + nOutputs);
          graph.connect(v["activations"], splitValuePartials);
          graph.setInitialValue(v["index"], batchOffset);
          graph.connect(v[lowerCase + "Value"], splitMinValue);
          graph.connect(v[lowerCase + "Index"], splitMinIndex);
          graph.setInitialValue(v["size"], partialsThisSplit);
          graph.setInitialValue(v["divisorLog2"], divisorLog2);
          graph.setTileMapping(splitMinValue, tile);
          graph.setTileMapping(splitMinIndex, tile);
          partialsIndex += nOutputs;
          batchOffset += partialsThisSplit;
        } else {
          const auto partialsThisSplit =
              std::min(lastBatchPartials - batchOffset, partialsFactor);
          auto splitValuePartials = lastValuePartials[b].slice(
              batchOffset, batchOffset + partialsThisSplit);
          auto splitIndexPartials = lastIndexPartials[b].slice(
              batchOffset, batchOffset + partialsThisSplit);
          graph.connect(v["activations"], splitValuePartials);
          graph.connect(v["labels"], splitIndexPartials);
          graph.connect(v[lowerCase + "Value"],
                        valuePartials[b][partialsIndex]);
          graph.connect(v[lowerCase + "Index"],
                        indexPartials[b][partialsIndex]);
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
    lastIndexPartials =
        graph.addVariable(argminType, {batchSize, 1},
                          layerPrefix + "/" + lowerCase + "IndexPartials");
    graph.setTileMapping(lastIndexPartials, numCorrectTile);
    for (std::size_t b = 0; b < batchSize; ++b) {
      graph.setInitialValue(lastIndexPartials[b][0], 0);
    }
  }

  for (const auto &cs : reductionCS) {
    prog.add(Execute(cs));
  }
  return lastIndexPartials;
}

Tensor argMax(Graph &graph, const Tensor &input, Sequence &prog,
              const std::string &debugPrefix) {
  // TODO: map the tensor to which the output goes correctly
  unsigned numCorrectTile = 0;

  if (input.rank() != 2) {
    throw poplibs_error("input tensor must be of rank 2");
  }

  if (input.elementType() != FLOAT && input.elementType() != HALF) {
    throw poplibs_error("arg max on input type is not supported");
  }
  auto output = argMinOrMax(graph, input, UNSIGNED_INT, prog, numCorrectTile,
                            debugPrefix);
  return output;
}

Tensor argMin(Graph &graph, const Tensor &input, Sequence &prog,
              const std::string &debugPrefix) {
  // TODO: map the tensor to which the output goes correctly
  unsigned numCorrectTile = 0;

  if (input.rank() != 2) {
    throw poplibs_error("input tensor must be of rank 2");
  }

  if (input.elementType() != FLOAT && input.elementType() != HALF) {
    throw poplibs_error("arg min on input type is not supported");
  }
  auto output = argMinOrMax(graph, input, UNSIGNED_INT, prog, numCorrectTile,
                            debugPrefix, false);
  return output;
}

Program
calcAccuracy(Graph &graph,
             const Tensor &modelOutputs,
             const Tensor &expected,
             const Tensor &numCorrect,
             const std::string &debugPrefix) {
  const auto layerPrefix = debugPrefix + "/Accuracy";

  // Normalize shape of numCorrect
  auto flatNumCorrect = numCorrect.flatten();
  if (flatNumCorrect.dim(0) != 1) {
    throw poplibs_error("numCorrect must be scalar or single element tensor");
  }
  const auto batchSize = modelOutputs.dim(0);
  if (expected.shape().size() > 1) {
    throw poplibs_error("expected must be a 1-dimensional tensor");
  }
  if (expected.dim(0) != batchSize) {
    throw poplibs_error("expected tensor must be of length equal the number of "
                       "batches given in modelOutputs tensor");
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

  Sequence prog;
  auto lastIndexPartials =
      argMinOrMax(graph, modelOutputs, expected.elementType(), prog,
                  *numCorrectTile, layerPrefix);

  // This would ideally be calculated with a popops::eq followed by a
  // popops::reduceWithOutput. At the moment popops::eq outputs bool
  // so this requires some ugly casting. For now this last step is its
  // own vertex. Doesn't particularly matter while batch size is generally
  // so small.
  const auto calcAccuracyVertexClass =
    templateVertex("popnn::CalcAccuracy", expected.elementType());

  const auto calcAccuracyCS =
    graph.addComputeSet(layerPrefix + "/CalcAccuracy");
  auto v = graph.addVertex(calcAccuracyCS, calcAccuracyVertexClass,
                           {{"maxPerBatch", lastIndexPartials.flatten()},
                            {"expected", expected},
                            {"numCorrect", flatNumCorrect[0]}});
  graph.setTileMapping(v, *numCorrectTile);

  // Add all the reductions and final accuracy.
  prog.add(Execute(calcAccuracyCS));
  return prog;
}

Program
calcLoss(Graph &graph,
         const Tensor &modelOutputs,
         const Tensor &expected,
         const Tensor &loss,
         const Tensor &deltas,
         const Tensor &numCorrect,
         LossType lossType,
         const std::string &debugPrefix) {
  Sequence prog(
    calcLoss(graph, modelOutputs, expected, loss, deltas, lossType,
             debugPrefix),
    calcAccuracy(graph, modelOutputs, expected, numCorrect, debugPrefix)
  );
  return prog;
}

} // end namespace popnn
