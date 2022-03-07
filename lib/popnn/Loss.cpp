// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "popnn/Loss.hpp"

#include "poplar/Graph.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Encoding.hpp"
#include "popops/Reduce.hpp"
#include "popops/TopK.hpp"
#include "poputil/Broadcast.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VarStructure.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/Algorithm.hpp>

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <cassert>
#include <limits>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs;
using namespace poputil;

namespace logging = poplibs_support::logging;

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const popnn::LossType &t) {
  switch (t) {
  case popnn::SUM_SQUARED_LOSS:
    return poplar::ProfileValue("SUM_SQUARED_LOSS");
  case popnn::CROSS_ENTROPY_LOSS:
    return poplar::ProfileValue("CROSS_ENTROPY_LOSS");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}
} // namespace poputil

namespace popnn {

namespace {

// Per-element of model outputs, on the tile these are mapped to
// compute the gradient and the contribution to loss.
Tensor onTileTransform(Graph &graph, const Tensor &modelOutputs,
                       const Tensor &expected, const Tensor &deltas,
                       boost::optional<Tensor> &_deltasScale,
                       boost::optional<Tensor> &_modelOutputScaling,
                       const std::string &vertexClassTemplate,
                       LossType lossType, Sequence &prog,
                       const DebugNameAndId &dnai) {
  const auto &target = graph.getTarget();
  const auto &dType = modelOutputs.elementType();

  Tensor deltasScale, modelOutputScaling;
  if (_deltasScale.is_initialized()) {
    if (lossType == LossType::CROSS_ENTROPY_LOSS) {
      deltasScale = _deltasScale.get();
      modelOutputScaling = _modelOutputScaling.get();
    } else {
      throw poplibs_error("Loss scaling not implemented for this loss type");
    }
  } else if (lossType == LossType::CROSS_ENTROPY_LOSS) {
    deltasScale = graph.addConstant(deltas.elementType(), {}, 1.0f, {dnai});
    modelOutputScaling =
        graph.addConstant(deltas.elementType(), {}, 1.0f, {dnai});
    graph.setTileMapping(deltasScale, 0);
    graph.setTileMapping(modelOutputScaling, 0);
  }

  const auto vertexClass = templateVertex(vertexClassTemplate, dType);
  auto transformCS = graph.addComputeSet({dnai, "on_tile_transform"});
  const auto batchSize = modelOutputs.dim(0);
  const auto perBatch = modelOutputs.numElements() / batchSize;
  auto transformed =
      graph.addVariable(dType, {batchSize, perBatch}, {dnai, "Transformed"});
  auto mapping = graph.getTileMapping(modelOutputs);
  for (std::size_t tile = 0; tile < mapping.size(); ++tile) {
    const auto &tileMapping = mapping[tile];
    if (tileMapping.empty())
      continue;
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

      auto transformV = graph.addVertex(
          transformCS, vertexClass,
          {{"probs", concat(modelOutputs.flatten().slices(vertexRegions))},
           {"expected", concat(expected.flatten().slices(vertexRegions))},
           {"deltas", vertexDeltas},
           {"transformed", vertexTransformed}});
      if (lossType == LossType::CROSS_ENTROPY_LOSS) {
        graph.connect(transformV["deltasScale"], deltasScale);
        graph.connect(transformV["modelOutputScaling"], modelOutputScaling);
      }
      graph.setInitialValue(transformV["size"],
                            vertexTransformed.numElements());
      graph.setTileMapping(vertexTransformed, tile);
      graph.setTileMapping(vertexDeltas, tile);
      graph.setTileMapping(transformV, tile);
    }
  }
  prog.add(Execute(transformCS, {dnai}));
  return transformed;
}

} // end anonymous namespace

Program calcLoss(Graph &graph, const Tensor &modelOutputs,
                 const Tensor &expected, const Tensor &loss,
                 const Tensor &deltas, boost::optional<Tensor> &deltasScale,
                 boost::optional<Tensor> &modelOutputScaling, LossType lossType,
                 const DebugNameAndId &dnai) {

  const auto getShape = [](const Tensor &t) {
    std::stringstream ss;
    ss << t.shape();
    return ss.str();
  };

  logging::popnn::info(
      "calcLoss modelOutputs={}, expected={}, loss={}, deltas={}, "
      "deltasScale={}, modelOutputScaling={}, type={}, name={}",
      modelOutputs.shape(), expected.shape(), loss.shape(), deltas.shape(),
      fmap(deltasScale, getShape), fmap(modelOutputScaling, getShape), lossType,
      dnai.getPathName());

  std::string transformVertexClass;
  if (modelOutputs.rank() != 2) {
    throw poplibs_error(
        "calcLoss: Rank mismatch. 'modelOutputs' tensor rank is: " +
        std::to_string(modelOutputs.rank()) + ". Must be a 2D tensor");
  }
  if (expected.rank() != 1) {
    throw poplibs_error("calcLoss: Rank mismatch. 'expected' tensor rank is: " +
                        std::to_string(expected.rank()) +
                        ". Must be a one-hot "
                        "encoded vector with the same number "
                        "of rows as 'modelOutputs': " +
                        std::to_string(modelOutputs.dim(0)));
  }
  if (expected.dim(0) != modelOutputs.dim(0)) {
    throw poplibs_error("calcLoss: Dimension mismatch of 'expected' tensor: {" +
                        std::to_string(expected.dim(0)) +
                        "}"
                        ". Must be a one-hot encoded vector with the "
                        "same number of rows as 'modelOutputs': {" +
                        std::to_string(modelOutputs.dim(0)) + "}");
  }
  if (loss.rank() != 1) {
    throw poplibs_error("calcLoss: Rank mismatch. 'loss' tensor rank is: " +
                        std::to_string(loss.rank()) +
                        ". Must be a 1D "
                        "vector with the same number "
                        "of rows as 'modelOutputs': " +
                        std::to_string(modelOutputs.dim(0)));
  }
  if (loss.dim(0) != modelOutputs.dim(0)) {
    throw poplibs_error("calcLoss: Dimension mismatch of 'loss' tensor: {" +
                        std::to_string(loss.dim(0)) +
                        "}"
                        ". Must be the same number of rows as "
                        "'modelOutputs': {" +
                        std::to_string(modelOutputs.dim(0)) + "}");
  }
  if (deltas.rank() != 2) {
    throw poplibs_error("calcLoss: Rank mismatch. 'deltas' tensor rank is: " +
                        std::to_string(deltas.rank()) +
                        " - Must be a 2D tensor of the same dimensions as "
                        "'modelOutputs': {" +
                        std::to_string(modelOutputs.dim(0)) + "," +
                        std::to_string(modelOutputs.dim(1)) + "}");
  }
  if ((deltas.dim(0) != modelOutputs.dim(0)) ||
      (deltas.dim(1) != modelOutputs.dim(1))) {
    throw poplibs_error("calcLoss: Dimension mismatch of 'deltas' tensor: {" +
                        std::to_string(deltas.dim(0)) + "," +
                        std::to_string(deltas.dim(1)) +
                        "} - Must be a 2D tensor of the same dimensions as "
                        "'modelOutputs': {" +
                        std::to_string(modelOutputs.dim(0)) + "," +
                        std::to_string(modelOutputs.dim(1)) + "}");
  }

  std::string layerPrefix;
  switch (lossType) {
  case LossType::SUM_SQUARED_LOSS:
    layerPrefix = "LossSumSquared";
    transformVertexClass = "popnn::LossSumSquaredTransform";
    break;
  case LossType::CROSS_ENTROPY_LOSS:
    layerPrefix = "LossCrossEntropy";
    transformVertexClass = "popnn::LossCrossEntropyTransform";
    break;
  default:
    throw poplibs_error("Unknown loss type requested in calcLoss");
    break;
  }

  Sequence prog({}, {dnai});
  const auto &target = graph.getTarget();
  const auto &dType = modelOutputs.elementType();
  const unsigned atomicStoreGranularity = target.getAtomicStoreGranularity();
  const unsigned exchangeBytesPerCycles = target.getExchangeBytesPerCycle();
  auto minBytes = std::max(atomicStoreGranularity, exchangeBytesPerCycles);
  auto dTypeSize = target.getTypeSize(dType);

  // Determine if we have to layout the one hot encoded tensor with the
  // innermost dimension having contiguous elements on a tile.
  const auto dimGrouping = detectInnermostGrouping(graph, modelOutputs);
  Tensor oneHot;
  if (dimGrouping * dTypeSize < minBytes) {
    oneHot = graph.addVariable(dType, modelOutputs.shape(),
                               {dnai, layerPrefix + "/OneHotEncoded"});
    mapTensorLinearly(graph, oneHot, 0, target.getVectorWidth(dType));
  } else {
    oneHot = graph.clone(modelOutputs.elementType(), modelOutputs,
                         {dnai, layerPrefix + "/OneHotEncoded"});
  }
  popops::encodeOneHot(graph, expected, oneHot, prog, {dnai, layerPrefix});

  // Compute loss partials and deltas
  auto transformed = onTileTransform(
      graph, modelOutputs, oneHot, deltas, deltasScale, modelOutputScaling,
      transformVertexClass, lossType, prog, {dnai, layerPrefix});
  // the gradients for masked labels are not masked out to 0 by the on tile
  // transform. This does this explicitly here for such label.
  if (lossType == CROSS_ENTROPY_LOSS && deltas.rank() == 2 &&
      expected.numElements() > 1) {
    auto maskedLabelCode = graph.addConstant(expected.elementType(), {},
                                             MASKED_LABEL_CODE, {dnai});
    graph.setTileMapping(maskedLabelCode, 0);
    auto nonMaskedLabels =
        popops::neq(graph, expected, maskedLabelCode, prog, {dnai});
    auto maskScale = popops::cast(graph, nonMaskedLabels, deltas.elementType(),
                                  prog, {dnai});
    popops::mulInPlace(graph, deltas.transpose(), maskScale, prog, {dnai});
  }
  // Reduce values in each batch
  popops::reduceWithOutput(graph, transformed, loss, {1},
                           popops::Operation::ADD, prog,
                           {dnai, layerPrefix + "/reduce_loss"});
  return std::move(prog);
}

Program calcLoss(Graph &graph, const Tensor &modelOutputs,
                 const Tensor &expected, const Tensor &loss,
                 const Tensor &deltas, const Tensor &_deltasScale,
                 const Tensor &_modelOutputScaling, LossType lossType,
                 const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(modelOutputs, expected, loss, deltas, _deltasScale,
                            _modelOutputScaling, lossType));

  boost::optional<Tensor> deltasScale = _deltasScale;
  boost::optional<Tensor> modelOutputScaling = _modelOutputScaling;
  return calcLoss(graph, modelOutputs, expected, loss, deltas, deltasScale,
                  modelOutputScaling, lossType, {di});
}

Program calcLoss(Graph &graph, const Tensor &modelOutputs,
                 const Tensor &expected, const Tensor &loss,
                 const Tensor &deltas, LossType lossType,
                 const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(modelOutputs, expected, loss, deltas, lossType));

  boost::optional<Tensor> deltasScale, modelOutputScaling;
  return calcLoss(graph, modelOutputs, expected, loss, deltas, deltasScale,
                  modelOutputScaling, lossType, {di});
}

Tensor topK(Graph &graph, const Tensor &input, Tensor &indices, unsigned K,
            bool sort, Sequence &prog,
            const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(input, indices, K, sort));

  logging::popnn::info("topK input={}, k={}, sort={}, name={}", input.shape(),
                       K, sort, debugContext.getPathName());

  if (input.rank() != 2) {
    throw poplibs_error("Topk: input tensor must be of rank 2");
  }

  if (input.elementType() != FLOAT && input.elementType() != HALF &&
      input.elementType() != INT && input.elementType() != UNSIGNED_INT &&
      input.elementType() != BOOL) {
    throw poplibs_error("TopK on input type is not supported");
  }

  if (input.dim(1) < K) {
    throw poplibs_error("K must be smaller or equal to the size of the "
                        "dimensions which the TopK is being calculated for.");
  }

  Tensor output;
  const bool largest = true;
  const auto sortOrder =
      sort ? popops::SortOrder::DESCENDING : popops::SortOrder::NONE;
  const popops::TopKParams params(K, largest, sortOrder, false);
  std::tie(output, indices) =
      popops::topKWithPermutation(graph, prog, input, params, {di});

  // Match behaviour of existing API that adds an extra singleton
  // dimension to the return
  output = output.expand({1});
  indices = indices.expand({1});

  di.addOutput(output);
  return output;
}

inline static std::pair<poplar::Tensor, poplar::Tensor>
getTop1Elem(Graph &graph, const Tensor &input, Sequence &prog, bool getMaxElem,
            const DebugNameAndId &dnai) {
  const popops::TopKParams params(1, getMaxElem, popops::SortOrder::NONE, true);
  return popops::topKWithPermutation(graph, prog, input, params, {dnai});
}

Tensor argMax(Graph &graph, const Tensor &input, Sequence &prog,
              const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  logging::popnn::info("argMax input={}, name={}", input.shape(),
                       debugContext.getPathName());

  auto output = getTop1Elem(graph, input, prog, true, {di}).second;
  di.addOutput(output);
  return output;
}

std::pair<poplar::Tensor, poplar::Tensor>
maxAndArgMax(Graph &graph, const Tensor &input, Sequence &prog,
             const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  logging::popnn::info("maxAndArgMax input={}, name={}", input.shape(),
                       debugContext.getPathName());
  const auto &topPair = getTop1Elem(graph, input, prog, true, {di});

  di.addOutputs({{"max", toProfileValue(topPair.first)},
                 {"argMax", toProfileValue(topPair.second)}});
  return std::move(topPair);
}

Tensor argMin(Graph &graph, const Tensor &input, Sequence &prog,
              const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  logging::popnn::info("argMin input={}, name={}", input.shape(),
                       debugContext.getPathName());

  auto output = getTop1Elem(graph, input, prog, false, {di}).second;
  di.addOutput(output);
  return output;
}

std::pair<poplar::Tensor, poplar::Tensor>
minAndArgMin(Graph &graph, const Tensor &input, Sequence &prog,
             const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(input));

  logging::popnn::info("minAndArgMin input={}, name={}", input.shape(),
                       debugContext.getPathName());
  const auto &topPair = getTop1Elem(graph, input, prog, false, {di});

  di.addOutputs({{"min", toProfileValue(topPair.first)},
                 {"argMin", toProfileValue(topPair.second)}});
  return std::move(topPair);
}

/// Compute the number of correct outputs against expected.
///
/// \param[in] graph        the graph for the tensor
/// \param[in] modelOutputs a 2D Tensor.
/// \param[in] expected     a 1D Tensor of integral type with the same number of
///                         elements as rows in 'modelOutputs'. Each element
///                         contains the index into the corresponding row of
///                         'modelOutputs' where we expect to find the maximum
///                         value for that row.
/// \param[out] numCorrect  a tensor containing a single element where this will
///                         place the result: the number of elements in
///                         'expected' that correctly indicate the max for their
//                          rows
/// \param[in] debugContext  as the name says
Program calcAccuracy(Graph &graph, const Tensor &modelOutputs,
                     const Tensor &expected, const Tensor &numCorrect,
                     const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(modelOutputs, expected));

  const std::string layerPrefix = "Accuracy";
  logging::popnn::info(
      "calcAccuracy modelOutputs={}, expected={}, numCorrect={}, name={}",
      modelOutputs.shape(), expected.shape(), numCorrect.shape(),
      debugContext.getPathName() + "/" + layerPrefix);

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
      assert(tileMapping.size() == 1 && tileMapping[0].size() == 1);
      numCorrectTile = tileMapping[0].begin();
    }
  }
  assert(numCorrectTile);

  // Get the indices of the max value of each row of 'modelOutput'
  Sequence prog({}, {di});

  auto maxIndices =
      getTop1Elem(graph, modelOutputs, prog, true, {di, layerPrefix}).second;
  if (maxIndices.elementType() != expected.elementType()) {
    maxIndices = popops::cast(graph, maxIndices, expected.elementType(), prog,
                              {di, layerPrefix});
  }

  // This would ideally be calculated with a popops::eq followed by a
  // popops::reduceWithOutput. At the moment popops::eq outputs bool
  // so this requires some ugly casting. For now this last step is its
  // own vertex. Doesn't particularly matter while batch size is generally
  // so small.
  const auto calcAccuracyVertexClass =
      templateVertex("popnn::CalcAccuracy", expected.elementType());

  const auto calcAccuracyCS =
      graph.addComputeSet({di, layerPrefix + "/CalcAccuracy"});
  auto v = graph.addVertex(calcAccuracyCS, calcAccuracyVertexClass,
                           {{"maxPerBatch", maxIndices.flatten()},
                            {"expected", expected},
                            {"numCorrect", flatNumCorrect[0]}});
  graph.setTileMapping(v, *numCorrectTile);
  // Add all the reductions and final accuracy.
  prog.add(Execute(calcAccuracyCS, {di}));

  di.addOutputs(DI_ARGS(numCorrect));

  return std::move(prog);
}

Program calcLoss(Graph &graph, const Tensor &modelOutputs,
                 const Tensor &expected, const Tensor &loss,
                 const Tensor &deltas, const Tensor &_deltasScale,
                 const Tensor &_modelOutputScaling, const Tensor &numCorrect,
                 LossType lossType, const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(modelOutputs, expected, loss, deltas, _deltasScale,
                            _modelOutputScaling, numCorrect, lossType));

  boost::optional<Tensor> deltasScale = _deltasScale;
  boost::optional<Tensor> modelOutputScaling = _modelOutputScaling;
  Sequence prog({calcLoss(graph, modelOutputs, expected, loss, deltas,
                          deltasScale, modelOutputScaling, lossType, {di}),
                 calcAccuracy(graph, modelOutputs, expected, numCorrect, {di})},
                {di});
  return std::move(prog);
}

Program calcLoss(Graph &graph, const Tensor &modelOutputs,
                 const Tensor &expected, const Tensor &loss,
                 const Tensor &deltas, const Tensor &numCorrect,
                 LossType lossType, const poplar::DebugContext &debugContext) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(modelOutputs, expected, loss, deltas, numCorrect, lossType));

  boost::optional<Tensor> deltasScale, modelOutputScaling;
  Sequence prog({calcLoss(graph, modelOutputs, expected, loss, deltas,
                          deltasScale, modelOutputScaling, lossType, {di}),
                 calcAccuracy(graph, modelOutputs, expected, numCorrect, {di})},
                {di});
  return std::move(prog);
}

} // end namespace popnn
