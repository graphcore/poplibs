// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include "poplibs_support/logging.hpp"
#include <poplar/Graph.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <popnn/CTCLoss.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poplibs_test;
using namespace poputil;

namespace {

enum class VertexType { ALPHA, BETA, GRAD_GIVEN_ALPHA, GRAD_GIVEN_BETA };

void generateVertex(Graph &graph, const Tensor &data, const Tensor &labels,
                    const Tensor &validLabels, const Tensor &tempAlphaOrBeta,
                    boost::optional<Tensor &> alphaOrBeta, const Tensor &out,
                    ComputeSet &cs, unsigned tile, VertexType vertexType,
                    unsigned blankClass) {

  const auto inType = data.elementType();
  const auto outType = out.elementType();
  const auto labelType = labels.elementType();
  std::string vertexName;
  if (vertexType == VertexType::ALPHA) {
    vertexName = templateVertex("popnn::CTCAlpha", inType, outType, labelType);
  } else if (vertexType == VertexType::BETA) {
    vertexName = templateVertex("popnn::CTCBeta", inType, outType, labelType);
  } else if (vertexType == VertexType::GRAD_GIVEN_ALPHA) {
    vertexName =
        templateVertex("popnn::CTCGradGivenAlpha", inType, outType, labelType);
  } else if (vertexType == VertexType::GRAD_GIVEN_BETA) {
    vertexName =
        templateVertex("popnn::CTCGradGivenBeta", inType, outType, labelType);
  }
  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, tile);

  graph.setInitialValue(v["maxT"], data.shape()[0]);
  graph.setInitialValue(v["numClasses"], data.shape()[2]);
  graph.setInitialValue(v["blankClass"], blankClass);

  graph.connect(v["probabilities"], data.flatten());
  graph.connect(v["labels"], labels.flatten());
  graph.connect(v["validLabels"], validLabels.reshape({}));

  if (vertexType == VertexType::ALPHA) {
    graph.connect(v["alphas"], out.flatten());
    graph.connect(v["alphaTemp"], tempAlphaOrBeta.flatten());
  } else if (vertexType == VertexType::BETA) {
    graph.connect(v["betas"], out.flatten());
    graph.connect(v["betaTemp"], tempAlphaOrBeta.flatten());
  } else if (vertexType == VertexType::GRAD_GIVEN_ALPHA) {
    graph.connect(v["grads"], out.flatten());
    graph.connect(v["betaTemp"], tempAlphaOrBeta.flatten());
    graph.connect(v["alphas"], alphaOrBeta.get().flatten());
  } else if (vertexType == VertexType::GRAD_GIVEN_BETA) {
    graph.connect(v["grads"], out.flatten());
    graph.connect(v["alphaTemp"], tempAlphaOrBeta.flatten());
    graph.connect(v["betas"], alphaOrBeta.get().flatten());
  }
}

void mapAccordingToPlan(Graph &graph, const Tensor &tensor,
                        const popnn::ctc_loss::Plan::Impl &plan) {
  // Simple initial logic to map to incrementing tiles according to batch split
  // (which is assumed to equal batchSize)
  const auto batchDim = tensor.rank() == 3 ? 1 : 0;
  for (unsigned i = 0; i < plan.parallel.batch; i++) {
    graph.setTileMapping(tensor.slice(i, i + 1, batchDim), i);
  }
}

} // namespace
namespace popnn {
namespace ctc_loss {

poplar::Tensor createDataInput(poplar::Graph &graph, const poplar::Type &type,
                               const std::size_t batchSize,
                               const std::size_t maxTime,
                               const std::size_t numClasses, const Plan &plan,
                               const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(type, batchSize, maxTime, numClasses, plan));

  logging::popnn::debug("Creating data tensor for CTC Loss with Time:{}"
                        " Batches:{} Classes:{}",
                        maxTime, batchSize, numClasses);
  const auto data =
      graph.addVariable(type, {maxTime, batchSize, numClasses}, {di, "data"});
  mapAccordingToPlan(graph, data, plan.getImpl());
  di.addOutput(data);
  return data;
}

poplar::Tensor createLabelsInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::size_t batchSize,
                                 const std::size_t maxLabels, const Plan &plan,
                                 const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(type, batchSize, maxLabels, plan));

  logging::popnn::debug("Creating labels tensor for CTC Loss with"
                        " Batches:{} Labels:{}",
                        batchSize, maxLabels);
  const auto labels =
      graph.addVariable(type, {batchSize, maxLabels}, {di, "labels"});
  mapAccordingToPlan(graph, labels, plan.getImpl());
  di.addOutput(labels);
  return labels;
}

poplar::Tensor
gradient(poplar::Graph &graph, const poplar::Type &outType,
         const poplar::Tensor &data, const poplar::Tensor &labels,
         const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
         poplar::program::Sequence &prog, const unsigned blankClass,
         const Plan &plan_, const poplar::DebugContext &debugContext) {

  const auto plan = plan_.getImpl();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(outType, data, labels, dataLengths,
                                         labelLengths, blankClass, plan_));
  const std::string layer = "CTCGradient";

  logging::popnn::debug("Creating CTCLoss using {}", plan_);
  const auto batchSize = data.dim(1);
  const auto maxT = data.dim(0);
  const auto extendedLabelsLength = 2 * labels.dim(1) + 1;

  auto gradient =
      graph.addVariable(outType, data.shape(), {di, layer + "/gradient"});
  mapAccordingToPlan(graph, gradient, plan);

  auto betas = graph.addVariable(
      outType, {maxT, batchSize, extendedLabelsLength}, {di, layer + "/betas"});
  mapAccordingToPlan(graph, betas, plan);

  // In our arithmetic, 0 is probability = 1, log::min is probability =  0
  // Create constants with which to initialise the temp vertex inputs and the
  // gradient
  auto initialZeros = graph.addConstant(outType, {extendedLabelsLength - 1},
                                        log::min, {di, layer + "/initalZeros"});
  auto initialOne =
      graph.addConstant(outType, {1}, 0, {di, layer + "/initalOne"});
  auto initialZero =
      graph.addConstant(outType, {1}, log::min, {di, layer + "/initalZero"});
  graph.setTileMapping(initialZeros, 0);
  graph.setTileMapping(initialZero, 0);
  graph.setTileMapping(initialOne, 0);

  prog.add(Copy(initialZero.broadcast(gradient.numElements(), 0),
                gradient.flatten(), false, {di}));

  auto cs1 = graph.addComputeSet({di, layer + "/beta"});
  for (unsigned i = 0; i < batchSize; i++) {
    auto betaTemp = graph.addVariable(outType, {1, extendedLabelsLength},
                                      {di, layer + "/betaTemp"});
    graph.setTileMapping(betaTemp, i);
    prog.add(Copy(concat(initialZeros, initialOne), betaTemp, false, {di}));

    auto tileData = data.slice(i, i + 1, 1);
    auto tileLabels = labels.slice(i, i + 1, 0);
    auto tileLabelLength = labelLengths.slice(i, i + 1);
    auto tileBetas = betas.slice(i, i + 1, 1);
    generateVertex(graph, tileData, tileLabels, tileLabelLength, betaTemp,
                   boost::none, tileBetas, cs1, i, VertexType::BETA,
                   blankClass);
  }

  prog.add(Execute(cs1, {di, layer}));

  auto cs2 = graph.addComputeSet({di, layer + "/alphaGrad"});
  for (unsigned i = 0; i < batchSize; i++) {
    auto alphaTemp = graph.addVariable(outType, {2, extendedLabelsLength},
                                       {di, layer + "/alphaTemp"});
    graph.setTileMapping(alphaTemp, i);
    prog.add(Copy(concat(initialOne, initialZeros), alphaTemp.slice(0, 1, 0),
                  false, {di}));

    auto tileData = data.slice(i, i + 1, 1);
    auto tileLabels = labels.slice(i, i + 1, 0);
    auto tileLabelLength = labelLengths.slice(i, i + 1);
    auto tileBetas = betas.slice(i, i + 1, 1);
    auto tileGradient = gradient.slice(i, i + 1, 1);
    generateVertex(graph, tileData, tileLabels, tileLabelLength, alphaTemp,
                   tileBetas, tileGradient, cs2, i, VertexType::GRAD_GIVEN_BETA,
                   blankClass);
  }

  prog.add(Execute(cs2, {di, layer}));
  di.addOutput(gradient);
  return gradient;
}

} // end namespace ctc_loss

} // end namespace popnn
