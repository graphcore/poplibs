// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include "poplibs_support/logging.hpp"
#include <poplar/CSRFunctions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <popnn/CTCLoss.hpp>
#include <popnn/LogSoftmax.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops;
using namespace popops::expr;
using namespace poputil;

template <unsigned size> using Slice = std::array<std::size_t, size>;
namespace {

enum class VertexType { ALPHA, BETA, GRAD_GIVEN_ALPHA, GRAD_GIVEN_BETA };

struct TempTensors {
  // Timestep counter to count loop passes with a counter on each tile
  // [timePartitions][batchSize][labelPartitions]
  Tensor counter;
  // A slice of the label input per tile, including previous/next symbol
  // to satisfy dependencies
  //[batchSize][timePartitions][labelPartitions][labelLengthPerPartition+2]
  Tensor label;
  // The actual valid length of each label in the batch, broadcast over all
  // time, label partitions.
  // [batchSize][timePartitions][labelPartitions]
  Tensor validLabelLengths;
  // The actual number of valid timesteps in the data input per batch entry,
  // broadcast over all time, label partitions.
  // [batchSize][timePartitions][labelPartitions]
  Tensor validTimeSteps;

  // The last timeslice of alpha/beta result exchanged
  // from the previous time partition while calculating alpha/beta
  // [maxT][batchSize][maxExtendedLabelLength]
  Tensor timeAlphaBeta1;
  // The gradGivenAlpha/beta vertex ping-pong working buffer containing 2
  // timeslices
  // [2 * maxT][batchSize][maxExtendedLabelLength]
  Tensor timeAlphaBeta2;
  // The timeAlphaBeta2 ping-pong data excahnged from the previous time
  // partition so that a gradGivenAlpha/beta vertex can continue to compute
  // alpha/beta
  // [2 * maxT][batchSize][maxExtendedLabelLength]
  Tensor timeAlphaBetaPrevPartition;

  // A single alpha value written by the vertex to exchange to a tile in the
  // next label, or label,time partition
  // [labelPartitions][batchSize][timePartitions][1]
  Tensor labelAlphaOut;
  // A single alpha value exchanged from the previous label partition
  // [labelPartitions][batchSize][timePartitions][1]
  Tensor labelAlphaIn;
  // A single alpha value exchanged from the previous label and time partition
  // [labelPartitions][batchSize][timePartitions][1]
  Tensor labelTimeAlphaIn; // Exchanged into

  // A pair of beta values written by the vertex to exchange to a tile in the
  // next label, or label,time partition
  // [labelPartitions][batchSize][timePartitions][2]
  Tensor labelBetaOut;
  // A pair of beta values exchanged from the previous label partition
  // [labelPartitions][batchSize][timePartitions][2]
  Tensor labelBetaIn;
  // A pair of beta values exchanged from the previous label and time partition
  // [labelPartitions][batchSize][timePartitions][2]
  Tensor labelTimeBetaIn;
};

TempTensors extractVertexSlices(const TempTensors &input, unsigned batch,
                                unsigned time, unsigned label,
                                const Interval &exLabelPartition,
                                const Interval &timePartition, bool isAlpha) {

  TempTensors output;
  output.validLabelLengths = input.validLabelLengths[batch][time][label];
  output.validTimeSteps = input.validTimeSteps[batch][time][label];
  output.counter = input.counter[time][batch][label];

  auto tempLabel = [&](const Tensor &toSlice) {
    const auto innerDimSize = toSlice.dim(toSlice.rank() - 1);
    Slice<4> begin = {label, batch, time, 0};
    Slice<4> end = {label + 1, batch + 1, (time + 1), innerDimSize};
    return toSlice.slice(begin, end).flatten();
  };
  output.labelAlphaIn = tempLabel(input.labelAlphaIn);
  output.labelAlphaOut = tempLabel(input.labelAlphaOut);
  output.labelTimeAlphaIn = tempLabel(input.labelTimeAlphaIn);

  output.labelBetaIn = tempLabel(input.labelBetaIn);
  output.labelBetaOut = tempLabel(input.labelBetaOut);
  output.labelTimeBetaIn = tempLabel(input.labelTimeBetaIn);

  // The temp.label input tensor contains a section of label:
  // prev,a,b,c,...,next. This vertex is responsible for the results for
  // a,b,c.... but when finding alpha there is a dependency on prev - so attach
  // the vertex to prev,a,b,c,.. for beta there is a dependency on next = so
  // attach a,b,c,...next
  unsigned startOffset = isAlpha ? 0 : 1;
  unsigned endOffset = isAlpha ? 1 : 0;

  unsigned tileLabelSize = 2 + ceildiv(exLabelPartition.size() - 1, 2u);
  output.label = input.label.slice(
      {batch, time, label, startOffset},
      {batch + 1, time + 1, label + 1, tileLabelSize - endOffset});

  // Provide the temporary time input slices for a grad given
  // alpha/beta vertex
  auto tileTimeInputSlice = [&](const Tensor &in,
                                unsigned elemsPerTimePartition) {
    Slice<3> begin = {elemsPerTimePartition * time, batch,
                      exLabelPartition.begin()};
    Slice<3> end = {elemsPerTimePartition * (time + 1), batch + 1,
                    exLabelPartition.end()};
    return in.slice(begin, end).flatten();
  };
  output.timeAlphaBeta1 = tileTimeInputSlice(input.timeAlphaBeta1, 1);
  output.timeAlphaBeta2 = tileTimeInputSlice(input.timeAlphaBeta2, 2);
  output.timeAlphaBetaPrevPartition =
      tileTimeInputSlice(input.timeAlphaBetaPrevPartition, 2);
  return output;
}

void generateVertex(Graph &graph, const Tensor &data, const Tensor &alphaOrBeta,
                    const Tensor &loss, boost::optional<Tensor &> grad,
                    const TempTensors &temp, ComputeSet &cs, unsigned tile,
                    VertexType vertexType, unsigned batch, unsigned time,
                    unsigned label, const Interval &exLabelPartition,
                    unsigned labelOffset, const Interval &timePartition,
                    bool processExtraBlank, unsigned blankClass) {

  auto isAlpha = vertexType == VertexType::ALPHA ||
                 vertexType == VertexType::GRAD_GIVEN_BETA;

  const auto numClasses = data.dim(3);

  // Extract the slices of temporary tensors needed for this vertex
  const auto tempSlices = extractVertexSlices(
      temp, batch, time, label, exLabelPartition, timePartition, isAlpha);

  // A single time partition slice for the input data, alphaBeta and grad
  // outputs.
  auto timeBegin = timePartition.begin();
  auto timeEnd = timePartition.end();
  Slice<4> beginData = {label, timeBegin, batch, 0};
  Slice<4> endData = {label + 1, timeEnd, batch + 1, numClasses};
  auto tileData = data.slice(beginData, endData);

  Slice<3> beginAlphaBeta = {timeBegin, batch, exLabelPartition.begin()};
  Slice<3> endAlphaBeta = {timeEnd, batch + 1, exLabelPartition.end()};
  auto tileAlphaOrBeta =
      alphaOrBeta.slice(beginAlphaBeta, endAlphaBeta).flatten();

  const auto inType = data.elementType();
  const auto outType = alphaOrBeta.elementType();
  const auto labelType = temp.label.elementType();
  std::string vertexName;
  if (vertexType == VertexType::ALPHA) {
    vertexName = templateVertex("popnn::CTCAlpha", inType, outType, labelType,
                                processExtraBlank);
  } else if (vertexType == VertexType::BETA) {
    vertexName = templateVertex("popnn::CTCBeta", inType, outType, labelType,
                                processExtraBlank);
  } else if (vertexType == VertexType::GRAD_GIVEN_ALPHA) {
    vertexName = templateVertex("popnn::CTCGradGivenAlpha", inType, outType,
                                labelType, processExtraBlank);
  } else if (vertexType == VertexType::GRAD_GIVEN_BETA) {
    vertexName = templateVertex("popnn::CTCGradGivenBeta", inType, outType,
                                labelType, processExtraBlank);
  }
  logging::popnn::trace("Making {} vertex on tile {} with label offset {}"
                        " and time offset {}",
                        vertexName, tile, labelOffset, timeBegin);
  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, tile);

  graph.setInitialValue(v["maxT"], timePartition.size());
  graph.setInitialValue(v["numClasses"], numClasses);
  graph.setInitialValue(v["blankClass"], blankClass);
  graph.setInitialValue(v["labelOffset"], labelOffset);
  graph.setInitialValue(v["timeOffset"], timeBegin);

  graph.connect(v["probabilities"], tileData.flatten());
  graph.connect(v["label"], tempSlices.label.flatten());
  graph.connect(v["validLabel"], tempSlices.validLabelLengths.reshape({}));
  graph.connect(v["validTime"], tempSlices.validTimeSteps.reshape({}));
  graph.connect(v["count"], tempSlices.counter);

  Slice<4> beginGrad = {label, timeBegin, batch, 0};
  Slice<4> endGrad = {label + 1, timeEnd, batch + 1, numClasses};
  if (vertexType == VertexType::ALPHA) {
    graph.connect(v["alphas"], tileAlphaOrBeta);
    graph.connect(v["alphaPrevTime"], tempSlices.timeAlphaBeta1);
    graph.connect(v["alphaPrevLabel"], tempSlices.labelAlphaIn.flatten());
    graph.connect(v["alphaPrevLabelOut"], tempSlices.labelAlphaOut.flatten());
    graph.connect(v["alphaPrevLabelTime"],
                  tempSlices.labelTimeAlphaIn.flatten());
    graph.connect(v["loss"], loss[batch][time][label]);
  } else if (vertexType == VertexType::BETA) {
    graph.connect(v["betas"], tileAlphaOrBeta);
    graph.connect(v["betaPrevTime"], tempSlices.timeAlphaBeta1);
    graph.connect(v["betaPrevLabel"], tempSlices.labelBetaIn.flatten());
    graph.connect(v["betaPrevLabelOut"], tempSlices.labelBetaOut.flatten());
    graph.connect(v["betaPrevLabelTime"], tempSlices.labelTimeBetaIn.flatten());
  } else if (vertexType == VertexType::GRAD_GIVEN_ALPHA) {
    graph.connect(v["alphas"], tileAlphaOrBeta);
    graph.connect(v["grads"], grad.get().slice(beginGrad, endGrad).flatten());
    graph.connect(v["betaPrevTime"], tempSlices.timeAlphaBeta2);
    graph.connect(v["betaPrevPartition"],
                  tempSlices.timeAlphaBetaPrevPartition);
    graph.connect(v["betaPrevLabel"], tempSlices.labelBetaIn.flatten());
    graph.connect(v["betaPrevLabelOut"], tempSlices.labelBetaOut.flatten());
    graph.connect(v["betaPrevLabelTime"], tempSlices.labelTimeBetaIn.flatten());
  } else if (vertexType == VertexType::GRAD_GIVEN_BETA) {
    graph.connect(v["betas"], tileAlphaOrBeta);
    graph.connect(v["grads"], grad.get().slice(beginGrad, endGrad).flatten());
    graph.connect(v["alphaPrevTime"], tempSlices.timeAlphaBeta2);
    graph.connect(v["alphaPrevPartition"],
                  tempSlices.timeAlphaBetaPrevPartition);
    graph.connect(v["alphaPrevLabel"], tempSlices.labelAlphaIn.flatten());
    graph.connect(v["alphaPrevLabelOut"], tempSlices.labelAlphaOut.flatten());
    graph.connect(v["alphaPrevLabelTime"],
                  tempSlices.labelTimeAlphaIn.flatten());
    graph.connect(v["loss"], loss[batch][time][label]);
  }
}

void mapAlphaBetaAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 const popnn::ctc::Plan::Impl &plan) {
  // Map the alpha beta tensor to tiles according to the plan. The extended
  // label dimension needs to be partitioned to match the label partitions.
  // We need every el partition to be 2x the size of the corresponding label
  // partition (except the last one which has an extra blank symbol)
  const auto timeSize = tensor.dim(0);
  const auto batchSize = tensor.dim(1);
  const auto labelSize = (tensor.dim(2) - 1) / 2;

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned time = 0; time < plan.parallel.time; time++) {
      for (unsigned label = 0; label < plan.parallel.label; label++) {

        auto tile = plan.getTile(batch, time, label);
        auto b = plan.partitionBatch(batchSize, batch);
        auto t = plan.partitionTime(timeSize, time);
        auto l = plan.partitionExtendedLabel(labelSize, label);
        graph.setTileMapping(tensor.slice({t.begin(), b.begin(), l.begin()},
                                          {t.end(), b.end(), l.end()}),
                             tile);
      }
    }
  }
}

void mapLossAccordingToPlan(Graph &graph, const Tensor &tensor,
                            const popnn::ctc::Plan::Impl &plan) {
  const auto batchSize = tensor.dim(0);
  const auto timeSize = tensor.dim(1);
  const auto labelSize = tensor.dim(2);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned time = 0; time < plan.parallel.time; time++) {
      for (unsigned label = 0; label < plan.parallel.label; label++) {

        auto tile = plan.getTile(batch, time, label);
        auto b = plan.partitionBatch(batchSize, batch);
        auto t = plan.partitionTime(timeSize, time);
        auto l = plan.partitionLabel(labelSize, label);
        graph.setTileMapping(tensor.slice({b.begin(), t.begin(), l.begin()},
                                          {b.end(), t.end(), l.end()}),
                             tile);
      }
    }
  }
}

void mapAccordingToPlan(Graph &graph, const Tensor &tensor,
                        const popnn::ctc::Plan::Impl &plan) {
  // Map any rank 3 tensors used in this process to the correct tiles according
  // to the plan.
  const auto timeSize = tensor.dim(0);
  const auto batchSize = tensor.dim(1);
  const auto labelSize = tensor.dim(2);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned time = 0; time < plan.parallel.time; time++) {
      for (unsigned label = 0; label < plan.parallel.label; label++) {

        auto tile = plan.getTile(batch, time, label);
        auto b = plan.partitionBatch(batchSize, batch);
        auto t = plan.partitionTime(timeSize, time);
        auto l = plan.partitionLabel(labelSize, label);
        graph.setTileMapping(tensor.slice({t.begin(), b.begin(), l.begin()},
                                          {t.end(), b.end(), l.end()}),
                             tile);
      }
    }
  }
}

void mapDataInputAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 const popnn::ctc::Plan::Impl &plan) {
  // Map the data input according to the plan, but the innermost dimension
  // isn't really compatible with the plan, as it is the number of classes
  // whereas we planned for the label length.
  // Choose to split the time dimension as much as possible over the combined
  // time and label partitions.  This avoids splitting the innermost dimension
  // which would result in increased exchange code size.
  const unsigned timeSize = tensor.dim(0);
  const auto batchSize = tensor.dim(1);
  const auto numClasses = tensor.dim(2);

  auto numNonBatchPartitions = plan.parallel.time * plan.parallel.label;
  auto remappedTimePartitions = std::min(numNonBatchPartitions, timeSize);
  auto timePartitionSize = ceildiv(timeSize, remappedTimePartitions);

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned time = 0; time < remappedTimePartitions; time++) {
      const auto timeBegin = time * timePartitionSize;
      if (timeBegin < timeSize) {
        auto tile = plan.getTile(batch, time);
        auto b = plan.partitionBatch(batchSize, batch);
        const auto timeEnd = std::min(timeSize, (time + 1) * timePartitionSize);

        graph.setTileMapping(tensor.slice({timeBegin, b.begin(), 0},
                                          {timeEnd, b.end(), numClasses}),
                             tile);
      }
    }
  }
}

void mapGradientAccordingToPlan(Graph &graph, const Tensor &tensor,
                                const popnn::ctc::Plan::Impl &plan) {
  // Map the rank 4 gradient tensor used in this process to the correct tiles
  // according to the plan.
  const auto labelSize = tensor.dim(0);
  const auto timeSize = tensor.dim(1);
  const auto batchSize = tensor.dim(2);
  const auto numSymbols = tensor.dim(3);

  for (unsigned label = 0; label < plan.parallel.label; label++) {
    for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {

        auto tile = plan.getTile(batch, time, label);
        auto l = plan.partitionLabel(labelSize, label);
        auto b = plan.partitionBatch(batchSize, batch);
        auto t = plan.partitionTime(timeSize, time);
        graph.setTileMapping(
            tensor.slice({l.begin(), t.begin(), b.begin(), 0},
                         {l.end(), t.end(), b.end(), numSymbols}),
            tile);
      }
    }
  }
}

void mapTempLabelAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 const popnn::ctc::Plan::Impl &plan) {
  // Map the rank 4 temporary "split by label tensor" according to the plan.
  // Note that time is the innermost dimension which matches the ordering in
  // which the vertices write the temporary data.
  const auto labelSize = tensor.dim(0);
  const auto batchSize = tensor.dim(1);
  const auto timeSize = tensor.dim(2);
  const auto tempTimeSlices = tensor.dim(3);

  for (unsigned label = 0; label < plan.parallel.label; label++) {
    for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {

        const auto tile = plan.getTile(batch, time, label);
        auto l = plan.partitionLabel(labelSize, label);
        const auto b = plan.partitionBatch(batchSize, batch);
        const auto t = plan.partitionTime(timeSize, time);

        graph.setTileMapping(
            tensor.slice({l.begin(), b.begin(), t.begin(), 0},
                         {l.end(), b.end(), t.end(), tempTimeSlices}),
            tile);
      }
    }
  }
}

void mapLabelsAccordingToPlan(Graph &graph, const Tensor &tensor,
                              const popnn::ctc::Plan::Impl &plan) {
  // Map the labels tensor used in this process to the correct tiles according
  // to the plan.
  const auto batchSize = tensor.dim(0);
  const auto labelSize = tensor.dim(1);
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned label = 0; label < plan.parallel.label; label++) {
      auto tile = plan.getTile(batch, 0, label);
      auto b = plan.partitionBatch(batchSize, batch);
      auto l = plan.partitionLabel(labelSize, label);
      graph.setTileMapping(
          tensor.slice({b.begin(), l.begin()}, {b.end(), l.end()}), tile);
    }
  }
}
// Broadcast and explicitly copy the label lengths or data lengths tensors.
// Map to the tile where they are used to avoid any exchange between
// compute steps.
Tensor createTempLengths(Graph &graph, const Tensor &input,
                         const popnn::ctc::Plan::Impl &plan, Sequence &prog,
                         const poplar::DebugContext &di) {

  const auto tilesForAllTimePartitions = plan.parallel.time;
  const auto batchSize = input.dim(0);
  auto result = graph.addVariable(
      input.elementType(), {batchSize, plan.parallel.time, plan.parallel.label},
      {di});

  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned label = 0; label < plan.parallel.label; label++) {
      for (unsigned time = 0; time < tilesForAllTimePartitions; time++) {
        auto tile = plan.getTile(batch, time, label);
        auto b = plan.partitionBatch(batchSize, batch);
        graph.setTileMapping(result.slice({b.begin(), time, label},
                                          {b.end(), time + 1, label + 1}),
                             tile);
      }
    }
  }
  auto inReshape = input.reshape({batchSize, 1, 1});
  inReshape = inReshape.broadcast(plan.parallel.time, 1);
  inReshape = inReshape.broadcast(plan.parallel.label, 2);
  prog.add(Copy(inReshape, result));

  return result;
}

// Broadcast the labels tensor and copy into place on tiles.  Each tile's
// vertices require  a window of [previousSymbol, sym0, sym1, ... , nextSymbol]
// Where it actually only processes [sym0, sym1, ...] but relies on the
// value of the previous/next symbol to determine dependencies.
// Each tile used over the time dimension needs a copy of this tensor to avoid
// exchange per compute step.
Tensor createBroadcastTempLabels(Graph &graph, const Tensor &labels,
                                 const popnn::ctc::Plan::Impl &plan,
                                 Sequence &prog,
                                 const poplar::DebugContext &di) {

  const auto batchSize = labels.dim(0);
  const auto maxLabelLength = labels.dim(1);
  // Maximum partition of label + previous and next symbols
  const auto perTileLength = 2 + plan.partitionLabel(maxLabelLength, 0).size();
  const auto tilesForAllTimePartitions = plan.parallel.time;

  auto broadcastLabels =
      graph.addVariable(labels.elementType(),
                        {batchSize, tilesForAllTimePartitions,
                         plan.parallel.label, perTileLength},
                        {di});
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    for (unsigned label = 0; label < plan.parallel.label; label++) {
      for (unsigned time = 0; time < tilesForAllTimePartitions; time++) {
        auto tile = plan.getTile(batch, time, label);
        auto b = plan.partitionBatch(batchSize, batch);
        graph.setTileMapping(broadcastLabels.slice(
                                 {b.begin(), time, label, 0},
                                 {b.end(), time + 1, label + 1, perTileLength}),
                             tile);
      }
    }
  }
  for (unsigned label = 0; label < plan.parallel.label; label++) {
    auto l = plan.partitionLabel(maxLabelLength, label);
    // The broadcast destination slice to copy into.  Note that l.size()+2
    // is not always equal to perTileLength in the case of uneven partitions
    // in the label dimension.
    Slice<4> beginOut = {0, 0, label, 0};
    Slice<4> endOut = {batchSize, tilesForAllTimePartitions, label + 1,
                       l.size() + 2};

    unsigned startOffset = label == 0 ? 0u : 1u;
    unsigned endOffset = (label == plan.parallel.label - 1) ? 0u : 1u;
    Slice<2> beginIn = {0, l.begin() - startOffset};
    Slice<2> endIn = {batchSize, l.end() + endOffset};

    auto previous = labels.slice({0, 0}, {batchSize, 1 - startOffset});
    auto next = labels.slice({0, maxLabelLength - 2 + endOffset},
                             {batchSize, maxLabelLength - 1});
    auto oneSlice =
        concat(concat(previous, labels.slice(beginIn, endIn), 1), next, 1)
            .expand({1, 1});

    prog.add(Copy(oneSlice.broadcast(tilesForAllTimePartitions, 1),
                  broadcastLabels.slice(beginOut, endOut)));
  }
  return broadcastLabels;
}

void initialise(Graph &graph, const Tensor &input, Sequence &prog,
                const poplar::DebugContext &di) {
  auto initialiser = graph.addConstant<float>(input.elementType(), {1},
                                              {log::probabilityZero}, {di});
  graph.setTileMapping(initialiser, 0);
  prog.add(Copy(initialiser.broadcast(input.numElements(), 0), input.flatten(),
                false, {di}));
}

// Initialise a loop counter per tile.  Those below "midPointTile" are
// initialised with lowerT, the rest with higherT
void initialiseCounters(Graph &graph, const Tensor &input,
                        unsigned short lowerT, unsigned short higherT,
                        Sequence &prog, const poplar::DebugContext &di) {

  const auto timePartitions = input.dim(0);
  auto midPointTile = ceildiv(timePartitions, 2u);

  auto initialiserZero = graph.addConstant<unsigned short>(input.elementType(),
                                                           {1}, {lowerT}, {di});
  graph.setTileMapping(initialiserZero, 0);

  auto initialiserMaxCount = graph.addConstant<unsigned short>(
      input.elementType(), {1}, {higherT}, {di});
  graph.setTileMapping(initialiserMaxCount, 0);

  auto firstTimePartitions = input.slice(0, midPointTile, 0).flatten();
  prog.add(Copy(initialiserZero.broadcast(firstTimePartitions.numElements(), 0),
                firstTimePartitions, false, {di}));

  auto lastTimePartitions =
      input.slice(midPointTile, input.dim(0), 0).flatten();
  prog.add(
      Copy(initialiserMaxCount.broadcast(lastTimePartitions.numElements(), 0),
           lastTimePartitions, false, {di}));
}

// Add a copy program to exchange the src tensor to the dst tensor which is
// offset by + or - 1 label partition
void exchangeToNextLabelPartition(Sequence &prog, const Tensor &src,
                                  const Tensor &dst, const Interval &time,
                                  const Interval &label, bool isAlpha,
                                  unsigned batchSize) {
  // Copy to the next label partition: 1 more or 1 less
  const auto dstOffsetBegin = isAlpha ? label.begin() + 1 : label.begin() - 1;
  const auto dstOffsetEnd = isAlpha ? label.end() + 1 : label.end() - 1;

  Slice<4> srcBegin = {label.begin(), 0, time.begin(), 0};
  Slice<4> srcEnd = {label.end(), batchSize, time.end(), src.dim(3)};
  Slice<4> dstBegin = {dstOffsetBegin, 0, time.begin(), 0};
  Slice<4> dstEnd = {dstOffsetEnd, batchSize, time.end(), src.dim(3)};
  prog.add(Copy(src.slice(srcBegin, srcEnd), dst.slice(dstBegin, dstEnd)));
}

// Add a copy program to exchange the src tensor to the dst tensor which is
// offset by + or - 1 label and time partition
void exchangeToNextLabelTimePartition(Sequence &prog, const Tensor &src,
                                      const Tensor &dst, const Interval &time,
                                      const Interval &label, bool isAlpha,
                                      unsigned batchSize) {
  // Copy to the next label and time  partition, both 1 more or 1 less
  const auto dstLabelBegin = isAlpha ? label.begin() + 1 : label.begin() - 1;
  const auto dstLabelEnd = isAlpha ? label.end() + 1 : label.end() - 1;
  const auto dstTimeBegin = isAlpha ? time.begin() + 1 : time.begin() - 1;
  const auto dstTimeEnd = isAlpha ? time.end() + 1 : time.end() - 1;

  Slice<4> srcBegin = {label.begin(), 0, time.begin(), 0};
  Slice<4> srcEnd = {label.end(), batchSize, time.end(), src.dim(3)};
  Slice<4> dstBegin = {dstLabelBegin, 0, dstTimeBegin, 0};
  Slice<4> dstEnd = {dstLabelEnd, batchSize, dstTimeEnd, src.dim(3)};
  prog.add(Copy(src.slice(srcBegin, srcEnd), dst.slice(dstBegin, dstEnd)));
}

// Add copy programs to exchange small temporary alpha/beta result tensors
// to the next time and label partitions
void addLabelExchangeAlphaBeta(Sequence &prog,
                               const popnn::ctc::Plan::Impl &plan,
                               TempTensors &temp, unsigned alphaTimeSteps,
                               unsigned batchSize) {
  if (plan.parallel.label > 1) {
    // Copy alphas to the next (1 more) label partition
    exchangeToNextLabelPartition(prog, temp.labelAlphaOut, temp.labelAlphaIn,
                                 {0, alphaTimeSteps},
                                 {0, plan.parallel.label - 1}, true, batchSize);
    // Copy betas to the next (1 less) label partition
    exchangeToNextLabelPartition(prog, temp.labelBetaOut, temp.labelBetaIn,
                                 {alphaTimeSteps, plan.parallel.time},
                                 {1, plan.parallel.label}, false, batchSize);

    if (plan.parallel.time > 1) {
      // Copy alphas to the next label AND time partition (1 more in each case)
      exchangeToNextLabelTimePartition(
          prog, temp.labelAlphaOut, temp.labelTimeAlphaIn,
          {0, alphaTimeSteps - 1}, {0, plan.parallel.label - 1}, true,
          batchSize);
      // Copy betas to the next label AND time partition (1 less in each case)
      exchangeToNextLabelTimePartition(
          prog, temp.labelBetaOut, temp.labelTimeBetaIn,
          {alphaTimeSteps + 1, plan.parallel.time}, {1, plan.parallel.label},
          false, batchSize);
    }
  }
}

// Add copy programs to exchange small temporary alpha/beta result tensors
// to the next time and label partitions
void addLabelExchangeAlphaBetaGrad(Sequence &prog,
                                   const popnn::ctc::Plan::Impl &plan,
                                   TempTensors &temp, unsigned alphaTimeSteps,
                                   unsigned batchSize) {
  if (plan.parallel.label > 1) {
    // Copy alphas to the next (1 more) label partition
    exchangeToNextLabelPartition(prog, temp.labelAlphaOut, temp.labelAlphaIn,
                                 {alphaTimeSteps, plan.parallel.time},
                                 {0, plan.parallel.label - 1}, true, batchSize);
    // Copy betas to the next (1 less) label partition
    exchangeToNextLabelPartition(prog, temp.labelBetaOut, temp.labelBetaIn,
                                 {0, alphaTimeSteps}, {1, plan.parallel.label},
                                 false, batchSize);

    if (plan.parallel.time > 1) {
      // Copy alphas to the next label AND time partition (1 more in each case)
      exchangeToNextLabelTimePartition(
          prog, temp.labelAlphaOut, temp.labelTimeAlphaIn,
          {alphaTimeSteps - 1, plan.parallel.time - 1},
          {0, plan.parallel.label - 1}, true, batchSize);
      // Copy betas to the next label AND time partition (1 less in each case)
      exchangeToNextLabelTimePartition(
          prog, temp.labelBetaOut, temp.labelTimeBetaIn,
          {1, alphaTimeSteps + 1}, {1, plan.parallel.label}, false, batchSize);
    }
  }
}

// Each vertex is connected to a partition (in time and extended label) of the
// alpha beta array representing the work it needs to do (al0,1,2) or (be3,4,5).
// In this pass we are filling that with alpha in the earlier time partitions or
// beta in the later time partitions.
// Although connected to the whole of the alpha beta array partition on the tile
// a vertex only writes a single timeslice per call, or nothing at all if the
// `count` references a timeslice in another partition.  Which timeslice is
// determined by a `count` variable.
// Picture a single time partition at the position (T=t, El=el) within the
// partitions of time, and extended label.
//
//      time  ------->
//                        Tile (T=t, El=el)
//      plptIn(t-1,el-1)            plIn(el-1)
//                       ---------------------------
// El   ptIn(t-1)        | al0       al1       al2 |
//  |   ptIn(t-1)        | al0       al1       al2 |
//  |   ptIn(t-1)        | al0       al1       al2 |
//  |   ptIn(t-1)        | al0       al1*      al2 |
//  v                    ---------------------------
//                                   plOut
//
// On every loop pass previous alpha results (plptIn, plIn, ptIn) are exchanged
// to this tile.  They aren't always all used by the vertex depending on the
// count.
// ptIn[elSizePerPartition]: (previous time In)
//            The last timeSlice from the previous time partition - partition
//            t-1 (al2 on this tile will be the ptIn of the next tile)
// plIn[1]:   (previous label In)
//            The plOut value from the previous label partition el-1
// plptIn[1]: (previous label and previous time In)
//            The plOut value from the previous label partition el-1
//            and previous time partiton t-1.
// plOut[1]:  (previous label out)
//            The vertex stores the last alpha value in the timeslice it
//            processes into plOut for exchange to other tiles in the next label
//            partition.
//            If processing the column al1 this would be equal to the value al1*
//
// The idea is to always have a complete previous timeslice of alpha including
// `one above` available in order to compute a column of alpha.
// So when the count references the timesteps in this partition this happens:
// Count** Computing `one above`                   previous timeslice
// 0       al0        plptIn == plOut(t-1,el-1)    ptIn
// 1       al1        plIn == plOut(t,el-1)        al0
// 2       al2        plIn == plOut(t,el-1)        al1
//
// -----------------------------------------------------------------------------
// When calculating beta (in the later time partitions) everything is in
// reverse, and the plptIn,plOut,plIn tensors each contain 2 elements per tile.
//
//          Tile (T=t, El=el)
//                      plOut
//          ---------------------------
// El       | be0       be1*      be2 |    ptIn(t+1)
//  |       | be0       be1*      be2 |    ptIn(t+1)
//  |       | be0       be1       be2 |    ptIn(t+1)
//  |       | be0       be1       be2 |    ptIn(t+1)
//  v       ---------------------------
//                      plIn(el+1)         plptIn(t+1,el+1)
//
// We compute be2,be1,be0 in that order as the `count` decrements so:
// Count** Computing `one below`                   previous timeslice
// 2       be2        plptIn == plOut(t+1,el+1)    ptIn
// 1       be1        plIn == plOut(t,el+1)        be2
// 0       be0        plIn == plOut(t,el+1)        be1
//
// ** For partitions computing alpha, the count increments 0...alphaPartitions
//    For partitions computing beta, the count decements maxT...alphaPartitions
//    Where if evenly divided alphaPartitions=maxT/2
//    Above we illustrate `count with in that tile's partition`.

// Make a program to run in the loop which exchanges temporary tensors over
// time and label partitions and makes an alpha or beta vertex on each tile.
Sequence createAlphaBetaProg(Graph &graph, const popnn::ctc::Plan::Impl &plan,
                             const Tensor &data, const Tensor &alphaBeta,
                             const Tensor &loss, TempTensors &temp,
                             unsigned labelsLength, unsigned blankClass,
                             const poplar::DebugNameAndId &dnai) {

  logging::popnn::trace("Creating program to find alpha/beta");
  Sequence prog;
  const auto maxT = data.dim(1);
  const auto batchSize = data.dim(2);

  auto cs = graph.addComputeSet({dnai, "/alphaBetaLoop"});
  // Partitions in the time dimension calculate either alpha or beta.
  // The time partitions that are less than this value calculate alpha.
  // The remaining partitions calculate beta.
  const auto alphaTimePartitions = ceildiv(plan.parallel.time, 2u);

  for (unsigned time = 0; time < plan.parallel.time; time++) {
    const auto vertexCalculatesAlpha = time < alphaTimePartitions;
    const auto timePartition = plan.partitionTime(maxT, time);
    // Copy the last timestep in the alphaBeta tensor from one tile into
    // the temporary time input for the next tile
    // We don't (can't) copy into the first partition from the one before it,
    // or into the last from the one after it. Therefore those partition's
    // timeAlphaBeta1 data remains as initialised before the loop started.
    if (vertexCalculatesAlpha && time != 0) {
      prog.add(Copy(
          alphaBeta.slice(timePartition.begin() - 1, timePartition.begin(), 0),
          temp.timeAlphaBeta1.slice(time, time + 1, 0)));
    } else if (!vertexCalculatesAlpha && time != plan.parallel.time - 1) {
      prog.add(
          Copy(alphaBeta.slice(timePartition.end(), timePartition.end() + 1, 0),
               temp.timeAlphaBeta1.slice(time, time + 1, 0)));
    }
  }
  addLabelExchangeAlphaBeta(prog, plan, temp, alphaTimePartitions, batchSize);

  // generate alpha, beta vertices for a single timestep to be run in a loop
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    const auto batchPartition = plan.partitionBatch(batchSize, batch);
    for (unsigned b = batchPartition.begin(); b < batchPartition.end(); b++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {
        for (unsigned label = 0; label < plan.parallel.label; label++) {
          const auto tile = plan.getTile(batch, time, label);

          auto vertexCalculatesAlpha = time < alphaTimePartitions;
          auto tileVertex =
              vertexCalculatesAlpha ? VertexType::ALPHA : VertexType::BETA;
          const auto labelOffset =
              plan.partitionLabel(labelsLength, 0).size() * label;
          const auto exLabelPartition =
              plan.partitionExtendedLabel(labelsLength, label);
          const auto timePartition = plan.partitionTime(maxT, time);
          const auto processExtraBlank = label == plan.parallel.label - 1;

          generateVertex(graph, data, alphaBeta, loss, boost::none, temp, cs,
                         tile, tileVertex, b, time, label, exLabelPartition,
                         labelOffset, timePartition, processExtraBlank,
                         blankClass);
        }
      }
    }
  }
  prog.add(Execute(cs, {dnai}));
  return prog;
}

// When computing gradGivenAlpha/beta the alphaBeta array is unchanged, an
// extra working data array is used to compute alpha (gradGivenBeta) or beta
// (gradGivenAlpha).  Other than the changes this detail implies this loop
// is the same as the createAlphaBetaProg loop.
// So we calculate grad given beta (calculating alpha in the pingPong working
// data buffer) as follows
//
//      time  ------->
//                        Tile (T=t, El=el)
//      plptIn(t-1,el-1)                                    plIn(el-1)
//                         ---------------------------      -----------
// El   ptPPIn(t-1)        | be0       be1       be2 |     | pp0  pp1 | (Ping
//  |   ptPPIn(t-1)        | be0       be1       be2 |     | pp0  pp1 |  pong)
//  |   ptPPIn(t-1)        | be0       ae1       be2 |     | pp0  pp1 |
//  |   ptPPIn(t-1)        | be0       be1       be2 |     | pp0* pp1*|
//  v                      ---------------------------      -----------
//                                                          plOut
//
// On every loop pass previous alpha results (plptIn,plIn,ptPPIn) are exchanged
// to this tile.  They aren't always all used by the vertex depending on the
// count.
// ptPPIn[elSizePerPartition]: (previous time Ping Pong In)
//            The pingPong buffer from the previous time partition (t-1)
//            (swapping between pp0 and pp1 as input/output continues over tile
//            partitions)
// plIn[1]:   (previous label In)
//            The plOut value from the previous label partition (el-1)
// plptIn[1]: (previous label and previous time In)
//            The plOut value from the previous label partition (el,1) and
//            previous time partiton (t-1).
// plOut[1]:  (previous label out)
//            The vertex stores the last alpha value in the timeslice it
//            processes into plOut for exchange to tiles in the next label
//            partitions. If processing the column pp0 this would be equal to
//            the value pp0*, or if processing pp1, pp1*.
//
// So when the count references the timesteps in this partition this happens:
// Count   Computing          `one above`                 previous timeslice
// 0       grad at be0        plptIn == plOut(t-1,el-1)   ptPPIn pp0/pp1
// 1       grad at be1        plIn == plOut(t,el-1)       PingPong pp0/pp1
// 2       grad at be2        plIn == plOut(t,el-1)       PingPong pp0/pp1
//
// The stored alpha values for any timeslice are always written to the pingPong
// buffer regardless of if the previous timeslice values came from that pingPong
// buffer or the ptPPIn input.

// Make a program to run in the loop which exchanges temporary tensors over
// time and label partitions and makes an gradGivenAlpha or gradGivenBeta
// vertex on each tile
Sequence createAlphaBetaGradProg(Graph &graph,
                                 const popnn::ctc::Plan::Impl &plan,
                                 const Tensor &data, const Tensor &alphaBeta,
                                 const Tensor &loss, Tensor &gradient,
                                 TempTensors &temp, unsigned labelsLength,
                                 unsigned blankClass,
                                 const poplar::DebugNameAndId &dnai) {

  logging::popnn::trace("Creating program to find gradient given alpha/beta");
  Sequence prog;
  const auto maxT = data.dim(1);
  const auto batchSize = data.dim(2);
  const auto extendedLabelSize = temp.timeAlphaBeta2.dim(2);

  auto cs = graph.addComputeSet({dnai, "/gradGivenAlphaBetaLoop"});
  // The partitions that were calculating alpha on the first pass are now
  // calculating beta (running a gradGivenAlpha vertex).  They are the time
  // partitions that are less than this value.  The remaining partitions
  // calculate alpha (running a gradGivenBeta vertex).
  const auto betaTimePartitions = ceildiv(plan.parallel.time, 2u);

  // We don't (can't) copy into the first partition from the one before it,
  // or into the last from the one after it. Therefore those partition's
  // timeAlphaBeta2 data remains as initialised before the loop started.
  if (betaTimePartitions > 0) {
    // Exchange temporary ping-pong buffers toward t=0 for the partitions that
    // are computing beta
    Slice<3> srcBegin = {2, 0, 0};
    Slice<3> srcEnd = {betaTimePartitions * 2, batchSize, extendedLabelSize};
    Slice<3> dstBegin = {0, 0, 0};
    Slice<3> dstEnd = {(betaTimePartitions - 1) * 2, batchSize,
                       extendedLabelSize};
    prog.add(Copy(temp.timeAlphaBeta2.slice(srcBegin, srcEnd),
                  temp.timeAlphaBetaPrevPartition.slice(dstBegin, dstEnd)));
  }
  if (plan.parallel.time - betaTimePartitions > 0) {
    // Exchange temporary ping-pong buffers toward t=maxT for the partitions
    // that are computing alpha
    Slice<3> srcBegin = {betaTimePartitions * 2, 0, 0};
    Slice<3> srcEnd = {(plan.parallel.time - 1) * 2, batchSize,
                       extendedLabelSize};
    Slice<3> dstBegin = {(betaTimePartitions + 1) * 2, 0, 0};
    Slice<3> dstEnd = {plan.parallel.time * 2, batchSize, extendedLabelSize};
    prog.add(Copy(temp.timeAlphaBeta2.slice(srcBegin, srcEnd),
                  temp.timeAlphaBetaPrevPartition.slice(dstBegin, dstEnd)));
  }
  addLabelExchangeAlphaBetaGrad(prog, plan, temp, betaTimePartitions,
                                batchSize);

  // generate gradGivenAlpha, beta vertices for a single timestep to be run
  // in a loop
  for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
    const auto batchPartition = plan.partitionBatch(batchSize, batch);
    for (unsigned b = batchPartition.begin(); b < batchPartition.end(); b++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {
        for (unsigned label = 0; label < plan.parallel.label; label++) {
          const auto tile = plan.getTile(batch, time, label);

          auto vertexCalculatesBeta = time < betaTimePartitions;
          auto tileVertex = vertexCalculatesBeta ? VertexType::GRAD_GIVEN_ALPHA
                                                 : VertexType::GRAD_GIVEN_BETA;
          const auto labelOffset =
              plan.partitionLabel(labelsLength, 0).size() * label;
          const auto exLabelPartition =
              plan.partitionExtendedLabel(labelsLength, label);
          const auto timePartition = plan.partitionTime(maxT, time);
          const auto processExtraBlank = label == plan.parallel.label - 1;

          generateVertex(graph, data, alphaBeta, loss, gradient, temp, cs, tile,
                         tileVertex, b, time, label, exLabelPartition,
                         labelOffset, timePartition, processExtraBlank,
                         blankClass);
        }
      }
    }
  }
  prog.add(Execute(cs, {dnai}));
  return prog;
}

TempTensors createAndInitialiseTemporaryTensors(
    Graph &graph, const popnn::ctc::Plan::Impl &plan, const Tensor &dataLengths,
    const Tensor &labelLengths, const Tensor &labels, const Type &outType,
    Sequence &prog, const poputil::PoplibsOpDebugInfo &di,
    const std::string &layer) {

  const auto batchSize = labels.dim(0);
  const auto labelsLength = labels.dim(1);
  const auto extendedLabelsLength = 2 * labelsLength + 1;

  TempTensors tempTensors;
  // Create and broadcast the label lengths, time lengths and the labels
  // themselves to avoid repeatedly exchanging every compute step
  tempTensors.validLabelLengths = createTempLengths(
      graph, labelLengths, plan, prog, {di, layer + "/broadcastLabelLengths"});
  tempTensors.validTimeSteps = createTempLengths(
      graph, dataLengths, plan, prog, {di, layer + "/broadcastDataLengths"});
  tempTensors.label = createBroadcastTempLabels(
      graph, labels, plan, prog, {di, layer + "/broadcastLabels"});

  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Time partitions"
                        " with Time:2 Batches:{} ExtendedLabelsLength:{}",
                        batchSize, extendedLabelsLength);
  tempTensors.timeAlphaBeta1 = graph.addVariable(
      outType, {plan.parallel.time, batchSize, extendedLabelsLength},
      {di, layer + "/tempTimeAlphaBeta1"});
  mapAlphaBetaAccordingToPlan(graph, tempTensors.timeAlphaBeta1, plan);

  tempTensors.timeAlphaBeta2 = graph.addVariable(
      outType, {2 * plan.parallel.time, batchSize, extendedLabelsLength},
      {di, layer + "/tempTimeAlphaBeta2"});
  mapAlphaBetaAccordingToPlan(graph, tempTensors.timeAlphaBeta2, plan);
  tempTensors.timeAlphaBetaPrevPartition =
      graph.clone(tempTensors.timeAlphaBeta2,
                  {di, layer + "/tempTimeAlphaBetaPrePartition"});

  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Label partitions"
                        " with Partitions:{} Time:{} Batches:{} Labels:2",
                        plan.parallel.label, plan.parallel.time, batchSize);

  tempTensors.labelAlphaIn = graph.addVariable(
      outType, {plan.parallel.label, batchSize, plan.parallel.time, 1},
      {di, layer + "/tempLabelAlphaIn"});
  mapTempLabelAccordingToPlan(graph, tempTensors.labelAlphaIn, plan);
  tempTensors.labelAlphaOut =
      graph.clone(tempTensors.labelAlphaIn, {di, layer + "/tempLabelAlphaOut"});
  tempTensors.labelTimeAlphaIn = graph.clone(
      tempTensors.labelAlphaIn, {di, layer + "/tempLabelTimeAlphaIn"});

  tempTensors.labelBetaIn = graph.addVariable(
      outType, {plan.parallel.label, batchSize, plan.parallel.time, 2},
      {di, layer + "/tempLabelBetaIn"});
  mapTempLabelAccordingToPlan(graph, tempTensors.labelBetaIn, plan);
  tempTensors.labelBetaOut =
      graph.clone(tempTensors.labelBetaIn, {di, layer + "/tempLabelBetaOut"});
  tempTensors.labelTimeBetaIn = graph.clone(
      tempTensors.labelBetaIn, {di, layer + "/tempLabelTimeBetaIn"});

  // Make a counter per tile for the vertices to use
  tempTensors.counter = graph.addVariable(
      UNSIGNED_SHORT, {plan.parallel.time, batchSize, plan.parallel.label},
      {di, "counter"});
  mapAccordingToPlan(graph, tempTensors.counter, plan);

  // Initialise the temporary inputs to the vertices, all to probabilityZero,
  // except for a single "previous alpha" element which equals probabilityOne.
  // The equivalent "previous beta" element is initialised at runtime in the
  // required place based on the time,label size of each individual input.
  initialise(graph, tempTensors.timeAlphaBeta1, prog, {di, layer});
  auto initialiserOne = graph.addConstant<float>(
      outType, {1, 1, 1}, static_cast<float>(log::probabilityOne), {di});
  graph.setTileMapping(initialiserOne, 0);
  auto tempTimeAlphaBeta1Slice =
      tempTensors.timeAlphaBeta1.slice({0, 0, 0}, {1, batchSize, 1});
  prog.add(Copy(initialiserOne.broadcast(batchSize, 1), tempTimeAlphaBeta1Slice,
                false, {di}));

  initialise(graph, tempTensors.timeAlphaBeta2, prog, {di, layer});
  initialise(graph, tempTensors.labelAlphaIn, prog, {di, layer});
  initialise(graph, tempTensors.labelBetaIn, prog, {di, layer});
  initialise(graph, tempTensors.labelAlphaOut, prog, {di, layer});
  initialise(graph, tempTensors.labelBetaOut, prog, {di, layer});
  initialise(graph, tempTensors.labelTimeAlphaIn, prog, {di, layer});
  initialise(graph, tempTensors.labelTimeBetaIn, prog, {di, layer});
  return tempTensors;
}
} // namespace
namespace popnn {
namespace ctc {

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
  mapDataInputAccordingToPlan(graph, data, plan.getImpl());
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
  mapLabelsAccordingToPlan(graph, labels, plan.getImpl());
  di.addOutput(labels);
  return labels;
}

void validateTensorTypes(const poplar::Tensor &data,
                         const poplar::Tensor &labels,
                         const poplar::Tensor &dataLengths,
                         const poplar::Tensor &labelLengths,
                         const poplar::Type &outType) {
  if (data.elementType() != poplar::HALF &&
      data.elementType() != poplar::FLOAT) {
    throw poputil::poplibs_error("data tensor must be of type HALF or FLOAT");
  }
  if (labels.elementType() != poplar::UNSIGNED_INT) {
    throw poputil::poplibs_error("labels tensor must be of type UNSIGNED_INT");
  }
  if (dataLengths.elementType() != poplar::UNSIGNED_INT) {
    throw poputil::poplibs_error(
        "dataLengths tensor must be of type UNSIGNED_INT");
  }
  if (labelLengths.elementType() != poplar::UNSIGNED_INT) {
    throw poputil::poplibs_error(
        "labelLengths tensor must be of type UNSIGNED_INT");
  }
  if (outType == poplar::HALF && data.elementType() == poplar::FLOAT) {
    throw poputil::poplibs_error(
        "outType HALF unsupported with input tensor type FLOAT");
  }
}

std::pair<poplar::Tensor, poplar::Tensor> calcLossAndGradientLogProbabilities(
    poplar::Graph &graph, const poplar::Type &outType,
    const poplar::Tensor &data, const poplar::Tensor &labels,
    const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
    poplar::program::Sequence &prog, const unsigned blankClass,
    const Plan &plan_, const poplar::DebugContext &debugContext) {
  validateTensorTypes(data, labels, dataLengths, labelLengths, outType);

  const auto plan = plan_.getImpl();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(outType, data, labels, dataLengths,
                                         labelLengths, blankClass, plan_));
  const std::string layer = "CTCGradient";

  logging::popnn::debug("Disabled NANOO for CTC Loss operation");
  poplar::FloatingPointBehaviour clear{false, false, false, false,
                                       true}; // Mask out nanoo
  poplar::FloatingPointBehaviour set{false, false, false, false, false};
  auto fpCSRToRestore =
      poplar::getAndModifyFloatingPointBehaviour(graph, prog, clear, set, {di});

  logging::popnn::debug("Creating CTCLoss using {}", plan_);
  const auto maxT = data.dim(0);
  const auto batchSize = data.dim(1);
  const auto numClasses = data.dim(2);
  const auto labelsLength = labels.dim(1);
  const auto extendedLabelsLength = 2 * labelsLength + 1;

  // A gradient tensor - either the final result, or a result per labels
  // partition which will need later reduction
  auto workingGradShape = data.shape();
  workingGradShape.insert(workingGradShape.begin(), plan.parallel.label);
  if (plan.parallel.label > 1) {
    logging::popnn::debug("Creating per label partition gradient result tensor"
                          " with Partitions:{} Time:{} Batches:{} Classes:{}",
                          plan.parallel.label, maxT, batchSize, numClasses);
  } else {
    logging::popnn::debug("Creating gradient tensor for CTC Loss with Time:{}"
                          " Batches:{} Classes:{}",
                          maxT, batchSize, numClasses);
  }

  auto gradient =
      graph.addVariable(outType, workingGradShape, {di, layer + "/gradient"});
  mapGradientAccordingToPlan(graph, gradient, plan);

  // Broadcast the data input and map according to the planned label splits
  // which require a copy of the data and the gradient while computing
  const auto workingData = [&]() {
    if (plan.parallel.label != 1) {
      auto result = graph.addVariable(data.elementType(), workingGradShape,
                                      {di, layer + "/broadcastInput"});
      mapGradientAccordingToPlan(graph, result, plan);
      auto broadcastData = data.expand({0}).broadcast(plan.parallel.label, 0);
      prog.add(Copy(broadcastData, result, false, {di}));
      return result;
    } else {
      // No broadcast/copy to do, so just add a dimension
      return data.expand({0});
    }
  }();
  logging::popnn::debug("Creating alpha/beta tensor for CTC Loss with Time:{}"
                        " Batches:{} ExtendedLabelLength:{}",
                        maxT, batchSize, extendedLabelsLength);
  auto alphaBeta =
      graph.addVariable(outType, {maxT, batchSize, extendedLabelsLength},
                        {di, layer + "/alphaBeta"});
  mapAlphaBetaAccordingToPlan(graph, alphaBeta, plan);

  // Make the temporary tensors to exchange between tiles, keep loop count etc
  auto tempTensors = createAndInitialiseTemporaryTensors(
      graph, plan, dataLengths, labelLengths, labels, outType, prog, di, layer);

  auto loss = graph.addVariable(
      outType, {batchSize, plan.parallel.time, plan.parallel.label},
      {di, layer + "/loss"});
  mapLossAccordingToPlan(graph, loss, plan);
  initialise(graph, loss, prog, {di, layer});

  // Initialise the gradient to probabilityZero, to accumulate into
  initialise(graph, gradient, prog, di);

  // Make the program to find alpha, beta and run it in a loop
  const auto alphaTimePartitions = ceildiv(plan.parallel.time, 2u);
  auto lastAlphaTimePartition =
      plan.partitionTime(maxT, alphaTimePartitions - 1);
  const auto numLoops = lastAlphaTimePartition.end();
  initialiseCounters(graph, tempTensors.counter, 0, maxT, prog, {di, layer});
  auto alphaBetaProg =
      createAlphaBetaProg(graph, plan, workingData, alphaBeta, loss,
                          tempTensors, labelsLength, blankClass, {di, layer});
  prog.add(Repeat(numLoops, alphaBetaProg, {di}));

  // Copy the last timestep in the alphaBeta tensor from one tile into
  // the temporary time input for the next tile
  if (plan.parallel.time > 1) {
    logging::popnn::trace("Creating copy program to interface alpha/beta "
                          " loop to gradGivenAlpha/Beta loop");

    // Copy the last computed alpha, broadcast into the 1st temporary alpha
    // input
    Slice<3> dstBegin = {alphaTimePartitions * 2, 0, 0};
    Slice<3> dstEnd = {(alphaTimePartitions + 1) * 2, batchSize,
                       extendedLabelsLength};
    prog.add(
        Copy(alphaBeta
                 .slice(lastAlphaTimePartition.end() - 1,
                        lastAlphaTimePartition.end(), 0)
                 .broadcast(2, 0),
             tempTensors.timeAlphaBetaPrevPartition.slice(dstBegin, dstEnd)));

    // Copy the last computed beta, broadcast into the 1st temporary beta
    // input
    dstBegin = {(alphaTimePartitions - 1) * 2, 0, 0};
    dstEnd = {(alphaTimePartitions - 1) * 2 + 2, batchSize,
              extendedLabelsLength};
    prog.add(
        Copy(alphaBeta
                 .slice(lastAlphaTimePartition.end(),
                        lastAlphaTimePartition.end() + 1, 0)
                 .broadcast(2, 0),
             tempTensors.timeAlphaBetaPrevPartition.slice(dstBegin, dstEnd)));
  }

  // Make the program to find gradient given alpha, beta and run it in a loop
  auto gradInitialCount = lastAlphaTimePartition.end();
  initialiseCounters(graph, tempTensors.counter, gradInitialCount,
                     gradInitialCount, prog, {di, layer});

  auto alphaBetaGradProg = createAlphaBetaGradProg(
      graph, plan, workingData, alphaBeta, loss, gradient, tempTensors,
      labelsLength, blankClass, {di, layer});
  prog.add(Repeat(numLoops, alphaBetaGradProg, {di}));

  di.addOutput(gradient);

  // Reduce where data was split over label.
  ReduceParams reduceParams = {popops::Operation::LOG_ADD, false};

  auto lossReduced = [&]() {
    loss = loss.flatten(1, 3);
    if (loss.dim(1) == 1) {
      return loss.reshape({batchSize});
    } else {
      return popops::reduce(graph, loss, {1}, reduceParams, prog, {di});
    }
  }();
  auto gradReduced = [&]() {
    if (gradient.dim(0) == 1) {
      return gradient.reshape(data.shape());
    } else {

      // Ensure we preserve mapping of the result to fit in with the plan
      auto gradientReduced = graph.clone(outType, data, debugContext);
      popops::reduceWithOutput(graph, gradient, gradientReduced, {0},
                               reduceParams, prog, {di});
      return gradientReduced;
    }
  }();

  popops::negInPlace(graph, lossReduced, prog, {di});

  di.addOutput(lossReduced);
  di.addOutput(gradReduced);

  poplar::setFloatingPointBehaviour(graph, prog, fpCSRToRestore, {di});
  return {lossReduced, gradReduced};
}

std::pair<poplar::Tensor, poplar::Tensor> calcLossAndGradientLogits(
    poplar::Graph &graph, const poplar::Type &outType,
    const poplar::Tensor &logits, const poplar::Tensor &labels,
    const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
    poplar::program::Sequence &prog, const unsigned blankClass,
    const Plan &plan_, const poplar::DebugContext &debugContext) {
  // TODO sort out debug info

  // Ensure we preserve mapping of the result to fit in with the plan
  auto logProbs = graph.clone(logits, debugContext);
  prog.add(Copy(logits, logProbs));
  logSoftmaxInPlace(graph, logProbs, prog, debugContext);

  return calcLossAndGradientLogProbabilities(graph, outType, logProbs, labels,
                                             dataLengths, labelLengths, prog,
                                             blankClass, plan_, debugContext);
}

} // end namespace ctc

} // end namespace popnn
