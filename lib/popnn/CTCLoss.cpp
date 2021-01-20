// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include "poplibs_support/logging.hpp"
#include <poplar/Graph.hpp>
#include <poplibs_test/CTCLoss.hpp>
#include <popnn/CTCLoss.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poplibs_test;
using namespace popops;
using namespace popops::expr;
using namespace poputil;

template <unsigned size> using Slice = std::array<std::size_t, size>;
namespace {

enum class VertexType { ALPHA, BETA, GRAD_GIVEN_ALPHA, GRAD_GIVEN_BETA };

void generateVertex(Graph &graph, const Tensor &data, const Tensor &labels,
                    const Tensor &validLabels,
                    const Tensor &tempTimeAlphaOrBeta,
                    const Tensor &tempLabelAlphaOrBeta,
                    const Tensor &alphaOrBeta, boost::optional<Tensor &> grad,
                    ComputeSet &cs, unsigned tile, VertexType vertexType,
                    unsigned batch, const Interval &timePartition,
                    unsigned label, const Interval &labelPartition,
                    const Interval &exLabelPartition, unsigned labelOffset,
                    bool processExtraBlank, unsigned blankClass) {

  const auto numClasses = data.dim(2);
  Slice<2> beginLabels = {batch, labelPartition.begin()};
  Slice<2> endLabels = {batch + 1, labelPartition.end()};
  auto tileLabels = labels.slice(beginLabels, endLabels);
  auto tileValidLabels = validLabels.slice(batch, batch + 1);

  auto isAlpha = vertexType == VertexType::ALPHA ||
                 vertexType == VertexType::GRAD_GIVEN_BETA;
  Tensor prevSymbol;
  if (isAlpha) {
    if (label == 0) {
      prevSymbol = tileLabels.flatten()[0];
    } else {
      Slice<2> beginLabels = {batch, labelPartition.begin() - 1};
      Slice<2> endLabels = {batch + 1, labelPartition.begin()};
      prevSymbol = labels.slice(beginLabels, endLabels);
    }
  } else {
    if (processExtraBlank) {
      prevSymbol = tileLabels.flatten()[tileLabels.numElements() - 1];
    } else {
      Slice<2> beginLabels = {batch, labelPartition.end()};
      Slice<2> endLabels = {batch + 1, labelPartition.end() + 1};
      prevSymbol = labels.slice(beginLabels, endLabels);
    }
  }

  Slice<3> beginData = {timePartition.begin(), batch, 0};
  Slice<3> endData = {timePartition.end(), batch + 1, numClasses};
  auto tileData = data.slice(beginData, endData);

  Slice<3> beginAlphaBeta = {timePartition.begin(), batch,
                             exLabelPartition.begin()};
  Slice<3> endAlphaBeta = {timePartition.end(), batch + 1,
                           exLabelPartition.end()};
  auto tileAlphaOrBeta = alphaOrBeta.slice(beginAlphaBeta, endAlphaBeta);

  Slice<4> beginGrad = {label, timePartition.begin(), batch, 0};
  Slice<4> endGrad = {label + 1, timePartition.end(), batch + 1, numClasses};

  const auto inType = data.elementType();
  const auto outType = alphaOrBeta.elementType();
  const auto labelType = labels.elementType();
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
  logging::popnn::trace("Making {} vertex on tile {}", vertexName, tile);
  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, tile);

  graph.setInitialValue(v["maxT"], tileData.shape()[0]);
  graph.setInitialValue(v["numClasses"], tileData.shape()[2]);
  graph.setInitialValue(v["blankClass"], blankClass);
  graph.setInitialValue(v["labelOffset"], labelOffset);

  graph.connect(v["probabilities"], tileData.flatten());
  graph.connect(v["labels"], tileLabels.flatten());
  graph.connect(v["validLabels"], tileValidLabels.reshape({}));

  if (vertexType == VertexType::ALPHA) {
    graph.connect(v["alphas"], tileAlphaOrBeta.flatten());
    graph.connect(v["alphaPrevTime"], tempTimeAlphaOrBeta.flatten());
    graph.connect(v["alphaPrevLabel"], tempLabelAlphaOrBeta.flatten());
    graph.connect(v["prevSymbol"], prevSymbol.reshape({}));
  } else if (vertexType == VertexType::BETA) {
    graph.connect(v["betas"], tileAlphaOrBeta.flatten());
    graph.connect(v["betaPrevTime"], tempTimeAlphaOrBeta.flatten());
    graph.connect(v["betaPrevLabel"], tempLabelAlphaOrBeta.flatten());
    graph.connect(v["prevSymbol"], prevSymbol.reshape({}));
  } else if (vertexType == VertexType::GRAD_GIVEN_ALPHA) {
    graph.connect(v["grads"], grad.get().slice(beginGrad, endGrad).flatten());
    graph.connect(v["betaPrevTime"], tempTimeAlphaOrBeta.flatten());
    graph.connect(v["betaPrevLabel"], tempLabelAlphaOrBeta.flatten());
    graph.connect(v["alphas"], tileAlphaOrBeta.flatten());
    graph.connect(v["prevSymbol"], prevSymbol.reshape({}));
  } else if (vertexType == VertexType::GRAD_GIVEN_BETA) {
    graph.connect(v["grads"], grad.get().slice(beginGrad, endGrad).flatten());
    graph.connect(v["alphaPrevTime"], tempTimeAlphaOrBeta.flatten());
    graph.connect(v["alphaPrevLabel"], tempLabelAlphaOrBeta.flatten());
    graph.connect(v["betas"], tileAlphaOrBeta.flatten());
    graph.connect(v["prevSymbol"], prevSymbol.reshape({}));
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
enum class VertexInitialiser { CONSTANT, PREVIOUS_RESULT, PREVIOUS_TEMP };

std::ostream &operator<<(std::ostream &o, const VertexInitialiser v) {
  if (v == VertexInitialiser::CONSTANT) {
    o << "CONSTANT";
  } else if (v == VertexInitialiser::PREVIOUS_RESULT) {
    o << "PREVIOUS_RESULT";
  } else if (v == VertexInitialiser::PREVIOUS_TEMP) {
    o << "PREVIOUS_TEMPORARY_RESULT";
  }
  return o;
}
// Struct to track and check which alpha, beta partitions have already been
// calculated and therefore what can be calculated next.
struct CompletionFlags {
  unsigned timeSplits;
  unsigned labelSplits;
  std::vector<bool> alphaImpl;
  std::vector<bool> betaImpl;

  CompletionFlags(unsigned label, unsigned time) {
    timeSplits = time;
    labelSplits = label;
    alphaImpl.resize(timeSplits * labelSplits, false);
    betaImpl.resize(timeSplits * labelSplits, false);
  }
  // Wrapper to access alphaImpl, betaImpl a bit like a 2D array.
  bool alpha(unsigned label, unsigned time) const {
    return alphaImpl[timeSplits * label + time];
  }
  bool beta(unsigned label, unsigned time) const {
    return betaImpl[timeSplits * label + time];
  }

  void set(unsigned label, unsigned time, VertexType vertex) {
    if (vertex == VertexType::ALPHA || vertex == VertexType::GRAD_GIVEN_BETA) {
      alphaImpl[timeSplits * label + time] = true;
    } else {
      betaImpl[timeSplits * label + time] = true;
    }
  }

  // Return the type of vertex that can be run on the partition (tile) specified
  boost::optional<VertexType> vertexToRun(unsigned label, unsigned time) const {

    if (alpha(label, time) && beta(label, time)) {
      // Nothing left to do
      return boost::none;
    }
    bool prevTimeBeta = time == timeSplits - 1 || beta(label, time + 1);
    bool prevLabelBeta = label == labelSplits - 1 || beta(label + 1, time);
    if (!beta(label, time) && prevTimeBeta && prevLabelBeta) {
      // We need to find beta, can find beta, we can find grad if we have alpha
      return alpha(label, time) ? VertexType::GRAD_GIVEN_ALPHA
                                : VertexType::BETA;
    }
    bool prevTimeAlpha = time == 0 || alpha(label, time - 1);
    bool prevLabelAlpha = label == 0 || alpha(label - 1, time);
    if (!alpha(label, time) && prevTimeAlpha && prevLabelAlpha) {
      // We need to find alpha, can find alpha, we can find grad if we have beta
      return beta(label, time) ? VertexType::GRAD_GIVEN_BETA
                               : VertexType::ALPHA;
    }
    return boost::none;
  }

  // Return the type of result to attach to the vertex.
  // split - the dimension which we are splitting and considering the previous
  //         result.
  // otherSplit - The other dimension, used to access dependencies.
  VertexInitialiser previousResult(unsigned split, unsigned otherSplit,
                                   bool isTime, VertexType vertex) const {
    unsigned numSplits = isTime ? timeSplits : labelSplits;
    if (numSplits == 1) {
      // No split, no tile to get a previous result from
      return VertexInitialiser::CONSTANT;
    }
    if (split == 0 && (vertex == VertexType::ALPHA ||
                       vertex == VertexType::GRAD_GIVEN_BETA)) {
      // First split, finding alpha - start up with a constant
      return VertexInitialiser::CONSTANT;
    }
    if (split == numSplits - 1 && (vertex == VertexType::BETA ||
                                   vertex == VertexType::GRAD_GIVEN_ALPHA)) {
      // Last split, finding beta - start up with a constant
      return VertexInitialiser::CONSTANT;
    }
    if (split > 0 && vertex == VertexType::GRAD_GIVEN_BETA) {
      bool prevAlpha =
          isTime ? alpha(otherSplit, split - 1) : alpha(split - 1, otherSplit);
      bool prevBeta =
          isTime ? beta(otherSplit, split - 1) : beta(split - 1, otherSplit);
      if (prevAlpha && prevBeta) {
        return VertexInitialiser::PREVIOUS_TEMP;
      }
    }
    if (split < numSplits - 1 && vertex == VertexType::GRAD_GIVEN_ALPHA) {
      bool prevAlpha =
          isTime ? alpha(otherSplit, split + 1) : alpha(split + 1, otherSplit);
      bool prevBeta =
          isTime ? beta(otherSplit, split + 1) : beta(split + 1, otherSplit);
      if (prevAlpha && prevBeta) {
        return VertexInitialiser::PREVIOUS_TEMP;
      }
    }
    return VertexInitialiser::PREVIOUS_RESULT;
  }
  // Wrapper for the above function where we want to find the initialiser for
  // the label dimension for the awkward single input on a diagonal
  VertexInitialiser previousResult(unsigned timeSplit, unsigned labelSplit,
                                   VertexType vertex) const {

    if (labelSplit > 0 && timeSplit > 1 &&
        vertex == VertexType::GRAD_GIVEN_BETA) {
      return previousResult(timeSplit - 2, labelSplit - 1, true, vertex);
    }
    if (labelSplit < labelSplits - 1 && timeSplit < timeSplits - 2 &&
        vertex == VertexType::GRAD_GIVEN_ALPHA) {
      return previousResult(timeSplit + 2, labelSplit + 1, true, vertex);
    }
    return VertexInitialiser::CONSTANT;
  }
};

// TODO - Hopefully we can reference a temporary tensor that is written into
//        by all the vertices.  Need an alpha, beta  version. That should
//        simplify this function a lot.  So leave untidy for now
Tensor tempLabelInputToVertex(Sequence &prog, VertexInitialiser initialiser,
                              VertexInitialiser timeAndLabelInitialiser,
                              VertexType tileVertex, unsigned batch,
                              unsigned label, const Interval &time,
                              const Interval &exLabel, bool lastTimePartition,
                              const Tensor &initialZeros,
                              const Tensor &alphaBeta,
                              const Tensor &tempAlphaBeta,
                              const poplar::DebugContext &di) {
  bool isAlpha = (tileVertex == VertexType::ALPHA ||
                  tileVertex == VertexType::GRAD_GIVEN_BETA);
  bool isGrad = (tileVertex == VertexType::GRAD_GIVEN_ALPHA ||
                 tileVertex == VertexType::GRAD_GIVEN_BETA);
  Tensor temp;

  // The piece of the temporary input tensor to attach to this vertex
  // Tensor shape: {plan.parallel.label, maxT, batchSize, 2}
  // Connect 2 rows if beta is being calculated as we need to propogate more
  // data due to the dependency on blank and symbols in the trellis.
  Slice<4> begin = {label, time.begin(), batch, 0};
  Slice<4> end = {label + 1, time.end(), batch + 1, isAlpha ? 1ul : 2ul};

  if (initialiser == VertexInitialiser::CONSTANT) {
    // TODO - this could be the right size to start with, smaller than presently
    // at least. Like the size of a tile's split?
    auto zeroSlice = initialZeros.slice(time, 0);
    auto zeroInit = isAlpha ? zeroSlice : concat(zeroSlice, zeroSlice);
    if (isGrad) {
      temp = tempAlphaBeta.slice(begin, end).flatten();
      prog.add(Copy(zeroInit, temp, false, {di}));
    } else {
      temp = zeroInit;
    }
  } else {
    temp = tempAlphaBeta.slice(begin, end).flatten();
    // Shape: [maxT,batch,ExtendedLabels]
    if (initialiser == VertexInitialiser::PREVIOUS_RESULT) {
      // Initialiser from the previous tile's alpha or beta
      if (isAlpha) {
        if (time.begin() == 0) {
          Slice<3> begin = {time.begin(), batch, exLabel.begin() - 1};
          Slice<3> end = {time.end() - 1, batch + 1, exLabel.begin()};
          prog.add(Copy(concat(initialZeros[0].flatten(),
                               alphaBeta.slice(begin, end).flatten()),
                        temp, false, {di}));
        } else {
          Slice<3> begin = {time.begin() - 1, batch, exLabel.begin() - 1};
          Slice<3> end = {time.end() - 1, batch + 1, exLabel.begin()};
          prog.add(
              Copy(alphaBeta.slice(begin, end).flatten(), temp, false, {di}));
        }
      } else {
        if (lastTimePartition) {

          Slice<3> begin = {time.begin() + 1, batch, exLabel.end()};
          Slice<3> end = {time.end(), batch + 1, exLabel.end() + 1};
          auto first = concat(alphaBeta.slice(begin, end).flatten(),
                              initialZeros[0].reshape({1}));
          Slice<3> begin2 = {time.begin() + 1, batch, exLabel.end() + 1};
          Slice<3> end2 = {time.end(), batch + 1, exLabel.end() + 2};
          auto second = concat(alphaBeta.slice(begin2, end2).flatten(),
                               initialZeros[0].reshape({1}));
          prog.add(Copy(concat(first, second), temp, false, {di}));
        } else {
          Slice<3> begin = {time.begin() + 1, batch, exLabel.end()};
          Slice<3> end = {time.end() + 1, batch + 1, exLabel.end() + 1};
          auto first = alphaBeta.slice(begin, end).flatten();
          Slice<3> begin2 = {time.begin() + 1, batch, exLabel.end() + 1};
          Slice<3> end2 = {time.end() + 1, batch + 1, exLabel.end() + 2};
          auto second = alphaBeta.slice(begin2, end2).flatten();
          prog.add(Copy(concat(first, second), temp, false, {di}));
        }
      }
    } else {
      // Initialiser from the previous tile's tempAlpha or Beta.
      auto diagonalWasTemp =
          timeAndLabelInitialiser == VertexInitialiser::PREVIOUS_TEMP;
      if (isAlpha) {
        // Copy the previous label partition's alpha output as the input to this
        // stage
        if (time.begin() == 0) {
          // Alpha - 1st slice so start with zero, then previous tile's result
          Slice<4> begin = {label - 1, time.begin(), batch, 0};
          Slice<4> end = {label, time.end() - 1, batch + 1, 1};
          prog.add(Copy(concat(initialZeros[0].flatten(),
                               tempAlphaBeta.slice(begin, end).flatten()),
                        temp, false, {di}));
        } else {
          // Alpha other time steps - previous tile's result including 1 element
          // from the previous timestep
          Slice<4> begin = {label - 1, time.begin(), batch, 0};
          Slice<4> end = {label, time.end() - 1, batch + 1, 1};
          // Deal with the case where the diagonally fetched input was
          // potentially a temporary one
          auto prevTimeElemTemp =
              tempAlphaBeta[label - 1][time.begin() - 1][batch][0].flatten();

          auto prevTimeElem =
              alphaBeta[time.begin() - 1][batch][exLabel.begin() - 1].flatten();
          auto first = diagonalWasTemp ? prevTimeElemTemp : prevTimeElem;
          prog.add(
              Copy(concat(first, tempAlphaBeta.slice(begin, end).flatten()),
                   temp, false, {di}));
        }
      } else {
        //! isAlpha
        if (lastTimePartition) {
          // Finding beta - last time step so start with zero, and previous
          // tile's result
          Slice<4> begin = {label + 1, time.begin(), batch, 0};
          Slice<4> end = {label + 2, time.end(), batch + 1, 2};
          auto thisTime =
              tempAlphaBeta.slice(begin, end).reshape({2, time.size()});
          thisTime = thisTime.slice(1, time.size(), 1);
          thisTime =
              concat(thisTime, initialZeros.slice(0, 2, 0).reshape({2, 1}), 1);
          prog.add(Copy(thisTime.flatten(), temp, false, {di}));

        } else {
          // Finding beta - other time steps so start with previous
          // tile's result. Including 2 elements from the previous timestep
          Slice<4> begin = {label + 1, time.begin(), batch, 0};
          Slice<4> end = {label + 2, time.end(), batch + 1, 2};
          auto thisTime =
              tempAlphaBeta.slice(begin, end).reshape({2, time.size()});
          thisTime = thisTime.slice(1, time.size(), 1);

          // Deal with the case where the diagonally fetched input was
          // potentially a temporary one
          auto prevTimeElem0 =
              alphaBeta[time.end()][batch][exLabel.end()].flatten();
          auto prevTimeElem1 =
              alphaBeta[time.end()][batch][exLabel.end() + 1].flatten();

          auto prevTimeElemTemp0 =
              tempAlphaBeta[label + 1][time.end()][batch][0].flatten();
          auto prevTimeElemTemp1 =
              tempAlphaBeta[label + 1][time.end()][batch][1].flatten();

          auto prevTime = concat(prevTimeElem0, prevTimeElem1).reshape({2, 1});
          auto prevTimeTemp =
              concat(prevTimeElemTemp0, prevTimeElemTemp1).reshape({2, 1});

          thisTime =
              concat(thisTime, diagonalWasTemp ? prevTimeTemp : prevTime, 1);
          prog.add(Copy(thisTime.flatten(), temp, false, {di}));
        }
      }
    }
  }
  return temp;
}

// Connect the required temporary time input to an alpha or beta vertex.
// As it is a non-gradient vertex it never changes the data content so we can
// just reference the input, there is no need to copy.
// The input can be a constant (Where alpha is the 1st timeSlice/beta the last)
// Or the input can come from a previous tile's result stored in the `alphaBeta`
// input. This can apply to alpha or beta.
// The input has no need to ever come from a temporary result. Temporary results
// come from vertices that calculate `gradGivenX`.  Depndencies mean that we
// will never call a `nonGradVertex` that relies on the temp data from a
// `gradVertex`
Tensor tempTimeInputToNonGradVertex(VertexInitialiser initialiser,
                                    VertexType tileVertex, unsigned batch,
                                    const Interval &timePartition,
                                    const Interval &exLabelPartition,
                                    const Tensor &initialAlpha,
                                    const Tensor &initialBeta,
                                    const Tensor &alphaBeta) {
  Tensor temp;
  if (initialiser == VertexInitialiser::CONSTANT) {
    temp = (tileVertex == VertexType::ALPHA)
               ? initialAlpha.slice(exLabelPartition)
               : initialBeta.slice(exLabelPartition);
  }
  if (initialiser == VertexInitialiser::PREVIOUS_RESULT) {
    // Initialiser from the previous tile's alpha or beta
    if (tileVertex == VertexType::ALPHA) {
      Slice<3> begin = {timePartition.begin() - 1, batch,
                        exLabelPartition.begin()};
      Slice<3> end = {timePartition.begin(), batch + 1, exLabelPartition.end()};
      temp = alphaBeta.slice(begin, end);
    } else {
      Slice<3> begin = {timePartition.end(), batch, exLabelPartition.begin()};
      Slice<3> end = {timePartition.end() + 1, batch + 1,
                      exLabelPartition.end()};
      temp = alphaBeta.slice(begin, end);
    }
  }
  return temp;
}

// Connect the required temporary time input to a gradGivenAlpha or
// gradGivenBeta vertex.
// The vertex will overwrite the data, so a copy is required.
// The vertex can use a constant input (Where finding alpha in the first
// timeslice or beta in the last timeslice)
// Otherwise if the tile that this tile's calculation depends on stored its
// data into the `alphaBeta` data we use that.  If it wrote into a temporary
// array it we use that as the input.
// For example - split in time by 4:
//      time -------------->
//      Tile 0 Tile 1 Tile 2  Tile 3
// Data:abcd   efgh   ijkl    mnop
// ComputeSet
//  0   alpha   -      -      beta   ;Not a gradVertex- not this fn but CONST
//  1     -    alpha  beta     -     ;Not a gradVertex- not this fn but PREV_RES
//  2     -  gradGivA gradGivB -     ;Select PREVIOUS_RESULT alpha or beta
//  3 gradGivA  -      -    gradGivB ;Select PREVIOUS_TEMP alpha or beta
//
// CS 2 - PREVIOUS_RESULT as the inptu required was stored in alphaBeta
// CS 3 - PREVIOUS_TEMP as the input required was stored in alphaBetaTemp
//        because the vertex that created it was a gradVertex.
Tensor tempTimeInputToGradVertex(
    Sequence &prog, VertexInitialiser initialiser, VertexType tileVertex,
    unsigned batch, unsigned time, const Interval &timePartition,
    const Interval &exLabelPartition, const Tensor &initialAlpha,
    const Tensor &initialBeta, const Tensor &alphaBeta,
    const Tensor &tempAlphaBeta, const popnn::ctc::Plan::Impl &plan,
    unsigned maxT, const poplar::DebugContext &di) {

  // The piece of the temporary input tensor to attach to this vertex
  Slice<3> beginT = {2 * time, batch, exLabelPartition.begin()};
  Slice<3> endT = {2 * (time + 1), batch + 1, exLabelPartition.end()};
  auto temp = tempAlphaBeta.slice(beginT, endT);

  // It is read-write so copy the required data into it
  if (initialiser == VertexInitialiser::PREVIOUS_RESULT) {
    // Initialiser from the previous tile's alpha or beta.
    // Take the last timeslice (alpha) or 1st (beta)
    if (tileVertex == VertexType::GRAD_GIVEN_ALPHA) {
      Slice<3> begin = {timePartition.end(), batch, exLabelPartition.begin()};
      Slice<3> end = {timePartition.end() + 1, batch + 1,
                      exLabelPartition.end()};
      prog.add(
          Copy(alphaBeta.slice(begin, end), temp.slice(0, 1, 0), false, {di}));
    } else {
      Slice<3> begin = {timePartition.begin() - 1, batch,
                        exLabelPartition.begin()};
      Slice<3> end = {timePartition.begin(), batch + 1, exLabelPartition.end()};
      prog.add(
          Copy(alphaBeta.slice(begin, end), temp.slice(0, 1, 0), false, {di}));
    }
  } else if (initialiser == VertexInitialiser::PREVIOUS_TEMP) {
    // Initialiser from the previous tile's temporary result.
    // Has shape[2][extendedLabels].  We may need either
    // shape[0][] or shape[1][] depending on the number of timesteps
    // the previous stage made (even steps: 0, odd steps 1), as the
    // vertex will have alternate buffer half each timestep
    if (tileVertex == VertexType::GRAD_GIVEN_ALPHA) {
      auto offset = plan.partitionTime(maxT, time + 1).size() % 2;
      Slice<3> begin = {2 * time + 2 + offset, batch, exLabelPartition.begin()};
      Slice<3> end = {2 * time + 3 + offset, batch + 1, exLabelPartition.end()};
      prog.add(Copy(tempAlphaBeta.slice(begin, end), temp.slice(0, 1, 0), false,
                    {di}));
    } else {
      auto offset = plan.partitionTime(maxT, time - 1).size() % 2;
      Slice<3> begin = {2 * time - 2 + offset, batch, exLabelPartition.begin()};
      Slice<3> end = {2 * time - 1 + offset, batch + 1, exLabelPartition.end()};
      prog.add(Copy(tempAlphaBeta.slice(begin, end), temp.slice(0, 1, 0), false,
                    {di}));
    }
  } else {
    // First time initialiser
    if (tileVertex == VertexType::GRAD_GIVEN_ALPHA) {
      prog.add(Copy(initialBeta.slice(exLabelPartition), temp.slice(0, 1, 0),
                    false, {di}));
    } else {
      prog.add(Copy(initialAlpha.slice(exLabelPartition), temp.slice(0, 1, 0),
                    false, {di}));
    }
  }

  return temp;
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
  mapLabelsAccordingToPlan(graph, labels, plan.getImpl());
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

  logging::popnn::debug("Creating alpha/beta tensor for CTC Loss with Time:{}"
                        " Batches:{} ExtendedLabelLength:{}",
                        maxT, batchSize, extendedLabelsLength);
  auto alphaBeta =
      graph.addVariable(outType, {maxT, batchSize, extendedLabelsLength},
                        {di, layer + "/alphaBeta"});
  mapAccordingToPlan(graph, alphaBeta, plan);

  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Time partitions"
                        " with Time:2 Batches:{} ExtendedLabelsLength:{}",
                        batchSize, extendedLabelsLength);
  auto tempTimeAlphaBeta = graph.addVariable(
      outType, {2 * plan.parallel.time, batchSize, extendedLabelsLength},
      {di, layer + "/tempTimeAlphaBeta"});
  mapAccordingToPlan(graph, tempTimeAlphaBeta, plan);

  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Label partitions"
                        " with Partitions:{} Time:{} Batches:{} Labels:2",
                        plan.parallel.label, maxT, batchSize);
  auto tempLabelAlphaBeta =
      graph.addVariable(outType, {plan.parallel.label, maxT, batchSize, 2},
                        {di, layer + "/tempLabelAlphaBeta"});
  // Same mapping process as the gradient tensor's allocation
  mapGradientAccordingToPlan(graph, tempLabelAlphaBeta, plan);

  // In our arithmetic, 0 is probability = 1, log::min is probability =  0
  // Create constants with which to initialise the temp vertex inputs and the
  // gradient
  auto initialZeros = graph.addConstant(outType, {numClasses}, log::min,
                                        {di, layer + "/initalZero"});
  graph.setTileMapping(initialZeros, 0);

  prog.add(
      Copy(initialZeros.broadcast(plan.parallel.label * maxT * batchSize, 0),
           gradient.flatten(), false, {di}));

  std::vector<float> initialiser(extendedLabelsLength, log::min);
  initialiser[0] = 0;
  auto initialAlpha = graph.addConstant<float>(
      outType, {initialiser.size()}, initialiser, {di, layer + "/initalAlpha"});
  initialiser[0] = log::min;
  initialiser.back() = 0;
  auto initialBeta = graph.addConstant<float>(
      outType, {initialiser.size()}, initialiser, {di, layer + "/initalBeta"});
  graph.setTileMapping(initialAlpha, 0);
  graph.setTileMapping(initialBeta, 0);

  std::vector<float> labelsInitialiser(maxT, log::min);
  auto initialLabelZeros = graph.addConstant<float>(
      outType, {maxT}, labelsInitialiser, {di, layer + "/initialLabels"});
  graph.setTileMapping(initialLabelZeros, 0);

  // Flags to remember which partitions of alpha and beta are already done, so
  // dependencies can be checked
  CompletionFlags complete(plan.parallel.label, plan.parallel.time);
  CompletionFlags updated(plan.parallel.label, plan.parallel.time);
  bool allComplete = false;
  unsigned csNum = 0;
  // Add more compute sets until we are done
  while (!allComplete) {
    allComplete = true;
    auto cs =
        graph.addComputeSet({di, layer + "/csNum" + std::to_string(csNum++)});
    logging::popnn::trace("");
    logging::popnn::trace("**** Compute Set:{}", csNum);
    for (unsigned time = 0; time < plan.parallel.time; time++) {
      for (unsigned label = 0; label < plan.parallel.label; label++) {

        // What vertex if any can be run on this time partition in this
        // computeSet?
        auto tileVertex = complete.vertexToRun(label, time);
        if (tileVertex) {
          updated.set(label, time, tileVertex.get());
        }
        if (!updated.alpha(label, time) || !updated.beta(label, time)) {
          allComplete = false;
        }
        if (!tileVertex) {
          // Nothing to run - either already done or not yet able to do more
          continue;
        }
        // What type of input is to initialise the vertex ?
        const auto timeInitialiser =
            complete.previousResult(time, label, true, tileVertex.get());
        const auto labelInitialiser =
            complete.previousResult(label, time, false, tileVertex.get());

        // The awkward diagonal previous input
        auto timeLabelInitialiser =
            complete.previousResult(time, label, tileVertex.get());
        ;

        const auto timePartition = plan.partitionTime(maxT, time);
        const auto labelPartition = plan.partitionLabel(labelsLength, label);
        const auto exLabelPartition =
            plan.partitionExtendedLabel(labelsLength, label);

        // We only now need to loop over the batch partitions as each will be
        // an identical copy of what was already decided
        for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
          const auto tile = plan.getTile(batch, time, label);
          bool isBeta = tileVertex.get() == VertexType::BETA ||
                        tileVertex.get() == VertexType::GRAD_GIVEN_ALPHA;
          logging::popnn::trace(
              "Tile: {} Initialiser: Time: {}, Label:{} Beta:{}", tile,
              timeInitialiser, labelInitialiser, isBeta);

          const auto batchPartition = plan.partitionBatch(batchSize, batch);
          const auto labelOffset =
              plan.partitionLabel(labelsLength, 0).size() * label;
          const auto processExtraBlank = label == plan.parallel.label - 1;
          const auto lastTimePartition = time == plan.parallel.time - 1;

          // Loop to cope with multiple batch entries per tile
          for (unsigned b = batchPartition.begin(); b < batchPartition.end();
               b++) {

            if (tileVertex == VertexType::ALPHA ||
                tileVertex == VertexType::BETA) {
              // Generate ALPHA or BETA vertices where possible
              auto tempLabelIn = tempLabelInputToVertex(
                  prog, labelInitialiser, timeLabelInitialiser,
                  tileVertex.get(), b, label, timePartition, exLabelPartition,
                  lastTimePartition, initialLabelZeros, alphaBeta,
                  tempLabelAlphaBeta, di);

              auto tempTimeIn = tempTimeInputToNonGradVertex(
                  timeInitialiser, tileVertex.get(), b, timePartition,
                  exLabelPartition, initialAlpha, initialBeta, alphaBeta);

              generateVertex(graph, data, labels, labelLengths, tempTimeIn,
                             tempLabelIn, alphaBeta, boost::none, cs, tile,
                             tileVertex.get(), b, timePartition, label,
                             labelPartition, exLabelPartition, labelOffset,
                             processExtraBlank, blankClass);
            }
            if (tileVertex == VertexType::GRAD_GIVEN_ALPHA ||
                tileVertex == VertexType::GRAD_GIVEN_BETA) {
              // Generate GRAD_GIVEN_ALPHA or BETA vertices where possible
              auto tempLabelIn = tempLabelInputToVertex(
                  prog, labelInitialiser, timeLabelInitialiser,
                  tileVertex.get(), b, label, timePartition, exLabelPartition,
                  lastTimePartition, initialLabelZeros, alphaBeta,
                  tempLabelAlphaBeta, di);

              auto tempTimeIn = tempTimeInputToGradVertex(
                  prog, timeInitialiser, tileVertex.get(), b, time,
                  timePartition, exLabelPartition, initialAlpha, initialBeta,
                  alphaBeta, tempTimeAlphaBeta, plan, maxT, di);

              generateVertex(graph, data, labels, labelLengths, tempTimeIn,
                             tempLabelIn, alphaBeta, gradient, cs, tile,
                             tileVertex.get(), b, timePartition, label,
                             labelPartition, exLabelPartition, labelOffset,
                             processExtraBlank, blankClass);
            }
          }
        }
      }
    }
    complete = updated;
    prog.add(Execute(cs, {di, layer}));
  }

  logging::popnn::debug("CTCLoss implemented in {} computeSets", csNum);
  di.addOutput(gradient);
  if (gradient.dim(0) == 1) {
    return gradient.reshape(data.shape());
  }
  // Reduce where data was split over label.
  // TODO - Inaccurate, we need reduceLogAdd.  Here we would find e^x, which
  // needs to be representable as a float.  Without denorm, float becomes -inf
  // at around 1*10^-38. So e^-87 = 1.6*10^-38 is about the smallest result that
  // get processed correctly here.
  popops::expInPlace(graph, gradient, prog, {di});
  ReduceParams reduceParams = {popops::Operation::ADD, false};
  auto gradReduce =
      popops::reduce(graph, gradient, {0}, reduceParams, prog, {di});
  popops::logInPlace(graph, gradReduce, prog, {di});
  return gradReduce;
}

} // end namespace ctc

} // end namespace popnn
