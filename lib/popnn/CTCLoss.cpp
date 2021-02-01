// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "CTCLossPlan.hpp"
#include "poplibs_support/logging.hpp"
#include <poplar/Graph.hpp>
#include <poplibs_support/LogArithmetic.hpp>
#include <popnn/CTCLoss.hpp>
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

void generateVertex(Graph &graph, const Tensor &data, const Tensor &labels,
                    const Tensor &validLabel, const Tensor &validTime,
                    const Tensor &tempTimeAlphaOrBeta,
                    const Tensor &tempLabelAlphaOrBeta,
                    const Tensor &alphaOrBeta, boost::optional<Tensor &> grad,
                    ComputeSet &cs, unsigned tile, VertexType vertexType,
                    unsigned batch, const Interval &timePartition,
                    unsigned label, const Interval &labelPartition,
                    const Interval &exLabelPartition, unsigned labelOffset,
                    unsigned timeOffset, bool processExtraBlank,
                    unsigned blankClass) {

  const auto numClasses = data.dim(3);
  Slice<2> beginLabels = {batch, labelPartition.begin()};
  Slice<2> endLabels = {batch + 1, labelPartition.end()};
  auto tileLabel = labels.slice(beginLabels, endLabels);
  auto tileValidLabel = validLabel.slice(batch, batch + 1);
  auto tileValidTime = validTime.slice(batch, batch + 1);

  auto isAlpha = vertexType == VertexType::ALPHA ||
                 vertexType == VertexType::GRAD_GIVEN_BETA;
  Tensor prevSymbol;
  if (isAlpha) {
    if (label == 0) {
      prevSymbol = tileLabel.flatten()[0];
    } else {
      Slice<2> beginLabels = {batch, labelPartition.begin() - 1};
      Slice<2> endLabels = {batch + 1, labelPartition.begin()};
      prevSymbol = labels.slice(beginLabels, endLabels);
    }
  } else {
    if (processExtraBlank) {
      prevSymbol = tileLabel.flatten()[tileLabel.numElements() - 1];
    } else {
      Slice<2> beginLabels = {batch, labelPartition.end()};
      Slice<2> endLabels = {batch + 1, labelPartition.end() + 1};
      prevSymbol = labels.slice(beginLabels, endLabels);
    }
  }
  Slice<4> beginData = {label, timePartition.begin(), batch, 0};
  Slice<4> endData = {label + 1, timePartition.end(), batch + 1, numClasses};
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
  logging::popnn::trace("Making {} vertex on tile {} with label offset {}"
                        " and time offset {}",
                        vertexName, tile, labelOffset, timeOffset);
  auto v = graph.addVertex(cs, vertexName);
  graph.setTileMapping(v, tile);

  graph.setInitialValue(v["maxT"], timePartition.size());
  graph.setInitialValue(v["numClasses"], numClasses);
  graph.setInitialValue(v["blankClass"], blankClass);
  graph.setInitialValue(v["labelOffset"], labelOffset);
  graph.setInitialValue(v["timeOffset"], timeOffset);

  graph.connect(v["probabilities"], tileData.flatten());
  graph.connect(v["label"], tileLabel.flatten());
  graph.connect(v["validLabel"], tileValidLabel.reshape({}));
  graph.connect(v["validTime"], tileValidTime.reshape({}));

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

void mapTempLabelAccordingToPlan(Graph &graph, const Tensor &tensor,
                                 bool isAlpha,
                                 const popnn::ctc::Plan::Impl &plan) {
  // Map the rank 4 temporary "split by label tensor" according to the plan.
  // Note that time is the innermost dimension which matches the ordering in
  // which the vertices write the temporary data.
  const auto labelSize = tensor.dim(0);
  const auto batchSize = tensor.dim(1);
  const auto tempTimeSlices = tensor.dim(2);
  const auto timeSize = tensor.dim(3);

  for (unsigned label = 0; label < plan.parallel.label; label++) {
    for (unsigned batch = 0; batch < plan.parallel.batch; batch++) {
      for (unsigned time = 0; time < plan.parallel.time; time++) {

        auto tile = plan.getTile(batch, time, label);
        auto l = plan.partitionLabel(labelSize, label);
        auto b = plan.partitionBatch(batchSize, batch);
        // Partition time ignoring the extra element that is added to account
        // for a copy with timeshift.
        auto t = plan.partitionTime(timeSize - 1, time);

        std::size_t startOffset, endOffset;
        if (isAlpha) {
          // The tensor has an extra timestep which is to be mapped to the
          // first tile
          startOffset = time == 0 ? 0 : 1;
          endOffset = 1;
        } else {
          // The tensor has an extra timestep which is to be mapped to the last
          // tile
          startOffset = 0;
          endOffset = (time == plan.parallel.time - 1) ? 1 : 0;
        }
        t = {t.begin() + startOffset, t.end() + endOffset};
        graph.setTileMapping(
            tensor.slice({l.begin(), b.begin(), 0, t.begin()},
                         {l.end(), b.end(), tempTimeSlices, t.end()}),
            tile);
      }
    }
  }
}

void mapTempAlphaLabelAccordingToPlan(Graph &graph, const Tensor &tensor,
                                      const popnn::ctc::Plan::Impl &plan) {
  mapTempLabelAccordingToPlan(graph, tensor, true, plan);
}

void mapTempBetaLabelAccordingToPlan(Graph &graph, const Tensor &tensor,
                                     const popnn::ctc::Plan::Impl &plan) {
  mapTempLabelAccordingToPlan(graph, tensor, false, plan);
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
    // Label initialiser always comes from PREVIOUS_TEMP if non constant
    return isTime ? VertexInitialiser::PREVIOUS_RESULT
                  : VertexInitialiser::PREVIOUS_TEMP;
  }
};
// *****************************************************************************
// Comments above the 3 functions below explain temporary inputs to the vertices
// They will make more sense if read in order!

// Connect the required temporary time input to an alpha or beta vertex.
// Suppose we split data by time as follows:
//
//  el      time ----->
//        tile 0          tile 1
//  |  1  a b c d         a'b'c'd'
//  |  0  e f g h         e'f'g'h'
//  V  0  i j k l         i'j'k'l
//  alpha propagates--->  <------- beta propagates
// tile 0 will use the CONSTANT initialiser column[1,0,0] in order to calculate
// alpha for the first tile step col[a,e,i]. col[a,e,i] is then used to
// calculate column[b,f,j] etc.
// Before execution, the data is set to the required CONSTANT initialiser
// with all elements set to zero except for the 1st column[1,0,0].
// The vertex writes alpha into the alphaBeta tensor as it goes.
// For tile 1 to continue, it needs the PREVIOUS_RESULT initialiser col[d,h,l]
// to calculate col[a',e',i'].  It can then continue through to col[d',h',l'].
// If there was a tile 2, col[d',h',l'] would be its initialiser and so on.
// Note that PREVIOUS_RESULT is the previously calculated alpha stored in the
// alphaBeta tensor by a previously executed vertex.
//
// The input will always come from the alphaBeta tensor, not a temporary result
// (see tempTimeInputToGradVertex). Temporary results come from
// vertices that calculate `gradGivenX`.
//
// All of this can apply to alpha or beta but with beta propagating towards
// t=0.  The initialiser for tile1 in the case of beta would be col[0,0,1] which
// would begin to the right of tile 1 col[d',h',l']

Tensor tempTimeInputToAlphaOrBetaVertex(
    Sequence &prog, VertexInitialiser initialiser, VertexType tileVertex,
    unsigned batch, unsigned time, const Interval &timePartition,
    const Interval &exLabelPartition, const Tensor &alphaBeta,
    const Tensor &tempAlphaBeta, const poplar::DebugContext &di) {
  // The piece of the temporary input tensor to attach to this vertex
  Slice<3> beginT = {time, batch, exLabelPartition.begin()};
  Slice<3> endT = {time + 1, batch + 1, exLabelPartition.end()};
  auto result = tempAlphaBeta.slice(beginT, endT).flatten();

  if (initialiser == VertexInitialiser::PREVIOUS_RESULT) {
    // Initialiser from the previous tile's alpha or beta
    if (tileVertex == VertexType::ALPHA) {
      Slice<3> begin = {timePartition.begin() - 1, batch,
                        exLabelPartition.begin()};
      Slice<3> end = {timePartition.begin(), batch + 1, exLabelPartition.end()};
      prog.add(
          Copy(alphaBeta.slice(begin, end).flatten(), result, false, {di}));
    } else {
      Slice<3> begin = {timePartition.end(), batch, exLabelPartition.begin()};
      Slice<3> end = {timePartition.end() + 1, batch + 1,
                      exLabelPartition.end()};
      prog.add(
          Copy(alphaBeta.slice(begin, end).flatten(), result, false, {di}));
    }
  }
  return result;
}

// Connect the required temporary time input to a gradGivenAlpha or
// gradGivenBeta vertex.
// Suppose we split data by time as follows:
//
//  el      time ----->
//        tile 0      tile 1       tile 2         tile 3
//  |     a b c d     a'b'c'd'     a"b"c"d"       a^b^c^d^
//  |     e f g h     e'f'g'h'     e"f"g"h"       e^f^g^h^
//  V     i j k l     i'j'k'l'     i"j"k"l"       i^j^k^l^
//  alpha propagates--->          <------- beta propagates
//
// Temp: u v          u'v'         u"v"           u^v^
//       w x          w'x'         w"x"           w^x^
//       y z          y'z'         y"z"           y^z^
//
// Data dependencies say that we need to run the following compute sets:
//      Tile 0        Tile 1       Tile 2         Tile 3
// Step
//  0   alpha         -            -              beta
//  1   -             alpha        beta           -
//  2   -             gradGivA     gradGivB       -
//  3   gradGivA      -            -              gradGivB
//
// Step0 and Step1
// Will run non-gradient vertices whose temporary inputs are dealt
// with by the tempTimeInputToAlphaOrBetaVertex function.  Those vertices write
// into the alphaBeta tensor.
// So: alphaBeta a..l and a'..l'  will be populated with alpha
//     alphaBeta a"..l" and a^..l^  will be populated with beta
// alphaBeta will not change in later compute sets as the data needs to be used
// for calculation of the gradient.
//
// Step2
// Will calculate gradientGivenAlpha on tile1 (a'..l' contains alpha).
// So it is calculating beta. It needs a temporary input from tile2 of
// column[a"e"i"]
// That is stored in a PREVIOUS_RESULT on tile 2 from where it is copied into
// tile 1 working temporary data col[u'w'y'].
// As tile 1 calculates beta it creates the beta that would be in col[d'h'l']
// but it can't overwrite. Instead this goes into col[v'x'z'], which is used as
// tile 1 calculates beta col[c'g'k'].
// Each timestep the use of col[u'w'y'] and col[v'x'z'] as input and output
// swaps. So after an even number of timesteps on tile 1 the last beta result
// will be in col[u'w'y'], if an odd number it will be col[v'x'z']
//
// In Step2 tile2 calculates gradientGivenBeta in a similar manner with temp
// input col[d'h'j'] which is the PREVIOUS_RESULT
//
// Step3
// Here we calculate gradientGivenAlpha on tile0 (a..l contains alpha).
// So it is calculating beta. It needs a temporary input from tile1 of
// col[a'e'i'], however that was stored in the PREVIOUS_TEMP data col[u'w'y'],
// so it needs to be initialised from there.  (If tile 1 had an odd number of
// timesteps it would come from col[v'x'z']. Ongoing calculation on tile 0 is
// similar to that described in Step2.
//
// In Step3 tile3 calculates gradientGivenBeta in a similar manner with temp
// input [d"h"j"] which is the PREVIOUS_TEMP col[u"w"y"]
//
// Further splits result in PREVIOUS_TEMP continuing to propagate with
// increasing time for alpha, and decreasing time for beta

Tensor tempTimeInputToGradVertex(
    Sequence &prog, VertexInitialiser initialiser, VertexType tileVertex,
    unsigned batch, unsigned time, const Interval &timePartition,
    const Interval &exLabelPartition, const Tensor &alphaBeta,
    const Tensor &tempAlphaBeta, const popnn::ctc::Plan::Impl &plan,
    unsigned maxT, const poplar::DebugContext &di) {

  // The piece of the temporary input tensor to attach to this vertex
  Slice<3> beginT = {2 * time, batch, exLabelPartition.begin()};
  Slice<3> endT = {2 * (time + 1), batch + 1, exLabelPartition.end()};
  auto result = tempAlphaBeta.slice(beginT, endT);

  // The vertex will overwrite the data, so a copy is required in each case
  if (initialiser == VertexInitialiser::PREVIOUS_RESULT) {
    // Initialiser from the previous tile's alpha or beta.
    // Take the last timeslice (alpha) or 1st (beta)
    if (tileVertex == VertexType::GRAD_GIVEN_ALPHA) {
      Slice<3> begin = {timePartition.end(), batch, exLabelPartition.begin()};
      Slice<3> end = {timePartition.end() + 1, batch + 1,
                      exLabelPartition.end()};
      prog.add(Copy(alphaBeta.slice(begin, end), result.slice(0, 1, 0), false,
                    {di}));
    } else {
      Slice<3> begin = {timePartition.begin() - 1, batch,
                        exLabelPartition.begin()};
      Slice<3> end = {timePartition.begin(), batch + 1, exLabelPartition.end()};
      prog.add(Copy(alphaBeta.slice(begin, end), result.slice(0, 1, 0), false,
                    {di}));
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
      prog.add(Copy(tempAlphaBeta.slice(begin, end), result.slice(0, 1, 0),
                    false, {di}));
    } else {
      auto offset = plan.partitionTime(maxT, time - 1).size() % 2;
      Slice<3> begin = {2 * time - 2 + offset, batch, exLabelPartition.begin()};
      Slice<3> end = {2 * time - 1 + offset, batch + 1, exLabelPartition.end()};
      prog.add(Copy(tempAlphaBeta.slice(begin, end), result.slice(0, 1, 0),
                    false, {di}));
    }
  }
  return result;
}
// Connect the required temporary label input to an alpha, beta, gradGivenAlpha
// or gradGivenBeta vertex.
//
// For a full explanation we need to consider a time and label split together.
// Time split temporary data is explained above the 2 functions involved.
//
// Suppose we split data by time as follows:
//
//  el        time ----->
//    sym     tile 0      tile 2
//  |  -      a b c d     a"b"c"d"
//  |  0      e f g h     e"f"g"h"
//  |
//  |       w x y z     w"x"y"z"  (tempAlpha - Tile temporary alpha data)
//  |
//  | sym     tile 1      tile 3
//  V  -      a'b'c'd'    a^b^c^d^
//     1      e'f'g'h'    e^f^g^h^
//
//          w'x'y'z     e^f^g^h^ (tempAlpha - Tile temporary alpha data)
//
//      alpha propagates tile 0, tiles (1 and 2), tile 3
//
// Each tile will contain data corresponding to a number of labels which will
// each expand: 0 to {-,0}, 1 to {-,1} etc.
// The sequence will end with an extra {-}.
// Given this decision at the split between tiles alpha depends only upon
// the row above it (consider trellis/blank dependencies).  Beta depends on the
// 2 rows below (As they contain {-,symbol} and the {-} can be skipped over
// when symbol!=previousSymbol, so the value of "symbol" can be used).
//
// Vertices take the 1st row input from `tempAlpha` but this also relies on
// an element from a previous timestep, so as an input the [w x y z] vector is
// used like this:
//  w  x y z
//  M  a b c d        (a=fn(w,M) b=fn(x,a) c=fn(y,b) and d=fn(z,c))
//  N  e f g h        ([M, N] come from the time initialiser - functions above)
//
//  And as an output:
//  M  a b c d
//  N  e f g h
//     w x y z        (w=copy of alpha at e, x=alpha[f], y=alpha[g] z=alpha[h])
//
// The same `tempAlpha` is used as input and output, data being overwritten as
// we go.
//
// So to initialise, and per compute set:
// Step0
// tile0 `tempAlpha` input [wxyz]=[0000](All zero as this is the 1st row of El)
//      produces an output [wxyz]=alpha at [efgh]
// Step1
// tile1 `tempAlpha` input [w`x`y`z`]=[0wxy]. (0 as there is no previous `t`)
//      produces an output [w`x`y`z`]=alpha at [e'f'g'h']
// tile2 `tempAlpha` input [w"x"y"z"]=[0000]
//      produces an output [w"x"y"z"] = alpha at [e"f"g"h"]
// Step2
// tile3 `tempAlpha` input [w^x^y^z^]=[zw"x"y"]
//      produces an output [w^x^y^z^] = alpha at [e^f^g^h^]
//
// This pattern continues. Note the detail:
// if (First row)        initialise with [0000]
// Else if(First column) initialise with [0wxy] or [0w'x'y'] etc
// Else                  initialise with [zw"x"y"]
//
// Exactly the same happens for beta, but 2 rows are maintained to pass up
// from tile to tile, and everything else is in reverse working from tile 3
// toward tile 0.  Call this [[stuv],[wxyz]].  The bottom row requires
// initialising with [[0000],[0000]] and
// the last column means [[tuv0],[xyz0]]

Tensor tempLabelInputToVertex(Sequence &prog, VertexInitialiser initialiser,
                              VertexType tileVertex, unsigned batch,
                              unsigned label, const Interval &time,
                              const Interval &exLabel, bool lastTimePartition,
                              const Tensor &tempAlpha, const Tensor &tempBeta,
                              const poplar::DebugContext &di) {

  bool vertexCalculatesAlpha = (tileVertex == VertexType::ALPHA ||
                                tileVertex == VertexType::GRAD_GIVEN_BETA);

  // The piece of the temporary input tensor to attach to this vertex
  // tempAlpha Tensor shape : {1, 1, 1, maxT}
  // tempBeta Tensor shape: {1, 1, 2, maxT}
  // dim 0: single slice in the label dimension
  // dim 1: single slice in the batch dimension
  // dim 2: 1 row for alpha, 2 rows for beta as we need to propagate more
  // data due to the dependency on blank and symbols in the trellis.

  // Reference input, output slices such that:
  // a copy for alpha shifts +1 timestep
  // a copy for beta shifts -1 timestep
  const std::size_t timeInOffset = vertexCalculatesAlpha ? 1 : 0;
  const std::size_t timeOutOffset = vertexCalculatesAlpha ? 0 : 1;

  Slice<4> begin = {label, batch, 0, time.begin() + timeInOffset};
  Slice<4> end = {label + 1, batch + 1, vertexCalculatesAlpha ? 1ul : 2ul,
                  time.end() + timeInOffset};

  const auto result = vertexCalculatesAlpha
                          ? tempAlpha.slice(begin, end).flatten()
                          : tempBeta.slice(begin, end).flatten();

  // Copy the required previous tile's alpha or beta temporary data into the
  // result tensor (which represents this tile's temporary memory input)
  if (initialiser == VertexInitialiser::PREVIOUS_TEMP) {
    // Initialise from the previous tile's temporary output
    if (vertexCalculatesAlpha) {
      // Take the previous label result, shifted by one timestep toward t=maxT
      Slice<4> begin = {label - 1, batch, 0, time.begin() + timeOutOffset};
      Slice<4> end = {label, batch + 1, 1, time.end() + timeOutOffset};
      prog.add(
          Copy(tempAlpha.slice(begin, end).flatten(), result, false, {di}));
    } else {
      // Take the previous label result, shifted by one timestep toward t=0
      Slice<4> begin = {label + 1, batch, 0, time.begin() + timeOutOffset};
      Slice<4> end = {label + 2, batch + 1, 2, time.end() + timeOutOffset};
      prog.add(Copy(tempBeta.slice(begin, end).flatten(), result, false, {di}));
    }
  }
  return result;
}

void initialise(Graph &graph, const Tensor &input, Sequence &prog,
                const poplar::DebugContext &di) {
  const auto rank = input.rank();
  const auto baseDim = input.shape()[rank - 1];
  // TODO - If there is no speed penalty we could initialise with a scalar
  // to save space
  std::vector<float> initialData(baseDim, log::probabilityZero);
  auto initialiser = graph.addConstant<float>(input.elementType(), {1, baseDim},
                                              initialData, {di});
  graph.setTileMapping(initialiser, 0);

  auto inputReshape = input.reshape({input.numElements() / baseDim, baseDim});
  prog.add(Copy(initialiser.broadcast(inputReshape.dim(0), 0), inputReshape,
                false, {di}));
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
  mapAccordingToPlan(graph, alphaBeta, plan);

  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Time partitions"
                        " with Time:2 Batches:{} ExtendedLabelsLength:{}",
                        batchSize, extendedLabelsLength);
  auto tempTimeAlphaBeta1 = graph.addVariable(
      outType, {plan.parallel.time, batchSize, extendedLabelsLength},
      {di, layer + "/tempTimeAlphaBeta1"});
  mapAccordingToPlan(graph, tempTimeAlphaBeta1, plan);

  auto tempTimeAlphaBeta2 = graph.addVariable(
      outType, {2 * plan.parallel.time, batchSize, extendedLabelsLength},
      {di, layer + "/tempTimeAlphaBeta2"});
  mapAccordingToPlan(graph, tempTimeAlphaBeta2, plan);

  // The temporary data for each label is copied and shifted by 1 timestep
  // Make it 1 element larger to account for this.
  const auto tempLabelTimeSteps = maxT + 1;
  logging::popnn::debug("Creating temporary alpha/beta tensor for CTC Loss "
                        "Label partitions"
                        " with Partitions:{} Time:{} Batches:{} Labels:2",
                        plan.parallel.label, tempLabelTimeSteps, batchSize);

  auto tempLabelAlpha = graph.addVariable(
      outType, {plan.parallel.label, batchSize, 1, tempLabelTimeSteps},
      {di, layer + "/tempLabelAlpha"});
  mapTempAlphaLabelAccordingToPlan(graph, tempLabelAlpha, plan);

  auto tempLabelBeta = graph.addVariable(
      outType, {plan.parallel.label, batchSize, 2, tempLabelTimeSteps},
      {di, layer + "/tempLabelBeta"});
  mapTempBetaLabelAccordingToPlan(graph, tempLabelBeta, plan);

  // Initialise the temporary inputs to the vertices, all to probabilityZero,
  // except for a single "previous alpha" element which equals probabilityOne.
  // The equivalent "previous beta" element is initialised at runtime in the
  // required place based on the time,label size of each individual input.
  initialise(graph, tempTimeAlphaBeta1, prog, {di, layer});
  auto initialiserOne = graph.addConstant<float>(
      outType, {1, 1, 1}, static_cast<float>(log::probabilityOne), {di});
  graph.setTileMapping(initialiserOne, 0);
  auto tempTimeAlphaBeta1Slice =
      tempTimeAlphaBeta1.slice({0, 0, 0}, {1, batchSize, 1});
  prog.add(Copy(initialiserOne.broadcast(batchSize, 1), tempTimeAlphaBeta1Slice,
                false, {di}));

  initialise(graph, tempTimeAlphaBeta2, prog, {di, layer});
  initialise(graph, tempLabelAlpha, prog, {di, layer});
  initialise(graph, tempLabelBeta, prog, {di, layer});
  // Initialise the gradient to probabilityZero, to accumulate into
  initialise(graph, gradient, prog, di);

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
          const auto timeOffset = plan.partitionTime(maxT, 0).size() * time;
          const auto processExtraBlank = label == plan.parallel.label - 1;
          const auto lastTimePartition = time == plan.parallel.time - 1;

          // Loop to cope with multiple batch entries per tile
          for (unsigned b = batchPartition.begin(); b < batchPartition.end();
               b++) {

            if (tileVertex == VertexType::ALPHA ||
                tileVertex == VertexType::BETA) {
              // Generate ALPHA or BETA vertices where possible
              auto tempLabelIn = tempLabelInputToVertex(
                  prog, labelInitialiser, tileVertex.get(), b, label,
                  timePartition, exLabelPartition, lastTimePartition,
                  tempLabelAlpha, tempLabelBeta, di);

              auto tempTimeIn = tempTimeInputToAlphaOrBetaVertex(
                  prog, timeInitialiser, tileVertex.get(), b, time,
                  timePartition, exLabelPartition, alphaBeta,
                  tempTimeAlphaBeta1, di);

              generateVertex(graph, workingData, labels, labelLengths,
                             dataLengths, tempTimeIn, tempLabelIn, alphaBeta,
                             boost::none, cs, tile, tileVertex.get(), b,
                             timePartition, label, labelPartition,
                             exLabelPartition, labelOffset, timeOffset,
                             processExtraBlank, blankClass);
            }
            if (tileVertex == VertexType::GRAD_GIVEN_ALPHA ||
                tileVertex == VertexType::GRAD_GIVEN_BETA) {
              // Generate GRAD_GIVEN_ALPHA or BETA vertices where possible
              auto tempLabelIn = tempLabelInputToVertex(
                  prog, labelInitialiser, tileVertex.get(), b, label,
                  timePartition, exLabelPartition, lastTimePartition,
                  tempLabelAlpha, tempLabelBeta, di);

              auto tempTimeIn = tempTimeInputToGradVertex(
                  prog, timeInitialiser, tileVertex.get(), b, time,
                  timePartition, exLabelPartition, alphaBeta,
                  tempTimeAlphaBeta2, plan, maxT, di);

              generateVertex(graph, workingData, labels, labelLengths,
                             dataLengths, tempTimeIn, tempLabelIn, alphaBeta,
                             gradient, cs, tile, tileVertex.get(), b,
                             timePartition, label, labelPartition,
                             exLabelPartition, labelOffset, timeOffset,
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
  // TODO: Mapping choice to spread according to the plan?  Or reduce into one
  //  of the copies of the gradient for less temporarary memory
  ReduceParams reduceParams = {popops::Operation::LOG_ADD, false};
  auto gradReduce =
      popops::reduce(graph, gradient, {0}, reduceParams, prog, {di});

  return gradReduce;
}

} // end namespace ctc

} // end namespace popnn
