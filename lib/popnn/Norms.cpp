// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplin/Norms.hpp"
#include "NormsInternal.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>

using namespace poplar;
using namespace poputil;

namespace popnn {

// This function is used to create normalisaton parameters for layers which
// have input activations of dimension 2 (i.e. fully connected layers)
static Tensor createNormParam(Graph &graph, const Tensor &acts,
                              const DebugNameAndId &dnai) {
  const unsigned numActs = acts.shape()[1];
  const auto dType = acts.elementType();
  // This should be replaced by a clone of the channel dimension of the
  // activations. It is not done because cloning part of a tensor is currently
  // expensive.
  auto param = graph.addVariable(dType, {numActs}, {dnai});
  poputil::mapTensorLinearly(graph, param);
  return param;
}

Tensor createNormGamma(Graph &graph, const Tensor &acts,
                       const DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));

  const auto rank = acts.rank();
  checkTensorShape(acts);

  Tensor output;
  if (rank > 2) {
    output = poplin::createNormGamma(graph, acts);
  } else {
    output = createNormParam(graph, acts, {di, "gamma"});
  }

  di.addOutput(output);
  return output;
}

Tensor createNormBeta(Graph &graph, const Tensor &acts,
                      const DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));

  const auto rank = acts.rank();
  checkTensorShape(acts);
  Tensor output;
  if (rank > 2) {
    output = poplin::createNormBeta(graph, acts);
  } else {
    output = createNormParam(graph, acts, {di, "beta"});
  }

  di.addOutput(output);
  return output;
}

std::pair<Tensor, Tensor>
createNormParams(Graph &graph, const Tensor acts,
                 const DebugContext &debugContext = {}) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(acts));

  Tensor gamma = createNormGamma(graph, acts, {di});
  Tensor beta = createNormBeta(graph, acts, {di});

  di.addOutputs(DI_ARGS(gamma, beta));
  return std::make_pair(gamma, beta);
}

std::uint64_t getNormFwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements, bool computeStats) {
  // Normalise:
  // - acts - beta (numActsElements)
  // - (acts - beta) * gamma (numActsElements)
  uint64_t normFlops = 2 * numActsElements;

  if (computeStats) {
    // Statistics:
    // - Sum of samples (numActsElements)
    // - Sum of squares (numActsElements)
    // - Divide sum of samples and sum of squares by numSamples
    //   (2*statisticsSize)
    // - power - mean * mean + eps (3*statisticsSize)
    // - sqrt (Not added)
    uint64_t statsflops = numActsElements +   // for sum of samples
                          numActsElements +   // for sum of squares
                          5 * statisticsSize; // divide by num samples,
                                              // power - mean * mean + eps
    // Whitening:
    // - acts - mean (numActsElements)
    // - (acts - mean) * istdDev (numActsElements)
    uint64_t whitenFlops = 2 * numActsElements;
    return statsflops + whitenFlops + normFlops;
  } else {
    return normFlops;
  }
}

std::uint64_t getNormBwdFlops(std::size_t statisticsSize,
                              std::size_t numActsElements) {

  // Norm gradients:
  // - grads * gamma  (numActsElements)
  uint64_t normGradFlops = numActsElements;

  // Statistics gradients:
  // - grads * whitenedActs (numActsElements)
  // - reduce1 (grads * whitenedActs) (numActsElements)
  // - reduce2 (grads) (numActsElements)
  // - scale reduce1 and reduce2 ( 2*statistics size)
  // - reduce1 * whitenedActs (numActsElements)
  // - grads - reduce1 * whitenedActs - reduce2 (2 * numActsElements)
  // - invStdDev * ( grads - reduce1 * whitenedActs - reduce2) (
  // numActsElements)
  uint64_t statisticsGradFlops =
      7 * static_cast<uint64_t>(numActsElements) + 2 * statisticsSize;
  return normGradFlops + statisticsGradFlops;
}

std::uint64_t getNormWuFlops(std::size_t paramsSize,
                             std::size_t numActsElements) {
  // Gradients w.r.t. params:
  // - gradsIn * actsWhitened  (numActsElements)
  // - reduce for dGamma (numActsElements)
  // - reduce for dBeta (numActsElements)
  uint64_t gradsFlops = 3 * numActsElements;

  // learning rate and update
  uint64_t updateFlops = 4 * paramsSize;
  return gradsFlops + updateFlops;
}

} // namespace popnn
