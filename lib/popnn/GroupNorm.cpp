// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "NormsInternal.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplin/Norms.hpp"
#include "popnn/BatchNorm.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Rearrange.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <functional>
#include <map>
#include <numeric>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace popnn {
namespace gn {
struct GroupNormOptions {
  bool stridedChannelGrouping = false;
};

static GroupNormOptions parseOptions(const OptionFlags &options) {
  GroupNormOptions optionFlags;
  const poplibs::OptionSpec groupNormOptionSpec{
      {"groupNormStridedChannelGrouping",
       poplibs::OptionHandler::createWithBool(
           optionFlags.stridedChannelGrouping)}};
  for (const auto &option : options) {
    groupNormOptionSpec.parse(option.first, option.second);
  }
  return optionFlags;
}

static Tensor groupActs(const Tensor &acts_, unsigned numGroups,
                        bool stridedChannelGrouping) {
  const auto numChannels = acts_.dim(1);
  const auto numBatches = acts_.dim(0);
  if (numChannels % numGroups != 0) {
    throw poplibs_error("Group Norm : Number of channels must be an integral "
                        "multiple of number of groups");
  }
  Tensor acts;
  if (stridedChannelGrouping) {
    acts = acts_.reshapePartial(1, 2, {numChannels / numGroups, numGroups})
               .dimRoll(2, 1)
               .reshapePartial(0, 2, {numGroups * numBatches})
               .dimRoll(1, 0);
  } else {
    acts = acts_.reshapePartial(1, 2, {numGroups, numChannels / numGroups})
               .reshapePartial(0, 2, {numGroups * numBatches})
               .dimRoll(1, 0);
  }
  return acts;
}

static Tensor ungroupActs(const Tensor &acts_, unsigned numChannels,
                          bool stridedChannelGrouping) {
  const auto numBatches = acts_.dim(0) * acts_.dim(1) / numChannels;
  Tensor acts;
  if (stridedChannelGrouping) {
    const auto numGroups = numChannels / acts_.dim(0);
    acts = acts_.reshapePartial(1, 2, {numBatches, numGroups})
               .dimRoll(0, 1)
               .reshapePartial(1, 3, {numChannels});
  } else {
    acts = acts_.dimRoll(0, 1).reshapePartial(0, 2, {numBatches, numChannels});
  }
  return acts;
}

std::pair<Tensor, Tensor>
groupNormStatistics(Graph &graph, const Tensor acts_, float eps, Sequence &prog,
                    unsigned numGroups, bool unbiasedVarEstimate,
                    bool stableAlgo, const Type &partialsType,
                    const poplar::DebugContext &debugContext,
                    const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(acts_, eps, numGroups, unbiasedVarEstimate,
                            stableAlgo, partialsType, options));

  checkTensorShape(acts_);
  const auto optionFlags = parseOptions(options);

  // Ensure grouping is suitable for group norm at this point.
  // TODO: T12904 Consider removing this check once T6174 is resolved.
  const auto preferredGrouping =
      graph.getTarget().getVectorWidth(acts_.elementType());
  auto acts = acts_;
  if (acts.dim(1) % preferredGrouping == 0) {
    acts = popops::rearrange::regroupIfBeneficial(
               graph, acts.dimRoll(1, acts.rank() - 1), preferredGrouping, prog,
               {di})
               .dimRoll(acts.rank() - 1, 1);
  }
  acts = groupActs(acts, numGroups, optionFlags.stridedChannelGrouping);
  auto outputs =
      poplin::normStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                             stableAlgo, partialsType, {di});
  di.addOutputs({{"mean", toProfileValue(outputs.first)},
                 {"iStdev", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor groupNormWhiten(Graph &graph, const Tensor &acts, const Tensor &mean,
                       const Tensor &iStdDev, Sequence &prog,
                       const poplar::DebugContext &debugContext,
                       const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, mean, iStdDev, options));

  const auto optionFlags = parseOptions(options);
  const auto rank = acts.rank();
  const auto numChannels = acts.dim(1);
  checkTensorShape(acts);
  const auto batchSize = acts.dim(0);
  assert(mean.dim(0) % batchSize == 0);
  const auto numGroups = mean.dim(0) / batchSize;
  auto groupedActs = groupActs(preProcessNormActs(acts), numGroups,
                               optionFlags.stridedChannelGrouping);
  auto whitenedActs =
      poplin::normWhiten(graph, groupedActs, mean, iStdDev, prog, {di});
  auto output =
      postProcessNormActs(ungroupActs(whitenedActs, numChannels,
                                      optionFlags.stridedChannelGrouping),
                          rank);
  di.addOutput(output);
  return output;
}

std::pair<Tensor, Tensor>
groupNormalise(Graph &graph, const Tensor &acts, const Tensor &gamma,
               const Tensor &beta, const Tensor &mean, const Tensor &iStdDev,
               Sequence &prog, const poplar::DebugContext &debugContext,
               const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(acts, mean, iStdDev, options));

  const auto rank = acts.rank();
  checkTensorShape(acts);
#ifndef NDEBUG
  const auto batchSize = acts.dim(0);
  assert(mean.dim(0) % batchSize == 0);
#endif
  auto preProcessedActs = preProcessNormActs(acts);
  auto whitenedActs = groupNormWhiten(graph, preProcessedActs, mean, iStdDev,
                                      prog, {di}, options);
  auto outputActs =
      poplin::normalise(graph, whitenedActs, gamma, beta, prog, {di});
  auto outputs = std::make_pair(postProcessNormActs(outputActs, rank),
                                postProcessNormActs(whitenedActs, rank));
  di.addOutputs({{"normlisedActs", toProfileValue(outputs.first)},
                 {"whitenedActs", toProfileValue(outputs.second)}});
  return outputs;
}

std::pair<Tensor, Tensor>
groupNormParamGradients(Graph &graph, const Tensor &actsWhitened,
                        const Tensor &gradsIn, Sequence &prog,
                        const Type &partialsType,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(actsWhitened, gradsIn, partialsType, options));

  checkTensorShape(gradsIn);
  checkTensorShape(actsWhitened);
  auto outputs = poplin::normParamGradients(graph, actsWhitened, gradsIn, prog,
                                            partialsType, {di});
  di.addOutputs({{"normlisedActs", toProfileValue(outputs.first)},
                 {"whitenedActs", toProfileValue(outputs.second)}});
  return outputs;
}

std::pair<Tensor, Tensor>
groupNormParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                        const Tensor &mean, const Tensor &iStdDev,
                        Sequence &prog, const Type &partialsType,
                        const poplar::DebugContext &debugContext,
                        const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts, gradsIn, mean, iStdDev, partialsType, options));

  checkTensorShape(acts);
  auto actsWhitened =
      groupNormWhiten(graph, acts, mean, iStdDev, prog, {di}, options);
  auto outputs = groupNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                         partialsType, {di}, options);
  di.addOutputs({{"meanGrad", toProfileValue(outputs.first)},
                 {"iStdDevGrad", toProfileValue(outputs.second)}});
  return outputs;
}

Tensor groupNormGradients(Graph &graph, const Tensor &actsWhitened_,
                          const Tensor &gradsIn_, const Tensor &iStdDev,
                          const Tensor &gamma, Sequence &prog,
                          const Type &partialsType,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(actsWhitened_, gradsIn_, gamma, iStdDev, partialsType, options));

  const auto optionFlags = parseOptions(options);
  const auto rank = actsWhitened_.rank();
  const auto numChans = actsWhitened_.dim(1);
  checkTensorShape(actsWhitened_);
  checkTensorShape(gradsIn_);
  const auto batchSize = actsWhitened_.dim(0);
  assert(iStdDev.dim(0) % batchSize == 0);
  const auto numGroups = iStdDev.dim(0) / batchSize;
  auto actsWhitened = preProcessNormActs(actsWhitened_);
  auto gradsIn = preProcessNormActs(gradsIn_);
  auto gradsNorm = poplin::normGradients(graph, gradsIn, gamma, prog, {di});
  auto groupedActsWhitened =
      groupActs(actsWhitened, numGroups, optionFlags.stridedChannelGrouping);
  auto groupedGradsNorm =
      groupActs(gradsNorm, numGroups, optionFlags.stridedChannelGrouping);
  auto gradsOut = poplin::normStatisticsGradients(graph, groupedActsWhitened,
                                                  groupedGradsNorm, iStdDev,
                                                  prog, partialsType, {di});
  auto output = postProcessNormActs(
      ungroupActs(gradsOut, numChans, optionFlags.stridedChannelGrouping),
      rank);
  di.addOutput(output);
  return output;
}

Tensor groupNormGradients(Graph &graph, const Tensor &acts_,
                          const Tensor &gradsIn_, const Tensor &mean,
                          const Tensor &iStdDev, const Tensor &gamma,
                          Sequence &prog, const Type &partialsType,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(acts_, gradsIn_, mean, gamma, iStdDev, partialsType, options));

  checkTensorShape(acts_);
  auto actsWhitened =
      groupNormWhiten(graph, acts_, mean, iStdDev, prog, {di}, options);
  auto output = groupNormGradients(graph, actsWhitened, gradsIn_, iStdDev,
                                   gamma, prog, partialsType, {di}, options);
  di.addOutput(output);
  return output;
}

void groupNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, float scale, Tensor &gamma,
                          Tensor &beta, Sequence &prog,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(gammaDelta, betaDelta, beta, gamma, scale, options));

  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, {di, "GN/paramUpdate"});
}

void groupNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, const Tensor &scale,
                          Tensor &gamma, Tensor &beta, Sequence &prog,
                          const poplar::DebugContext &debugContext,
                          const poplar::OptionFlags &options) {
  POPNN_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(gammaDelta, betaDelta, scale, beta, gamma, options));

  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, {di, "GN/paramUpdate"});
}
} // namespace gn
} // namespace popnn
