#include "NormsInternal.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Norms.hpp"
#include "popnn/BatchNorm.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
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

static Tensor groupActs(const Tensor &acts_, unsigned numGroups) {
  const auto numChannels = acts_.dim(1);
  const auto numBatches = acts_.dim(0);
  if (numChannels % numGroups != 0) {
    throw poplibs_error("Group Norm : Number of channels must be an integral "
                        "multiple of number of groups");
  }
  auto acts = acts_.reshapePartial(1, 2, {numChannels / numGroups, numGroups})
                  .dimRoll(2, 1)
                  .reshapePartial(0, 2, {numGroups * numBatches})
                  .dimRoll(1, 0);
  return acts;
}

static Tensor ungroupActs(const Tensor &acts_, unsigned numChannels) {
  const auto numBatches = acts_.dim(0) * acts_.dim(1) / numChannels;
  const auto numGroups = numChannels / acts_.dim(0);
  auto acts = acts_.reshapePartial(1, 2, {numBatches, numGroups})
                  .dimRoll(0, 1)
                  .reshapePartial(1, 3, {numChannels});
  return acts;
}

std::pair<Tensor, Tensor>
groupNormStatistics(Graph &graph, const Tensor acts_, float eps, Sequence &prog,
                    unsigned numGroups, bool unbiasedVarEstimate,
                    const Type &partialsType, const std::string &debugPrefix) {
  checkTensorShape(acts_);
  // TODO: T12904 Until T6174 is fixed, reductions deal terribly with
  // reducing along the innermost dimension in memory. Ensure
  // grouping is suitable for group norm at this point.
  const auto preferredGrouping =
      graph.getTarget().getVectorWidth(acts_.elementType());
  auto acts = acts_;
  if (acts.dim(1) % preferredGrouping == 0) {
    acts = poplin::regroupIfBeneficial(graph, acts, preferredGrouping, prog,
                                       debugPrefix);
  }
  acts = groupActs(acts, numGroups);
  return poplin::normStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                                partialsType, debugPrefix);
}

Tensor groupNormWhiten(Graph &graph, const Tensor &acts, const Tensor &mean,
                       const Tensor &iStdDev, Sequence &prog,
                       const std::string &debugPrefix) {
  const auto rank = acts.rank();
  const auto numChannels = acts.dim(1);
  checkTensorShape(acts);
  const auto batchSize = acts.dim(0);
  assert(mean.dim(0) % batchSize == 0);
  const auto numGroups = mean.dim(0) / batchSize;
  auto groupedActs = groupActs(preProcessNormActs(acts), numGroups);
  auto whitenedActs =
      poplin::normWhiten(graph, groupedActs, mean, iStdDev, prog, debugPrefix);
  return postProcessNormActs(ungroupActs(whitenedActs, numChannels), rank);
}

std::pair<Tensor, Tensor> groupNormalise(Graph &graph, const Tensor &acts,
                                         const Tensor &gamma,
                                         const Tensor &beta, const Tensor &mean,
                                         const Tensor &iStdDev, Sequence &prog,
                                         const std::string &debugPrefix) {
  const auto rank = acts.rank();
  checkTensorShape(acts);
#ifndef NDEBUG
  const auto batchSize = acts.dim(0);
  assert(mean.dim(0) % batchSize == 0);
#endif
  auto preProcessedActs = preProcessNormActs(acts);
  auto whitenedActs = groupNormWhiten(graph, preProcessedActs, mean, iStdDev,
                                      prog, debugPrefix);
  auto outputActs =
      poplin::normalise(graph, whitenedActs, gamma, beta, prog, debugPrefix);
  return std::make_pair(postProcessNormActs(outputActs, rank),
                        postProcessNormActs(whitenedActs, rank));
}

std::pair<Tensor, Tensor> groupNormParamGradients(
    Graph &graph, const Tensor &actsWhitened, const Tensor &gradsIn,
    Sequence &prog, const Type &partialsType, const std::string &debugPrefix) {
  checkTensorShape(gradsIn);
  checkTensorShape(actsWhitened);
  return poplin::normParamGradients(graph, actsWhitened, gradsIn, prog,
                                    partialsType, debugPrefix);
}

std::pair<Tensor, Tensor>
groupNormParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                        const Tensor &mean, const Tensor &iStdDev,
                        Sequence &prog, const Type &partialsType,
                        const std::string &debugPrefix) {
  checkTensorShape(acts);
  auto actsWhitened =
      groupNormWhiten(graph, acts, mean, iStdDev, prog, debugPrefix);
  return groupNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                 partialsType, debugPrefix);
}

Tensor groupNormGradients(Graph &graph, const Tensor &actsWhitened_,
                          const Tensor &gradsIn_, const Tensor &iStdDev,
                          const Tensor &gamma, Sequence &prog,
                          const Type &partialsType,
                          const std::string &debugPrefix) {
  const auto rank = actsWhitened_.rank();
  const auto numChans = actsWhitened_.dim(1);
  checkTensorShape(actsWhitened_);
  checkTensorShape(gradsIn_);
  const auto batchSize = actsWhitened_.dim(0);
  assert(iStdDev.dim(0) % batchSize == 0);
  const auto numGroups = iStdDev.dim(0) / batchSize;
  auto actsWhitened = preProcessNormActs(actsWhitened_);
  auto gradsIn = preProcessNormActs(gradsIn_);
  auto gradsNorm =
      poplin::normGradients(graph, gradsIn, gamma, prog, debugPrefix);
  auto groupedActsWhitened = groupActs(actsWhitened, numGroups);
  auto groupedGradsNorm = groupActs(gradsNorm, numGroups);
  auto gradsOut = poplin::normStatisticsGradients(
      graph, groupedActsWhitened, groupedGradsNorm, iStdDev, prog, partialsType,
      debugPrefix);
  return postProcessNormActs(ungroupActs(gradsOut, numChans), rank);
}

Tensor groupNormGradients(Graph &graph, const Tensor &acts_,
                          const Tensor &gradsIn_, const Tensor &mean,
                          const Tensor &iStdDev, const Tensor &gamma,
                          Sequence &prog, const Type &partialsType,
                          const std::string &debugPrefix) {
  checkTensorShape(acts_);
  auto actsWhitened =
      groupNormWhiten(graph, acts_, mean, iStdDev, prog, debugPrefix);
  return groupNormGradients(graph, actsWhitened, gradsIn_, iStdDev, gamma, prog,
                            partialsType, debugPrefix);
}

void groupNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, float scale, Tensor &gamma,
                          Tensor &beta, Sequence &prog,
                          const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/GN/paramUpdate";
  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, fnPrefix);
}

void groupNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, const Tensor &scale,
                          Tensor &gamma, Tensor &beta, Sequence &prog,
                          const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/GN/paramUpdate";
  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, fnPrefix);
}
} // namespace gn
} // namespace popnn
