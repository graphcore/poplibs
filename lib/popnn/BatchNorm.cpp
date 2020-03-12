// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popnn/BatchNorm.hpp"
#include "NormsInternal.hpp"
#include "poplin/Norms.hpp"
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
namespace bn {

std::pair<Tensor, Tensor>
batchNormStatistics(Graph &graph, const Tensor acts, float eps, Sequence &prog,
                    bool unbiasedVarEstimate, bool stableAlgo,
                    const Type &partialsType, const std::string &debugPrefix) {
  checkTensorShape(acts);
  return poplin::normStatistics(graph, acts, eps, prog, unbiasedVarEstimate,
                                stableAlgo, partialsType, debugPrefix);
}

Tensor batchNormWhiten(Graph &graph, const Tensor &acts_, const Tensor &mean,
                       const Tensor &iStdDev, Sequence &prog,
                       const std::string &debugPrefix) {
  const auto rank = acts_.rank();
  auto acts = preProcessNormActs(acts_);
  auto whitenedActs =
      poplin::normWhiten(graph, acts, mean, iStdDev, prog, debugPrefix);
  return postProcessNormActs(whitenedActs, rank);
}

std::pair<Tensor, Tensor> batchNormalise(Graph &graph, const Tensor &acts,
                                         const Tensor &gamma,
                                         const Tensor &beta, const Tensor &mean,
                                         const Tensor &iStdDev, Sequence &prog,
                                         const std::string &debugPrefix) {
  const auto rank = acts.rank();
  checkTensorShape(acts);
  auto preProcessActs = preProcessNormActs(acts);
  auto whitenedActs =
      batchNormWhiten(graph, preProcessActs, mean, iStdDev, prog, debugPrefix);
  auto outputActs =
      poplin::normalise(graph, whitenedActs, gamma, beta, prog, debugPrefix);
  return std::make_pair(postProcessNormActs(outputActs, rank),
                        postProcessNormActs(whitenedActs, rank));
}

Tensor batchNormalise(Graph &graph, const Tensor &acts,
                      const Tensor &combinedMultiplicand, const Tensor &addend,
                      Sequence &prog, const std::string &debugPrefix) {
  const auto rank = acts.rank();
  checkTensorShape(acts);
  auto preProcessedActs = preProcessNormActs(acts);
  auto actsNormalised = poplin::normalise(
      graph, preProcessedActs, combinedMultiplicand, addend, prog, debugPrefix);
  return postProcessNormActs(actsNormalised, rank);
}

std::pair<Tensor, Tensor> batchNormParamGradients(
    Graph &graph, const Tensor &actsWhitened, const Tensor &gradsIn,
    Sequence &prog, const Type &partialsType, const std::string &debugPrefix) {
  checkTensorShape(gradsIn);
  checkTensorShape(actsWhitened);
  return poplin::normParamGradients(graph, actsWhitened, gradsIn, prog,
                                    partialsType, debugPrefix);
}

std::pair<Tensor, Tensor>
batchNormParamGradients(Graph &graph, const Tensor &acts, const Tensor &gradsIn,
                        const Tensor &mean, const Tensor &iStdDev,
                        Sequence &prog, const Type &partialsType,
                        const std::string &debugPrefix) {
  checkTensorShape(gradsIn);
  checkTensorShape(acts);
  auto actsWhitened =
      batchNormWhiten(graph, acts, mean, iStdDev, prog, debugPrefix);
  return batchNormParamGradients(graph, actsWhitened, gradsIn, prog,
                                 partialsType, debugPrefix);
}

Tensor batchNormGradients(Graph &graph, const Tensor &actsWhitened_,
                          const Tensor &gradsIn_, const Tensor &iStdDev,
                          const Tensor &gamma, Sequence &prog,
                          const Type &partialsType,
                          const std::string &debugPrefix) {
  const auto rank = actsWhitened_.rank();
  checkTensorShape(actsWhitened_);
  checkTensorShape(gradsIn_);
  auto actsWhitened = preProcessNormActs(actsWhitened_);
  auto gradsIn = preProcessNormActs(gradsIn_);
  auto gradsNorm =
      poplin::normGradients(graph, gradsIn, gamma, prog, debugPrefix);
  auto gradsOut = poplin::normStatisticsGradients(
      graph, actsWhitened, gradsNorm, iStdDev, prog, partialsType, debugPrefix);
  return postProcessNormActs(gradsOut, rank);
}

Tensor batchNormGradients(Graph &graph, const Tensor &acts_,
                          const Tensor &gradsIn_, const Tensor &mean,
                          const Tensor &iStdDev, const Tensor &gamma,
                          Sequence &prog, const Type &partialsType,
                          const std::string &debugPrefix) {
  checkTensorShape(acts_);
  auto actsWhitened =
      batchNormWhiten(graph, acts_, mean, iStdDev, prog, debugPrefix);
  return batchNormGradients(graph, actsWhitened, gradsIn_, iStdDev, gamma, prog,
                            partialsType, debugPrefix);
}

void batchNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, float scale, Tensor &gamma,
                          Tensor &beta, Sequence &prog,
                          const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/BN/paramUpdate";
  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, fnPrefix);
}

void batchNormParamUpdate(Graph &graph, const Tensor &gammaDelta,
                          const Tensor &betaDelta, const Tensor &scale,
                          Tensor &gamma, Tensor &beta, Sequence &prog,
                          const std::string &debugPrefix) {
  const std::string fnPrefix = debugPrefix + "/BN/paramUpdate";
  // Do update of beta and gamma together
  scaledAddTo(graph, concat(beta, gamma), concat(betaDelta, gammaDelta), scale,
              prog, fnPrefix);
}

} // namespace bn
} // namespace popnn
