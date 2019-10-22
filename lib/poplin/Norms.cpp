#include "ChannelOps.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Convolution.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include <boost/functional/hash.hpp>
#include <cassert>
#include <cmath>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

static bool singletonSpatialDims(const Tensor &t) {
  std::size_t spatialDims;
  if (t.rank() > 2) {
    const auto tShape = t.shape();
    spatialDims = std::accumulate(tShape.begin() + 2, tShape.end(), 1ULL,
                                  std::multiplies<std::size_t>());
  } else {
    spatialDims = 1ULL;
  }
  return spatialDims == 1ULL;
}

namespace poplin {

// Create a variable of dimension {actsOrGrads.dim(1)} with start tile for the
// mapping a function of dimensions of \actsOrGrads.
static Tensor createAndMapParamOrReductionOutput(Graph &graph,
                                                 const Tensor &actsOrGrads,
                                                 const Type &type,
                                                 const std::string &name) {
  // Randomise the start tile for mapping of the variable
  std::size_t seed = 0x9e3779b9UL;
  const auto shape = actsOrGrads.shape();
  boost::hash_range(seed, shape.begin(), shape.end());
  const auto mappingGranularity = 4U;
  const auto &target = graph.getTarget();
  auto t = graph.addVariable(type, {shape[1]}, name);
  // TODO: use instrospection to map onto different IPUs
  mapTensorLinearly(graph, t, 0,
                    target.getDataPathWidth() / (8 * target.getTypeSize(type)));
  auto oldMapping = graph.getTileMapping(t);
  const auto numTiles = oldMapping.size();
  Graph::TileToTensorMapping newMapping(numTiles);
  std::size_t dstTile =
      ((seed / mappingGranularity) * mappingGranularity) % numTiles;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (!oldMapping[tile].empty())
      newMapping[dstTile] = std::move(oldMapping[tile]);
    if (++dstTile == numTiles) {
      dstTile = 0;
    }
  }
  graph.setTileMapping(t, newMapping);
  return t;
}

static Tensor normReduce(Graph &graph, const Tensor &actsUngrouped,
                         const Tensor &scale, bool doSquare,
                         std::vector<ComputeSet> &css,
                         const Type &, // partialsType,
                         const Type &outputType,
                         const std::string &debugPrefix) {
  std::string name = debugPrefix + "/ReduceResult";
  auto t = createAndMapParamOrReductionOutput(graph, actsUngrouped, outputType,
                                              name);

  if (actsUngrouped.rank() < 2)
    throw poplibs_error("NormReduce with rank " +
                        std::to_string(actsUngrouped.rank()) + " expected >=2");

  std::vector<std::size_t> reduceDims(actsUngrouped.rank() - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  popops::reduceWithOutput(
      graph, actsUngrouped, t, reduceDims,
      {doSquare ? popops::Operation::SQUARE_ADD : popops::Operation::ADD, false,
       scale},
      css, debugPrefix);
  return t;
}

static Tensor normReduce(Graph &graph, const Tensor &actsUngrouped, float scale,
                         bool doSquare, std::vector<ComputeSet> &css,
                         const Type &partialsType, const Type &outputType,
                         const std::string &debugPrefix) {
  auto constantScale =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/constantScale");
  graph.setTileMapping(constantScale, 0);

  return normReduce(graph, actsUngrouped, constantScale, doSquare, css,
                    partialsType, outputType, debugPrefix + "/ConstScale");
}

static Tensor computeInvStdDev(Graph &graph, const Tensor &mean,
                               const Tensor &power, float eps, float scaleVar,
                               Sequence &prog, const Type &invStdDevType,
                               const std::string debugPrefix) {
  const auto meanType = mean.elementType();
  const auto powerType = power.elementType();
  auto iStdDev = graph.clone(invStdDevType, mean, debugPrefix + "/iStdDev");

  const auto meanFlat = mean.flatten();
  const auto powerFlat = power.flatten();
  const auto iStdDevFlat = iStdDev.flatten();

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto cs = graph.addComputeSet(debugPrefix + "/iStdDev");

  const auto mapping = graph.getTileMapping(iStdDev);
  const auto grainSize = target.getVectorWidth(invStdDevType);

  for (auto tile = 0U; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(iStdDevFlat, mapping[tile]);
    auto vertexRegions = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v =
          graph.addVertex(cs,
                          templateVertex("poplin::InverseStdDeviation",
                                         meanType, powerType, invStdDevType),
                          {{"mean", meanFlat.slices(regions)},
                           {"power", powerFlat.slices(regions)},
                           {"iStdDev", iStdDevFlat.slices(regions)}});
      graph.setInitialValue(v["eps"], eps);
      graph.setInitialValue(v["scaleVar"], scaleVar);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
  return iStdDev;
}

std::pair<Tensor, Tensor> normStatistics(Graph &graph, const Tensor &acts,
                                         float eps, Sequence &prog,
                                         bool unbiasedVarEstimate,
                                         const Type &partialsType,
                                         const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/statistics";

  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(1);
  const float scaleVar =
      unbiasedVarEstimate ? static_cast<float>(numElements) / (numElements - 1)
                          : 1.0f;
  const auto powerOutputType = partialsType;
  const auto meanOutputType = acts.elementType();

  std::vector<ComputeSet> css;
  auto mean = normReduce(graph, acts, 1.0f / numElements, false, css,
                         partialsType, meanOutputType, fnPrefix + "/mean");
  // The actual output type for squared sum may be different as the dynamic
  // range is higher. The selection should be based on actual statistics
  // gathered from training experiments. For now keep it at reduced precision
  // to save memory
  auto power = normReduce(graph, acts, 1.0f / numElements, true, css,
                          partialsType, powerOutputType, fnPrefix + "/power");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }
  auto iStdDev = computeInvStdDev(graph, mean, power, eps, scaleVar, prog,
                                  acts.elementType(), debugPrefix);
  return std::make_pair(mean, iStdDev);
}

Tensor createNormGamma(Graph &graph, const Tensor &acts) {
  return createAndMapParamOrReductionOutput(graph, acts, acts.elementType(),
                                            "gamma");
}

Tensor createNormBeta(Graph &graph, const Tensor &acts) {
  return createAndMapParamOrReductionOutput(graph, acts, acts.elementType(),
                                            "beta");
}

std::pair<Tensor, Tensor> createNormParams(Graph &graph, const Tensor &acts) {
  // map beta and gamma the same way as biases
  auto gamma = createNormGamma(graph, acts);
  auto beta = createNormBeta(graph, acts);
  return std::make_pair(gamma, beta);
}

static Tensor broadcastChannelToMatch(const Tensor &ref, const Tensor &t) {
  return t.flatten().expand(std::vector<std::size_t>(ref.rank() - 2, 1));
}

Tensor normWhiten(Graph &graph, const Tensor &acts, const Tensor &mean,
                  const Tensor &iStdDev, Sequence &prog,
                  const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Whiten";

  auto meanBroadcast = broadcastChannelToMatch(acts, mean);
  auto actsWhitened = sub(graph, acts, meanBroadcast, prog, fnPrefix + "/mean");
  auto iStdDevBroadcast = broadcastChannelToMatch(actsWhitened, iStdDev);
  mulInPlace(graph, actsWhitened, iStdDevBroadcast, prog,
             fnPrefix + "/istdDev");
  return actsWhitened;
}

Tensor normalise(Graph &graph, const Tensor &actsWhitened, const Tensor &gamma,
                 const Tensor &beta, Sequence &prog,
                 const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/normalise";

  auto gammaBroadcast = broadcastChannelToMatch(actsWhitened, gamma);
  auto actsNormalised =
      mul(graph, actsWhitened, gammaBroadcast, prog, fnPrefix + "/gamma");
  auto betaBroadcast = broadcastChannelToMatch(actsNormalised, beta);
  addInPlace(graph, actsNormalised, betaBroadcast, prog, fnPrefix + "/beta");
  return actsNormalised;
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, float scale, Sequence &prog,
                   const Type &partialsType, bool attemptRegroup,
                   const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/Norm/deltas";
  auto gradsInMaybeRegrouped =
      attemptRegroup
          ? regroupIfBeneficial(graph, gradsIn, actsWhitened, prog, debugPrefix)
          : gradsIn;
  const auto gradsInMultActs =
      mul(graph, actsWhitened, gradsInMaybeRegrouped, prog, fnPrefix);

  auto numChannels = gradsInMultActs.dim(1);
  const auto concatInputs = concat({gradsInMultActs, gradsInMaybeRegrouped}, 1);

  std::vector<ComputeSet> css;

  // For beta = Re{gradsIn} where Re{x} reduces the tensor x along the
  //                              second dimension to produce a vector
  //                              of length x.dim(1)
  // For gamma = Re{actsWhitened .* gradsIn}
  //                              .* is element-wise multiplication operator
  //                              Reduction along second dimension

  auto scaleTensor =
      graph.addConstant(FLOAT, {}, scale, debugPrefix + "/scaleTensor");
  graph.setTileMapping(scaleTensor, 0);
  const auto concatDeltas = normReduce(
      graph, concatInputs, scaleTensor, false, css, partialsType,
      gradsInMaybeRegrouped.elementType(), fnPrefix + "/JointGammaDelta");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  return std::make_pair(concatDeltas.slice(0, numChannels),
                        concatDeltas.slice(numChannels, 2 * numChannels));
}

std::pair<Tensor, Tensor>
normParamGradients(Graph &graph, const Tensor &actsWhitened,
                   const Tensor &gradsIn, Sequence &prog,
                   const Type &partialsType, const std::string &debugPrefix) {
  return normParamGradients(graph, actsWhitened, gradsIn, 1.0, prog,
                            partialsType, true, debugPrefix);
}

Tensor normGradients(Graph &graph, const Tensor &gradsIn, const Tensor &gamma,
                     Sequence &prog, const std::string &debugPrefix) {
  auto gammaBroadcast = broadcastChannelToMatch(gradsIn, gamma);
  return mul(graph, gradsIn, gammaBroadcast, prog, debugPrefix + "/NormGrad");
}

Tensor normStatisticsGradients(Graph &graph, const Tensor &actsWhitened,
                               const Tensor &gradsIn, const Tensor &invStdDev,
                               Sequence &prog,
                               const Type &partialsType, // currently unused
                               const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/gradients";
  const auto actsShape = actsWhitened.shape();
  const auto numElements = actsWhitened.numElements() / actsWhitened.dim(1);
  const float rScale = 1.0f / numElements;

  auto gradsInMaybeRegrouped =
      regroupIfBeneficial(graph, gradsIn, actsWhitened, prog, debugPrefix);

  // split rScale = rScale1 * rScale2;
  // TODO: This split should actually be found by the research team (dependence
  // on model and field size)
  const auto scaleSplit = 3.0f / 4;
  const float rScale1 = std::pow(rScale, scaleSplit);
  const float rScale2 = rScale / rScale1;

  Tensor varDelta, meanDelta;
  // See Description of Re{} operator in normParamGradients
  // varDelta = Re{actsWhitened .* gradsIn} * -rScale
  //   Size of varDelta is the size of inverse standard deviation
  // meanDelta = Re{gradsIn} * -rScale
  std::tie(varDelta, meanDelta) =
      normParamGradients(graph, actsWhitened, gradsInMaybeRegrouped, -rScale1,
                         prog, partialsType, false, debugPrefix);

  auto gradient = graph.clone(actsWhitened, fnPrefix + "/gradsIn");
  prog.add(Copy(gradsInMaybeRegrouped, gradient));

  // gradOut = gradsIn - rScale * actsWhitened .* Br{varDelta}
  // where Br{x} broadcast x along all dimensions other than dim(1) of
  // actsWhitened
  // gradsOut = gradsIn - rScale * actsWhitened .* Br{varDelta} + Br{meanDelta}

  const auto singletonDims = singletonSpatialDims(actsWhitened);

  auto varDeltaBroadcast = broadcastChannelToMatch(actsWhitened, varDelta);
  auto varGrads =
      mul(graph, actsWhitened, varDeltaBroadcast, prog, fnPrefix + "/varGrads");
  mulInPlace(graph, meanDelta, rScale2, prog, fnPrefix + "/scaleMeanDelta");
  auto meanDeltaBroadcast = broadcastChannelToMatch(gradient, meanDelta);
  addInPlace(graph, gradient, meanDeltaBroadcast, prog,
             fnPrefix + "/meanGrads");
  // TODO: Once scaledAddTo is targeted efficiently in element-wise ops,
  // this should become a mapInPlace() expression.
  scaledAddTo(graph, gradient, varGrads, rScale2, prog, fnPrefix + "/addGrads");

  // Br{invStdDev} .* (gradsIn - rScale * actsWhitened .* Br{varDelta}
  //                   + Br{meanDelta})
  auto invStdDevBroadcast = broadcastChannelToMatch(gradient, invStdDev);
  mulInPlace(graph, gradient, invStdDevBroadcast, prog, fnPrefix);
  return gradient;
}

} // namespace poplin
