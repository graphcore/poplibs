#include <cassert>
#include <cmath>
#include "popops/ScaledAdd.hpp"
#include "poputil/TileMapping.hpp"
#include "popops/Reduce.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/Util.hpp"
#include "popops/ElementWise.hpp"
#include "ChannelOps.hpp"
#include "poplin/Convolution.hpp"
#include "poplin/ConvUtil.hpp"
#include <boost/functional/hash.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

static bool singletonSpatialDims(const Tensor &t) {
  std::size_t spatialDims;
  if (t.rank() > 2) {
    const auto tShape = t.shape();
    spatialDims = std::accumulate(tShape.begin() + 2, tShape.end(),
                                  1ULL, std::multiplies<std::size_t>());
  } else {
    spatialDims = 1ULL;
  }
  return spatialDims == 1ULL;
}

namespace poplin {

// Create a variable of dimension {actsOrGrads.dim(1)} with start tile for the
// mapping a function of dimensions of \actsOrGrads.
static Tensor
createAndMapParamOrReductionOutput(Graph &graph,
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

static Tensor
normReduce(Graph &graph,
           const Tensor &actsUngrouped,
           float scale,
           bool doSquare,
           std::vector<ComputeSet> &css,
           const Type &, //partialsType,
           const Type &outputType,
           const std::string &debugPrefix) {
  std::string name = debugPrefix + "/ReduceResult";
  auto t = createAndMapParamOrReductionOutput(graph, actsUngrouped, outputType,
                                              name);

  if (actsUngrouped.rank() < 2)
    throw poplibs_error("NormReduce with rank " +
                         std::to_string(actsUngrouped.rank()) +
                         " expected >=2");

  std::vector<std::size_t> reduceDims(actsUngrouped.rank()-1);
  std::iota(reduceDims.begin()+1, reduceDims.end(), 2);

  popops::reduceWithOutput(graph, actsUngrouped, t, reduceDims, {
                             doSquare ? popops::Operation::SQUARE_ADD
                                      : popops::Operation::ADD,
                             scale
                           }, css, debugPrefix);
  return t;
}

static Tensor computeInvStdDev(Graph &graph, const Tensor &mean,
                               const Tensor &power, float eps,
                               float scaleVar,
                               Sequence &prog,
                               const Type &invStdDevType,
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
    auto vertexRegions =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                 grainSize, 2 * grainSize);

    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs,
                               templateVertex("poplin::InverseStdDeviation",
                                              meanType, powerType,
                                              invStdDevType),
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

std::pair<Tensor, Tensor>
normStatistics(Graph &graph,
               const Tensor &acts,
               float eps,
               Sequence &prog,
               bool unbiasedVarEstimate,
               const Type &partialsType,
               const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/statistics";

  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(1);
  const float scaleVar = unbiasedVarEstimate ?
      static_cast<float>(numElements) / (numElements - 1) : 1.0f;
  const auto powerOutputType = partialsType;
  const auto meanOutputType = acts.elementType();

  std::vector<ComputeSet> css;

  auto mean =
      normReduce(graph, acts, 1.0f / numElements, false, css,
                 partialsType, meanOutputType, fnPrefix + "/mean");
  // The actual output type for squared sum may be different as the dynamic
  // range is higher. The selection should be based on actual statistics
  // gathered from training experiments. For now keep it at reduced precision
  // to save memory
  auto power =
      normReduce(graph, acts, 1.0f / numElements, true, css,
                 partialsType, powerOutputType, fnPrefix + "/power");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }
  auto iStdDev = computeInvStdDev(graph, mean, power, eps, scaleVar, prog,
                                  acts.elementType(), debugPrefix);
  return std::make_pair(mean, iStdDev);
}

Tensor createNormGamma(Graph &graph,const Tensor &acts) {
  return createAndMapParamOrReductionOutput(graph, acts, acts.elementType(),
                                            "gamma");
}

Tensor createNormBeta(Graph &graph, const Tensor &acts) {
  return createAndMapParamOrReductionOutput(graph, acts, acts.elementType(),
                                            "beta");
}

std::pair<Tensor, Tensor>
createNormParams(Graph &graph, const Tensor &acts) {
  // map beta and gamma the same way as biases
  auto gamma = createNormGamma(graph, acts);
  auto beta = createNormBeta(graph, acts);
  return std::make_pair(gamma, beta);
}

Tensor
normWhiten(Graph &graph,
           const Tensor &acts,
           const Tensor &mean,
           const Tensor &iStdDev,
           Sequence &prog,
           const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Whiten";

  Tensor actsWhitened;

  // When T4987 is fixed, the special casing for singleton spatial dimensions
  // may be removed. We could check for grouping of the acts and addend tensor
  // to decide on using one or the other but is not done because T4987 should
  // do this anyway.
  if (singletonSpatialDims(acts)) {
    actsWhitened =
        popops::sub(graph, acts, mean.broadcast(acts.dim(0), 0)
                                     .reshape(acts.shape()), prog,
                    fnPrefix + "/mean");
    mulInPlace(graph, actsWhitened, iStdDev.broadcast(acts.dim(0), 0)
                                           .reshape(acts.shape()), prog,
               fnPrefix + "/iStdDev");
  } else {
    actsWhitened = duplicate(graph, acts, prog, fnPrefix + "/actsZeroMean");

    addToChannel(graph, actsWhitened, mean, -1.0, prog, fnPrefix + "/mean");
    actsWhitened =
      channelMul(graph, actsWhitened, iStdDev, prog, fnPrefix + "/istdDev");
  }
  return actsWhitened;
}

Tensor
normalise(Graph &graph,
          const Tensor &actsWhitened,
          const Tensor &gamma,
          const Tensor &beta,
          Sequence &prog,
          const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/Norm/normalise";

  Tensor actsNormalised;

  // When T4987 is fixed, the special casing for singleton spatial dimensions
  // may be removed. We could check for grouping of the acts, beta, and gamma
  // tensor to decide on using one or the other but is not done because T4987
  // should do this anyway.
  if (singletonSpatialDims(actsWhitened)) {
    const auto actsShape = actsWhitened.shape();
    const auto dim0 = actsWhitened.dim(0);
    actsNormalised =
          popops::mul(graph, actsWhitened, gamma.broadcast(dim0, 0)
                                                .reshape(actsShape),
                      prog, fnPrefix + "/gamma");
    popops::scaledAddTo(graph, actsNormalised, beta.broadcast(dim0, 0)
                                           .reshape(actsShape),
                        1.0, prog, fnPrefix + "/beta");
   } else {
    actsNormalised =
      channelMul(graph, actsWhitened, gamma, prog, fnPrefix + "/gamma");
    addToChannel(graph, actsNormalised, beta, 1.0, prog, fnPrefix + "/beta");
  }
  return actsNormalised;
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph,
                   const Tensor &actsWhitened,
                   const Tensor &gradsIn,
                   float scale,
                   Sequence &prog,
                   const Type &partialsType,
                   bool attemptRegroup,
                   const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/Norm/deltas";
  auto gradsInMaybeRegrouped = attemptRegroup ?
      regroupIfBeneficial(graph, gradsIn, actsWhitened, prog, debugPrefix) :
      gradsIn;
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
  const auto concatDeltas =
      normReduce(graph, concatInputs, scale, false, css, partialsType,
                 gradsInMaybeRegrouped.elementType(),
                 fnPrefix + "/JointGammaDelta");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }

  return std::make_pair(concatDeltas.slice(0, numChannels),
                        concatDeltas.slice(numChannels, 2 * numChannels));
}

std::pair<Tensor, Tensor>
normParamGradients(Graph &graph,
                   const Tensor &actsWhitened,
                   const Tensor &gradsIn,
                   Sequence &prog,
                   const Type &partialsType,
                   const std::string &debugPrefix) {
  return normParamGradients(graph, actsWhitened, gradsIn, 1.0,
                            prog, partialsType, true, debugPrefix);
}

Tensor normGradients(Graph &graph,
                     const Tensor &gradsIn,
                     const Tensor &gamma,
                     Sequence &prog,
                     const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/NormGrad";

  // When T4987 is fixed, the special casing for singleton spatial dimensions
  // may be removed. We could check for grouping of the gradsIn and gamma tensor
  // to decide on using one or the other but is not done because T4987 should
  // do this anyway.
  if (singletonSpatialDims(gradsIn)) {
    return popops::mul(graph, gradsIn, gamma.broadcast(gradsIn.dim(0), 0)
                                            .reshape(gradsIn.shape()),
                       prog, fnPrefix);

  } else {
     return channelMul(graph, gradsIn, gamma, prog, fnPrefix);
  }
}

Tensor normStatisticsGradients(Graph &graph,
                     const Tensor &actsWhitened,
                     const Tensor &gradsIn,
                     const Tensor &invStdDev,
                     Sequence &prog,
                     const Type &partialsType, //currently unused
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
  const auto scaleSplit = 3.0f/4;
  const float rScale1 = std::pow(rScale, scaleSplit);
  const float rScale2 = rScale / rScale1;

  auto gradient = graph.clone(actsWhitened, fnPrefix + "/gradsIn");
  Tensor varDelta, meanDelta;
  // See Description of Re{} operator in normParamGradients
  // varDelta = Re{actsWhitened .* gradsIn} * -rScale
  //   Size of varDelta is the size of inverse standard deviation
  // meanDelta = Re{gradsIn} * -rScale
  std::tie(varDelta, meanDelta) =
      normParamGradients(graph, actsWhitened, gradsInMaybeRegrouped, -rScale1,
                         prog, partialsType, false, debugPrefix);
  prog.add(Copy(gradsInMaybeRegrouped, gradient));

  // gradOut = gradsIn - rScale * actsWhitened .* Br{varDelta}
  // where Br{x} broadcast x along all dimensions other than dim(1) of
  // actsWhitened
  // gradsOut = gradsIn - rScale * actsWhitened .* Br{varDelta} + Br{meanDelta}

  auto cs = graph.addComputeSet(debugPrefix + "/varGrads+meanGrads");

  Tensor varGrads;
  const auto singletonDims = singletonSpatialDims(actsWhitened);

  // When T4987 is fixed, the special casing for singleton spatial dimensions
  // may be removed. We could check for grouping of the acts and
  // varGrads/meanDelta tensor to decide on using one or the other but is not
  // done because T4987 should do this anyway.
  if (singletonDims) {
    const auto dim0 = actsWhitened.dim(0);
    varGrads = popops::mul(graph, actsWhitened,
                           varDelta.broadcast(dim0, 0)
                                   .reshape(actsShape), prog, fnPrefix);
    popops::scaledAddTo(graph, gradient,
                        meanDelta.broadcast(dim0, 0)
                                 .reshape(actsShape), rScale2, prog, fnPrefix);
  } else {
    varGrads = channelMul(graph, actsWhitened, varDelta, cs, fnPrefix);
    addToChannel(graph, gradient, meanDelta, rScale2, cs, fnPrefix);
  }
  prog.add(Execute(cs));

  scaledAddTo(graph, gradient, varGrads, rScale2, prog, fnPrefix + "/addGrads");

  // Br{invStdDev} .* (gradsIn - rScale * actsWhitened .* Br{varDelta}
  //                   + Br{meanDelta})
  // When T4987 is fixed, the special casing for singleton spatial dimensions
  // may be removed. We could check for grouping of the gradient and invStdDev
  // tensor to decide on using one or the other but is not
  // done because T4987 should do this anyway.
  if (singletonSpatialDims(gradient)) {
    const auto dim0 = gradient.dim(0);
    const auto gradShape = gradient.shape();
    popops::mulInPlace(graph, gradient, invStdDev.broadcast(dim0, 0)
                                                 .reshape(gradShape),
                       prog, fnPrefix);
  } else {
    gradient = channelMul(graph, gradient, invStdDev, prog, fnPrefix);
  }
  return gradient;
}

} // namespace poplin
