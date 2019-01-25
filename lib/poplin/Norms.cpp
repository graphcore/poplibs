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

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace popops;

namespace poplin {

static Tensor
normReduce(Graph &graph,
           const Tensor &actsUngrouped,
           float scale,
           bool doSquare,
           std::vector<ComputeSet> &css,
           const Type &partialsType,
           const Type &outputType,
           const std::string &debugPrefix) {
  // TODO: partialsType is unused.
  std::string name = debugPrefix + "/ReduceResult";

  // TODO: When this is moved to popnn, a function to create parameters to which
  // the result is written to may be needed. Alternately, we could just take
  // the output and then cast it down.
  auto t = createBiases(graph, actsUngrouped, name);

  if (actsUngrouped.elementType() != outputType) {
    t = graph.clone(outputType, t, name);
  }

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
  const auto &outputType = acts.elementType();

  std::vector<ComputeSet> css;

  auto mean =
      normReduce(graph, acts, 1.0f / numElements, false, css,
                 partialsType, outputType, fnPrefix + "/mean");
  // The actual output type for squared sum may be different as the dynamic
  // range is higher. The selection should be based on actual statistics
  // gathered from training experiments. For now keep it at reduced precision
  // to save memory
  auto power =
      normReduce(graph, acts, 1.0f / numElements, true, css,
                 partialsType, outputType, fnPrefix + "/power");

  for (const auto &cs : css) {
    prog.add(Execute(cs));
  }
  auto iStdDev = computeInvStdDev(graph, mean, power, eps, scaleVar, prog,
                                  acts.elementType(), debugPrefix);
  return std::make_pair(mean, iStdDev);
}

Tensor createNormGamma(Graph &graph,const Tensor &acts) {
  return createBiases(graph, acts, "gamma");
}

Tensor createNormBeta(Graph &graph, const Tensor &acts) {
  return createBiases(graph, acts, "beta");
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
  auto actsWhitened = duplicate(graph, acts, prog, fnPrefix + "/actsZeroMean");
  addToChannel(graph, actsWhitened, mean, -1.0, prog, fnPrefix + "/beta");
  actsWhitened =
    channelMul(graph, actsWhitened, iStdDev, prog, fnPrefix + "/istdDev");
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
  auto actsOut =
    channelMul(graph, actsWhitened, gamma, prog, fnPrefix + "/gamma");
  addToChannel(graph, actsOut, beta, 1.0, prog, fnPrefix + "/beta");
  return actsOut;
}

static std::pair<Tensor, Tensor>
normParamGradients(Graph &graph,
                   const Tensor &actsWhitened,
                   const Tensor &gradsIn,
                   float scale,
                   Sequence &prog,
                   const Type &partialsType,
                   const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/Norm/deltas";
  const auto gradsInMultActs =
    mul(graph, actsWhitened, gradsIn, prog, fnPrefix);

  auto numChannels = gradsInMultActs.dim(1);
  const auto concatInputs = concat({gradsInMultActs, gradsIn}, 1);

  std::vector<ComputeSet> css;

  // For beta = Re{gradsIn} where Re{x} reduces the tensor x along the
  //                              second dimension to produce a vector
  //                              of length x.dim(1)
  // For gamma = Re{actsWhitened .* gradsIn}
  //                              .* is element-wise multiplication operator
  //                              Reduction along second dimension
  const auto concatDeltas =
      normReduce(graph, concatInputs, scale, false, css, partialsType,
                 gradsIn.elementType(), fnPrefix + "/JointGammaDelta");

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
  return normParamGradients(graph, actsWhitened, gradsIn, 1.0, prog,
                            partialsType, debugPrefix);
}

Tensor normGradients(Graph &graph,
                     const Tensor &gradsIn,
                     const Tensor &gamma,
                     Sequence &prog,
                     const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/NormGrad";
  return channelMul(graph, gradsIn, gamma, prog, fnPrefix);
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
  const float rScale = 1.0 / numElements;

  auto gradient = graph.clone(actsWhitened, fnPrefix + "/gradsIn");
  Tensor varDelta, meanDelta;
  // See Description of Re{} operator in normParamGradients
  // varDelta = Re{actsWhitened .* gradsIn} * -rScale
  //   Size of varDelta is the size of inverse standard deviation
  // meanDelta = Re{gradsIn} * -rScale
  std::tie(varDelta, meanDelta) =
      normParamGradients(graph, actsWhitened, gradsIn, -rScale, prog,
                         partialsType, debugPrefix);
  prog.add(Copy(gradsIn, gradient));

  // gradOut = gradsIn - rScale * actsWhitened .* Br{varDelta}
  // where Br{x} broadcast x along all dimensions other than dim(1) of
  // actsWhitened
  // gradsOut = gradsIn - rScale * actsWhitened .* Br{varDelta} + Br{meanDelta}
  auto cs = graph.addComputeSet(debugPrefix + "/varGrads+meanGrads");
  auto varGrads = channelMul(graph, actsWhitened, varDelta, cs, fnPrefix);
  addToChannel(graph, gradient, meanDelta, 1.0, cs, fnPrefix);
  prog.add(Execute(cs));

  scaledAddTo(graph, gradient, varGrads, 1.0, prog, fnPrefix + "/addGrads");

  // Br{invStdDev} .* (gradsIn - rScale * actsWhitened .* Br{varDelta}
  //                   + Br{meanDelta})
  return channelMul(graph, gradient, invStdDev, prog, fnPrefix);
}

} // namespace poplin
