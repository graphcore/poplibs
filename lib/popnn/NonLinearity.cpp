#include "popnn/NonLinearity.hpp"
#include "popstd/TileMapping.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/Operations.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/Util.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popstd;


// computes softmax along the innermost dimension
// This is not an optimal implementation in terms of precision and order of
// operations
static Tensor softmaxImpl(Graph &graph, Tensor t, Sequence &prog,
                          const std::string &debugStr = "") {
  // exchange innermost dimension as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0}, {rank - 1});

  const auto fnStr = debugStr + "/SoftMax";
  auto e = exp(graph, tShuf, prog, fnStr);
  auto r =
    popreduce::reduce(graph, e, {0}, popreduce::Operation::ADD, prog, fnStr);

  auto rShuf = r.expand({0}).broadcast(t.dim(rank - 1), 0);
  auto outShuf = div(graph, e, rShuf, prog, fnStr);

  return outShuf.dimShufflePartial({0}, {rank - 1});
}

namespace popnn {

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          ComputeSet &cs,
                          const std::string &debugPrefix) {
  if (nonLinearityType == NON_LINEARITY_SOFTMAX) {
    throw popstd::poplib_error("SOFTMAX gradient not implemented");
  }
  const auto dType = out.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto inGradient = graph.clone(outGradient, debugPrefix + "/NonLinearityGrad");
  auto outFlat = out.flatten();
  auto outGradFlat = outGradient.flatten();
  auto inGradFlat = inGradient.flatten();
  auto outGradMapping = graph.getTileMapping(outGradFlat);
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outFlat, outGradMapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, templateVertex("popnn::NonLinearityGrad",
                                                  dType),
                               {{"out", outFlat.slices(regions)},
                                {"outGrad", outGradFlat.slices(regions)},
                                {"inGrad", inGradFlat.slices(regions)}});
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  return inGradient;
}

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix) {

  auto cs = graph.addComputeSet(debugPrefix + "/NonLinearityGrad");
  auto t = nonLinearityInputGradient(graph, nonLinearityType, out, outGradient,
                                     cs, debugPrefix);
  prog.add(Execute(cs));
  return t;
}


void nonLinearity(poplar::Graph &graph, NonLinearityType nonLinearityType,
                  poplar::Tensor t, ComputeSet &cs,
                  const std::string &debugPrefix) {
  if (nonLinearityType == NON_LINEARITY_SOFTMAX) {
    throw popstd::poplib_error("Compute set variant of softmax not "
                               "implemented");
  }
  if (!t.isParallelWriteable())
    throw popstd::poplib_error("Trying to update tensor that cannot be "
                               "written in parallel");
  const auto dType = t.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto mapping = graph.getTileMapping(t);
  const auto numTiles = deviceInfo.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(t, mapping[tile]);

    auto numElements = intervalSequenceNumElements(tileContiguousRegions);
    auto minVectors =
        numElements <= vectorWidth * deviceInfo.numWorkerContexts ? 1 : 2;
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                   vectorWidth, minVectors * vectorWidth);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, templateVertex("popnn::NonLinearity",
                                                  dType),
                               {{"data", tFlat.slices(regions)}});
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
}

void nonLinearity(Graph &graph, NonLinearityType nonLinearityType,
                  Tensor t, Sequence &prog, const std::string &debugPrefix) {
  if (nonLinearityType == NON_LINEARITY_SOFTMAX) {
    auto out = softmaxImpl(graph, t, prog, debugPrefix);
    prog.add(Copy(out, t));
    return;
  }
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/Nonlinearity");
  nonLinearity(graph,nonLinearityType,t,cs,debugPrefix);
  prog.add(Execute(cs));
}

} // end namespace popnn
