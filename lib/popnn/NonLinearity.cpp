#include "popnn/NonLinearity.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include "poputil/VertexTemplates.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "poputil/Util.hpp"
using namespace poplar;
using namespace poplar::program;
using namespace poputil;


// computes softmax along the innermost dimension
// This is not an optimal implementation in terms of precision and order of
// operations
static Tensor softmaxImpl(Graph &graph, Tensor t, Sequence &prog,
                          const std::string &debugStr = "") {
  // exchange innermost dimension as softmax is done over it
  const auto rank = t.rank();
  auto tShuf = t.dimShufflePartial({0}, {rank - 1});

  const auto fnStr = debugStr + "/SoftMax";
  auto e = popops::exp(graph, tShuf, prog, fnStr);
  auto r =
    popops::reduce(graph, e, {0}, popops::Operation::ADD, prog, fnStr);

  auto rShuf = r.expand({0}).broadcast(t.dim(rank - 1), 0);
  auto outShuf = popops::div(graph, e, rShuf, prog, fnStr);

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
    throw poputil::poplib_error("SOFTMAX gradient not implemented");
  }
  const auto dType = out.elementType();
  const auto &target = graph.getTarget();
  auto inGradient = graph.clone(outGradient, debugPrefix + "/NonLinearityGrad");
  auto outFlat = out.flatten();
  auto outGradFlat = outGradient.flatten();
  auto inGradFlat = inGradient.flatten();
  graph.reorderToSimplify(&inGradFlat, {&outFlat, &outGradFlat});
  auto outGradMapping = graph.getTileMapping(outGradFlat);
  const auto numTiles = target.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = target.getVectorWidth(dType);
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(outGradFlat, outGradMapping[tile]);
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      VertexRef v;
      if (regions.size() == 1 && regions[0].size() == 1) {
        const auto region = regions[0][0];
        v = graph.addVertex(cs,
                            templateVertex("popnn::NonLinearityGrad", dType,
                                       static_cast<unsigned>(nonLinearityType)),
                            {{"out", outFlat.slice(region)},
                             {"outGrad", outGradFlat.slice(region)},
                             {"inGrad", inGradFlat.slice(region)}});
        graph.setInitialValue(v["n"], region.size());

      } else {
        v = graph.addVertex(cs,
                            templateVertex("popnn::NonLinearityGrad2D", dType,
                                       static_cast<unsigned>(nonLinearityType)),
                            {{"out", outFlat.slices(regions)},
                             {"outGrad", outGradFlat.slices(regions)},
                             {"inGrad", inGradFlat.slices(regions)}});
      }
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
    throw poputil::poplib_error("Compute set variant of softmax not "
                               "implemented");
  }
  if (!t.isParallelWriteable())
    throw poputil::poplib_error("Trying to update tensor that cannot be "
                               "written in parallel");
  t = t.flatten();
  graph.reorderToSimplify(&t, {});
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  auto mapping = graph.getTileMapping(t);
  const auto numTiles = target.getNumTiles();
  const auto tFlat = t.flatten();
  const auto vectorWidth = target.getVectorWidth(dType);
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
        numElements <= vectorWidth * target.getNumWorkerContexts() ? 1 : 2;
    auto vertexRegions =
        splitRegionsBetweenWorkers(target, tileContiguousRegions,
                                   vectorWidth, minVectors * vectorWidth);

    for (const auto &regions : vertexRegions) {
      VertexRef v;
      if (regions.size() == 1 && regions[0].size() == 1) {
        const auto region = regions[0][0];
        v = graph.addVertex(cs,
                            templateVertex("popnn::NonLinearity", dType,
                                       static_cast<unsigned>(nonLinearityType)),
                            {{"data", tFlat.slice(region)}});
        graph.setInitialValue(v["n"], region.size());

      } else {
        v =
          graph.addVertex(cs,
                          templateVertex("popnn::NonLinearity2D",dType,
                                       static_cast<unsigned>(nonLinearityType)),
                          {{"data", tFlat.slices(regions)}});
      }
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
