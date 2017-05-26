#include "popnn/NonLinearity.hpp"
#include "popstd/ActivationMapping.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/Regroup.hpp"
#include "popstd/Util.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popnn {

Tensor
nonLinearityInputGradient(Graph &graph,
                          NonLinearityType nonLinearityType,
                          Tensor out, Tensor outGradient,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix) {
  auto cs = graph.addComputeSet(debugPrefix + "/NonLinearityGrad");
  const auto dType = out.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  auto inGradient = graph.addTensor(dType, outGradient.shape(),
                                    debugPrefix + "/NonLinearityGrad");
  graph.setTileMapping(inGradient, graph.getTileMapping(outGradient));

  Tensor outRegrouped;
  // TODO: This could possible be made more efficient by merging the
  // regrouping with the calculation of the non linearity derivative.
  if (out.rank() == 2 ||
      out.dim(4) == out.dim(4)) {
    outRegrouped = out;
  } else {
    outRegrouped = graph.addTensor(dType, outGradient.shape(), "regroupedActs");
    mapActivations(graph, outRegrouped);
    prog.add(Copy(regroup(out, outGradient.dim(4)), outRegrouped));
  }
  auto outFlat = outRegrouped.flatten();
  auto outGradFlat = outGradient.flatten();
  auto inGradFlat = inGradient.flatten();
  auto inGradMapping = graph.getTileMapping(inGradFlat);
  const auto numTiles = deviceInfo.getNumTiles();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, inGradMapping[tile],
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

  prog.add(Execute(cs));
  return inGradient;
}

void nonLinearity(Graph &graph, NonLinearityType nonLinearityType,
                  Tensor t, Sequence &prog, const std::string &debugPrefix) {
  const auto dType = t.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/Nonlinearity");
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
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                   vectorWidth, 2 * vectorWidth);
    for (const auto &regions : vertexRegions) {
      auto v = graph.addVertex(cs, templateVertex("popnn::NonLinearity",
                                                  dType),
                               {{"data", tFlat.slices(regions)}});
      graph.setInitialValue(v["nonLinearityType"], nonLinearityType);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));
}

} // end namespace popnn
