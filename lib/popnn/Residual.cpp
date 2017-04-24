#include "popnn/Residual.hpp"

#include "popstd/ActivationMapping.hpp"
#include "popconv/Convolution.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/Regroup.hpp"
#include "popstd/Pad.hpp"
#include "util/gcd.hpp"
#include "popstd/Util.hpp"

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popnn {

std::uint64_t getNumberOfAdds(unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans) {
  // An addition is required to add the residual information
  return inNumChans * inDimX * inDimY;
}

uint64_t getFlops(unsigned batchSize,
                  unsigned inDimY, unsigned inDimX, unsigned inNumChans) {
  auto flopsPerItem = getNumberOfAdds(inDimY, inDimX, inNumChans);
  return batchSize * flopsPerItem;
}

double getPerfectCycleCount(const Graph &graph,
                            std::string dType, unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getFlops(batchSize, inDimY, inDimX, numChannels);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}


Tensor
arrangeResidualInput(Graph &graph,
                     Tensor resIn,
                     std::vector<size_t> outDims,
                     std::string dType,
                     ResidualMethod resMethod) {
  auto resDimY = resIn.dim(2), outDimY = outDims[2];
  auto resDimX = resIn.dim(3), outDimX = outDims[3];
  if (resDimX < outDimX || resDimY < outDimY) {
    throw popstd::poplib_error("Residual layers must use previous layers "
                               "with X and Y dimensions that are larger"
                               "than the current layer's output.");
  }
  unsigned resStride = resDimX / outDimX;
  if (resDimY / outDimY != resStride) {
    throw popstd::poplib_error("Only residual layers with the same X/Y stride"
                               "are supported");
  }
  auto resNumChanGroups = resIn.dim(1), outNumChanGroups = outDims[1];
  auto resChansPerGroup = resIn.dim(4), outChansPerGroup = outDims[4];
  auto resNumChans = resNumChanGroups * resChansPerGroup;
  auto outNumChans = outNumChanGroups * outChansPerGroup;
  if (resMethod == RESIDUAL_PAD &&
      resNumChans == outNumChans &&
      resNumChanGroups == outNumChanGroups) {
    // We can directly add the output of the previous layer to this
    // layer's output.
    return resIn;
  }

  Tensor residual;

  switch (resMethod) {
  case RESIDUAL_PAD:
    {
      Tensor resampled = resIn.subSample(resDimY / outDimY, 2)
                              .subSample(resDimX / outDimX, 3);
      // pad channel depth to match out then regroup to match it
      Tensor zPadded = pad(graph,
                           regroup(resampled, resNumChans),
                                   {resampled.dim(0), 1, resampled.dim(2),
                                    resampled.dim(3), outNumChans},
                                   {0, 0, 0, 0, 0});
      residual = regroup(zPadded, outNumChans / outNumChanGroups);
    }
    break;
  case RESIDUAL_CONCATENATE:
    assert(0 && "Weighted calculation of residual input not implemented");
    break;
  default:
    assert(0 && "Unknown residual calculation method");
  }
  return residual;
}

Program
joinResidual(Graph &graph,
             Tensor in0,
             Tensor in1,
             Tensor out,
             const std::string &debugPrefix) {
  assert(in0.rank() == 5); //[batch][nCG][y][g][chan]
  assert(in0.shape() == out.shape());
  assert(in1.shape() == in0.shape());

  const auto outType = graph.getTensorElementType(out);
  const auto inType = graph.getTensorElementType(in0);
  assert(inType == graph.getTensorElementType(in1));
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/JoinResidual");
  Program prog = Execute(cs);

  unsigned vectorWidth = (outType == "float")
                          ? deviceInfo.getFloatVectorWidth()
                          : deviceInfo.getHalfVectorWidth();
  Tensor flattenedIn0 = in0.flatten();
  Tensor flattenedIn1 = in1.flatten();
  Tensor flattenedOut = out.flatten();
  buildTransform2D(graph, graph.getTileMapping(out), vectorWidth,
                   [&](const std::vector<Interval<std::size_t>> &regions,
                       unsigned tile)
  {
    for (auto &region : regions) {
      const auto regionBegin = region.begin();
      const auto regionEnd = region.end();
      auto v = graph.addVertex(
          cs,
          templateVertex("popnn::AddTensors", inType, outType),
          {{"in0", flattenedIn0.slice(regionBegin, regionEnd)},
           {"in1", flattenedIn1.slice(regionBegin, regionEnd)},
           {"out", flattenedOut.slice(regionBegin, regionEnd)}});
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  });
  return prog;
}

}
