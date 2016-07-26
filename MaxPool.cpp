#include "MaxPool.hpp"
#include "VertexTemplates.hpp"
#include "ActivationMapping.hpp"
#include "ConvUtil.hpp"
#include "gcd.hpp"
#include "exceptions.hpp"
#include <cassert>

using namespace poplar;
using namespace poplar::program;

namespace maxpool {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSize,
             unsigned stride, unsigned padding) {
  unsigned outDimX = (inDimX + (padding * 2) - kernelSize) / stride + 1;
  unsigned outDimY = (inDimY + (padding * 2) - kernelSize) / stride + 1;
  return {outDimY, outDimX};
}

uint64_t getNumFlops(unsigned inDimY, unsigned inDimX,
                     unsigned numChannels, unsigned kernelSize,
                     unsigned stride, unsigned padding) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX,
                                            kernelSize, stride, padding);
  std::uint64_t numFlops = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                               padding, inDimY, true);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, inDimX, true);
      const auto width = inXEnd - inXBegin;
      numFlops += width * height;
    }
  }
  return numFlops;
}

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            std::string dType,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels, unsigned kernelSize,
                            unsigned stride, unsigned padding) {
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getNumFlops(inDimY, inDimX, numChannels, kernelSize,
                                    stride, padding);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

Program
maxPool(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping,
        DeviceInfo &deviceInfo,
        unsigned kernelSize, unsigned stride, unsigned padding,
        std::string dType,
        Tensor in, Tensor out) {
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  unsigned prevChanGroups = in.dim(0);
  unsigned numChannels = in.dim(0) * in.dim(3);
  unsigned numChanGroups = out.dim(0);
  unsigned prevChansPerGroup = numChannels / prevChanGroups;
  unsigned chansPerGroup = numChannels / numChanGroups;
  unsigned inDimY = in.dim(1), inDimX = in.dim(2);
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX,
                                            kernelSize, stride, padding);
  assert(outDimY == out.dim(1));
  assert(outDimX == out.dim(2));
  const auto chunkSize = gcd<unsigned>(prevChansPerGroup, chansPerGroup);
  const auto chunksPerChanGroup = chansPerGroup / chunkSize;
  ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
  const auto outMapping = computeActivationsMapping(out, deviceInfo);
  const auto numTiles = deviceInfo.getNumIPUs() * deviceInfo.getTilesPerIPU();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileOutBegin = outMapping[tile];
    const auto tileOutEnd = outMapping[tile + 1];
    assert(tileOutBegin % chansPerGroup == 0);
    assert(tileOutEnd % chansPerGroup == 0);
    const auto tileGroupBegin = tileOutBegin / chansPerGroup;
    const auto tileGroupEnd = tileOutEnd / chansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    for (unsigned i = tileGroupBegin; i != tileGroupEnd; ++i) {
      unsigned y = i % outDimY;
      unsigned x = (i / outDimY) % outDimX;
      unsigned chanGroup = i / (outDimY * outDimX);
      unsigned inYBegin, inYEnd;
      std::tie(inYBegin, inYEnd) = getInputRange(y, stride, kernelSize,
                                                 padding, inDimY, true);
      const auto inYSize = inYEnd - inYBegin;
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, stride, kernelSize,
                                                 padding, inDimX, true);
      const auto inXSize = inXEnd - inXBegin;
      auto v =
        graph.addVertex(fwd, templateVertex("MaxPooling", dType),
          { {"activationOut", out[chanGroup][y][x]} });
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      mapping.setMapping(v, tile);
      graph.setFieldSize(v["activationIn"],
                         chunksPerChanGroup * inYSize * inXSize);
      for (unsigned j = 0; j != chunksPerChanGroup; ++j) {
        unsigned chan = chanGroup * chansPerGroup + j * chunksPerChanGroup;
        unsigned prevChanGroup = chan / prevChansPerGroup;
        unsigned prevChanInGroup = chan % prevChansPerGroup;
        for (unsigned inY = inYBegin; inY != inYEnd; ++inY) {
          for (unsigned inX = inXBegin; inX != inXEnd; ++inX) {
            Tensor inWindow =
                in[prevChanGroup][inY][inX].slice(prevChanInGroup,
                                                  prevChanInGroup + chunkSize);
            const auto chunkIndex =
                (inX - inXBegin) + inXSize * ((inY - inYBegin) + inYSize * j);
            graph.connect(inWindow, v["activationIn"][chunkIndex]);
          }
        }
      }
    }
  }
  return Execute(fwd);
}

Program
maxPoolBackward(Graph &graph, IPUModelEngineBuilder::TileMapping &mapping,
                DeviceInfo &deviceInfo,
                unsigned kernelSize, unsigned stride, unsigned padding,
                std::string dType, Tensor actIn, Tensor actOut,
                Tensor deltasIn, Tensor deltasOut) {
  // Currently this code does not target the IPU model (no tile mapping)
  // but is just used for the purely functional tests
  if (deviceInfo.isIPU)
    return Sequence();
  const auto numChanGroups = deltasIn.dim(0);
  const auto chansPerGroup = deltasIn.dim(3);
  const auto numChannels = numChanGroups * chansPerGroup;
  const auto layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  const auto yDimOut = deltasIn.dim(1);
  const auto xDimOut = deltasIn.dim(2);
  const auto xDim = actIn.dim(2);
  const auto yDim = actIn.dim(1);
  if (padding != 0 || xDim % stride != 0 || yDim % stride != 0 ||
      kernelSize != stride) {
    throw net_creation_error("padding, overlapped pooling or pooling that "
                             "doesn't exactly divide "
                             "input not implemented yet");
  }
  for (unsigned xIn = 0; xIn < xDim; ++xIn) {
    const auto xOut = xIn / stride;
    assert(xOut < xDimOut);
    for (unsigned yIn = 0; yIn < yDim; ++yIn) {
      const auto yOut = yIn / stride;
      assert(yOut < yDimOut);
      for (unsigned chan = 0; chan < numChannels; ++chan) {
        graph.addVertex(bwdCS, templateVertex("MaxPoolingBwd", dType),
          { {"actOut", actOut[chan / actOut.dim(3)][yOut][xOut]
                             [chan % actOut.dim(3)]},
            {"actIn", actIn[chan / actIn.dim(3)][yIn][xIn]
                           [chan % actIn.dim(3)]},
            {"errIn", deltasIn[chan / deltasIn.dim(3)][yOut][xOut]
                              [chan % deltasIn.dim(3)]},
            {"errOut", deltasOut[chan / deltasOut.dim(3)][yIn][xIn]
                                [chan % deltasOut.dim(3)]} });
      }
    }
  }
  return Execute(bwdCS);
}


}
