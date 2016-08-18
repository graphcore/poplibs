#include "popnn/MaxPool.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"
#include "ConvUtil.hpp"
#include "gcd.hpp"
#include "popnn/exceptions.hpp"
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
  const auto prevNumChanGroups = in.dim(0);
  const auto prevChansPerGroup = in.dim(3);
  const auto prevNumChannels = prevNumChanGroups * prevChansPerGroup;
  const auto numChanGroups = out.dim(0);
  const auto chansPerGroup = out.dim(3);
  const auto numChannels = numChanGroups * chansPerGroup;
  assert(numChannels == prevNumChannels);
  unsigned inDimY = in.dim(1), inDimX = in.dim(2);
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX,
                                            kernelSize, stride, padding);
  assert(outDimY == out.dim(1));
  assert(outDimX == out.dim(2));
  const auto chunkSize = gcd<unsigned>(prevChansPerGroup, chansPerGroup);
  const auto chunksPerChanGroup = chansPerGroup / chunkSize;
  ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
  //mapping over this MaxPool layer's outputs
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
      unsigned x = i % outDimX;
      unsigned y = (i / outDimX) % outDimY;
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
      unsigned chunkIndex = 0;

      for (unsigned j = 0; j != chunksPerChanGroup; ++j) {
        unsigned chan = chanGroup * chansPerGroup + j * chunksPerChanGroup;
        unsigned prevChanGroup = chan / prevChansPerGroup;
        unsigned prevChanInGroup = chan % prevChansPerGroup;

        for (unsigned inY = inYBegin; inY != inYEnd; ++inY) {
          for (unsigned inX = inXBegin; inX != inXEnd; ++inX) {
            Tensor inWindow =
                in[prevChanGroup][inY][inX].slice(prevChanInGroup,
                                                  prevChanInGroup + chunkSize);

            graph.connect(inWindow, v["activationIn"][chunkIndex]);
            chunkIndex++;
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
  // actIn is from the previous layer
  // actOut went to the next layer
  // deltasIn is from the next layer
  // deltasOut is to the previous layer
  assert(actIn.dim(1) == deltasOut.dim(1));
  assert(actIn.dim(2) == deltasOut.dim(2));
  assert(actOut.dim(1) == deltasIn.dim(1));
  assert(actOut.dim(2) == deltasIn.dim(2));

  // "prev" refers to the layer nearer the input
  // "next" refers to the layer nearer the output
  const auto layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  const auto nextNumChanGroups = actOut.dim(0);
  const auto nextChansPerGroup = actOut.dim(3);
  const auto nextNumChannels = nextNumChanGroups * nextChansPerGroup;
  const auto prevNumChanGroups = actIn.dim(0);
  const auto prevChansPerGroup = actIn.dim(3);
  const auto prevNumChannels = prevNumChanGroups * prevChansPerGroup;
  //MaxPool so no change in channel dimension
  assert(nextNumChannels == prevNumChannels);

  const auto yDimPrev = actIn.dim(1);
  const auto xDimPrev = actIn.dim(2);
  const auto yDimNext = actOut.dim(1);
  const auto xDimNext = actOut.dim(2);
  unsigned calcNextX, calcNextY;

  if (deviceInfo.isIPU
      && (padding != 0 || xDimPrev % stride != 0 || yDimPrev % stride != 0 ||
          kernelSize != stride)) {
    std::cerr << "WARNING: padding, overlapped pooling or pooling that"
                 " doesn't exactly divide\n"
                 "input not implemented yet, skipping tile mapping for"
                 " this layer\n";
    return Sequence();
  }
  std::tie(calcNextY, calcNextX) = getOutputDim(yDimPrev, xDimPrev,
                                              kernelSize, stride, padding);
  assert(calcNextY == yDimNext);
  assert(calcNextX == xDimNext);
  // The input and output tensors may have different group sizes
  const auto prevChunkSize = gcd<unsigned>(actOut.dim(3), deltasOut.dim(3));
  const auto nextChunkSize = gcd<unsigned>(actIn.dim(3), deltasIn.dim(3));
  const auto chunkSize = gcd<unsigned>(prevChunkSize, nextChunkSize);
  const auto nextChunksPerChanGroup = nextChansPerGroup / chunkSize;

  auto bwdCS = graph.createComputeSet(layerName + ".bwd");
  // map over the Pool kernels - all will be mapped to the same tile
  const auto nextMapping = computeActivationsMapping(actOut, deviceInfo);
  const auto numTiles = deviceInfo.getNumIPUs() * deviceInfo.getTilesPerIPU();

  for (auto tile = 0; tile != numTiles; ++tile) {
    const auto tileBegin = nextMapping[tile];
    const auto tileEnd = nextMapping[tile + 1];
    assert(tileBegin % nextChansPerGroup == 0);
    assert(tileEnd % nextChansPerGroup == 0);
    const auto tileGroupBegin = tileBegin / nextChansPerGroup;
    const auto tileGroupEnd = tileEnd / nextChansPerGroup;
    const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
    if (tileNumGroups == 0)
      continue;
    for (unsigned i = tileGroupBegin; i != tileGroupEnd; ++i) {
      unsigned xNext = i % xDimNext;
      unsigned yNext = (i / xDimNext) % yDimNext;
      unsigned chanGroupNext = i / (yDimNext * xDimNext);
      unsigned prevYBegin, prevYEnd;
      std::tie(prevYBegin, prevYEnd) = getInputRange(yNext, stride, kernelSize,
                                                     padding, yDimPrev, true);
      const auto prevYSize = prevYEnd - prevYBegin;
      unsigned prevXBegin, prevXEnd;
      std::tie(prevXBegin, prevXEnd) = getInputRange(xNext, stride, kernelSize,
                                                     padding, xDimPrev, true);
      const auto prevXSize = prevXEnd - prevXBegin;
      for (unsigned chunk = 0; chunk != nextChunksPerChanGroup; chunk++) {
        // Generate one vertex for each individual output pixel
        unsigned baseChan = chanGroupNext * nextChansPerGroup + chunk * chunkSize;
        unsigned prevChanGroup = baseChan / prevChansPerGroup;
        unsigned prevChanInGroup = baseChan % prevChansPerGroup;
        for (auto yPrev = prevYBegin; yPrev != prevYEnd; ++yPrev) {
          for (auto xPrev = prevXBegin; xPrev != prevXEnd; ++xPrev) {
            for (auto chan = baseChan; chan != baseChan + chunkSize; ++chan) {
              // add a new vertex for every input activation
              auto v = graph.addVertex(bwdCS, templateVertex("MaxPoolingBwd", dType),
                {
                  {"actOut", actOut[chan / actOut.dim(3)][yNext][xNext]
                                   [chan % actOut.dim(3)]},
                  {"actIn", actIn[chan / actIn.dim(3)][yPrev][xPrev]
                                 [chan % actIn.dim(3)]},
                  {"errIn", deltasIn[chan / deltasIn.dim(3)][yNext][xNext]
                                    [chan % deltasIn.dim(3)]},
                  {"errOut", deltasOut[chan / deltasOut.dim(3)][yPrev][xPrev]
                                      [chan % deltasOut.dim(3)]}
                });
              mapping.setMapping(v, tile);

            }
          }
        }
      }
    }
  }

  return Execute(bwdCS);
}


}
