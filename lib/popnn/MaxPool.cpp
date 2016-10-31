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

uint64_t getNumFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
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
      numFlops += numChannels * width * height;
    }
  }
  return batchSize * numFlops;
}

double getPerfectCycleCount(const Graph &graph,
                            std::string dType, unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels, unsigned kernelSize,
                            unsigned stride, unsigned padding) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getNumFlops(batchSize,
                                    inDimY, inDimX, numChannels, kernelSize,
                                    stride, padding);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

Program
maxPool(Graph &graph,
        unsigned kernelSize, unsigned stride, unsigned padding,
        Tensor in, Tensor out) {
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  const auto batchSize = in.dim(0);
  const auto prevNumChanGroups = in.dim(1);
  const auto prevChansPerGroup = in.dim(4);
  const auto prevNumChannels = prevNumChanGroups * prevChansPerGroup;
  const auto numChanGroups = out.dim(1);
  const auto chansPerGroup = out.dim(4);
  const auto numChannels = numChanGroups * chansPerGroup;
  if (numChannels != prevNumChannels)
    assert(!"maxPool's input and output numChannels differ");
  unsigned inDimY = in.dim(2), inDimX = in.dim(3);
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX,
                                            kernelSize, stride, padding);
  assert(outDimY == out.dim(2));
  assert(outDimX == out.dim(3));
  const auto chunkSize = gcd<unsigned>(prevChansPerGroup, chansPerGroup);
  const auto chunksPerChanGroup = chansPerGroup / chunkSize;
  ComputeSet fwd = graph.createComputeSet(layerName + ".fwd");
  // Iterate through the batch adding vertices to the same compute set (so
  // batch is executed in parallel).
  for (unsigned b = 0; b < batchSize; ++b) {
    //mapping over this MaxPool layer's outputs
    const auto outMapping = computeActivationsMapping(graph, out[b], b,
                                                      batchSize);
    const auto numTiles = deviceInfo.getNumTiles();

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
          graph.addVertex(fwd, templateVertex("popnn::MaxPooling", dType),
            { {"activationOut", out[b][chanGroup][y][x]} });
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setTileMapping(v, tile);
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
                  in[b][prevChanGroup][inY][inX]
                     .slice(prevChanInGroup,
                            prevChanInGroup + chunkSize);
              graph.connect(inWindow, v["activationIn"][chunkIndex]);
              chunkIndex++;
            }
          }
        }
      }
    }
  }
  return Execute(fwd);
}

Program
maxPoolBackward(Graph &graph,
                unsigned kernelSize, unsigned stride, unsigned padding,
                Tensor actIn, Tensor actOut,
                Tensor deltasIn, Tensor deltasOut) {
  const auto dType = graph.getTensorElementType(actIn);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto batchSize = actIn.dim(0);
  // actIn is from the previous layer
  // actOut went to the next layer
  // deltasIn is from the next layer
  // deltasOut is to the previous layer
  assert(actIn.dim(2) == deltasOut.dim(2));
  assert(actIn.dim(3) == deltasOut.dim(3));
  assert(actOut.dim(2) == deltasIn.dim(2));
  assert(actOut.dim(3) == deltasIn.dim(3));
  assert(deltasIn.dim(1) * deltasIn.dim(4)
         == deltasOut.dim(1) * deltasOut.dim(4));

  // "prev" refers to the layer nearer the input
  // "next" refers to the layer nearer the output
  const auto layerName = "MaxPool" + std::to_string(kernelSize) + "x" +
                          std::to_string(kernelSize);
  const auto nextNumChanGroups = actOut.dim(1);
  const auto nextChansPerGroup = actOut.dim(4);
  const auto nextNumChannels = nextNumChanGroups * nextChansPerGroup;
  const auto prevNumChanGroups = actIn.dim(1);
  const auto prevChansPerGroup = actIn.dim(4);
  const auto prevNumChannels = prevNumChanGroups * prevChansPerGroup;
  //MaxPool so no change in channel dimension
  if(nextNumChannels != prevNumChannels)
    assert(!"maxPoolBackwards: prev and next NumChannels must match");

  const auto yDimPrev = deltasOut.dim(2);
  const auto xDimPrev = deltasOut.dim(3);
  const auto yDimNext = deltasIn.dim(2);
  const auto xDimNext = deltasIn.dim(3);
  unsigned calcNextX, calcNextY;
  std::tie(calcNextY, calcNextX) = getOutputDim(yDimPrev, xDimPrev,
                                                kernelSize, stride, padding);
  assert(calcNextY == yDimNext);
  assert(calcNextX == xDimNext);
  // The input and output tensors may have different group sizes
  // \a chunkSize is the group size we will operate on
  const auto prevChunkSize = gcd<unsigned>(actOut.dim(4), deltasOut.dim(4));
  const auto nextChunkSize = gcd<unsigned>(actIn.dim(4), deltasIn.dim(4));
  const auto chunkSize = gcd<unsigned>(prevChunkSize, nextChunkSize);
  const auto prevChunksPerChanGroup = prevChansPerGroup / chunkSize;

  if (graph.getDevice().getDeviceType() == DeviceType::IPU_MODEL
      && dType == "half"
      && chunkSize % 2 != 0) {
    // Possible race with different vertices writing to the same 32bit word
    std::cerr << "WARNING: When Z is odd and dType is half the IPU Model only"
                 "supports\nnon-overlapping, even-sized kernels\n";
  }

  auto bwdCS = graph.createComputeSet(layerName + ".bwd");

  for (unsigned b = 0; b != batchSize; ++b) {
    // map over deltaOut so that no reduce will be required.
    const auto prevDeltaMapping = computeActivationsMapping(graph, deltasOut[b],
                                                            b, batchSize);
    const auto numTiles = deviceInfo.getNumTiles();

    for (unsigned tile = 0; tile != numTiles; ++tile) {
      const auto tileBegin = prevDeltaMapping[tile];
      const auto tileEnd = prevDeltaMapping[tile + 1];
      assert(tileBegin % nextChansPerGroup == 0);
      assert(tileEnd % nextChansPerGroup == 0);
      const auto tileGroupBegin = tileBegin / nextChansPerGroup;
      const auto tileGroupEnd = tileEnd / nextChansPerGroup;
      const auto tileNumGroups = tileGroupEnd - tileGroupBegin;
      if (tileNumGroups == 0)
        continue;
      for (unsigned i = tileGroupBegin; i != tileGroupEnd; ++i) {
        unsigned xPrev = i % xDimPrev;
        unsigned yPrev = (i / xDimPrev) % yDimPrev;
        unsigned chanGroupPrev = i / (yDimPrev * xDimPrev);
        auto nextYRange = getInputRange(yPrev, stride, kernelSize,
                                        padding, yDimNext, false);
        auto nextXRange = getInputRange(xPrev, stride, kernelSize,
                                        padding, xDimNext, false);
        const auto nextYSize = nextYRange.second - nextYRange.first;
        const auto nextXSize = nextXRange.second - nextXRange.first;

        for (unsigned chunk = 0; chunk != prevChunksPerChanGroup; chunk++) {
          unsigned chanBase = chanGroupPrev * prevChansPerGroup
                                + chunk * chunkSize;
          auto v =
              graph.addVertex(bwdCS, templateVertex("popnn::MaxPoolingBwd",
                                                    dType));
          graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
          graph.setTileMapping(v, tile);

          unsigned chunkBase = chanBase % deltasOut.dim(4);
          Tensor chunkErrOut =
              deltasOut[b][chanBase / deltasOut.dim(4)][yPrev][xPrev]
                   .slice(chunkBase, chunkBase + chunkSize);
          graph.connect(chunkErrOut, v["errOut"]);

          chunkBase = chanBase % actIn.dim(4);
          Tensor chunkActIn = actIn[b][chanBase / actIn.dim(4)][yPrev][xPrev]
              .slice(chunkBase, chunkBase + chunkSize);
          graph.connect(chunkActIn, v["actIn"]);

          graph.setFieldSize(v["actOut"], nextXSize * nextYSize);
          graph.setFieldSize(v["errIn"],  nextXSize * nextYSize);

          unsigned chunkIndex = 0;
//TODO: can we combine multiple X into a contiguous vector? If so pointer
// storage would be considerably reduced. Revisit once we're using a decision
// vector rather than actIn/actOut comparisons
          for (auto yNext = nextYRange.first; yNext != nextYRange.second;
               ++yNext) {
            for (auto xNext = nextXRange.first; xNext != nextXRange.second;
                 ++xNext) {

              auto chunkBase = chanBase % actOut.dim(4);
              Tensor chunkActOut =
                  actOut[b][chanBase / actOut.dim(4)][yNext][xNext]
                       .slice(chunkBase, chunkBase + chunkSize);
              graph.connect(chunkActOut, v["actOut"][chunkIndex]);

              chunkBase = chanBase % deltasIn.dim(4);
              Tensor chunkDeltasIn =
                  deltasIn[b][chanBase / deltasIn.dim(4)][yNext][xNext]
                       .slice(chunkBase, chunkBase + chunkSize);
              graph.connect(chunkDeltasIn, v["errIn"][chunkIndex]);
              chunkIndex++;
            }
          }
        }
      }
    }
  }
  return Execute(bwdCS);
}


}
