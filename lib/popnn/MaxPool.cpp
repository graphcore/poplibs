#include "popnn/MaxPool.hpp"
#include "VertexTemplates.hpp"
#include "popnn/ActivationMapping.hpp"
#include "ConvUtil.hpp"
#include "gcd.hpp"
#include "popnn/exceptions.hpp"
#include "Util.hpp"
#include <cassert>
#include <map>

using namespace poplar;
using namespace poplar::program;
using namespace convutil;
using std::tie;

namespace maxpool {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX, unsigned strideY,
             unsigned strideX, unsigned paddingY,
             unsigned paddingX) {
  return convutil::getOutputDim(inDimY, inDimX, kernelSizeY, kernelSizeX,
                                strideY, strideX, paddingY, paddingX);
}


uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX) {
  unsigned outDimY, outDimX;
  std::tie(outDimY, outDimX) = getOutputDim(inDimY, inDimX,
                                            kernelSizeY, kernelSizeX,
                                            strideY, strideX,
                                            paddingY, paddingX);
  std::uint64_t numFlops = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(y, strideY, kernelSizeY,
                                               paddingY, inDimY, false);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(x, strideX, kernelSizeX,
                                                 paddingX, inDimX, false);
      const auto width = inXEnd - inXBegin;
      numFlops += numChannels * width * height;
    }
  }
  return batchSize * numFlops;
}

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX) {
  return 0;
}


double getFwdPerfectCycleCount(const Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getFwdFlops(batchSize,
                                    inDimY, inDimX,
                                    numChannels,
                                    kernelSizeY, kernelSizeX,
                                    strideY, strideX,
                                    paddingY, paddingX);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX) {
  return getFwdPerfectCycleCount(graph, dType, batchSize, inDimY, inDimX,
                                 numChannels, kernelSizeY, kernelSizeX,
                                 strideY, strideX, paddingY, paddingX) * 2;
}

// A utility type representing a pixel in the field being pooled over.
namespace {
struct Pixel {
  unsigned batch; unsigned y; unsigned x;
  Pixel(unsigned batch, unsigned y, unsigned x) : batch(batch), y(y), x(x) {}
  bool operator<(const Pixel &o) const { return tie(batch, y, x) <
                                                tie(o.batch, o.y, o.x); }
};
}

Program maxPool(Graph &graph,  unsigned kernelSizeY, unsigned kernelSizeX,
                unsigned strideY, unsigned strideX,
                unsigned paddingY, unsigned paddingX,
                Tensor in, Tensor out, const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(in);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = debugPrefix + "/MaxPool"
                         + std::to_string(kernelSizeX) + "x"
                         + std::to_string(kernelSizeY);
  ComputeSet cs = graph.addComputeSet(layerName);
  // Turn the tensors back into their natural dimensions of
  // {batch x height x width x channels}.
  out = out.dimShuffle({0, 2, 3, 1, 4})
           .reshape({out.dim(0), out.dim(2), out.dim(3),
                     out.dim(1) * out.dim(4)});
  in = in.dimShuffle({0, 2, 3, 1, 4})
         .reshape({in.dim(0), in.dim(2), in.dim(3),
                   in.dim(1) * in.dim(4)});

  const auto batchSize = in.dim(0);
  const auto inHeight = in.dim(1);
  const auto inWidth = in.dim(2);
  const auto numChannels = in.dim(3);
  const auto outHeight = out.dim(1);
  const auto outWidth = out.dim(2);
  unsigned expectedOutHeight, expectedOutWidth;
  tie(expectedOutHeight, expectedOutWidth) =
      getOutputDim(inHeight, inWidth, kernelSizeY, kernelSizeX, strideY,
                   strideX, paddingY, paddingX);

  if (out.dim(0) != batchSize)
    throw popnn::popnn_error("Input and output batchsize does not match");
  if (out.dim(3) != numChannels)
    throw popnn::popnn_error("Input and output number of channels does not "
                             "match");
  if (outHeight != expectedOutHeight)
    throw popnn::popnn_error("Input and output height dimensions do not match");
  if (outWidth != expectedOutWidth)
     throw popnn::popnn_error("Input and output width dimensions do not match");

  const auto numTiles = deviceInfo.getNumTiles();
  auto outTileMapping = graph.getTileMapping(out);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, outTileMapping[tile],
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      // A list of output vectors for the vertex to update
      std::vector<Tensor> vertexOut;
      // A list of input vectors which is kernelSize times bigger than
      // vertexOut. Each output element in vertexOut corresponds to
      // kernelSize number of input elements in vertexIn.
      std::vector<Tensor> vertexIn;
      std::vector<unsigned> windowSizes;
      for (const auto &region : regions) {
        // For each contiguous regions of output points group them by
        // pixel location.
        std::map<Pixel, std::vector<std::size_t>> groupedByPixel;
        for (unsigned i = region.begin(); i < region.end(); ++i) {
          auto coord = unflattenIndex(out.shape(), i);
          auto pixel = Pixel(coord[0], coord[1], coord[2]);
          auto channel = coord[3];
          groupedByPixel[pixel].push_back(channel);
        }
        // For each pixel add the vector of output channels to write to
        // and the vectors of input channels to pool over to the input/output
        // lists.
        for (const auto &entry : groupedByPixel) {
          // Construct a vector of channel values for each field position.
          const auto &pixel = entry.first;
          const auto batch = pixel.batch;
          const auto y = pixel.y;
          const auto x = pixel.x;
          const auto &channels = entry.second;
          Tensor outVector = graph.addTensor(dType, {0}, "");
          for (const auto chan : channels) {
            outVector = append(outVector, out[batch][y][x][chan]);
          }
          vertexOut.push_back(outVector);
          unsigned windowSize = 0;
          for (unsigned ky = 0; ky < kernelSizeY; ++ky) {
            auto inY = getInputIndex(y, strideY, kernelSizeY, paddingY,
                                     inHeight, ky, false);
            if (inY == ~0U)
              continue;
            for (unsigned kx = 0; kx < kernelSizeX; ++kx) {
              auto inX = getInputIndex(x, strideX, kernelSizeX, paddingX,
                                       inWidth, kx, false);
              if (inX == ~0U)
                continue;
              Tensor inVector = graph.addTensor(dType, {0}, "");
              for (const auto chan : channels) {
                inVector = append(inVector, in[batch][inY][inX][chan]);
              }
              vertexIn.push_back(inVector);
              ++windowSize;
            }
          }
          windowSizes.push_back(windowSize);
        }
      }
      auto v = graph.addVertex(cs, templateVertex("popnn::MaxPooling",
                                                  dType),
                               {{"in", vertexIn}, {"out", vertexOut}});
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setFieldSize(v["windowSizes"], windowSizes.size());
      for (unsigned i = 0; i < windowSizes.size(); ++i)
        graph.setInitialValue(v["windowSizes"][i], windowSizes[i]);
    }
  }

  return Execute(cs);
}

Program
maxPoolBackward(Graph &graph,
                unsigned kernelSizeY, unsigned kernelSizeX,
                unsigned strideY, unsigned strideX,
                unsigned paddingY, unsigned paddingX,
                Tensor fwdIn, Tensor fwdPooled,
                Tensor bwdIn, Tensor bwdOut,
                const std::string &debugPrefix) {
  const auto dType = graph.getTensorElementType(fwdIn);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = debugPrefix + "/MaxPoolBwd"
                         + std::to_string(kernelSizeX) + "x"
                         + std::to_string(kernelSizeY);
  ComputeSet cs = graph.addComputeSet(layerName);
  // Turn the tensors back into their natural dimensions of
  // {batch x height x width x channels}.
  bwdIn = bwdIn.dimShuffle({0, 2, 3, 1, 4})
         .reshape({bwdIn.dim(0), bwdIn.dim(2), bwdIn.dim(3),
                   bwdIn.dim(1) * bwdIn.dim(4)});
  bwdOut = bwdOut.dimShuffle({0, 2, 3, 1, 4})
           .reshape({bwdOut.dim(0), bwdOut.dim(2), bwdOut.dim(3),
                     bwdOut.dim(1) * bwdOut.dim(4)});
  fwdIn = fwdIn.dimShuffle({0, 2, 3, 1, 4})
               .reshape({fwdIn.dim(0), fwdIn.dim(2),
                         fwdIn.dim(3), fwdIn.dim(1) * fwdIn.dim(4)});
  fwdPooled = fwdPooled.dimShuffle({0, 2, 3, 1, 4})
                 .reshape({fwdPooled.dim(0), fwdPooled.dim(2),
                           fwdPooled.dim(3),
                           fwdPooled.dim(1) * fwdPooled.dim(4)});

  const auto batchSize = bwdIn.dim(0);
  const auto inHeight = bwdIn.dim(1);
  const auto inWidth = bwdIn.dim(2);
  const auto numChannels = bwdIn.dim(3);

  if (bwdOut.dim(0) != batchSize)
    throw popnn::popnn_error("Input and output batchsize does not match");
  if (bwdOut.dim(3) != numChannels)
    throw popnn::popnn_error("Input and output number of channels does not "
                             "match");
  if (fwdIn.dim(0) != batchSize || fwdPooled.dim(0) != batchSize)
    throw popnn::popnn_error("Forward pass batch size does not match gradient"
                             "calculation pass");
  if (fwdIn.dim(3) != numChannels || fwdPooled.dim(3) != numChannels)
    throw popnn::popnn_error("Forward pass number of channels does not match "
                             "gradient calculation pass");
  if (fwdIn.dim(1) != bwdOut.dim(1) || fwdIn.dim(2) != bwdOut.dim(2))
    throw popnn::popnn_error("Forward pass input height and width does not "
                             "match gradient calculation output height and "
                             "width");
  if (fwdPooled.dim(1) != bwdIn.dim(1) || fwdPooled.dim(2) != bwdIn.dim(2))
    throw popnn::popnn_error("Forward pass output height and width does not "
                             "match gradient calculation input height and "
                             "width");

  const auto numTiles = deviceInfo.getNumTiles();
  auto outTileMapping = graph.getTileMapping(bwdOut);

  for (unsigned tile = 0; tile != numTiles; ++tile) {
    // On each tile split the elements of the output up between the workers.
    // The grainSize is set to the vector width so vectors will not be split
    // up when allocating work to vertices.
    // The minimum amount of work per vertex is set to 2 * vectorwidth to
    // balance memory and loop overhead against parallel performance.
    const auto grainSize = dType == "float" ? deviceInfo.getFloatVectorWidth()
                                            : deviceInfo.getHalfVectorWidth();
    auto vertexRegions =
        splitRegionsBetweenWorkers(deviceInfo, outTileMapping[tile],
                                   grainSize, 2 * grainSize);
    for (const auto &regions : vertexRegions) {
      // A list of output vectors for the vertex to update
      std::vector<Tensor> vertexOut;
      // A list of input vectors from the forward pass (the activations
      // of the previous layer).
      std::vector<Tensor> vertexFwdIn;
      // A list of input vectors which is kernelSize times bigger than
      // vertexOut. Each output element in vertexOut corresponds to
      // kernlSize number of input elements in vertexIn.
      std::vector<Tensor> vertexIn;
      // A list of output vectors from the forward pass (the activations
      // going into the next layer).
      std::vector<Tensor> vertexFwdOut;
      std::vector<unsigned> windowSizes;
      for (const auto &region : regions) {
        // For each contiguous regions of output points group them by
        // pixel location.
        std::map<Pixel, std::vector<std::size_t>> groupedByPixel;
        for (unsigned i = region.begin(); i < region.end(); ++i) {
          auto coord = unflattenIndex(bwdOut.shape(), i);
          auto pixel = Pixel(coord[0], coord[1], coord[2]);
          auto channel = coord[3];
          groupedByPixel[pixel].push_back(channel);
        }

        // For each pixel add the vector of output channels to write to
        // and the vectors of channels from other tensors required to
        // calculate the output.
        for (const auto &entry : groupedByPixel) {
          const auto &pixel = entry.first;
          const auto batch = pixel.batch;
          const auto y = pixel.y;
          const auto x = pixel.x;
          const auto &channels = entry.second;
          Tensor outVector = graph.addTensor(dType, {0}, "");
          Tensor fwdInVector = graph.addTensor(dType, {0}, "");
          for (const auto chan : channels) {
            outVector = append(outVector, bwdOut[batch][y][x][chan]);
            fwdInVector = append(fwdInVector, fwdIn[batch][y][x][chan]);
          }
          vertexOut.push_back(outVector);
          vertexFwdIn.push_back(fwdInVector);
          unsigned windowSize = 0;
          for (unsigned ky = 0; ky < kernelSizeY; ++ky) {
            auto inY = getInputIndex(y, strideY, kernelSizeY, paddingY,
                                     inHeight, ky, true);
            if (inY == ~0U)
              continue;
            for (unsigned kx = 0; kx < kernelSizeX; ++kx) {
              auto inX = getInputIndex(x, strideX, kernelSizeX, paddingX,
                                       inWidth, kx, true);
              if (inX == ~0U)
                continue;
              Tensor inVector = graph.addTensor(dType, {0}, "");
              Tensor fwdOutVector = graph.addTensor(dType, {0}, "");
              for (const auto chan : channels) {
                inVector = append(inVector, bwdIn[batch][inY][inX][chan]);
                fwdOutVector = append(fwdOutVector,
                                      fwdPooled[batch][inY][inX][chan]);
              }
              vertexIn.push_back(inVector);
              vertexFwdOut.push_back(fwdOutVector);
              ++windowSize;
            }
          }
          windowSizes.push_back(windowSize);
        }
      }
      assert(vertexFwdIn.size() == vertexOut.size());
      assert(vertexFwdOut.size() == vertexIn.size());
      auto v = graph.addVertex(cs, templateVertex("popnn::MaxPoolingBwd",
                                                  dType),
                               {{"in", vertexIn}, {"out", vertexOut},
                                {"fwdIn", vertexFwdIn},
                                {"fwdOut", vertexFwdOut}});
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setFieldSize(v["windowSizes"], windowSizes.size());
      for (unsigned i = 0; i < windowSizes.size(); ++i)
        graph.setInitialValue(v["windowSizes"][i], windowSizes[i]);
    }
  }

  return Execute(cs);
}


}
