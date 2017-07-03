#include "popnn/MaxPool.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/ActivationMapping.hpp"
#include "popconv/ConvUtil.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Util.hpp"
#include <cassert>
#include <map>

using namespace poplar;
using namespace poplar::program;
using namespace popconv;
using std::tie;
using namespace popstd;

namespace popnn {
namespace maxpool {

static void
checkWindowParameters(std::vector<std::size_t> kernelShape,
                      std::vector<unsigned> stride,
                      std::vector<int> inputPaddingLower,
                      std::vector<int> inputPaddingUpper) {
  if (kernelShape.size() != stride.size() ||
      stride.size() != inputPaddingLower.size() ||
      inputPaddingLower.size() != inputPaddingUpper.size()) {
    throw popstd::poplib_error("Mismatched window dimensions on poplibs "
                               "maxpool operation");
  }
  if (kernelShape.size() != 2) {
    throw popstd::poplib_error("poplibs maxpool only supports 2D operation");
  }
}

// Create dummy convolution parameters with same special characteristics
// as a pooling operation.
static ConvParams
makeConvParams(unsigned inDimY, unsigned inDimX,
               const std::vector<std::size_t> &kernelShape,
               const std::vector<unsigned> &stride,
               const std::vector<int> &inputPaddingLower,
               const std::vector<int> &inputPaddingUpper) {
  return  {"", {1, inDimY, inDimX, 1},
           {kernelShape[0], kernelShape[1], 1, 1},
           stride, inputPaddingLower, inputPaddingUpper,
           {1, 1}, {0, 0}, {0, 0}, {1, 1}};
}

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX,
             const std::vector<std::size_t> &kernelShape,
             const std::vector<unsigned> &stride,
             const std::vector<int> &inputPaddingLower,
             const std::vector<int> &inputPaddingUpper) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  auto params = makeConvParams(inDimY, inDimX, kernelShape, stride,
                               inputPaddingLower, inputPaddingUpper);
  return {params.getOutputHeight(), params.getOutputWidth()};
}


uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     const std::vector<std::size_t> &kernelShape,
                     const std::vector<unsigned> &stride,
                     const std::vector<int> &inputPaddingLower,
                     const std::vector<int> &inputPaddingUpper) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  auto params = makeConvParams(inDimY, inDimX, kernelShape, stride,
                               inputPaddingLower, inputPaddingUpper);
  auto outDimY = params.getOutputHeight();
  auto outDimX = params.getOutputWidth();
  std::uint64_t numFlops = 0;
  for (unsigned y = 0; y < outDimY; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(0, y, params);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outDimX; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(1, x, params);
      const auto width = inXEnd - inXBegin;
      numFlops += numChannels * width * height;
    }
  }
  return batchSize * numFlops;
}

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     const std::vector<std::size_t> &kernelShape,
                     const std::vector<unsigned> &stride,
                     const std::vector<int> &inputPaddingLower,
                     const std::vector<int> &inputPaddingUpper) {
  return 0;
}


double getFwdPerfectCycleCount(const Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               const std::vector<std::size_t> &kernelShape,
                               const std::vector<unsigned> &stride,
                               const std::vector<int> &inputPaddingLower,
                               const std::vector<int> &inputPaddingUpper) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  unsigned dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numFLOPs = getFwdFlops(batchSize,
                                    inDimY, inDimX,
                                    numChannels,
                                    kernelShape,
                                    stride,
                                    inputPaddingLower,
                                    inputPaddingUpper);
  const auto vectorWidth = deviceInfo.dataPathWidth / (8 * dTypeSize);
  return static_cast<double>(numFLOPs) / (vectorWidth * numTiles);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               const std::vector<std::size_t> &kernelShape,
                               const std::vector<unsigned> &stride,
                               const std::vector<int> &inputPaddingLower,
                               const std::vector<int> &inputPaddingUpper) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  return getFwdPerfectCycleCount(graph, dType, batchSize, inDimY, inDimX,
                                 numChannels, kernelShape,
                                 stride, inputPaddingLower,
                                 inputPaddingUpper) * 2;
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

Tensor maxPool(Graph &graph,
               const std::vector<std::size_t> &kernelShape,
               const std::vector<unsigned> &stride,
               const std::vector<int> &inputPaddingLower,
               const std::vector<int> &inputPaddingUpper,
               Tensor in, Sequence &prog, const std::string &debugPrefix) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  const auto dType = in.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = debugPrefix + "/MaxPool"
                         + std::to_string(kernelShape[0]) + "x"
                         + std::to_string(kernelShape[1]);
  ComputeSet cs = graph.addComputeSet(layerName);

  const auto batchSize = in.dim(0);
  const auto inHeight = in.dim(1);
  const auto inWidth = in.dim(2);
  const auto numChannels = in.dim(3);
  unsigned outHeight, outWidth;
  tie(outHeight, outWidth) =
      getOutputDim(inHeight, inWidth, kernelShape, stride, inputPaddingLower,
                   inputPaddingUpper);

  // Create output
  auto chansPerGroup = detectChannelGrouping(in);
  auto outGrouped =
      graph.addTensor(dType, {batchSize, numChannels / chansPerGroup,
                              outHeight, outWidth, chansPerGroup},
                      debugPrefix + "/maxPooled");
  mapActivations(graph, outGrouped);
  auto out = outGrouped.dimShuffle({0, 2, 3, 1, 4})
                       .reshape({batchSize, outHeight, outWidth, numChannels});

  const auto numTiles = deviceInfo.getNumTiles();
  auto outTileMapping = graph.getTileMapping(out);
  const auto params = makeConvParams(in.dim(1), in.dim(2),
                                     kernelShape, stride,
                                     inputPaddingLower, inputPaddingUpper);

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
        std::map<Pixel, std::vector<Interval<std::size_t>>> groupedByPixel;
        for (unsigned i = region.begin(); i < region.end(); ++i) {
          auto coord = unflattenIndex(out.shape(), i);
          auto pixel = Pixel(coord[0], coord[1], coord[2]);
          auto channel = coord[3];
          if (!groupedByPixel[pixel].empty() &&
              groupedByPixel[pixel].back().end() == channel) {
            groupedByPixel[pixel].back() =
                Interval<std::size_t>(groupedByPixel[pixel].back().begin(),
                                      channel + 1);
          } else {
            groupedByPixel[pixel].emplace_back(channel, channel + 1);
          }
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
          const auto outPixel = out[batch][y][x];
          std::vector<Tensor> outChans;
          for (const auto chanSlice : channels) {
            outChans.push_back(outPixel.slice(chanSlice));
          }
          vertexOut.push_back(concat(outChans));
          unsigned windowSize = 0;
          for (unsigned ky = 0; ky < kernelShape[0]; ++ky) {
            auto inY = getInputIndex(0, y, ky, params);
            if (inY == ~0U)
              continue;
            for (unsigned kx = 0; kx < kernelShape[1]; ++kx) {
              auto inX = getInputIndex(1, x, kx, params);
              if (inX == ~0U)
                continue;
              const auto inPixel = in[batch][inY][inX];
              std::vector<Tensor> inChans;
              for (const auto chanSlice : channels) {
                inChans.push_back(inPixel.slice(chanSlice));
              }
              vertexIn.push_back(concat(inChans));
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

  prog.add(Execute(cs));
  return out;
}

Tensor
maxPoolInputGradient(Graph &graph,
                     const std::vector<std::size_t> &kernelShape,
                     const std::vector<unsigned> &stride,
                     const std::vector<int> &inputPaddingLower,
                     const std::vector<int> &inputPaddingUpper,
                     Tensor in, Tensor pooled,
                     Tensor pooledGradient, Sequence &prog,
                     const std::string &debugPrefix) {
  checkWindowParameters(kernelShape, stride, inputPaddingLower,
                        inputPaddingUpper);
  const auto dType = in.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto layerName = debugPrefix + "/MaxPoolBwd"
                         + std::to_string(kernelShape[0]) + "x"
                         + std::to_string(kernelShape[1]);
  ComputeSet cs = graph.addComputeSet(layerName);


  const auto batchSize = pooledGradient.dim(0);
  const auto numChannels = pooledGradient.dim(3);
  const auto inHeight = in.dim(1);
  const auto inWidth = in.dim(2);

  if (in.dim(0) != batchSize || pooled.dim(0) != batchSize)
    throw popstd::poplib_error("Forward pass batch size does not match gradient"
                               "calculation pass");
  if (in.dim(3) != numChannels || pooled.dim(3) != numChannels)
    throw popstd::poplib_error("Forward pass number of channels does not match "
                               "gradient calculation pass");
  if (pooled.dim(1) != pooledGradient.dim(1) ||
      pooled.dim(2) != pooledGradient.dim(2))
    throw popstd::poplib_error("Forward pass output height and width does not "
                               "match gradient calculation input height and "
                               "width");

  auto inGradient = graph.clone(in);

  const auto numTiles = deviceInfo.getNumTiles();
  auto outTileMapping = graph.getTileMapping(inGradient);
  auto params = makeConvParams(inHeight, inWidth,
                               kernelShape, stride,
                               inputPaddingLower, inputPaddingUpper);
  auto bwdParams = getGradientParams(params);

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
      // A list of input gradient vectors for the vertex to update
      std::vector<Tensor> vertexInGrad;
      // A list of input vectors from the forward pass (the activations
      // of the previous layer).
      std::vector<Tensor> vertexIn;
      // A list of output gradient vectors which is kernelSize times
      // bigger than vertexOut. Each output element in vertexOut corresponds to
      // kernelSize number of input elements in vertexIn.
      std::vector<Tensor> vertexPooledGrad;
      // A list of output vectors from the forward pass (the activations
      // going into the next layer).
      std::vector<Tensor> vertexPooled;
      std::vector<unsigned> windowSizes;
      for (const auto &region : regions) {
        // For each contiguous regions of output points group them by
        // pixel location.
        std::map<Pixel, std::vector<Interval<std::size_t>>> groupedByPixel;
        for (unsigned i = region.begin(); i < region.end(); ++i) {
          auto coord = unflattenIndex(inGradient.shape(), i);
          auto pixel = Pixel(coord[0], coord[1], coord[2]);
          auto channel = coord[3];
          if (!groupedByPixel[pixel].empty() &&
              groupedByPixel[pixel].back().end() == channel) {
            groupedByPixel[pixel].back() =
                Interval<std::size_t>(groupedByPixel[pixel].back().begin(),
                                      channel + 1);
          } else {
            groupedByPixel[pixel].emplace_back(channel, channel + 1);
          }
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
          const auto inGradPixel = inGradient[batch][y][x];
          const auto inPixel = in[batch][y][x];
          std::vector<Tensor> inGradChans;
          std::vector<Tensor> inChans;
          for (const auto chanSlice : channels) {
            inGradChans.push_back(inGradPixel.slice(chanSlice));
            inChans.push_back(inPixel.slice(chanSlice));
          }
          vertexInGrad.push_back(concat(inGradChans));
          vertexIn.push_back(concat(inChans));
          unsigned windowSize = 0;
          for (unsigned ky = 0; ky < kernelShape[0]; ++ky) {
            auto outY = getInputIndex(0, y, ky, bwdParams);
            if (outY == ~0U)
              continue;
            for (unsigned kx = 0; kx < kernelShape[1]; ++kx) {
              auto outX = getInputIndex(1, x, kx, bwdParams);
              if (outX == ~0U)
                continue;
              const auto pooledGradPixel = pooledGradient[batch][outY][outX];
              const auto pooledPixel = pooled[batch][outY][outX];
              std::vector<Tensor> pooledGradChans;
              std::vector<Tensor> pooledChans;
              for (const auto chanSlice : channels) {
                pooledGradChans.push_back(pooledGradPixel.slice(chanSlice));
                pooledChans.push_back(pooledPixel.slice(chanSlice));
              }
              vertexPooledGrad.push_back(concat(pooledGradChans));
              vertexPooled.push_back(concat(pooledChans));
              ++windowSize;
            }
          }
          windowSizes.push_back(windowSize);
        }
      }
      assert(vertexPooled.size() == vertexPooledGrad.size());
      assert(vertexIn.size() == vertexInGrad.size());
      auto v = graph.addVertex(cs, templateVertex("popnn::MaxPoolingGrad",
                                                  dType),
                               {{"outGrad", vertexPooledGrad},
                                {"inGrad", vertexInGrad},
                                {"in", vertexIn},
                                {"out", vertexPooled}});
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setFieldSize(v["windowSizes"], windowSizes.size());
      for (unsigned i = 0; i < windowSizes.size(); ++i)
        graph.setInitialValue(v["windowSizes"][i], windowSizes[i]);
    }
  }

  prog.add(Execute(cs));
  return inGradient;
}


}
}
