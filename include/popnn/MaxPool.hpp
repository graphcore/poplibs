#ifndef __popnn_MaxPool_hpp__
#define __popnn_MaxPool_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <cstdint>

namespace popnn {
namespace maxpool {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX);

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX);

poplar::Tensor
maxPool(poplar::Graph &graph, unsigned kernelSizeY, unsigned kernelSizeX,
        unsigned strideY, unsigned strideX, unsigned paddingY,
        unsigned paddingX, poplar::Tensor in, poplar::program::Sequence &prog,
        const std::string &debugPrefix = "");

// Calculate the gradient w.r.t. to the input of a max pooling operation given
// the gradient of the output.
poplar::Tensor
maxPoolInputGradient(poplar::Graph &graph, unsigned kernelSizeY,
                     unsigned kernelSizeX, unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX,
                     poplar::Tensor in,
                     poplar::Tensor pooled,
                     poplar::Tensor pooledGradient,
                     poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "");
}
}

#endif  // __MaxPool_hpp__
