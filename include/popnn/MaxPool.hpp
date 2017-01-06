#ifndef __MaxPool_hpp__
#define __MaxPool_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <cstdint>

namespace maxpool {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

uint64_t getNumFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX,
                     unsigned paddingY, unsigned paddingX);

double getPerfectCycleCount(const poplar::Graph &graph,
                            std::string dType, unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels,
                            unsigned kernelSizeY, unsigned kernelSizeX,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX);

poplar::program::Program
maxPool(poplar::Graph &graph,
        unsigned kernelSizeY, unsigned kernelSizeX,
        unsigned strideY, unsigned strideX,
        unsigned paddingY, unsigned paddingX,
        poplar::Tensor in, poplar::Tensor out,
        const std::string &debugPrefix = "");

poplar::program::Program
maxPoolBackward(poplar::Graph &graph,
                unsigned kernelSizeY, unsigned kernelSizeX,
                unsigned strideY, unsigned strideX,
                unsigned paddingY, unsigned paddingX,
                poplar::Tensor actIn,
                poplar::Tensor actOut, poplar::Tensor deltasIn,
                poplar::Tensor deltasOut,
                const std::string &debugPrefix = "");
}

#endif  // __MaxPool_hpp__
