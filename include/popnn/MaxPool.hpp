#ifndef __MaxPool_hpp__
#define __MaxPool_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <cstdint>

namespace maxpool {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSize,
             unsigned stride, unsigned padding);

uint64_t getNumFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels, unsigned kernelSize,
                     unsigned stride, unsigned padding);

double getPerfectCycleCount(const poplar::Graph &graph,
                            std::string dType,
                            unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels, unsigned kernelSize,
                            unsigned stride, unsigned padding);

poplar::program::Program
maxPool(poplar::Graph &graph,
        unsigned kernelSize, unsigned stride, unsigned padding,
        poplar::Tensor in, poplar::Tensor out);

poplar::program::Program
maxPoolBackward(poplar::Graph &graph,
                unsigned kernelSize, unsigned stride, unsigned padding,
                poplar::Tensor actIn,
                poplar::Tensor actOut, poplar::Tensor deltasIn,
                poplar::Tensor deltasOut);
}

#endif  // __MaxPool_hpp__
