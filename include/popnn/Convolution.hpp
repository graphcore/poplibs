#ifndef __Convolution_hpp__
#define __Convolution_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>
#include "popnn/ConvPlan.hpp"
#include "popnn/ResidualDef.hpp"
#include "popnn/NonLinearityDef.hpp"

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

uint64_t getFlops(unsigned batchSize,
                  unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSizeY, unsigned kernelSizeX,
                  unsigned strideY, unsigned strideX, unsigned paddingY,
                  unsigned paddingX,
                  unsigned outNumChans, bool forwardOnly);

double getPerfectCycleCount(const poplar::Graph &graph,
                            std::string dType,
                            unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSizeY, unsigned kernelSizeX,
                            unsigned strideY, unsigned strideX,
                            unsigned paddingY, unsigned paddingX,
                            unsigned outNumChans, bool forwardOnly);

poplar::Tensor
createWeights(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned outNumChans,
             const ConvPlan &plan);

poplar::Tensor
createBiases(poplar::Graph &graph, std::string dType,
             unsigned outNumChans);

poplar::program::Program
convolution(poplar::Graph &graph, const ConvPlan &plan,
            unsigned kernelSizeY, unsigned kernelSizeX,
            unsigned strideY, unsigned strideX, unsigned paddingY,
            unsigned paddingX,
            unsigned numChannels,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out,
            const std::string &partialsType,
            bool useWinogradConv = false,
            unsigned winogradPatchSize = 4);

void mapWeights(poplar::Tensor w, poplar::Graph &graph, const ConvPlan &plan,
                unsigned batchSize);

void mapBiases(poplar::Tensor b, poplar::Graph &graph,
               poplar::Tensor activations);

poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    const ConvPlan &plan,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX);

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const ConvPlan &plan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned kernelSizeY, unsigned kernelSizeX,
                        unsigned strideY, unsigned strideX, unsigned paddingY,
                        unsigned paddingX, float learningRate);

extern poplar::program::Program winogradConvolution(poplar::Graph &graph,
            unsigned kernelSizeY, unsigned kernelSizeX, unsigned strideY,
            unsigned strideX, unsigned paddingY, unsigned paddingX,
            unsigned xDim, unsigned yDim,
            unsigned outNumChans, unsigned patchSizeX, unsigned patchSizeY,
            const std::string &dType,
            const std::string &partialsType,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor activations);

}
#endif  // __Convolution_hpp__
