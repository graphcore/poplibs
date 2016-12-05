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
             const Plan &plan);

poplar::Tensor
createBiases(poplar::Graph &graph, std::string dType,
             unsigned outNumChans);

poplar::program::Program
convolution(poplar::Graph &graph, const Plan &plan,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out, const std::string &partialsType,
            bool isFractional, bool useWinogradConv = false,
            unsigned winogradPatchSize = 4,
            const std::string &debugPrefix = "");

void mapWeights(poplar::Tensor w, poplar::Graph &graph, const Plan &plan,
                unsigned batchSize);

void mapBiases(poplar::Tensor b, poplar::Graph &graph,
               poplar::Tensor activations);

poplar::program::Program
weightsTransposeChansFlipXY(poplar::Graph &graph,
                            poplar::Tensor weightsIn,
                            poplar::Tensor WeightsOut);

poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    const Plan &plan,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX, bool isFractional,
                    const std::string &debugPrefix = "");

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const Plan &plan, const Plan &fwdPlan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned strideY, unsigned strideX, unsigned paddingY,
                        unsigned paddingX, float learningRate,
                        const std::string &debugPrefix = "");

extern poplar::program::Program winogradConvolution(poplar::Graph &graph,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            unsigned xDim, unsigned yDim,
            unsigned outNumChans, unsigned patchSizeX, unsigned patchSizeY,
            const std::string &dType,
            const std::string &partialsType,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor activations,
            const std::string &debugPrefix = "");

}
#endif  // __Convolution_hpp__
