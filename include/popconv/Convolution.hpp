#ifndef __popconv_Convolution_hpp__
#define __popconv_Convolution_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>
#include "popconv/ConvPlan.hpp"

namespace popconv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSizeY,
             unsigned kernelSizeX,
             unsigned strideY, unsigned strideX, unsigned paddingY,
             unsigned paddingX);

uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX, unsigned paddingY,
                     unsigned paddingX, unsigned outNumChans);

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                     unsigned kernelSizeY, unsigned kernelSizeX,
                     unsigned strideY, unsigned strideX, unsigned paddingY,
                     unsigned paddingX, unsigned outNumChans);

uint64_t getWuFlops(unsigned batchSize,
                    unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX, unsigned paddingY,
                    unsigned paddingX, unsigned outNumChans);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType,
                               unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned inNumChans,
                               unsigned kernelSizeY, unsigned kernelSizeX,
                               unsigned strideY, unsigned strideX,
                               unsigned paddingY, unsigned paddingX,
                               unsigned outNumChans);

double getWuPerfectCycleCount(const poplar::Graph &graph,
                              std::string dType,
                              unsigned batchSize,
                              unsigned inDimY, unsigned inDimX,
                              unsigned inNumChans,
                              unsigned kernelSizeY, unsigned kernelSizeX,
                              unsigned strideY, unsigned strideX,
                              unsigned paddingY, unsigned paddingX,
                              unsigned outNumChans);

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
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &padding,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out, const std::string &partialsType,
            bool isFractional, const std::string &debugPrefix = "");

inline poplar::program::Program
convolution(poplar::Graph &graph, const Plan &plan,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out, const std::string &partialsType,
            bool isFractional, const std::string &debugPrefix = "") {
  return convolution(graph, plan, {strideY, strideX}, {paddingY, paddingX},
                     in, weights, biases, out, partialsType, isFractional,
                     debugPrefix);
}

void mapActivations(poplar::Graph &graph, const Plan &plan,
                    poplar::Tensor acts);

void mapWeights(poplar::Tensor w, poplar::Graph &graph, const Plan &plan,
                unsigned batchSize);

void mapBiases(poplar::Tensor b, poplar::Graph &graph,
               poplar::Tensor activations);

poplar::program::Program
weightsTransposeChansFlipXY(poplar::Graph &graph,
                            poplar::Tensor weightsIn,
                            poplar::Tensor WeightsOut,
                            const std::string &debugPrefix = "");

poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    const Plan &plan,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    const std::vector<unsigned> &stride,
                    const std::vector<unsigned> &padding, bool isFractional,
                    const std::string &debugPrefix = "");

inline poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    const Plan &plan,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX, bool isFractional,
                    const std::string &debugPrefix = "") {
  return convolutionBackward(graph, plan, zDeltas, weights, deltasOut,
                             {strideY, strideX}, {paddingY, paddingX},
                             isFractional, debugPrefix);
}

poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, const Plan &plan,
                      const Plan &fwdPlan, poplar::Tensor zDeltas,
                      unsigned kernelSizeY, unsigned kernelSizeX,
                      poplar::Tensor activations,
                      const std::vector<unsigned> &stride,
                      const std::vector<unsigned> &padding,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "");

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const Plan &plan, const Plan &fwdPlan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        const std::vector<unsigned> &stride,
                        const std::vector<unsigned> &padding,
                        float learningRate,
                        const std::string &debugPrefix = "");

inline poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const Plan &plan, const Plan &fwdPlan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned strideY, unsigned strideX, unsigned paddingY,
                        unsigned paddingX, float learningRate,
                        const std::string &debugPrefix = "") {
  return convolutionWeightUpdate(graph, plan, fwdPlan, zDeltas, weights,
                                 biases, activations, {strideY, strideX},
                                 {paddingY, paddingX}, learningRate,
                                 debugPrefix);
}

}
#endif  // __popconv_Convolution_hpp__
