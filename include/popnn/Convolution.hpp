#ifndef __Convolution_hpp__
#define __Convolution_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Engine.hpp>
#include "popnn/ConvPlan.hpp"
#include "popnn/ConvDef.hpp"
#include "popnn/NonLinearityDef.hpp"

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSize,
             unsigned stride, unsigned padding);

uint64_t getFlops(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSize, unsigned stride, unsigned padding,
                  unsigned outNumChans, bool doResidual, bool forwardOnly);

double getPerfectCycleCount(const poplar::Graph &graph,
                            std::string dType,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding,
                            unsigned outNumChans, bool doResidual,
                            bool forwardOnly);

poplar::Tensor
createWeights(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSize,
             unsigned outNumChans,
             const ConvPlan &plan);

poplar::Tensor
createBiases(poplar::Graph &graph, std::string dType,
             unsigned outNumChans);

poplar::program::Program
convolution(poplar::Graph &graph, const ConvPlan &plan,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned numChannels, NonLinearityType nonLinearityType,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out,
            ResidualMethod resMethod = RESIDUAL_NONE,
            poplar::Tensor residual={});

void mapWeights(poplar::Tensor w, poplar::Graph &graph, const ConvPlan &plan);

void mapBiases(poplar::Tensor b, poplar::Graph &graph,
               poplar::Tensor activations);

poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    const ConvPlan &plan,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    unsigned kernelSize, unsigned stride,
                    unsigned padding);

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        const ConvPlan &plan,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned kernelSize, unsigned stride,
                        unsigned padding, float learningRate);

}
#endif  // __Convolution_hpp__
