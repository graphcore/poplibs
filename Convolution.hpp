#ifndef __Convolution_hpp__
#define __Convolution_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/IPUModelEngine.hpp>
#include "DeviceInfo.hpp"
#include "neural_net_common.h"
#include "ConvPlan.hpp"

namespace conv {

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX, unsigned kernelSize,
             unsigned stride, unsigned padding);

uint64_t getFlops(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  unsigned kernelSize, unsigned stride, unsigned padding,
                  unsigned outNumChans, bool doResidual, bool forwardOnly);

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            std::string dType,
                            unsigned inDimY, unsigned inDimX,
                            unsigned inNumChans,
                            unsigned kernelSize, unsigned stride,
                            unsigned padding,
                            unsigned outNumChans, bool doResidual,
                            bool forwardOnly);

std::pair<poplar::Tensor, poplar::Tensor>
createParams(poplar::Graph &graph, std::string dType,
             unsigned inNumChans,
             unsigned kernelSize,
             unsigned outNumChans,
             const ConvPlan &plan);

poplar::program::Program
convolution(poplar::Graph &graph,
            poplar::IPUModelEngineBuilder::TileMapping &mapping,
            const DeviceInfo &deviceInfo,
            const ConvPlan &plan,
            unsigned kernelSize, unsigned stride, unsigned padding,
            unsigned numChannels, NonLinearityType nonLinearityType,
            std::string dType,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor z, poplar::Tensor out,
            ResidualMethod resMethod = RESIDUAL_NONE,
            poplar::Tensor residual={});

void mapWeights(poplar::Tensor w,
                poplar::IPUModelEngineBuilder::TileMapping &mapping,
                const DeviceInfo &deviceInfo, const ConvPlan &plan);

void mapBiases(poplar::Tensor b,
               poplar::IPUModelEngineBuilder::TileMapping &mapping,
               const DeviceInfo &deviceInfo,
               poplar::Tensor activations);

poplar::program::Program
convolutionBackward(poplar::Graph &graph,
                    poplar::IPUModelEngineBuilder::TileMapping &mapping,
                    const DeviceInfo &deviceInfo,
                    const ConvPlan &plan,
                    std::string dType,
                    poplar::Tensor zDeltas, poplar::Tensor weights,
                    poplar::Tensor deltasOut,
                    unsigned kernelSize, unsigned stride,
                    unsigned padding);

poplar::program::Program
convolutionWeightUpdate(poplar::Graph &graph,
                        poplar::IPUModelEngineBuilder::TileMapping &mapping,
                        const DeviceInfo &deviceInfo,
                        std::string dType,
                        poplar::Tensor zDeltas, poplar::Tensor weights,
                        poplar::Tensor biases,
                        poplar::Tensor activations,
                        unsigned kernelSize, unsigned stride,
                        unsigned padding, float learningRate);

}
#endif  // __Convolution_hpp__
