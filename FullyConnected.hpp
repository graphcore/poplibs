#ifndef __FullyConnected_hpp__
#define __FullyConnected_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/IPUModelEngine.hpp>
#include "DeviceInfo.hpp"
#include "neural_net_common.h"

namespace fc {

struct Plan;

uint64_t getNumFlops(unsigned inSize, unsigned outSize);

double getPerfectCycleCount(const DeviceInfo &deviceInfo,
                            unsigned inSize, unsigned outSize,
                            std::string dType);

std::pair<poplar::Tensor, poplar::Tensor>
createParams(poplar::Graph &graph, std::string dType, unsigned inSize,
             unsigned outSize);

poplar::program::Program
fullyConnected(poplar::Graph &graph,
               poplar::IPUModelEngineBuilder::TileMapping &mapping,
               DeviceInfo &deviceInfo,
               unsigned size, NonLinearityType nonLinearityType,
               std::string dType,
               poplar::Tensor in, poplar::Tensor weights,
               poplar::Tensor biases,
               poplar::Tensor z,
               poplar::Tensor out,
               const Plan &plan);

poplar::program::Program
fullyConnectedBackward(poplar::Graph &graph,
                       poplar::IPUModelEngineBuilder::TileMapping &mapping,
                       DeviceInfo &deviceInfo,
                       std::string dType,
                       poplar::Tensor zDeltas,
                       poplar::Tensor weights, poplar::Tensor deltasOut,
                       const Plan &plan);

poplar::program::Program
fullyConnectedWeightUpdate(poplar::Graph &graph,
                           poplar::IPUModelEngineBuilder::TileMapping &mapping,
                           DeviceInfo &deviceInfo,
                           std::string dType,
                           poplar::Tensor zDeltas,
                           poplar::Tensor activations,
                           poplar::Tensor weights, poplar::Tensor biases,
                           float learningRate,
                           const Plan &plan);


} // namespace fc

#endif // __FullyConnected_hpp__
