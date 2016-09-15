#ifndef __FullyConnected_hpp__
#define __FullyConnected_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include "popnn/NonLinearityDef.hpp"

namespace fc {

struct Plan;

uint64_t getNumFlops(unsigned batchSize,
                     unsigned inSize, unsigned outSize, bool forwardOnly);

double getPerfectCycleCount(const poplar::Graph &graph,
                            unsigned batchSize,
                            unsigned inSize, unsigned outSize,
                            std::string dType, bool forwardOnly);

std::pair<poplar::Tensor, poplar::Tensor>
createParams(poplar::Graph &graph, std::string dType, unsigned inSize,
             unsigned outSize);

poplar::program::Program
fullyConnected(poplar::Graph &graph,
               unsigned size, NonLinearityType nonLinearityType,
               poplar::Tensor in, poplar::Tensor weights,
               poplar::Tensor biases,
               poplar::Tensor out,
               const Plan &plan);

poplar::program::Program
fullyConnectedBackward(poplar::Graph &graph,
                       poplar::Tensor zDeltas,
                       poplar::Tensor weights, poplar::Tensor deltasOut,
                       const Plan &plan);

poplar::program::Program
fullyConnectedWeightUpdate(poplar::Graph &graph,
                           poplar::Tensor zDeltas,
                           poplar::Tensor activations,
                           poplar::Tensor weights, poplar::Tensor biases,
                           float learningRate,
                           const Plan &plan);


} // namespace fc

#endif // __FullyConnected_hpp__
