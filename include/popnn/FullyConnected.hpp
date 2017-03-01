#ifndef __FullyConnected_hpp__
#define __FullyConnected_hpp__
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include "popnn/NonLinearityDef.hpp"

namespace fc {

struct Plan;

uint64_t getFwdFlops(unsigned batchSize, unsigned inSize, unsigned outSize);

uint64_t getBwdFlops(unsigned batchSize, unsigned inSize, unsigned outSize);

uint64_t getWuFlops(unsigned batchSize, unsigned inSize, unsigned outSize);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               unsigned batchSize,
                               unsigned inSize, unsigned outSize,
                               std::string dType);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               unsigned batchSize,
                               unsigned inSize, unsigned outSize,
                               std::string dType);

double getWuPerfectCycleCount(const poplar::Graph &graph,
                              unsigned batchSize,
                              unsigned inSize, unsigned outSize,
                              std::string dType);

std::pair<poplar::Tensor, poplar::Tensor>
createParams(poplar::Graph &graph, std::string dType, unsigned inSize,
             unsigned outSize);

void mapBiases(poplar::Graph &graph, poplar::Tensor biases,
               const std::vector<unsigned> &outMapping);

void mapWeights(poplar::Graph &graph, poplar::Tensor weights,
                const std::vector<unsigned> &outMapping,
                const Plan &plan);

poplar::program::Program
fullyConnected(poplar::Graph &graph,
               unsigned size, NonLinearityType nonLinearityType,
               poplar::Tensor in, poplar::Tensor weights,
               poplar::Tensor biases,
               poplar::Tensor out,
               const Plan &plan,
               const std::string &debugPrefix="");

poplar::program::Program
fullyConnectedBackward(poplar::Graph &graph,
                       poplar::Tensor zDeltas,
                       poplar::Tensor weights, poplar::Tensor deltasOut,
                       const Plan &plan,
                       const std::string &debugPrefix = "");

poplar::program::Program
fullyConnectedWeightUpdate(poplar::Graph &graph,
                           poplar::Tensor zDeltas,
                           poplar::Tensor activations,
                           poplar::Tensor weights, poplar::Tensor biases,
                           float learningRate,
                           const Plan &plan,
                           const std::string &debugPrefix = "");


} // namespace fc

#endif // __FullyConnected_hpp__
