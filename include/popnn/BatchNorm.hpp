#ifndef __popnn_BatchNorm_hpp__
#define __popnn_BatchNorm_hpp__
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include <utility>

namespace popnn {
namespace bn {

// Create and map parameter tensors used for batch normalisation in fully
// connected layers
std::pair<poplar::Tensor, poplar::Tensor>
createBatchNormParams(poplar::Graph &graph, const poplar::Tensor acts);

// Estimate mean and inverse of standard deviation of batched activaions
std::pair<poplar::Tensor, poplar::Tensor>
batchNormEstimates(poplar::Graph &graph, const poplar::Tensor acts,
                   float eps,
                   poplar::program::Sequence &prog,
                   const std::string &partialsTypeStr = "float",
                   const std::string &debugPrefix = "");

// Batch normalise activations given mean, standard deviation and batch norm
// parameters. The outputs produced are
// 1) whitened activations
// 2) batch normalised activations
std::pair<poplar::Tensor, poplar::Tensor>
batchNormalise(poplar::Graph &graph,
               const poplar::Tensor &acts,
               const poplar::Tensor &gamma,
               const poplar::Tensor &beta,
               const poplar::Tensor &mean,
               const poplar::Tensor &invStdDev,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

// Computes the output of batch normalisation given
//  combinedMultiplicand = gamma / stdDev
//  addend = beta - gamma * mean / stdDev
poplar::Tensor
batchNormalise(poplar::Graph &graph,
               const poplar::Tensor &acts,
               const poplar::Tensor &combinedMultiplicand,
               const poplar::Tensor &addend,
               poplar::program::Sequence &prog,
               const std::string &debugPrefix = "");

// Compute deltas required for both input gradient and parameter
// update computations
std::pair<poplar::Tensor, poplar::Tensor>
batchNormDeltas(poplar::Graph &graph,
                const poplar::Tensor &actsWhitened,
                const poplar::Tensor &gradsIn,
                poplar::program::Sequence &prog,
                const std::string &partialsType = "float",
                const std::string &debugPrefix = "");

poplar::Tensor
batchNormGradients(poplar::Graph &graph,
                   const poplar::Tensor &actsWhitened,
                   const poplar::Tensor &gradsIn,
                   const poplar::Tensor &gammaDelta,
                   const poplar::Tensor &betaDelta,
                   const poplar::Tensor &invStdDev,
                   const poplar::Tensor &gamma,
                   poplar::program::Sequence &prog,
                   const std::string &partialsTypeStr = "float",
                   const std::string &debugPrefix = "");

void batchNormParamUpdate(poplar::Graph &graph,
                          const poplar::Tensor &gammaDelta,
                          const poplar::Tensor &betaDelta,
                          float learningRate,
                          poplar::Tensor &gamma,
                          poplar::Tensor &beta,
                          poplar::program::Sequence &prog,
                          const std::string &debugPrefix = "");

std::size_t numChannels(poplar::Tensor acts);

std::size_t numActsPerChannel(poplar::Tensor acts);

// In flop computation, following applies
// Acts per channel:
// - for fc layers is the total number of batches.
// - for conv layers it is the field size per channel * batch size
//
// Number of channels:
// - for fc layers is the total number of activations in a batch
// - for conv layers is the total number of channels

uint64_t getFwdFlops(uint64_t numChannels, uint64_t actsPerChannel,
                     bool computeEstimates);
uint64_t getBwdFlops(uint64_t numChannels, uint64_t actsPerChannel);
uint64_t getWuFlops(uint64_t numChannels, uint64_t actsPerChannel);

} // namespace bn
} // namespace popnn
#endif // __popnn_BatchNorm_hpp__
