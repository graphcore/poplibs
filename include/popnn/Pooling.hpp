#ifndef __popnn_MaxPool_hpp__
#define __popnn_MaxPool_hpp__
#include <popnn/PoolingDef.hpp>
#include <tuple>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <cstdint>

namespace popnn {
namespace pooling {

const char *asString(const PoolingType &method);

std::pair<unsigned, unsigned>
getOutputDim(unsigned inDimY, unsigned inDimX,
             const std::vector<std::size_t> &kernelShape,
             const std::vector<unsigned> &stride,
             const std::vector<int> &inputPaddingLower,
             const std::vector<int> &inputPaddingUpper);

uint64_t getFwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     const std::vector<std::size_t> &kernelShape,
                     const std::vector<unsigned> &stride,
                     const std::vector<int> &inputPaddingLower,
                     const std::vector<int> &inputPaddingUpper,
                     PoolingType poolingType);

uint64_t getBwdFlops(unsigned batchSize,
                     unsigned inDimY, unsigned inDimX,
                     unsigned numChannels,
                     const std::vector<std::size_t> &kernelShape,
                     const std::vector<unsigned> &stride,
                     const std::vector<int> &inputPaddingLower,
                     const std::vector<int> &inputPaddingUpper,
                     PoolingType poolingType);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               const std::vector<std::size_t> &kernelShape,
                               const std::vector<unsigned> &stride,
                               const std::vector<int> &inputPaddingLower,
                               const std::vector<int> &inputPaddingUpper,
                               PoolingType poolingType);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               std::string dType, unsigned batchSize,
                               unsigned inDimY, unsigned inDimX,
                               unsigned numChannels,
                               const std::vector<std::size_t> &kernelShape,
                               const std::vector<unsigned> &stride,
                               const std::vector<int> &inputPaddingLower,
                               const std::vector<int> &inputPaddingUpper,
                               PoolingType poolingType);

/** Add a pooling operation to the graph
 *
 * This performs a pooling over the spatial dimensions [Y, X].  The shape of
 * the input should be [B x Y x X x inChans].
 *
 * \param graph             The operation will be added to this graph
 * \param poolingType       Type of pooling operation to perform
 *                          (\see PoolingType)
 * \param kernelShape       Shape of the pooling kernel
 * \param stride            Stride over the pooling dimensions
 * \param inputPaddingLower Lower padding over the pooling dimensions
 * \param inputPaddingUpper Upper padding over the pooling dimensions
 * \param in                Input tensor
 * \param prog              Program sequence to append the operation to
 * \param debugPrefix       Debug name for the operation
 * \return                  A tensor with the results of the pooling operation
 */
poplar::Tensor
pool(poplar::Graph &graph,
     PoolingType poolingType,
     const std::vector<std::size_t> &kernelShape,
     const std::vector<unsigned> &stride,
     const std::vector<int> &inputPaddingLower,
     const std::vector<int> &inputPaddingUpper,
     poplar::Tensor in, poplar::program::Sequence &prog,
     const std::string &debugPrefix = "");

// Calculate the gradient w.r.t. to the input of a pooling operation given
// the gradient of the output.
poplar::Tensor
poolInputGradient(poplar::Graph &graph,
                  PoolingType poolingType,
                  const std::vector<std::size_t> &kernelShape,
                  const std::vector<unsigned> &stride,
                  const std::vector<int> &inputPaddingLower,
                  const std::vector<int> &inputPaddingUpper,
                  poplar::Tensor in,
                  poplar::Tensor pooled,
                  poplar::Tensor pooledGradient,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");
}
}

#endif  // __MaxPool_hpp__
