// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#ifndef popnn_Pooling_hpp
#define popnn_Pooling_hpp
#include <cstdint>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <popnn/PoolingDef.hpp>
#include <tuple>

namespace popnn {
namespace pooling {

struct PoolParams {
  PoolingType poolingType;
  std::vector<std::size_t> inputFieldShape;
  std::vector<std::size_t> kernelShape;
  std::vector<unsigned> stride;
  std::vector<int> inputTruncationOrPaddingLower;
  std::vector<int> inputTruncationOrPaddingUpper;
  std::size_t numChannels;
  std::size_t batchSize;
  poplar::Type dType;

  PoolParams(PoolingType poolingType, std::vector<std::size_t> inputFieldShape,
             std::vector<std::size_t> kernelShape, std::vector<unsigned> stride,
             std::vector<int> inputTruncationOrPaddingLower,
             std::vector<int> inputTruncationOrPaddingUpper,
             std::size_t numChannels, std::size_t batchSize, poplar::Type dType)
      : poolingType(poolingType), inputFieldShape(std::move(inputFieldShape)),
        kernelShape(std::move(kernelShape)), stride(std::move(stride)),
        inputTruncationOrPaddingLower(std::move(inputTruncationOrPaddingLower)),
        inputTruncationOrPaddingUpper(std::move(inputTruncationOrPaddingUpper)),
        numChannels(numChannels), batchSize(batchSize), dType(dType) {}

  std::size_t getNumFieldDims() const { return inputFieldShape.size(); }
  std::vector<std::size_t> getOutputFieldShape() const;
};

std::ostream &operator<<(std::ostream &o, const PoolParams &params);

const char *asString(const PoolingType &method);

std::vector<std::size_t> getOutputFieldShape(const PoolParams &params);

uint64_t getFwdFlops(const PoolParams &params);

uint64_t getBwdFlops(const PoolParams &params);

double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               const PoolParams &params);

double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               const PoolParams &params);

/** Add a pooling operation to the graph
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of
 * the input should be [B x inChans x ...].
 *
 * \param graph             The operation will be added to this graph
 * \param params            Pooling parameters
 * \param in                Input tensor
 * \param prog              Program sequence to append the operation to
 * \param debugPrefix       Debug name for the operation
 * \param options           Pooling options (not currently used)
 * \return                  A tensor with the results of the pooling operation
 */
/*[INTERNAL]
 * **Pooling options**
 *
 *    * `poolUseIntrospectiveMapping` (true, false) [=true]
 *
 *      If true, take into account the tile mapping of the output tensor (where
 *      it is provided in as an argument) or the input tensor when deciding how
 *      to map the pooling operation across tiles.
 */
poplar::Tensor pool(poplar::Graph &graph, const PoolParams &params,
                    const poplar::Tensor &in, poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "",
                    const poplar::OptionFlags &options = {});

/** For MAX, AVG or SUM pooling.
 *  Note - recommend the specific function for AVG or SUM pooling, below.
 *  Calculate the gradient w.r.t. to the input of a pooling operation given
 *  the gradient of the output.
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of
 * the input should be [B x inChans x ...].
 *
 * \param graph             The operation will be added to this graph
 * \param params            Pooling parameters
 * \param in                Forward activations tensor input to pooling
 * \param pooled            Output of pooling in the forward pass
 * \param pooledGradient    Gradients to the pooling operation
 * \param useScaledGradient Use scaled gradient if set to true. Otherwise, the
 *                          gradient is propagated to all the positions which
 *                          matched pooled value in forward pass.
 * \param prog              Program sequence to append the operation to
 * \param debugPrefix       Debug name for the operation
 * \param options           Pooling options. See pool().
 * \return                  A tensor with the results of the pooling operation
 */
poplar::Tensor poolInputGradient(poplar::Graph &graph, const PoolParams &params,
                                 const poplar::Tensor &in,
                                 const poplar::Tensor &pooled,
                                 const poplar::Tensor &pooledGradient,
                                 bool useScaledGradient,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {});
/** For AVG and SUM pooling
 *  Calculate the gradient w.r.t. to the input of a pooling operation given
 *  the gradient of the output.
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of
 * the output will be [B x inChans x ...].
 *
 * \param graph             The operation will be added to this graph
 * \param params            Pooling parameters
 * \param fwdChansPerGroup  Used in creating the output tensor
 * \param pooledGradient    Gradients to the pooling operation
 * \param prog              Program sequence to append the operation to
 * \param debugPrefix       Debug name for the operation
 * \param options           Pooling options. See pool().
 * \return                  A tensor with the results of the pooling operation
 */
poplar::Tensor poolInputGradient(poplar::Graph &graph, const PoolParams &params,
                                 const unsigned fwdChansPerGroup,
                                 const poplar::Tensor &pooledGradient,
                                 poplar::program::Sequence &prog,
                                 const std::string &debugPrefix = "",
                                 const poplar::OptionFlags &options = {});

} // namespace pooling
} // namespace popnn

#endif // popnn_Pooling_hpp
