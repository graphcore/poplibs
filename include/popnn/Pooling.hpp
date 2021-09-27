// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file
 *  Support for pooling operations.
 */

#ifndef popnn_Pooling_hpp
#define popnn_Pooling_hpp
#include <cstdint>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <popnn/PoolingDef.hpp>
#include <tuple>

namespace popnn {
/// Functions for pooling operations
namespace pooling {

struct PoolParams {
  /// The type of pooling to be performed (for example, maximum, sum or
  /// average).
  PoolingType poolingType;
  /// The input shape not including the dimensions for batch size `batchSize`
  /// and number of channels `numChannels`.
  std::vector<std::size_t> inputFieldShape;
  /// The shape of the pooling kernel.
  std::vector<std::size_t> kernelShape;
  /// The stride per input dimension.
  std::vector<unsigned> stride;
  /// The lower padding (for values >0) or truncation (for values <0) for each
  /// input dimension specifying the padding/truncation at the start of the
  /// dimension.
  std::vector<int> inputTruncationOrPaddingLower;
  /// The upper padding (for values >0) or truncation (for values <0) for each
  /// input dimension specifying the padding/truncation at the end of the
  /// dimension.
  std::vector<int> inputTruncationOrPaddingUpper;
  /// The number of channels in the input tensor.
  std::size_t numChannels;
  /// The batch size of the input tensor.
  std::size_t batchSize;
  /// The output type.
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

/** Add a pooling operation to the graph.
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of
 * the input should be [`batchSize * numChannels * ...`].
 *
 * \param graph             The graph that the operation will be added to.
 * \param params            The pooling parameters.
 * \param in                The input tensor.
 * \param prog              The program sequence to append the operation to.
 * \param debugContext      Optional debug information.
 * \param options           Pooling options (not currently used).
 * \return                  A tensor with the results of the pooling operation.
 */
/*[INTERNAL]
 * **Pooling options**
 *
 *    * `poolUseIntrospectiveMapping` (true, false) [=true]
 *
 *      If true, take into account the tile mapping of the output tensor (where
 *      it is provided as an argument) or the input tensor when deciding how
 *      to map the pooling operation across tiles.
 */
poplar::Tensor pool(poplar::Graph &graph, const PoolParams &params,
                    const poplar::Tensor &in, poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext = {},
                    const poplar::OptionFlags &options = {});

// clang-format off
// Need long lines for Doxygen
/** Calculate the gradient with respect to the input of a pooling operation
 * given the gradient of the output.
 *
 * While this function can be used for maximum, average or sum pooling, poolInputGradient(poplar::Graph&, const PoolParams&, const unsigned, const poplar::Tensor&, poplar::program::Sequence&, const poplar::DebugContext&, const poplar::OptionFlags&) is recommended for average or sum pooling.
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of
 * the input should be [`batchSize * numChannels * ...`].
 *
 * \param graph             The graph the operation will be added to.
 * \param params            The pooling parameters.
 * \param in                The forward activations tensor input to pooling.
 * \param pooled            The output of pooling in the forward pass.
 * \param pooledGradient    The gradients to the pooling operation.
 * \param useScaledGradient Use a scaled gradient if set to true. Otherwise, the
 *                          gradient is propagated to all the positions which
 *                          matched the pooled value in the forward pass.
 * \param prog              The program sequence to append the operation to.
 * \param debugContext      Optional debug information.
 * \param options           The pooling options. See pool().
 * \return                  A tensor with the results of the pooling operation.
 */
// clang-format on
poplar::Tensor poolInputGradient(poplar::Graph &graph, const PoolParams &params,
                                 const poplar::Tensor &in,
                                 const poplar::Tensor &pooled,
                                 const poplar::Tensor &pooledGradient,
                                 bool useScaledGradient,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext = {},
                                 const poplar::OptionFlags &options = {});

/** Calculate the gradient with respect to the input of a pooling operation
 * given the gradient of the output.
 *
 * This function should be used for average and sum pooling.
 *
 * This performs a pooling over the spatial dimensions [...].  The shape of the
 * output will be [` batchSize * numChannels * ...`].
 *
 * \param graph             The graph the operation will be added to.
 * \param params            The pooling parameters.
 * \param fwdChansPerGroup  Used in creating the output tensor.
 * \param pooledGradient    The gradients to the pooling operation.
 * \param prog              The program sequence to append the operation to.
 * \param debugContext      Optional debug information.
 * \param options           The pooling options. See pool().
 * \return                  A tensor with the results of the pooling operation.
 */
poplar::Tensor poolInputGradient(poplar::Graph &graph, const PoolParams &params,
                                 const unsigned fwdChansPerGroup,
                                 const poplar::Tensor &pooledGradient,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext = {},
                                 const poplar::OptionFlags &options = {});

} // namespace pooling
} // namespace popnn

#endif // popnn_Pooling_hpp
