// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for Connectionist Temporal Classification (CTC) Loss.
 *
 */

#ifndef popnn_CTCLoss_hpp
#define popnn_CTCLoss_hpp

#include "CTCPlan.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace ctc {

/** Create a plan for implementing the CTC Loss (gradient) function
 *
 * **CTC Loss options**
 *
 *    * `partialsType` poplar::Type [=poplar::FLOAT]
 *
 *      The type to use for partial results.
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive) [=0.6]
 *
 *      The maximum proportion of available memory on each tile that this
 *      layer should consume temporarily during the course of the operation.
 *
 * \param graph            The graph the operation will be added to
 * \param inType           The data type of the probability data input
 * \param outType          The data type of the gradient output
 * \param batchSize        The size of the batch to be processed at once
 * \param maxTime          The maximum time of any data input to be planned for
 * \param maxLabelLength   The maximum length of any label to be planned for
 * \param numClasses       The number of symbols/classes in the "alphabet",
 *                         including the blankClass
 * \param options          Any implementation/debug options for the operation
 *
 * \return plan            The plan produced, which will specify how the
 *                         operation is to be implemented
 */
Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          const poplar::Type &outType, unsigned batchSize, unsigned maxTime,
          unsigned maxLabelLength, unsigned numClasses,
          const poplar::OptionFlags &options = {});

/** Create and map a data input [maxTime, batchSize, numClasses] tensor which
 *  the gradient function will use.  Mapping is according to the plan provided.
 *
 * \param graph        The graph the data tensor will be added to
 * \param type         The data type of the tensor to be added to the graph
 * \param batchSize    The size of the batch to be processed at once
 * \param maxTime      The time dimension of the tensor to be created
 * \param numClasses   The number of symbols/classes in the "alphabet",
 *                     including the blankClass
 * \param plan         The plan which will specify how the tensor is to be
 *                     mapped
 * \param debugContext Optional debug information
 * \return             The data input [maxTime, batchSize, numClasses] tensor
 */
poplar::Tensor createDataInput(poplar::Graph &graph, const poplar::Type &type,
                               const std::size_t batchSize,
                               const std::size_t maxTime,
                               const std::size_t numClasses, const Plan &plan,
                               const poplar::DebugContext &debugContext = {});

/** Create and map a labels input [batchSize, maxLabelLength] tensor which the
 *  gradient function will use. Mapping is according to the plan provided.
 *
 * \param graph          The graph the labels tensor will be added to
 * \param type           The data type of the tensor to be added to the graph
 * \param batchSize      The size of the batch to be processed at once
 * \param maxLabelLength The maximum length of any label
 * \param plan           The plan which will specify how the tensor is to be
 *                       mapped
 * \param debugContext   Optional debug information
 * \return               The labels input [batchSize, maxLabelLength] tensor
 */
poplar::Tensor createLabelsInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::size_t batchSize,
                                 const std::size_t maxLabelLength,
                                 const Plan &plan,
                                 const poplar::DebugContext &debugContext = {});

/** Calculate the CTC loss & gradient, creating and mapping the result tensor
 *  according to the plan provided
 *
 * **calcLossAndGradientLogProbabilities options**
 *
 *    * `includeSoftmaxGradient` (true, false) [=true]
 *
 *      Whether or not to include LogSoftmax in gradient calculation. To avoid
 *      numerical issues, it is recommended to be included. But care must be
 *      taken to not include gradient of the LogSoftmax (created external to
 *      this function call) twice.
 *
 * \param graph        The graph the operation will be added to
 * \param outType      The data type of the gradient output
 * \param logProbs     The data input [maxTime, batchSize, numClasses] tensor
 * \param labels       The labels input [batchSize, maxLabelLength] tensor
 * \param dataLengths  A tensor of shape [batchSize] containing the number of
 *                     valid timesteps in each data[] batch entry
 * \param labelLengths A tensor of shape [batchSize] containing the number of
 *                     valid labels in each labels[] batch entry
 * \param prog         A program sequence to append the operation to
 * \param blankClass   The value associated with the blankClass
 * \param plan         The plan which will specify how the output tensor is to
 *                     be mapped and how the operation is to be carried out
 * \param debugContext Optional debug information
 * \param options      Any implementation/debug options for the operation
 *
 * \return             The loss[batchSize] (negative log probability),
 *                     and gradient [maxTime, batchSize, numClasses] tensor
 */
std::pair<poplar::Tensor, poplar::Tensor> calcLossAndGradientLogProbabilities(
    poplar::Graph &graph, const poplar::Type &outType,
    const poplar::Tensor &logProbs, const poplar::Tensor &labels,
    const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
    poplar::program::Sequence &prog, const unsigned blankClass,
    const Plan &plan, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

/** Calculate the CTC loss & gradient, creating and mapping the result tensor
 *  according to the plan provided. Prior to performing the gradient
 *  calculation, applies log softmax to logits input.
 *
 * \param graph        The graph the operation will be added to
 * \param outType      The data type of the gradient output
 * \param logits       The data input [maxTime, batchSize, numClasses] tensor
 * \param labels       The labels input [batchSize, maxLabelLength] tensor
 * \param dataLengths  A tensor of shape [batchSize] containing the number of
 *                     valid timesteps in each data[] batch entry
 * \param labelLengths A tensor of shape [batchSize] containing the number of
 *                     valid labels in each labels[] batch entry
 * \param prog         A program sequence to append the operation to
 * \param blankClass   The value associated with the blankClass
 * \param plan         The plan which will specify how the output tensor is to
 *                     be mapped and how the operation is to be carried out
 * \param debugContext Optional debug information
 * \param options      Any implementation/debug options for the operation
 *
 * \return             The loss[batchSize] (negative log probability),
 *                     and gradient [maxTime, batchSize, numClasses] tensor
 */
std::pair<poplar::Tensor, poplar::Tensor> calcLossAndGradientLogits(
    poplar::Graph &graph, const poplar::Type &outType,
    const poplar::Tensor &logits, const poplar::Tensor &labels,
    const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
    poplar::program::Sequence &prog, const unsigned blankClass,
    const Plan &plan, const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {});

} // namespace ctc
} // namespace popnn

#endif // popnn_CTCLoss_hpp
