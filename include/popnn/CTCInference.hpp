// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for Connectionist Temporal Classification (CTC) Beam search decoder.
 *
 */

#ifndef popnn_CTCInference_hpp
#define popnn_CTCInference_hpp

#include "CTCPlan.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace ctc_infer {

/** Create a plan for implementing the CTC Beam search inference function
 *
 * **CTC Beam search inference options**
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
 * \param batchSize        The size of the batch to be processed at once
 * \param maxTime          The maximum time of any sequence input
 * \param numClasses       The number of symbols/classes in the "alphabet",
 *                         including the blankClass
 * \param beamwidth        The number of beams to maintain during beamsearch
 * \param options          Any implementation/debug options for the operation
 *
 * \return plan            The plan produced, which will specify how the
 *                         operation is to be implemented
 */
ctc::Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
               unsigned batchSize, unsigned maxTime, unsigned numClasses,
               unsigned beamwidth, const poplar::OptionFlags &options = {});

/** Create and map a data input [maxTime, batchSize, numClasses] tensor which
 *  the beam search function will use.  Mapping is according to the plan
 *  provided.
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
poplar::Tensor createDataInput(poplar::Graph &graph, poplar::Type &type,
                               const std::size_t batchSize,
                               const std::size_t maxTime,
                               const std::size_t numClasses,
                               const ctc::Plan &plan,
                               const poplar::DebugContext &debugContext = {});

/** Calculate the most likely \p topPaths labels and their probabilities given
 * the input \p logProbs with lengths \p dataLengths, creating and mapping the
 * result tensors according to the plan provided
 *
 * \param graph        The graph the operation will be added to
 * \param logProbs     The data input [maxTime, batchSize, numClasses] tensor
 * \param dataLengths  A tensor of shape [batchSize] containing the number of
 *                     valid timesteps in each \p logProbs batch entry
 * \param prog         A program sequence to append the operation to
 * \param blankClass   The value associated with the blankClass
 * \param beamWidth    The number of beams to use when decoding
 * \param topPaths     The number of most likely decoded paths to return,
 *                     must be less than or equal to \p beamWidth
 * \param plan         The plan which will specify how the output tensor is to
 *                     be mapped and how the operation is to be carried out
 * \param debugContext Optional debug information
 * \param options      Any implementation/debug options for the operation
 *
 * \return             The labelProbs[batchSize, topPaths] (negative log
 *                     probability with the same type as \p logProbs),
 *                     labelLengths[batchSize, topPaths]
 *                     and decodedLabels [batchSize, topPaths, maxTime] tensors
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
beamSearchDecoderLogProbabilities(poplar::Graph &graph,
                                  const poplar::Tensor &logProbs,
                                  const poplar::Tensor &dataLengths,
                                  poplar::program::Sequence &prog,
                                  unsigned blankClass, unsigned beamwidth,
                                  unsigned topPaths, const ctc::Plan &plan,
                                  const poplar::DebugContext = {},
                                  const poplar::OptionFlags &options = {});

/** Calculate the most likely \p topPaths labels and their probabilities given
 * the input \p logits with lengths \p dataLengths, creating and mapping the
 * result tensors according to the plan provided. Prior to performing the
 * beam search, applies log softmax to logits input.
 *
 * \param graph        The graph the operation will be added to
 * \param logits       The data input [maxTime, batchSize, numClasses] tensor
 * \param dataLengths  A tensor of shape [batchSize] containing the number of
 *                     valid timesteps in each \p logits batch entry
 * \param prog         A program sequence to append the operation to
 * \param blankClass   The value associated with the blankClass
 * \param beamWidth    The number of beams to use when decoding
 * \param topPaths     The number of most likely decoded paths to return,
 *                     must be less than or equal to \p beamWidth
 * \param plan         The plan which will specify how the output tensor is to
 *                     be mapped and how the operation is to be carried out
 * \param debugContext Optional debug information
 * \param options      Any implementation/debug options for the operation
 *
 * \return             The labelProbs[batchSize, topPaths] (negative log
 *                     probability with the same type as \p logits),
 *                     labelLengths[batchSize, topPaths]
 *                     and decodedLabels [batchSize, topPaths, maxTime] tensors
 */
std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
beamSearchDecoderLogits(poplar::Graph &graph, const poplar::Tensor &logits,
                        const poplar::Tensor &dataLengths,
                        poplar::program::Sequence &prog, unsigned blankClass,
                        unsigned beamwidth, unsigned topPaths,
                        const ctc::Plan &plan, const poplar::DebugContext = {},
                        const poplar::OptionFlags &options = {});
} // namespace ctc_infer
} // namespace popnn

#endif // popnn_CTCLoss_hpp
