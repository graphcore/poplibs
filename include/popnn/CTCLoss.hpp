// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for Connectionist Temporal Classification (CTC) Loss.
 *
 */

#ifndef popnn_CTCLoss_hpp
#define popnn_CTCLoss_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poputil/DebugInfo.hpp>

namespace popnn {
namespace ctc {

/** An object representing a plan that describes how to map tensors and
 *  implement the CTC Loss function.
 */
class Plan {
public:
  Plan();
  ~Plan();
  Plan &operator=(Plan &&);

  friend std::ostream &operator<<(std::ostream &o, const Plan &p);
  friend poplar::ProfileValue poputil::toProfileValue<>(const Plan &p);

  // Internal implementation
  class Impl;
  Impl &getImpl() const { return *impl; }
  Plan(std::unique_ptr<Impl> impl);

private:
  std::unique_ptr<Impl> impl;
};

/** Create a plan for implementing the CTC Loss (gradient) function
 *
 * \param graph       The graph the operation will be added to
 * \param inType      The data type of the probability data input
 * \param outType     The data type of the gradient output
 * \param batchSize   The size of the batch to be processed at once
 * \param maxTime     The maximum time of any data input to be planned for
 * \param maxLabels   The maximum length of any label to be planned for
 * \param numClasses  The number of symbols/classes in the "alphabet", including
 *                    the blankClass
 * \return plan       The plan produced, which will specify how the operation
 *                    is to be implemented
 */
Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          const poplar::Type &outType, unsigned batchSize, unsigned maxTime,
          unsigned maxLabels, unsigned numClasses);

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

/** Create and map a labels input [batchSize, maxLabels] tensor which the
 *  gradient function will use. Mapping is according to the plan provided.
 *
 * \param graph        The graph the labels tensor will be added to
 * \param type         The data type of the tensor to be added to the graph
 * \param batchSize    The size of the batch to be processed at once
 * \param maxLabels    The labels dimension of the tensor to be created
 * \param plan         The plan which will specify how the tensor is to be
 *                     mapped
 * \param debugContext Optional debug information
 * \return             The labels input [batchSize, maxLabels] tensor
 */
poplar::Tensor createLabelsInput(poplar::Graph &graph, const poplar::Type &type,
                                 const std::size_t batchSize,
                                 const std::size_t maxLabels, const Plan &plan,
                                 const poplar::DebugContext &debugContext = {});

/** Calculate the CTC loss gradient, creating and mapping the result tensor
 *  according to the plan provided
 *
 * \param graph        The graph the operation will be added to
 * \param outType      The data type of the gradient output
 * \param data         The data input [maxTime, batchSize, numClasses] tensor
 * \param labels       The labels input [batchSize, maxLabels] tensor
 * \param dataLengths  A tensor of shape [batchSize] containing the number of
 *                     valid timesteps in each data[] batch entry
 * \param labelLengths A tensor of shape [batchSize] containing the number of
 *                     valid labels in each labels[] batch entry
 * \param prog         A program sequence to append the operation to
 * \param blankClass   The value associated with the blankClass
 * \param plan         The plan which will specify how the output tensor is to
 *                     be mapped and how the operation is to be carried out
 * \param debugContext Optional debug information
 * \return             The gradient [maxTime, batchSize, numClasses] tensor
 */
poplar::Tensor
gradient(poplar::Graph &graph, const poplar::Type &outType,
         const poplar::Tensor &data, const poplar::Tensor &labels,
         const poplar::Tensor &dataLengths, const poplar::Tensor &labelLengths,
         poplar::program::Sequence &prog, const unsigned blankClass,
         const Plan &plan, const poplar::DebugContext &debugContext = {});

} // namespace ctc
} // namespace popnn

#endif // popnn_CTCLoss_hpp