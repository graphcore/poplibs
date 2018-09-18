// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popnn_Loss_hpp
#define popnn_Loss_hpp

namespace popnn {

enum LossType {
  SUM_SQUARED_LOSS,
  SOFTMAX_CROSS_ENTROPY_LOSS
};

} // end namespace popnn

#ifndef __POPC__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace popnn {

/** Calculate loss and gradient for a set of activations and
 *  expected labels.
 *
 *  \param graph          Graph to add operations and tensors to.
 *  \param activations    Set of activations per-batch to calculate loss for.
 *  \param expected       Labels per-batch.
 *  \param loss           Tensor to store the loss per-batch
 *  \param deltas         Tensor to store deltas for each activation from
 *                        the expected per-batch.
 *  \param activationType Device type used for activations.
 *  \param expectedType   Device type used for expected labels.
 *  \param stabilityOptimization If true, potentially perform extra calculation
 *                        to improve the numerical stability of the results.
 *  \param lossType       Method for calculating loss measurement.
 *  \param debugPrefix    Optional debug prefix for operations and tensors
 *                        for this operation.
 */
poplar::program::Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor& activations,
         const poplar::Tensor& expected,
         const poplar::Tensor& loss,
         const poplar::Tensor& deltas,
         const poplar::Type& activationType,
         const poplar::Type& expectedType,
         LossType lossType,
         const bool& stabilityOptimization=true,
         const std::string &debugPrefix = "");

/** Calculate loss, gradient, and number of correct classifications
 *  per-batch for a set of activations and expected labels.
 *
 *  \see calcLoss, and \see calcAccuracy which this function is simply
 *  a combination of.
 */
poplar::program::Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor &activations,
         const poplar::Tensor &expected,
         const poplar::Tensor &loss,
         const poplar::Tensor &deltas,
         const poplar::Tensor &numCorrect,
         const poplar::Type &activationType,
         const poplar::Type &expectedType,
         LossType lossType,
         const bool &stabilityOptimization=true,
         const std::string &debugPrefix = "");

/** Calculate the number of correct classifications for a set of
 *  activations and expected labels.
 *
 *  \param graph          Graph to add operations and tensors to.
 *  \param activations    Set of activations per-batch to calculate loss for.
 *  \param expected       Labels per-batch.
 *  \param numCorrect     Tensor to store the number of correct
 *                        classifications. Must be scalar, or single-element
 *                        Tensor.
 *  \param activationType Device type used for activations.
 *  \param expectedType   Device type used for expected labels.
 *  \param debugPrefix    Optional debug prefix for operations and tensors
 *                        for this operation.
 */
poplar::program::Program
calcAccuracy(poplar::Graph &graph,
             const poplar::Tensor &activations,
             const poplar::Tensor &expected,
             const poplar::Tensor &numCorrect,
             const poplar::Type &activationType,
             const poplar::Type &expectedType,
             const std::string &debugPrefix = "");

} // end namespace popnn

#endif // !__POPC__


#endif // popnn_Loss_hpp
