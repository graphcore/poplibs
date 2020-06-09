// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplin_MultiConvolution_hpp
#define poplin_MultiConvolution_hpp

#include "poplin/Convolution.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>

#include <vector>

namespace poplin {

namespace multiconv {

/** Multiconvolutions allow for a set of convolutions to be executed in
 * parallel. The benefit of executing convolutions in parallel is an increase in
 * data throughput. Specifically, executing N independent convolutions in
 * parallel will be faster than sequentially executing them because less time is
 * spent on the ~constant vertex overhead per tile.
 *
 * Note that the allocation of associated tensors for convolutions should be
 * done through the same api such that they are mapped across tiles
 * appropriately for the operation.
 *
 * See Convolution.hpp for information about convolutions and each individual
 * operation.
 */

/**
 * \param params  Parameters specifying the convolution.
 * \param options Options controlling the implementation.
 * \param name    Debugging name for the tensor.
 */
struct CreateTensorArgs {
  ConvParams params;
  poplar::OptionFlags options;
  std::string name;
};

/** Create the set of weights tensors.
 *
 * \param graph   The graph that the tensors will be added to.
 * \param args    The same set of parameters as used by convolution().
 * \param cache   Optional pointer to a planning cache to use.
 * \return        The set of weights tensors suitable for use with
 * convolution().
 */
std::vector<poplar::Tensor>
createWeights(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args,
              poplin::PlanningCache *cache = nullptr);

/** Create a specific weights tensor for the multiconvolution.
 *
 * \param graph        The graph that the tensors will be added to.
 * \param args         The same set of parameters as used by convolution().
 * \param weightsIndex Index into args describing the convolution which to
 *                     create the weights for.
 * \param cache        Optional pointer to a planning cache to use.
 * \return             A weights tensor suitable for use with convolution().
 */
poplar::Tensor createWeights(poplar::Graph &graph,
                             const std::vector<CreateTensorArgs> &args,
                             unsigned weightsIndex,
                             poplin::PlanningCache *cache = nullptr);

/** Create the set of input tensors.
 *
 * \param graph   The graph that the tensors will be added to.
 * \param args    The same set of parameters as used by convolution().
 * \param cache   Optional pointer to a planning cache to use.
 * \return        The set of input tensors suitable for use with convolution().
 */
std::vector<poplar::Tensor>
createInput(poplar::Graph &graph, const std::vector<CreateTensorArgs> &args,
            poplin::PlanningCache *cache = nullptr);

/** Create a specific input tensor for the multiconvolution.
 *
 * \param graph      The graph that the tensors will be added to.
 * \param args       The same set of parameters as used by convolution().
 * \param inputIndex Index into args describing the convolution which to
 *                   create the input for.
 * \param cache      Optional pointer to a planning cache to use.
 * \return           A input tensor suitable for use with convolution().
 */
poplar::Tensor createInput(poplar::Graph &graph,
                           const std::vector<CreateTensorArgs> &args,
                           unsigned inputIndex,
                           poplin::PlanningCache *cache = nullptr);

/**
 * \param in                      Input tensor.
 * \param weights                 Weights tensor.
 * \param params                  Parameters specifying the convolution.
 * \param options                 Options controlling the implementation.
 */
struct ConvolutionArgs {
  poplar::Tensor inputs;
  poplar::Tensor weights;
  ConvParams params;
  poplar::OptionFlags options;
};

/** Convolve a set of inputs with a set of weights.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of inputs, weights, and convolution parameters
 *                    specifying each convolution in the multiconvolution.
 * \param transposeAndFlipWeights Prepare the weights for the backwards pass.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param debugPrefix Name of the operation, for debugging.
 * \param cache       Optional pointer to a planning cache to use.
 * \return            Set of convolved output tensors.
 */
std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args,
            bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const std::string &debugPrefix = "",
            poplin::PlanningCache *cache = nullptr);

/**
 * \param zDeltas     Tensor containing gradients with respect to
 *                    the output of the convolution.
 * \param activations Tensor containing the inputs of the convolution in the
 *                    forward pass.
 * \param params      Parameters specifying the convolution.
 * \param options     Options controlling the implementation.
 */
struct CalculateWeightDeltasArgs {
  poplar::Tensor zDeltas;
  poplar::Tensor activations;
  ConvParams params;
  poplar::OptionFlags options;
};

/** Append an operation to generate the set of weight delta tensors.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of zDeltas, activations, and convolution
 *                    parameters specifying each convolution in the
 *                    multiconvolution.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param debugPrefix Name of the operation, for debugging.
 * \param cache       Optional pointer to a planning cache to use.
 * \return            Set of weight deltas.
 */
std::vector<poplar::Tensor> calculateWeightDeltas(
    poplar::Graph &graph, const std::vector<CalculateWeightDeltasArgs> &args,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    poplin::PlanningCache *cache = nullptr);

/**
 * \param zDeltas       Tensor containing gradients with respect to the output
 *                      of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs of the convolution in the
 *                      forward pass.
 * \param scale         Scale to apply to the zDeltas.
 * \param params        Parameters specifying the convolution.
 * \param options       Options controlling the implementation.
 */
struct ConvWeightUpdateArgs {
  poplar::Tensor zDeltas;
  poplar::Tensor weights;
  poplar::Tensor activations;
  poplar::Tensor scale;
  ConvParams params;
  poplar::OptionFlags options;
};

/** Append operations to \p prog to generate and apply the weight update.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of zDeltas, activations, scale, and convolution
 *                    parameters for the weight updates in the multiconvolution.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param debugPrefix Name of the operation, for debugging.
 * \param cache       Optional pointer to a planning cache to use.
 */
void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const std::string &debugPrefix = "",
                             poplin::PlanningCache *cache = nullptr);

/**
 * \param zDeltas       Tensor containing gradients with respect to the output
 *                      of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs of the convolution in the
 *                      forward pass.
 * \param scale         Scale to apply to the zDeltas.
 * \param params        Parameters specifying the convolution.
 * \param options       Options controlling the implementation.
 */
struct ConvWeightUpdateArgsScalar {
  poplar::Tensor zDeltas;
  poplar::Tensor weights;
  poplar::Tensor activations;
  float scale;
  ConvParams params;
  poplar::OptionFlags options;
};

/** Append operations to \p prog to generate and apply the weight update.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of zDeltas, activations, scale, and convolution
 *                    parameters for the weight updates in the multiconvolution.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param debugPrefix Name of the operation, for debugging.
 * \param cache       Optional pointer to a planning cache to use.
 */
void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog, const std::string &debugPrefix = "",
    poplin::PlanningCache *cache = nullptr);

} // namespace multiconv
} // namespace poplin

#endif // poplin_MultiConvolution_hpp
