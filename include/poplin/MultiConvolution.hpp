// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support performing convolutions in parallel.
 *
 */

#ifndef poplin_MultiConvolution_hpp
#define poplin_MultiConvolution_hpp

#include "poplin/Convolution.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>

#include <vector>

namespace poplin {

namespace multiconv {

/** Multi-convolutions allow for a set of convolutions to be executed in
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
 *
 * **Multi-Convolution options**
 *
 *    * `planType` (serial, parallel) [=parallel]
 *
 *      Which multi-conv implementation to use. Serial is the same as using the
 *      normal API for each convolution.
 *
 *    * `perConvReservedTiles` Integer [=50]
 *
 *      The amount of tiles to reserve for each convolution when planning.
 *
 *    * `cycleBackOff` Double [=0.1]
 *
 *      A percentage, represented as a proportion between 0 and 1 of how much
 *      off the fastest plan when attempting to plan the largest convolution
 *      using the least amount of tiles.
 *
 *      This number is scaled up according to how many convolutions are being
 *      run in parallel.
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

/** Create a specific weights tensor for the multiconvolution.
 *
 * \param graph        The graph that the tensors will be added to.
 * \param args         The same set of parameters as used by convolution().
 * \param weightsIndex Index into args describing the convolution which to
 *                     create the weights for.
 * \param options      Options controlling the implementation.
 * \param cache        Optional pointer to a planning cache to use.
 * \return             A weights tensor suitable for use with convolution().
 */
poplar::Tensor createWeights(poplar::Graph &graph,
                             const std::vector<CreateTensorArgs> &args,
                             unsigned weightsIndex,
                             const poplar::OptionFlags &options = {},
                             poplin::PlanningCache *cache = nullptr);

/** Create a specific input tensor for the multiconvolution.
 *
 * \param graph      The graph that the tensors will be added to.
 * \param args       The same set of parameters as used by convolution().
 * \param inputIndex Index into args describing the convolution which to
 *                   create the input for.
 * \param options    Options controlling the implementation.
 * \param cache      Optional pointer to a planning cache to use.
 * \return           A tensor suitable for use as an input to convolution().
 */
poplar::Tensor createInput(poplar::Graph &graph,
                           const std::vector<CreateTensorArgs> &args,
                           unsigned inputIndex,
                           const poplar::OptionFlags &options = {},
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

/** For each element in the multi-convolution set, copy the corresponding
 * \p weightsIn element into the convolution weight input such that each
 * element of the kernel is transposed with respect to the input and
 * output channels and each spatial dimension of the kernel is flipped.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of inputs, weights, and convolution parameters
 *                    specifying each convolution in the multiconvolution.
 * \param weightsIn   Collection of weights tensor to copy from, the arrangement
 *                    of which must correspond with the arrangement of the
 *                    collection of convolution parameters.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param options     Options controlling the implementation.
 * \param debugContext Optional debug information.
 * \param cache       Optional pointer to a planning cache to use.
 */
void weightsTransposeChansFlipXY(poplar::Graph &graph,
                                 std::vector<ConvolutionArgs> &args,
                                 const std::vector<poplar::Tensor> &weightsIn,
                                 poplar::program::Sequence &prog,
                                 const poplar::OptionFlags &options,
                                 const poplar::DebugContext &debugContext,
                                 poplin::PlanningCache *cache);

/** Convolve a set of inputs with a set of weights.
 *
 * See Convolution.hpp for more information.
 *
 * \param graph       The graph that the operations will be added to.
 * \param args        Collection of inputs, weights, and convolution parameters
 *                    specifying each convolution in the multiconvolution.
 * \param transposeAndFlipWeights Prepare the weights for the backwards pass.
 * \param prog        Poplar program sequence to append the operations onto.
 * \param debugContext Optional debug information.
 * \param options     Options controlling the implementation.
 * \param cache       Optional pointer to a planning cache to use.
 * \return            Set of convolved output tensors.
 */
std::vector<poplar::Tensor>
convolution(poplar::Graph &graph, const std::vector<ConvolutionArgs> &args,
            bool transposeAndFlipWeights, poplar::program::Sequence &prog,
            const poplar::DebugContext &debugContext = {},
            const poplar::OptionFlags &options = {},
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
 * \param debugContext Optional debug information.
 * \param options     Options controlling the implementation.
 * \param cache       Optional pointer to a planning cache to use.
 * \return            Set of weight deltas.
 */
std::vector<poplar::Tensor>
calculateWeightDeltas(poplar::Graph &graph,
                      const std::vector<CalculateWeightDeltasArgs> &args,
                      poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      poplin::PlanningCache *cache = nullptr);

/**
 * \param zDeltas       Tensor containing gradients with respect to the output
 *                      of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs of the convolution in the
 *                      forward pass.
 * \param scale         Scale to apply to the \p zDeltas.
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
 * \param graph        The graph that the operations will be added to.
 * \param args         Collection of zDeltas, activations, scale, and
 * convolution parameters for the weight updates in the multiconvolution. \param
 * prog         Poplar program sequence to append the operations onto. \param
 * debugContext Optional debug information. \param options      Options
 * controlling the implementation. \param cache        Optional pointer to a
 * planning cache to use.
 */
void convolutionWeightUpdate(poplar::Graph &graph,
                             const std::vector<ConvWeightUpdateArgs> &args,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext = {},
                             const poplar::OptionFlags &options = {},
                             poplin::PlanningCache *cache = nullptr);

/**
 * \param zDeltas       Tensor containing gradients with respect to the output
 *                      of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs of the convolution in the
 *                      forward pass.
 * \param scale         Scale to apply to the \p zDeltas.
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
 * \param debugContext Optional debug information.
 * \param options     Options controlling the implementation.
 * \param cache       Optional pointer to a planning cache to use.
 */
void convolutionWeightUpdate(
    poplar::Graph &graph, const std::vector<ConvWeightUpdateArgsScalar> &args,
    poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {},
    poplin::PlanningCache *cache = nullptr);

} // namespace multiconv
} // namespace poplin

#endif // poplin_MultiConvolution_hpp
