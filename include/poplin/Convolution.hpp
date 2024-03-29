// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions and data types to support performing convolutions.
 *
 */

#ifndef poplin_Convolution_hpp
#define poplin_Convolution_hpp
#include "ConvParams.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <set>
#include <tuple>

namespace poplin {

/** Class used to cache the calculation of plans for convolution operations.
 */
class PlanningCache;

/// Calculate the minimum number of floating point operations required to
/// perform the forward pass convolution given a set of \p params.
uint64_t getFwdFlops(const ConvParams &params);
/// Calculate the minimum number of floating point operations required to
/// perform the backward pass convolution given a set of \p params.
uint64_t getBwdFlops(const ConvParams &params);
/// Calculate minimum number of floating point operations required to
/// perform the weight update pass convolution given a set of \p params.
uint64_t getWuFlops(const ConvParams &params);

/** Calculate the number of cycles to perform the forward pass assuming maximal
 * utilisation of target hardware performing the minimum number of floating
 * point operations. This takes into account the number of tiles available and
 * vectorization support on the target.
 *
 * This is an optimistic number useful for estimating efficiency:
 *      `cycleCount =` getFwdFlops() `/ maximumHardwareVectorization`.
 *
 * \param graph     Provides target the convolution will run on.
 * \param params    Description of convolution.
 * \return Estimated number of cycles to perform the forward pass.
 */
double getFwdPerfectCycleCount(const poplar::Graph &graph,
                               const ConvParams &params);

/** Calculate the number of cycles to perform the backward pass assuming maximal
 * utilisation of the target hardware, performing the minimum number of floating
 * point operations. This takes into account the number of tiles available and
 * vectorization support on the target.
 *
 * This is an optimistic number useful for estimating efficiency:
 *      `cycleCount = getBwdFlops() / maximumHardwareVectorization`.
 *
 * \param graph     Provides target the convolution will run on.
 * \param params    Description of convolution.
 * \return Estimated number of cycles to perform the backward pass.
 */
double getBwdPerfectCycleCount(const poplar::Graph &graph,
                               const ConvParams &params);

/** Calculate the number of cycles to perform the weight update pass assuming
 * maximal utilisation of the target hardware, performing the minimum number of
 * floating point operations. This takes into account the number of tiles
 * available and vectorization support on the target.
 *
 * This is an optimistic number useful for estimating efficiency.
 *      cycleCount = getWuFlops() / maximumHardwareVectorization
 *
 * \param graph     Provides target the convolution will run on.
 * \param params    Description of convolution.
 * \return Estimated number of cycles to perform the weight update pass.
 */
double getWuPerfectCycleCount(const poplar::Graph &graph,
                              const ConvParams &params);

/** Create a weight tensor suitable for use with convolution()
 *
 * The shape of the tensor will be [convGroups x outChansPerConvGroup  x
 * inChansPerConvGroup x H x W]
 *
 * **Convolution options**
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive) [=0.6]
 *
 *      The amount of memory allocated for use for temporary data whilst the
 *      operation is executing (for example, for intermediate calculated values
 *      or temporary values passed between tiles on the IPU). The value is
 *      specified as a proportion of available memory on the IPU. So, for
 *      example, a value of 0.1 will constrain the library to use 10% of the
 *      total memory for temporary data.
 *
 *      The library will try and constrain the use of temporary memory to below
 *      this value. An operation that has more temporary memory
 *      available to use will run in the same or fewer cycles.
 *
 *      For a specific operation, the minimum amount of temporary memory the
 *      library is able to use may be more than the amount specified by this
 *      option. In this case, if `POPLIBS_LOG_LEVEL=WARN` or
 *      `POPLIBS_POPLIN_LOG_LEVEL=WARN`, a warning
 *      message will be output, and the amount specified by this option is
 *      ignored.
 *
 *      Note: if this value is set to less than 5% of memory (so, a value less
 *      than 0.05) then it is often the case that the library will need to
 *      create a large amount of code and data structures to keep the temporary
 *      memory low which could have a permanent memory overhead larger than the
 *      saving of temporary memory. You should take great care when setting a
 *      value this low.
 *
 *      \sa [Optimising Temporary Memory Usage for
 *      Convolutions and Matmuls on the IPU]
 *      (https://docs.graphcore.ai/projects/available-memory/)
 *       technical note for some practical examples of using
 *       `availableMemoryProportion`
 *
 *    * `partialsType` (half, float) [=float]
 *
 *      Data type used for intermediate calculations. If the type specified
 *      is smaller than the output type then the option is ignored and the
 *      output type is used instead.
 *
 *    * `pass` (NONE, INFERENCE_FWD, TRAINING_FWD, TRAINING_BWD, TRAINING_WU,
 *      FC_INFERENCE_FWD, FC_TRAINING_FWD, FC_TRAINING_BWD, FC_TRAINING_WU)
 *      [=NONE]
 *
 *      Optimize the plan for the specified type of pass. Note the
 *      abbreviations:
 *      FWD (forward), BWD (backward), WU (weight-update), FC (fully-connected).
 *
 *    * `use128BitConvUnitLoad` (true, false) [=false]
 *
 *      If true, convolution weights are loaded 128-bits at a time. Otherwise,
 *      they are loaded 64-bits at a time. Not all codelets support 128-bit
 *      loads. This option affects memory usage and cycle count.
 *
 *    * `enableMultiStageReduce` (true, false) [=true]
 *
 *      If true, perform the reduction following the convolution in multiple
 *      stages if it would significantly reduce code size. This comes at the
 *      cost of increasing the number of cycles.
 *
 *    * `enableFastReduce` (true, false) [=false]
 *
 *      If true, use a faster reduction vertex if the data types and widths
 *      allow it.  This comes at the cost of further constraints on memory
 *      allocation
 *
 *    * `enableConvDithering`       (true, false) [=false]
 *
 *       If true, then convolutions with different parameters will be laid out
 *       from different tiles in an effort to improve tile balance in models.
 */
/*[INTERNAL]
 *    * `numIPUs` Integer [=target.getNumIPUs()]
 *
 *      Number of IPUs to be used.
 *
 *   * `remapOutputTensor`       (true, false) [=true]
 *
 *      If true, the output of the convolution is remapped if the output
 *      is detected to have a poor layout. The convolutions planner will try
 *      to map the channels in groups of 16, 8 or 4. This typically results
 *      in better performance for the operation(s) consuming the output
 *      of the convolution.
 *
 *
 *    * `planConstraints` JSON string
 *
 *      Constraints on the chosen convolution plan. Example:
 *
 *          {"0": {"transform": {"swapOperands": true},
 *                 "partition": {"fieldSplit":{"1": 4},
 *                               "inChanSplit": 4,
 *                               "outChanSplit": {"parallel": 4}}
 *                }
 *          }
 *
 *      Where the outer-most index in the plan is an index into the plan
 *      hierarchy, and any multi-dimensional fields are sparsely indexed
 *      objects. Therefore, constraining dimension 1 of fieldSplit to be 4 is
 *      specified as:
 *
 *          {"fieldSplit": {"1": 4}}
 *
 *      This is only implemented for `partitioning` and for the `swapOperands`
 *      transform for now.
 *
 *    * `planConstraintsOutputFilename` String
 *
 *      If set, plan constraints for each plan used by a convolution will be
 *      saved to file. The file path will be the value of this option appended
 *      with _FWD, _BWD, or _WU (depending on the pass), with a file extension
 *      of .json. The content of these files may be used as input to the
 *      `planConstraints` option (above). The constraints will be complete,
 *      meaning they can only be satisfied by one specific plan - this allows
 *      reliable reproduction regardless of changes to the planner.
 *
 *    * `partialsType.interIPU` (half, float) [=`partialsType`]
 *
 *      Data type of inter-IPU partials. If the type specified
 *      is smaller than the output type then the option is ignored and the
 *      output type is used instead.
 *
 *    * `partialsType.interTile` (half, float) [=`partialsType`]
 *
 *      Data type of inter-tile partials. If the type specified
 *      is smaller than the output type then the option is ignored and the
 *      output type is used instead.
 *
 *    * `tilesPerIPU` Integer [=target.getTilesPerIPU()]
 *
 *      Number of tiles per IPU to be used.
 *
 *   * `gatherConvOutput` (true, false) [=false]
 *
 *     Gather output of the matrix multiply into a single variable
 *
 *   * 'experimental.slicVmac16' (true, false) [=false]
 *
 *     Restricts convolution planner to use SLIC/VMAC vertices with
 *     grouping of 16
 *
 *   * 'disableSRForAMPVertices' (true, false) [=false]
 *
 *     Disable stochastic rounding for vertices that use AMP
 *
 */
/**
 * \param graph   The graph that the tensor will be added to.
 * \param params  The same parameters as used by the convolution().
 * \param name    Debugging name for the tensor.
 * \param options Options controlling the implementation.
 * \param cache   Optional pointer to planning cache to use.
 * \return        The weights tensor suitable for use with convolution().
 */
poplar::Tensor createWeights(poplar::Graph &graph, const ConvParams &params,
                             const poplar::DebugContext &debugContext = {},
                             const poplar::OptionFlags &options = {},
                             PlanningCache *cache = nullptr);

/** Create a bias tensor suitable for input to the addBias() function
 *
 * The tensor will have the shape [outChans]
 *
 * \param graph         The graph that the tensor will be added to.
 * \param activations   The activation tensor which is output from the
 *                      convolution.
 * \param name          Debugging name for the tensor.
 * \return              The tensor of biases.
 */
poplar::Tensor
createBiases(poplar::Graph &graph, const poplar::Tensor &activations,
             const poplar::DebugContext &debugContext = {"biases"});

/** Create a bias tensor suitable for input to the addBias() function
 *  with allocation consistent with plan parameters
 *
 * The tensor will have the shape [outChans]
 *
 * \param graph         The graph that the tensor will be added to.
 * \param activations   The activation tensor which is output from the
 *                      convolution.
 * \param params        Parameters as passed to the target convolution.
 * \param name          Debugging name for the tensor.
 * \param options       Options controlling the implementation. See
 *                      createWeights().
 * \param cache         Optional pointer to planning cache to use.
 * \return              The tensor of biases.
 */
poplar::Tensor
createBiases(poplar::Graph &graph, const poplar::Tensor &activations,
             const ConvParams &params,
             const poplar::DebugContext &debugContext = {"biases"},
             const poplar::OptionFlags &options = {},
             PlanningCache *cache = nullptr);

/** Create an input tensor for a convolution.
 *
 * Use this when you need to create an input data tensor for a convolution. The
 * same set of parameters which will be passed to the convolution() should also
 * be passed to createInput().
 *
 * The returned tensor has the shape [B x inChans x H x W].
 *
 * \param graph    The tensor will be added to this graph.
 * \param params   Parameters as passed to the target convolution.
 * \param name     Debugging name for the tensor.
 * \param options  Options controlling the implementation. See createWeights().
 * \param cache    Optional pointer to planning cache to use.
 * \return         The allocated input tensor.
 */
poplar::Tensor createInput(poplar::Graph &graph, const ConvParams &params,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags &options = {},
                           PlanningCache *cache = nullptr);

/** Create an output tensor for a convolution.
 *
 * Use this when you need to create an output data tensor for a convolution. The
 * same set of parameters which will be passed to the convolution() should also
 * be passed to createInput().
 *
 * The returned tensor has the shape [B x inChans x H x W].
 *
 * \param graph         The tensor will be added to this graph.
 * \param params        Parameters as passed to the target convolution.
 * \param debugContext  Debugging name for the tensor.
 * \param options       Options controlling the implementation. See
 *                      createWeights().
 * \param cache         Optional pointer to planning cache to use.
 * \return              The allocated output tensor.
 */
poplar::Tensor createConvOutput(poplar::Graph &graph, const ConvParams &params,
                                const poplar::DebugContext &debugContext = {},
                                const poplar::OptionFlags &options = {},
                                PlanningCache *cache = nullptr);

/** Convolve an input with a set of weights.
 *
 * The input tensor is in the form [B x inChans x H x W], and can be allocated
 * using createInput().  The weights tensor is in the form
 * [convGroups x outChansPerConvGroup x inChansPerConvGroup x H x W], and can be
 * allocated using createWeights().
 *
 * The returned tensor has the shape [B x outChans x H x W]
 *
 * Padding and striding are specified in the ConvParams structure.
 *
 * \param graph                   The graph that the operation will be added to.
 * \param in                      Input data tensor.
 * \param weights                 Weights tensor.
 * \param params                  Parameters for the form of the convolution.
 * \param transposeAndFlipWeights For the weight update pass.
 * \param prog                    Poplar program sequence to append the
 *                                operation onto.
 * \param debugContext            Optional debug information.
 * \param options                 Options that control the implementation. See
 *                                createWeights().
 * \param cache                   Optional pointer to planning cache to use.
 * \return                        The convolved output tensor.
 */
poplar::Tensor convolution(poplar::Graph &graph, const poplar::Tensor &in,
                           const poplar::Tensor &weights,
                           const ConvParams &params,
                           bool transposeAndFlipWeights,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags &options = {},
                           PlanningCache *cache = nullptr);

/** Convolve an input with a set of weights into a pre-allocated output tensor.
 *
 * The output tensor is in the form [B x OutChans x H x W], and can be allocated
 * using createConvOutput().  The weights tensor is in the form
 * [convGroups x outChansPerConvGroup x inChansPerConvGroup x H x W], and can be
 * allocated using createWeights(). The input tensor is in the form
 * [B x inChans x H x W], and can be allocated using createInput().
 *
 * Padding and striding are specified in the ConvParams structure.
 *
 * \param graph                   The graph that the operation will be added to.
 * \param in                      Input data tensor.
 * \param weights                 Weights tensor.
 * \param out                     Pre-allocated output tensor.
 * \param params                  Parameters for the form of the convolution.
 * \param transposeAndFlipWeights For the weight update pass.
 * \param prog                    Poplar program sequence to append the
 *                                operation onto.
 * \param debugContext            Optional debug information.
 * \param options                 Options that control the implementation. See
 *                                createWeights().
 * \param cache                   Optional pointer to planning cache to use.
 */
void convolutionWithOutput(poplar::Graph &graph, const poplar::Tensor &in,
                           const poplar::Tensor &weights,
                           const poplar::Tensor &out, const ConvParams &params,
                           bool transposeAndFlipWeights,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext = {},
                           const poplar::OptionFlags &options = {},
                           PlanningCache *cache = nullptr);

using ConvPlanParams = std::tuple<const poplar::Target *, const ConvParams,
                                  const poplar::OptionFlags *>;
/** \deprecated Use preplan() instead.
 *
 * Plan the specified convolutions.
 *
 * \param convs   A set of tuples of:
 *                  - conv-specific target for tile / IPU sizing
 *                  - convolution parameters
 *                  - implementation options. See createWeights().
 *
 *                All entries must have matching machine parameters.
 * \param cache   The planning cache to update.
 */
void preplanConvolutions(const std::set<ConvPlanParams> &convs,
                         PlanningCache &cache);

/** \deprecated Use preplan() instead.
 *
 * Plan the specified convolutions.
 *
 * \param graph   The graph the convolutions will belong to
 * \param convs   A set of tuples of:
 *                  - conv-specific target for tile / IPU sizing
 *                  - convolution parameters
 *                  - implementation options. See createWeights().
 *
 *                All entries must have matching machine parameters.
 * \param cache   The planning cache to update.
 */
void preplanConvolutions(poplar::Graph &graph,
                         const std::set<ConvPlanParams> &convs,
                         PlanningCache &cache);

/** Copy the weights in \p weightsIn into \p weightsOut such that
 * each element of the kernel is transposed with respect to the input and
 * output channels and flip each spatial dimension of the kernel.
 *
 * See the `transposeAndFlipWeights` parameter in convolution().
 *
 * \param graph         The graph that the operation will be added to.
 * \param weightsIn     The input weights tensor.
 * \param weightsOut    The output weights tensor.
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 */
void weightsTransposeChansFlipXY(poplar::Graph &graph,
                                 const poplar::Tensor &weightsIn,
                                 const poplar::Tensor &weightsOut,
                                 poplar::program::Sequence &prog,
                                 const poplar::DebugContext &debugContext = {},
                                 const poplar::OptionFlags &options = {});

/** Append an operation to a poplar::Program to generate the tensor of
 *  weight deltas.
 *
 * \param graph         The tensor will be added to this graph.
 * \param zDeltas       Tensor containing the gradients with respect to the
 *                      output of the convolution.
 * \param activation    Tensor containing the inputs to the convolution in the
 *                      forward pass.
 * \param params        Parameters of the convolution.
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 *
 * \return              The weight deltas are the gradients with respect to the
 *                      weights of the convolution. These are populated when the
 *                      operation runs.
 */
poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                      const poplar::Tensor &activations,
                      const ConvParams &params, poplar::program::Sequence &prog,
                      const poplar::DebugContext &debugContext = {},
                      const poplar::OptionFlags &options = {},
                      PlanningCache *cache = nullptr);

/** Append operations to a poplar::Program to generate and apply the
 *  weight update.
 *
 * \see calculateWeightDeltas().
 *
 * \param graph         The graph that the operation will be added to.
 * \param zDeltas       Tensor containing the gradients with respect to the
 *                      output of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs to the convolution in
 *                      the forward pass.
 * \param params        Parameters of the convolution.
 * \param scale         Scale to apply to the \p zDeltas.
 * \param prog          Poplar program sequence to append the operations onto.
 * \param debugContext  Optional debug information.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 */
void convolutionWeightUpdate(poplar::Graph &graph,
                             const poplar::Tensor &zDeltas,
                             const poplar::Tensor &weights,
                             const poplar::Tensor &activations,
                             ConvParams params, const poplar::Tensor &scale,
                             poplar::program::Sequence &prog,
                             const poplar::DebugContext &debugContext = {},
                             const poplar::OptionFlags &options = {},
                             PlanningCache *cache = nullptr);

/** Append operations to a poplar::Program to generate and apply the
 *  weight update.
 *
 * \see calculateWeightDeltas().
 *
 * \param graph         The graph that the operation will be added to.
 * \param zDeltas       Tensor containing the gradients with respect to the
 *                      output of the convolution.
 * \param weights       Weights tensor.
 * \param activations   Tensor containing the inputs to the convolution in
 *                      the forward pass.
 * \param params        Parameters of the convolution.
 * \param scale         Scale to apply to the zDeltas.
 * \param prog          Poplar program sequence to append the operations onto.
 * \param debugContext  Optional debug information.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 */
void convolutionWeightUpdate(
    poplar::Graph &graph, const poplar::Tensor &zDeltas,
    const poplar::Tensor &weights, const poplar::Tensor &activations,
    ConvParams params, float scale, poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/** Add a program to update \p biases tensor with the gradients derived from
 * the \p zDeltas tensor.
 *
 * \param graph         The graph that the operation will be added to.
 * \param zDeltas       Tensor containing the gradients with respect to the
 *                      output of the convolution.
 * \param biases        Biases tensor to update.
 * \param scale         Scale to apply to to zDeltas tensor.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 */
void convolutionBiasUpdate(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                           const poplar::Tensor &biases,
                           const poplar::Tensor &scale,
                           const poplar::OptionFlags &options,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext = {});

/** Add a program to update \p biases tensor with the gradients derived from
 * the \p zDeltas tensor.
 *
 * \param graph         The graph that the operation will be added to.
 * \param zDeltas       Tensor containing the gradients with respect to the
 *                      output of the convolution.
 * \param biases        Biases tensor to update.
 * \param scale         Scale to apply to to \p zDeltas tensor.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 */
void convolutionBiasUpdate(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                           const poplar::Tensor &biases, float scale,
                           const poplar::OptionFlags &options,
                           poplar::program::Sequence &prog,
                           const poplar::DebugContext &debugContext = {});

/** Adds a program to \p prog which adds \p biases to \p activations tensor.
 *
 * \param graph         The graph that the operation will be added to.
 * \param input         Tensor containing values which to add the biases.
 * \param biases        Biases to add to the \p input tensor.
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 */
void addBias(poplar::Graph &graph, const poplar::Tensor &in,
             const poplar::Tensor &biases, poplar::program::Sequence &prog,
             const poplar::DebugContext &debugContext = {});

/** Report the convolution plan corresponding to the \p params and \p options
 * provided.
 *
 * \param out           Output stream to report the plan to.
 * \param graph         The graph that the convolution is planned with.
 * \param params        The same parameters as used by the convolution().
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 */
void reportPlanInfo(std::ostream &out, const poplar::Graph &graph,
                    const ConvParams &params,
                    const poplar::OptionFlags &options = {},
                    PlanningCache *cache = nullptr);

/** Structure for estimated costs returned by reportPlanEstimatedCosts() */
struct PlanCosts {
  std::size_t cycles;
  std::size_t memory;
  constexpr static size_t unknown = std::numeric_limits<std::size_t>::max();
};

/** Report the estimated cycles and memory costs of the convolution plan
 * corresponding to the \p params and \p options provided.
 *
 * \param graph         The graph that the convolution is planned with.
 * \param params        The same parameters as used by the convolution().
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 *
 * \return              Cycles and memory cost estimates for the planned
 *                      convolution.
 */
PlanCosts reportPlanEstimatedCosts(const poplar::Graph &graph,
                                   const ConvParams &params,
                                   const poplar::OptionFlags &options = {},
                                   PlanningCache *cache = nullptr);

namespace internal {

/** Structure for detailed estimated costs returned by
 * reportDetailedPlanEstimatedCosts(). */
struct DetailedPlanCosts {
  std::size_t parallelSplit = 1;
  std::size_t serialSplit = 1;
  PlanCosts broadcast = {};
  PlanCosts rearrangement = {};
  PlanCosts dynamicSlice = {};
  PlanCosts transform = {};
  PlanCosts exchange = {};
  PlanCosts tileLevelTransform = {};
  PlanCosts inputsCast = {};
  PlanCosts compute = {};
  PlanCosts reduction = {};
  PlanCosts dynamicUpdate = {};
  PlanCosts addInPlace = {};
  PlanCosts outputCast = {};
  PlanCosts total = {};

  template <typename Function>
  void apply(Function fn, bool includeTotal = true,
             bool serialSplitOnly = false) const {
    // Call the non-const overload to avoid duplication. Prevent fn from being
    // able to modify state by wrapping it in another lambda that applies const
    // to the reference.
    const_cast<DetailedPlanCosts *>(this)->apply(
        [&fn](const PlanCosts &c) { fn(c); }, includeTotal, serialSplitOnly);
  }
  template <typename Function>
  void apply(Function fn, bool includeTotal = true,
             bool serialSplitOnly = false) {
    if (!serialSplitOnly) {
      fn(broadcast);
      fn(rearrangement);
    }
    fn(dynamicSlice);
    fn(transform);
    fn(exchange);
    fn(tileLevelTransform);
    fn(inputsCast);
    fn(compute);
    fn(reduction);
    fn(dynamicUpdate);
    if (!serialSplitOnly) {
      fn(addInPlace);
      fn(outputCast);
    }
    if (includeTotal)
      fn(total);
  }
};

std::ostream &operator<<(std::ostream &os, DetailedPlanCosts const &c);

std::istream &operator>>(std::istream &is, DetailedPlanCosts &c);

/** Like reportPlanEstimatedCosts() but return an itemised breakdown of the
 *  estimates based on what the planner did.
 */
DetailedPlanCosts reportDetailedPlanEstimatedCosts(
    const poplar::Graph &graph, const ConvParams &params,
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

} // namespace internal

/** Report the convolution plan corresponding to the weight update pass given
 * the forward pass \p params and \p options.
 *
 * \param out           ostream to report the plan to.
 * \param graph         The graph that the convolution is planned with.
 * \param fwdParams     Forward pass parameters as used by the convolution().
 * \param fwdOptions    Forward pass options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 */
void reportWeightUpdatePlanInfo(std::ostream &out, const poplar::Graph &graph,
                                const ConvParams &fwdParams,
                                const poplar::OptionFlags &fwdOptions = {},
                                PlanningCache *cache = nullptr);

/** Arranges the weights (activations) such that they are suited for the
 * backward pass in a fully connected layer.
 * \param graph         The graph that the operation will be added to.
 * \param activations   Tensor containing the inputs to the convolution.
 * \param params        Parameters of the convolution.
 * \param prog          Poplar program sequence to append the operation onto.
 * \param debugContext  Optional debug information.
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 * \param cache         Optional pointer to planning cache to use.
 * \return A tensor with the weights suitably arranged.
 */
poplar::Tensor fullyConnectedWeightTranspose(
    poplar::Graph &graph, poplar::Tensor weights, const ConvParams &params,
    poplar::program::Sequence &prog,
    const poplar::DebugContext &debugContext = {},
    const poplar::OptionFlags &options = {}, PlanningCache *cache = nullptr);

/** Provides an interface to validate the convolution options. Presence of
 *  invalid key or a value will throw an exception.
 *
 * \param options       Options controlling the implementation.
 *                      See createWeights().
 */
void convolutionValidateOptions(const poplar::OptionFlags &options);

struct Plan;

class PlanningCacheImpl;
class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();

  /** Returns the number of entries currently stored in the cache. */
  std::size_t size() const;

  std::unique_ptr<PlanningCacheImpl> impl;
};

} // namespace poplin

#endif // poplin_Convolution_hpp
