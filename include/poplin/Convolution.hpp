// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplin_Convolution_hpp
#define poplin_Convolution_hpp
#include <tuple>
#include <set>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/OptionFlags.hpp>
#include "ConvParams.hpp"

namespace poplin {

/** Class used to cache the calculation of plans for convolution operations.
 */
class PlanningCache;

uint64_t getFwdFlops(const ConvParams &params);
uint64_t getBwdFlops(const ConvParams &params);
uint64_t getWuFlops(const ConvParams &params);

double
getFwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getBwdPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

double
getWuPerfectCycleCount(const poplar::Graph &graph, const ConvParams &params);

/** Create a weight tensor suitable for use with convolution()
 *
 * The shape of the tensor will be [convGroups x outChans x inChans x H x W]
 *
 * \param graph   The tensor will be added to this graph
 * \param params  The same parameters as used by the convolution()
 * \param name    Debugging name for the tensor
 * \param options Options controlling the implementation
 * \param cache   Optional pointer to planning cache to use
 * \return        The weights tensor suitable for use with convolution()
 */
poplar::Tensor
createWeights(poplar::Graph &graph,
              const ConvParams &params,
              const std::string &name,
              const poplar::OptionFlags &options = {},
              PlanningCache *cache = nullptr);

/** Create a bias tensor suitable for input to addBias() function
 *
 * The tensor will have the shape [outChans]
 *
 * \param graph  The tensor will be added to this graph
 * \param acts   The activation tensor which is output from the convolution
 * \param name   Debugging name for the tensor
 * \return       The tensor of biases
 */
poplar::Tensor
createBiases(poplar::Graph &graph, const poplar::Tensor &acts,
             const std::string &name = "biases");

/** Create an input tensor for a convolution
 *
 * Use this when required to create an input data tensor for a convoution. The
 * same set of parameters which will be passed to the convolution() should also
 * be passed to createInput()
 *
 * The returned tensor has the shape [B x inChans x H x W].
 *
 * \param graph    The tensor will be added to this graph
 * \param params   Parameters as passed to the target convolution.
 * \param name     Debugging name for the tensor
 * \param options  Options controlling the implementation
 * \param cache    Optional pointer to planning cache to use
 * \return         The allocated input tensor
 */
poplar::Tensor
createInput(poplar::Graph &graph,
            const ConvParams &params,
            const std::string &name,
            const poplar::OptionFlags &options = {},
            PlanningCache *cache = nullptr);

/** Convolve an input with a set of weights.
 *
 * This is for a 2D convolution.
 *
 * The input tensor is in the form [B x inChans x H x W], and can be allocated
 * using createInput().  The weights tensor is in the form
 * [convGroups x outChans x inChans x H x W], and can be allocated using
 * createWeights().
 *
 * Padding and striding are specified in the ConvParams structure.
 *
 * \param graph                   The operation will be added to this graph
 * \param in                      Input data tensor
 * \param weights                 Weights tensor
 * \param params                  Parameters for the form of the convolution
 * \param transposeAndFlipWeights For the weight update pass
 * \param prog                    Poplar program sequence to append to op onto
 * \param debugPrefix             Name of the operation, for debugging
 * \param options                 Options that control the implementation
 * \param cache                   Optional pointer to planning cache to use
 * \return                        The convolved output tensor
 */
poplar::Tensor
convolution(poplar::Graph &graph,
            const poplar::Tensor &in,
            const poplar::Tensor &weights,
            const ConvParams &params,
            bool transposeAndFlipWeights,
            poplar::program::Sequence &prog,
            const std::string &debugPrefix = "",
            const poplar::OptionFlags &options = {},
            PlanningCache *cache = nullptr);

/*
 * Plan the specified convolutions

 * \param convs   set of tuples of
 *                - conv-specific target for tile / ipu sizing
 *                - convolution parameters
 *                - implementation options
 *                All entries must have matching machine parameters
 * \param cache   The planning cache to update
 */
void preplanConvolutions(
    const std::set<std::tuple<const poplar::Target *,
                              const poplin::ConvParams,
                              const poplar::OptionFlags *>> &convs,
    PlanningCache &cache);

void
weightsTransposeChansFlipXY(poplar::Graph &graph,
                            const poplar::Tensor &weightsIn,
                            const poplar::Tensor &WeightsOut,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");

poplar::Tensor
calculateWeightDeltas(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                      const poplar::Tensor &activations,
                      const ConvParams &params,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix = "",
                      const poplar::OptionFlags &options = {},
                      PlanningCache *cache = nullptr);

void
convolutionWeightUpdate(poplar::Graph &graph,
                        const poplar::Tensor &zDeltas,
                        const poplar::Tensor &weights,
                        const poplar::Tensor &activations,
                        ConvParams params,
                        const poplar::Tensor &scale,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {},
                        PlanningCache *cache = nullptr);

void
convolutionWeightUpdate(poplar::Graph &graph,
                        const poplar::Tensor &zDeltas,
                        const poplar::Tensor &weights,
                        const poplar::Tensor &activations,
                        ConvParams params,
                        float scale,
                        poplar::program::Sequence &prog,
                        const std::string &debugPrefix = "",
                        const poplar::OptionFlags &options = {},
                        PlanningCache *cache = nullptr);

void
convolutionBiasUpdate(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                  const poplar::Tensor &biases,
                  const poplar::Tensor &scale,
                  const poplar::Type &partialsType,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

void
convolutionBiasUpdate(poplar::Graph &graph, const poplar::Tensor &zDeltas,
                  const poplar::Tensor &biases,
                  float scale,
                  const poplar::Type &partialsType,
                  poplar::program::Sequence &prog,
                  const std::string &debugPrefix = "");

void
addBias(poplar::Graph &graph, const poplar::Tensor &acts,
        const poplar::Tensor &biases,
        poplar::program::Sequence &prog,
        const std::string &debugPrefix = "");

poplar::Tensor
fullyConnectedWeightTranspose(poplar::Graph &graph,
                              poplar::Tensor activations,
                              const ConvParams &params,
                              poplar::program::Sequence &prog,
                              const std::string &debugPrefix,
                              const poplar::OptionFlags &options,
                              PlanningCache *cache = nullptr);

void reportPlanInfo(std::ostream &out,
                    const poplar::Graph &graph,
                    const ConvParams &params,
                    const poplar::OptionFlags &options = {},
                    PlanningCache *cache = nullptr);

void reportWeightUpdatePlanInfo(std::ostream &out,
                                const poplar::Graph &graph,
                                const ConvParams &params,
                                const poplar::OptionFlags &options = {},
                                PlanningCache *cache = nullptr);

struct Plan;

class PlanningCacheImpl;
class PlanningCache {
public:
  PlanningCache();
  ~PlanningCache();
  std::unique_ptr<PlanningCacheImpl> impl;
};

} // namespace poplin

#endif // poplin_Convolution_hpp
