// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ConvPlan.hpp"
#include "ConvPlanTypes.hpp"
#include "ConvVertexType.hpp"
#include "PlanningCache.hpp"
#include "PlanningObjective.hpp"
#include <poplar/Target.hpp>
#include <popsolver/Model.hpp>

namespace poplin {

/// Check if we can use the ConvPartial1x1 vertex instead of the ConvPartialnx1
/// vertex.
///
/// The 1x1 vertex can only be used if:
///
///   - The planner has not already selected a different conv-width.
///   - The number of input channels is non-zero. The 1x1 vertex requires the
///     input channels to be non-zero to write zero to the output
///   - The number of kernel elements is 1.
///   - There will be only one work-list entry per worker. If either the batch
///     size or outer output-field shape is > 1 then we end up with multiple
///     partitions and multiple work-list entries per worker.
///   - The entire output range is written by the vertex. This may not be the
///     case if output padding is applied, because the padding may be added by
///     a separate tile/vertex.
///
/// \return True if we can use the 1x1 vertex instead of the nx1 vertex.
bool canUseConvPartial1x1Vertex(
    unsigned convUnitWeightHeight, unsigned inputChannels, unsigned batchSize,
    const std::vector<unsigned> &transformedInputDilation,
    const std::vector<unsigned> &transformedOutputStride,
    const std::vector<unsigned> &tileKernelShape,
    const std::vector<unsigned> &outputFieldShape,
    const ConvParams::OutputTransform &outputTransform);

Estimates<popsolver::Variable> constructModel(
    const poplar::Target &target, const std::vector<ConvTransform> &transforms,
    const std::vector<ConvTypes> &types,
    const std::vector<unsigned> &fieldGrainSize,
    const ConvVertexType &convVertexType, const ConvParams &untransformedParams,
    bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
    const boost::optional<Plan> &referencePlan,
    const boost::optional<Cost> &referenceCost,
    PlanningCacheImpl::CycleEstimationImpl *cache, const ConvOptions &options,
    popsolver::Model &m, std::vector<PartitionVariables> &partitionVars,
    popsolver::Variable &broadcastInputBeforeLoop);

void expandDim(ConvParams &params, unsigned dim);
bool canDeferDilation(const ConvParams &params, unsigned dim);

ConvParams calculateSwappedParams(const ConvParams &params, bool swapOperands);
ConvParams calculateExpandedParams(const ConvParams &params,
                                   const std::vector<unsigned> &expandDims);
ConvParams
calculateFlattenedParams(const ConvParams &params,
                         const std::vector<unsigned> &outChanFlattenDims,
                         std::vector<unsigned> &flattenDims);
ConvParams calculateGroupedParams(ConvParams groupedParams,
                                  unsigned combineConvGroups);

bool isFullyConnected(Pass pass);

ConvVertexType getFullyConnectedWuConvVertexType(
    const poplar::Target &target, Plan::Method fwdMethod,
    unsigned fwdOutChansPerGroup, unsigned fwdInChansPerGroup,
    const ConvParams &untransformedFwdParams, const ConvOptions &fwdOptions);

ConvVertexType getFullyConnectedBwdConvVertexType(
    const poplar::Target &target, Plan::Method fwdMethod,
    unsigned bwdInChansPerGroup, unsigned bwdOutChansPerGroup,
    const ConvParams &untransformedFwdParams, const ConvOptions &fwdOptions);

std::vector<ConvTypes> getConvTypes(const poplar::Target &target,
                                    poplar::Type vertexOutputType,
                                    poplar::Type resultType,
                                    const ConvOptions &options);

unsigned getMaxMACsPerCyclePerTile(const poplar::Target &target,
                                   const ConvVertexType &convVertexType);

} // namespace poplin
