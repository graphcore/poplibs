// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "ConvPlan.hpp"
#include "ConvPlanTypes.hpp"
#include "PlanningCache.hpp"
#include "PlanningObjective.hpp"
#include <poplar/Target.hpp>
#include <popsolver/Model.hpp>

namespace poplin {

Estimates<popsolver::Variable> constructModel(
    const poplar::Target &target, const std::vector<ConvTransform> &transforms,
    const std::vector<ConvTypes> &types, const std::vector<unsigned> &hierarchy,
    const std::vector<double> &perLevelExchangeBytesPerCycle,
    const std::vector<unsigned> &fieldGrainSize,
    const ConvVertexType &convVertexType, const ConvParams &untransformedParams,
    bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
    const boost::optional<Plan> &referencePlan,
    const boost::optional<Cost> &referenceCost,
    PlanningCacheImpl::CycleEstimationImpl *cache, const ConvOptions &options,
    popsolver::Model &m, std::vector<PartitionVariables> &partitionVars);

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
Plan::Method getFullyConnectedWUMethod(const ConvParams &fwdParams,
                                       Plan::Method fwdMethod,
                                       unsigned fwdOutChansPerGroups,
                                       unsigned fwdInChansPerGroup);
Plan::Method getFullyConnectedBwdMethod(Plan::Method fwdMethod);

} // namespace poplin