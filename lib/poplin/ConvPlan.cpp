// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "ConvPlan.hpp"
#include "CanonicalConvParams.hpp"
#include "ConvModel.hpp"
#include "ConvOptions.hpp"
#include "ConvPlanTypes.hpp"
#include "ConvReducePlan.hpp"
#include "ConvUtilInternal.hpp"
#include "ConvValidation.hpp"
#include "PlanningCache.hpp"
#include "PlanningObjective.hpp"
#include "poplar/Graph.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poplibs_support/print.hpp"
#include "poplin/ConvUtil.hpp"
#include "poplin/Convolution.hpp"
#include "popsolver/Model.hpp"
#include "poputil/exceptions.hpp"
#include "tbb/concurrent_unordered_map.h"
#include "tbb/parallel_for.h"
#include <boost/functional/hash.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_set>

using namespace poplibs_support;

namespace poplin {

static const char *asString(Plan::Method m) {
  switch (m) {
  case Plan::Method::AMP:
    return "AMP";
  case Plan::Method::SLIC:
    return "SLIC";
  case Plan::Method::MAC:
    return "MAC";
  case Plan::Method::OUTER_PRODUCT:
    return "OUTER_PRODUCT";
  }
  POPLIB_UNREACHABLE();
}

bool operator<(const Partition &a, const Partition &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &Partition::fieldSplit, &Partition::batchSplit, &Partition::outChanSplit,
      &Partition::kernelSplit, &Partition::inChanSplit,
      &Partition::convGroupSplit, &Partition::fieldAxisGrainSize,
      &Partition::convGroupGrainSize, &Partition::inChanGrainSize,
      &Partition::outChanGrainSize);

  return helper.lt(a, b);
}

std::ostream &operator<<(std::ostream &os, const Partition &p) {
  // T10408: Splitting the batch and in channel dimensions serially has not been
  // implemented yet so we don't bother printing them out for now.
  os << "  Partition: fieldSplit            ";
  printContainer(p.fieldSplit, os);
  os << "\n"
     << "             batchSplit            " << p.batchSplit << "\n"
     << "             outChanSplit.serial   " << p.outChanSplit.serial << "\n"
     << "             outChanSplit.parallel " << p.outChanSplit.parallel << "\n"
     << "             kernelSplit           ";
  printContainer(p.kernelSplit, os);
  os << "\n"
     << "             inChanSplit.serial    " << p.inChanSplit.serial << "\n"
     << "             inChanSplit.parallel  " << p.inChanSplit.parallel << "\n"
     << "             convGroupSplit        " << p.convGroupSplit << "\n"
     << "             fieldAxisGrainSize    ";
  printContainer(p.fieldAxisGrainSize, os);
  os << "\n"
     << "             inChanGrainSize       " << p.inChanGrainSize << "\n"
     << "             outChanGrainSize      " << p.outChanGrainSize << "\n";
  return os;
}

bool operator<(const ConvTransform &a, const ConvTransform &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &ConvTransform::extraFieldDims, &ConvTransform::dilatePostConv,
      &ConvTransform::swapOperands, &ConvTransform::expandDims,
      &ConvTransform::outChanFlattenDims, &ConvTransform::flattenDims,
      &ConvTransform::combineConvGroupsFactor);

  return helper.lt(a, b);
}

std::ostream &operator<<(std::ostream &os, const ConvTransform &t) {
  os << "  Transform:\n"
        "        extraFieldDims          "
     << t.extraFieldDims
     << "\n"
        "        dilatePostConv          ";
  printContainer(t.dilatePostConv, os);
  os << "\n"
     << "        swapOperands            "
     << (t.swapOperands ? "true" : "false") << "\n"
     << "        expandDims              ";
  printContainer(t.expandDims, os);
  os << "\n"
     << "        outChanFlattenDims      ";
  printContainer(t.outChanFlattenDims, os);
  os << "\n"
     << "        flattenDims             ";
  printContainer(t.flattenDims, os);
  os << "\n"
     << "        combineConvGroupsFactor       " << t.combineConvGroupsFactor
     << "\n";
  return os;
}

bool operator<(const ConvTypes &a, const ConvTypes &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &ConvTypes::partialType, &ConvTypes::resultType);

  return helper.lt(a, b);
}

std::ostream &operator<<(std::ostream &os, const ConvTypes &t) {
  os << "  Types: partialType        " << t.partialType << "\n";
  os << "         resultType         " << t.resultType << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Plan::Method &m) {
  os << asString(m);
  return os;
}

std::istream &operator>>(std::istream &is, Plan::Method &m) {
  std::string token;
  is >> token;
  if (token == "MAC") {
    m = Plan::Method::MAC;
  } else if (token == "AMP") {
    m = Plan::Method::AMP;
  } else if (token == "SLIC") {
    m = Plan::Method::SLIC;
  } else if (token == "OUTER_PRODUCT") {
    m = Plan::Method::OUTER_PRODUCT;
  } else {
    throw poputil::poplibs_error("Unrecognised convolution method '" + token +
                                 "'");
  }
  return is;
}

std::ostream &operator<<(std::ostream &os, Plan::LinearizeTileDirection d) {
  switch (d) {
  case Plan::LinearizeTileDirection::ASCENDING:
    return os << "ASCENDING";
  case Plan::LinearizeTileDirection::DESCENDING:
    return os << "DESCENDING";
  }

  auto id = static_cast<std::underlying_type_t<decltype(d)>>(d);
  throw poputil::poplibs_error("Unrecognised tile direction <" +
                               std::to_string(id) + ">");
}

bool operator<(const Plan &a, const Plan &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &Plan::transforms, &Plan::partitions, &Plan::types,
      &Plan::convGroupsPerGroup, &Plan::inChansPerGroup,
      &Plan::partialChansPerGroup, &Plan::slicWindowWidth,
      &Plan::numConvUnitsRequired, &Plan::method, &Plan::linearizeTileOrder,
      &Plan::startTile, &Plan::linearizeTileDirection, &Plan::isJointPlan);

  return helper.lt(a, b);
}

std::ostream &operator<<(std::ostream &os, const Plan &p) {
  os << "  Plan:";
  const auto numLevels = p.transforms.size();
  for (std::size_t i = 0; i != numLevels; ++i) {
    os << "        transform #" << i << "\n";
    os << p.transforms[i] << "\n";
    if (i + 1 != numLevels) {
      os << "        partition #" << i << "\n";
      os << p.partitions[i];
    }
    os << "        types #" << i << "\n";
    os << p.types[i];
  }
  os << "        convGroupsPerGroup      " << p.convGroupsPerGroup << "\n"
     << "        inChansPerGroup         " << p.inChansPerGroup << "\n"
     << "        partialChansPerGroup    " << p.partialChansPerGroup << "\n"
     << "        method                  " << p.method << "\n"
     << "        isJointPlan             " << p.isJointPlan << "\n"
     << "        startTile               " << p.startTile << "\n"
     << "        linearizeTileDirection  " << p.linearizeTileDirection << "\n"
     << "        totalTiles              " << p.totalTiles() << "\n";
  return os;
}

PlanningCache::PlanningCache() {
  impl = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl());
}

PlanningCache::~PlanningCache() = default;

// Pick a tile to start laying out the convolution on. We pick a "random" tile
// by hashing the forward shape in an attempt to evenly distribute across the
// entire tile range. The start tile granularity is such that we always start
// on a new column, and we also decide whether to lay the data out in ascending
// or descending tile order. We make an effort (using the Pass) to give the
// forward, backward and weight update passes the same start tile and direction.
static std::pair<unsigned, Plan::LinearizeTileDirection>
getStartTile(const poplar::Target &target,
             unsigned startTileIdxForVirtualHierarchy, const ConvParams &params,
             const ConvOptions &options) {
  if (!options.enableConvDithering) {
    return std::make_pair(startTileIdxForVirtualHierarchy,
                          Plan::LinearizeTileDirection::ASCENDING);
  } else {
    if (startTileIdxForVirtualHierarchy != 0) {
      // This is a quick get out for multiplans for now while it's unsupported
      // (where startTileIdxForVirtualHierarchy is not 0 as the IPU is split up
      // for each plan)
      throw poputil::poplibs_error(
          "Unsupported conv dithering with multi plans");
    }
  }

  const auto seed = [&] {
    // starting seed: 2^32/phi, where phi is the golden ratio.
    std::size_t seed = 0x9e3779b9UL;
    boost::hash_combine(seed, params.numConvGroups);

    // fully connected layers swap the channels and field dimensions around so
    // for those to remain pass oblivious we must handle them separately. this
    // basically means that all non-inference fully connected layers will have
    // the same dithering, T19546 tracks improving this and also once T16758 is
    // fixed we can remove this.
    if (options.pass == Pass::FC_TRAINING_FWD ||
        options.pass == Pass::FC_TRAINING_BWD ||
        options.pass == Pass::FC_TRAINING_WU) {
      boost::hash_combine(seed, params.batchSize);
      assert(params.inputFieldShape.size() == 1);
      const auto x = params.inputFieldShape.front() *
                     params.inputChannelsPerConvGroup *
                     params.outputChannelsPerConvGroup;
      boost::hash_combine(seed, x);
      return seed;
    }

    // use the forward pass shape to determine the start column and direction.
    // this is easier than hashing the whole params in a pass oblivious manner.
    auto shape = [&] {
      switch (options.pass) {
      // if no pass, assume forward and training.
      case Pass::NONE:
      case Pass::FC_INFERENCE_FWD:
      case Pass::INFERENCE_FWD:
      case Pass::TRAINING_FWD:
        return params.inputFieldShape;

      case Pass::TRAINING_BWD:
        return params.getOutputFieldShape();

      case Pass::TRAINING_WU:
        return params.inputFieldShape;

      case Pass::FC_TRAINING_FWD:
      case Pass::FC_TRAINING_BWD:
      case Pass::FC_TRAINING_WU:
        // handled above.
        break;
      }

      throw poputil::poplibs_error("Unknown pass to determine start tile.");
    }();

    boost::hash_range(seed, std::begin(shape), std::end(shape));
    if (options.pass == Pass::INFERENCE_FWD ||
        options.pass == Pass::FC_INFERENCE_FWD) {
      boost::hash_combine(seed, params.batchSize);
      boost::hash_combine(seed, params.outputChannelsPerConvGroup);
      boost::hash_combine(seed, params.inputChannelsPerConvGroup);
    } else {
      // we must combine the batch and channels in a commutative way to get the
      // same result for each pass.
      auto x = params.batchSize * params.outputChannelsPerConvGroup *
               params.inputChannelsPerConvGroup;
      boost::hash_combine(seed, x);
    }

    return seed;
  }();

  // we always do start tile dithering per-IPU because when we wrap around we
  // need to stay on the same IPU.
  const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();

  const auto numSuperTiles = ceildiv(options.tilesPerIPU, tilesPerSuperTile);

  unsigned startTile = (seed % numSuperTiles) * tilesPerSuperTile;

  const auto numDirections = 2;
  auto direction =
      static_cast<Plan::LinearizeTileDirection>(seed % numDirections);

  return std::make_pair(startTile, direction);
}

static unsigned getConvUnitsPerTile(const poplar::Target &target,
                                    bool floatActivations, bool floatPartials) {
  if (floatActivations) {
    return floatPartials ? target.getFp32InFp32OutConvUnitsPerTile() : 0;
  }
  return floatPartials ? target.getFp16InFp32OutConvUnitsPerTile()
                       : target.getFp16InFp16OutConvUnitsPerTile();
}

static bool canUseConvolutionInstruction(bool floatActivations,
                                         bool floatPartials,
                                         const poplar::Target &target) {
  if (getConvUnitsPerTile(target, floatActivations, floatPartials) == 0) {
    return false;
  }

  if (floatActivations) {
    // the case where activations are float but partials are not is handled by
    // getConvUnitsPerTile above.
    assert(floatPartials);
  }

  return true;
}

static bool canUseConvolutionInstruction(bool floatActivations,
                                         bool floatPartials,
                                         unsigned inChansPerGroup,
                                         unsigned numConvUnitsRequired,
                                         unsigned outChansPerGroup,
                                         const poplar::Target &target) {
  if (!canUseConvolutionInstruction(floatActivations, floatPartials, target)) {
    return false;
  }
  unsigned usedWeightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);
  // Any other configuration than 4 uses full set of weights hence no need for
  // extra constraint
  if (numConvUnitsRequired == 4) {
    usedWeightsPerConvUnit =
        (usedWeightsPerConvUnit * numConvUnitsRequired) /
        getConvUnitsPerTile(target, floatActivations, floatPartials);
  }
  if (usedWeightsPerConvUnit % inChansPerGroup != 0) {
    return false;
  }
  // Output channels grouping shall be great or equal to number of engines
  if ((outChansPerGroup % numConvUnitsRequired) != 0) {
    return false;
  }
  // Check we can use aligned loads.
  if ((inChansPerGroup * (floatActivations ? 32 : 16)) %
          target.getDataPathWidth() !=
      0) {
    return false;
  }
  return true;
}

static Partition makePartition(const popsolver::Solution &s,
                               const PartitionVariables &vars) {
  std::vector<unsigned> fieldSplitValues;
  for (const auto var : vars.fieldSplit) {
    fieldSplitValues.push_back(s[var].getAs<unsigned>());
  }
  std::vector<unsigned> kernelSplitValues;
  for (const auto var : vars.kernelSplit) {
    kernelSplitValues.push_back(s[var].getAs<unsigned>());
  }

  Partition partition(
      std::move(fieldSplitValues), s[vars.batchSplit].getAs<unsigned>(),
      {s[vars.outChanSplit.serial].getAs<unsigned>(),
       s[vars.outChanSplit.parallel].getAs<unsigned>()},
      std::move(kernelSplitValues),
      {s[vars.inChanSplit.serial].getAs<unsigned>(),
       s[vars.inChanSplit.parallel].getAs<unsigned>()},
      s[vars.convGroupSplit].getAs<unsigned>(), vars.fieldGrainSize,
      vars.convGroupGrainSize, vars.inChanGrainSize, vars.outChanGrainSize);
  return partition;
}

static std::tuple<Plan, Cost, popsolver::ConstraintEvaluationSummary>
choosePlan(const poplar::Target &target,
           const std::vector<ConvTransform> &transforms,
           const std::vector<ConvTypes> &types,
           const std::vector<unsigned> &hierarchy,
           const std::vector<double> &perLevelExchangeBytesPerCycle,
           const std::vector<unsigned> &fieldGrainSize,
           const ConvVertexType &convVertexType, const ConvParams &params,
           bool isJointPlan, Cost bestCost, const PlanningObjective &objective,
           unsigned startTileIdxForVirtualHierarchy,
           const boost::optional<Plan> &referencePlan,
           const boost::optional<Cost> &referenceCost,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           const ConvOptions &options) {
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  Estimates<popsolver::Variable> e = constructModel(
      target, transforms, types, hierarchy, perLevelExchangeBytesPerCycle,
      fieldGrainSize, convVertexType, params, isJointPlan, bestCost, objective,
      referencePlan, referenceCost, cache, options, m, partitionVars);
  popsolver::Solution s;

  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    s = m.minimize({e.totalCycles, e.totalTempBytes});
    break;
  case PlanningObjective::MINIMIZE_COST_DIFF: {
    const auto secondaryObjective =
        objective.getMinimizeForTiles() ? e.totalTiles : e.totalTempBytes;
    s = m.minimize({e.totalPerStepCycleDiff, secondaryObjective});
    break;
  }
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    s = m.minimize({e.totalTempBytes, e.totalCycles});
    break;
  case PlanningObjective::MINIMIZE_TILES:
    s = m.minimize({e.totalTiles, e.totalCycles});
    break;
  }

  if (!s.validSolution()) {
    return {Plan(), highestCost, s.constraintsEvaluated()};
  }

  std::vector<Partition> partitions;
  for (const auto &p : partitionVars) {
    partitions.push_back(makePartition(s, p));
  }
  auto startTile =
      getStartTile(target, startTileIdxForVirtualHierarchy, params, options);
  Plan plan(std::move(partitions), std::move(types),
            convVertexType.convGroupsPerGroup, convVertexType.inChansPerGroup,
            convVertexType.partialChansPerGroup, convVertexType.slicWindowWidth,
            convVertexType.numConvUnitsRequired, convVertexType.method,
            Plan::LinearizeTileOrder::STANDARD, startTile.first,
            startTile.second, isJointPlan, convVertexType.useLimitedVersion);
  plan.transforms = transforms;

  Cost cost;
  cost.totalCycles = s[e.totalCycles];
  cost.totalTempBytes = s[e.totalTempBytes];
  cost.totalPerStepCycleDiff = s[e.totalPerStepCycleDiff];
  cost.totalTiles = s[e.totalTiles];

  cost.rearrangeBeforeSliceCycles = s[e.rearrangeBeforeSliceCycles];
  cost.memsetZeroBeforeAddInPlace = s[e.memsetZeroBeforeAddInPlace];
  cost.dynamicSliceCycles = s[e.dynamicSliceCycles];
  cost.transformCycles = s[e.transformCycles];

  cost.totalExchangeCycles = s[e.totalExchangeCycles];
  cost.itemisedExchangeCycles.inputExchangeCycles =
      s[e.itemisedExchangeCycles.inputExchangeCycles];
  cost.itemisedExchangeCycles.weightExchangeCycles =
      s[e.itemisedExchangeCycles.weightExchangeCycles];
  cost.itemisedExchangeCycles.reduceFirstStageExchangeCycles =
      s[e.itemisedExchangeCycles.reduceFirstStageExchangeCycles];
  cost.itemisedExchangeCycles.reduceRemainingStagesExchangeCycles =
      s[e.itemisedExchangeCycles.reduceRemainingStagesExchangeCycles];

  cost.tileLevelTransformCycles = s[e.tileLevelTransformCycles];
  cost.partialCalcCycles = s[e.partialCalcCycles];
  cost.reduceCycles = s[e.reduceCycles];
  cost.dynamicUpdateCycles = s[e.dynamicUpdateCycles];
  cost.addInPlaceCycles = s[e.addInPlaceCycles];
  cost.castCycles = s[e.castCycles];

  cost.rearrangeBeforeSliceTempBytes = s[e.rearrangeBeforeSliceTempBytes];
  cost.rearrangeBeforeSliceTempDuringRearrangeBytes =
      s[e.rearrangeBeforeSliceTempDuringRearrangeBytes];
  cost.transformTempBytes = s[e.transformTempBytes];
  cost.tileLevelTransformTempBytes = s[e.tileLevelTransformTempBytes];
  cost.convTempBytes = s[e.convTempBytes];
  cost.reduceTempBytes = s[e.reduceTempBytes];
  cost.addInPlaceTempBytes = s[e.addInPlaceTempBytes];

  return {plan, cost, s.constraintsEvaluated()};
}

static void getConvVertexMACCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<popsolver::DataType>("convGroupsPerGroup");
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");
  const auto constrainedUseLimitedVersion =
      planConstraints.get_optional<bool>("useLimitedVersion");

  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnits =
      getNumConvUnits(floatActivations, ampFloatPartials, target);
  bool useLimitedVersion = true;

  // For the test purposes constrain vertex to use unsigned type for
  // vertex states
  if (constrainedUseLimitedVersion) {
    useLimitedVersion = *constrainedUseLimitedVersion;
  }

  // Constrain the input channel grouping to a multiple of two if the activation
  // type is half. This ensures that we never need to apply padding when sending
  // activations over the exchange.
  auto grainSize = floatActivations ? 1u : 2u;
  const auto roundedNumInChans =
      roundUp(params.getNumInputChansPerConvGroup(), grainSize);

  const unsigned convGroupsPerGroup = 1;
  // This is the only supported convGroupsPerGroup for this method.
  if (constrainedConvGroupsPerGroup &&
      *constrainedConvGroupsPerGroup !=
          popsolver::DataType{convGroupsPerGroup}) {
    return;
  }

  unsigned inChansLower = grainSize;
  unsigned inChansUpper = roundedNumInChans;
  if (constrainedInChansPerGroup) {
    // Must be within bounds of the input channels and divisible by
    // the grain size for this type to use this vertex.
    if (*constrainedInChansPerGroup > popsolver::DataType{roundedNumInChans} ||
        *constrainedInChansPerGroup % popsolver::DataType{grainSize} !=
            popsolver::DataType{0}) {
      return;
    }
    inChansLower = inChansUpper =
        (*constrainedInChansPerGroup).getAs<unsigned>();
  }

  unsigned partialChansPerGroup = 1;
  // MAC codelet for half partials processes 2 partials inside inner loop
  // to have most optimal load/store pipeline
  if (!floatPartials) {
    partialChansPerGroup = 2;
  }

  // This is the only supported partialChansPerGroup for this method.
  if (constrainedPartialChansPerGroup &&
      *constrainedPartialChansPerGroup !=
          popsolver::DataType{partialChansPerGroup}) {
    return;
  }

  unsigned previousInChanGroups = 0;
  for (unsigned inChansPerGroup = inChansLower; inChansPerGroup <= inChansUpper;
       inChansPerGroup += grainSize) {
    unsigned inChanGroups =
        (roundedNumInChans + inChansPerGroup - 1) / inChansPerGroup;
    if (inChanGroups == previousInChanGroups) {
      // There is no point considering a larger group size if it doesn't
      // decrease the number of groups - the zero padding increases the
      // amount of work per group and we can't use fewer groups per tile.
      continue;
    }
    if (isJointPlan) {
      assert(options.pass == Pass::FC_TRAINING_FWD);
      // The input channels in the forward pass become the output channels of
      // the weight update pass. Make sure it is a multiple of the supported
      // output channels per group.
      if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
        continue;
    }

    // The MAC vertex does not require a grouping of the conv groups.
    const unsigned convGroupsPerGroup = 1;

    candidates.emplace_back(Plan::Method::MAC, inputType, partialType,
                            convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup, numConvUnits, numConvUnits,
                            useLimitedVersion);
    previousInChanGroups = inChanGroups;
  }
}

static void getConvVertexAMPCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");
  const auto constrainedNumConvUnits =
      planConstraints.get_optional<popsolver::DataType>("numAmpConvUnits");

  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnitsOnIpu =
      getNumConvUnits(floatActivations, ampFloatPartials, target);
  if (numConvUnitsOnIpu == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnitsOnIpu =
        getNumConvUnits(floatActivations, ampFloatPartials, target);
  }
  auto ampPartialType = ampFloatPartials ? poplar::FLOAT : poplar::HALF;
  if (canUseConvolutionInstruction(floatActivations, ampFloatPartials,
                                   target)) {
    const auto weightsPerConvUnit =
        target.getWeightsPerConvUnit(floatActivations);

    std::vector<unsigned> partialChansCandidates = {numConvUnitsOnIpu,
                                                    weightsPerConvUnit};
    std::vector<unsigned> numConvUnitsCandidates = {numConvUnitsOnIpu};
    // On IPU1 we support half of conv units configuration for HALF types
    const bool canUseAmp4 = options.enableAmpHalfEnginesPlan &&
                            target.getFp16InFp16OutConvUnitsPerTile() == 8 &&
                            !floatActivations;

    // On IPU2 we need to enable 8 engines config as well
    const bool canUseAmp8 = numConvUnitsOnIpu == 16;

    if (canUseAmp4 || canUseAmp8) {
      numConvUnitsCandidates.push_back(numConvUnitsOnIpu / 2);
      partialChansCandidates.push_back(numConvUnitsOnIpu / 2);
    }

    for (const auto convUnits : numConvUnitsCandidates) {
      for (unsigned inputs = weightsPerConvUnit; inputs >= 1; inputs--) {
        for (const auto partials : partialChansCandidates) {
          // Input channels constrain
          if (constrainedInChansPerGroup &&
              popsolver::DataType{inputs} != *constrainedInChansPerGroup) {
            continue;
          }

          // Partial channels constrain
          if (constrainedPartialChansPerGroup &&
              popsolver::DataType{partials} !=
                  *constrainedPartialChansPerGroup) {
            continue;
          }

          // Number of conv units constrain
          if (constrainedNumConvUnits &&
              popsolver::DataType{convUnits} != *constrainedNumConvUnits) {
            continue;
          }

          const auto usedWeightsPerConvUnit =
              weightsPerConvUnit * convUnits / numConvUnitsOnIpu;
          if (partials != convUnits && partials != usedWeightsPerConvUnit) {
            continue;
          }

          if (!canUseConvolutionInstruction(floatActivations, floatPartials,
                                            inputs, convUnits, partials,
                                            target)) {
            continue;
          }

          // There are two reasons we might choose to make partialChansPerGroup
          // not equal to numConvUnitsOnIpu:
          // - The output of a convolution is likely to be fed into another
          //   convolution that wants its input grouped by weightsPerConvUnit
          //   so there will be a small cost (estimated by the planner) if
          //   partialChansPerGroup != weightsPerConvUnit
          // - The output channel grouping of a fully connected forward pass
          //   becomes the input channel grouping of the fully connected weight
          //   update pass and so if partialChansPerGroup != weightsPerConvUnit
          //   we can't fully utilize AMP in the weight update pass.
          // Neither of these reasons apply to fully connected inference (we
          // must always rearrange the output regardless of the grouping and
          // there is no weight update pass).
          if (options.pass == Pass::FC_INFERENCE_FWD && partials != convUnits) {
            continue;
          }

          if (isJointPlan) {
            assert(options.pass == Pass::FC_TRAINING_FWD);
            // The input channels in the forward pass become the output channels
            // of the weight update pass. Make sure it is a multiple of the
            // supported output channels per group.
            if (inputs % convUnits != 0) {
              continue;
            }
          }

          // AMP only supports a conv group grouping of 1.
          const unsigned convGroupsPerGroup = 1;

          candidates.emplace_back(Plan::Method::AMP, inputType, ampPartialType,
                                  convGroupsPerGroup, inputs, partials, 0,
                                  convUnits, true);
        }
      }
    }
  }
}

static void getConvVertexSLICCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {

  if (inputType != poplar::HALF) {
    return;
  }

  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<popsolver::DataType>("convGroupsPerGroup");
  const auto constrainedSlicWindowWidth =
      planConstraints.get_optional<popsolver::DataType>("slicWindowWidth");

  const auto constrainedChansPerGroup =
      [&]() -> boost::optional<popsolver::DataType> {
    const auto constrainedInChansPerGroup =
        planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
    const auto constrainedPartialChansPerGroup =
        planConstraints.get_optional<popsolver::DataType>(
            "partialChansPerGroup");

    if (constrainedInChansPerGroup && constrainedPartialChansPerGroup &&
        *constrainedInChansPerGroup != *constrainedPartialChansPerGroup) {
      throw poputil::poplibs_error("SLIC requires the input and output channel "
                                   "grouping to be the same.");
    }

    if (constrainedInChansPerGroup) {
      return constrainedInChansPerGroup;
    } else if (constrainedPartialChansPerGroup) {
      return constrainedPartialChansPerGroup;
    } else {
      return boost::none;
    }
  }();
  const bool floatActivations = inputType == poplar::FLOAT;
  const bool floatPartials = partialType == poplar::FLOAT;
  bool ampFloatPartials = floatPartials;
  auto numConvUnits =
      getNumConvUnits(floatActivations, ampFloatPartials, target);
  if (numConvUnits == 0 && !floatPartials) {
    ampFloatPartials = true;
    numConvUnits = getNumConvUnits(floatActivations, ampFloatPartials, target);
  }
  // List the number of conv units used in the candidate vertices which are
  // available - either on this hardware or implemented at present
  std::vector<unsigned> convUnitsCandidates;
  if (floatPartials) {
    convUnitsCandidates.push_back(8);
  } else {
    if (numConvUnits == 16) {
      convUnitsCandidates.push_back(16);
    }
    // This is always available with 8, or 16 conv units - let cycle estimates
    // reject it in favour of the 16 conv unit version if that's available
    convUnitsCandidates.push_back(8);
  }

  const auto ampPartialType = ampFloatPartials ? poplar::FLOAT : poplar::HALF;
  const unsigned weightsPerConvUnit =
      target.getWeightsPerConvUnit(floatActivations);

  // the numbers below are hardcoded but dependent on the expected machine
  // model that the real hardware models. ie. we expect 16 weights per conv unit

  if (weightsPerConvUnit != 16) {
    throw poputil::poplibs_error("Unsupported number of weights per conv "
                                 "unit for the SLIC instruction.");
  }

  // TODO: T14626, add a vertex for the the 1x3 kernel window size.
  const unsigned slicWindowWidth =
      constrainedSlicWindowWidth.value_or(popsolver::DataType{4})
          .getAs<unsigned>();

  if (isJointPlan) {
    assert(options.pass == Pass::FC_TRAINING_FWD);
    // There are a number of transformations between different passes when a
    // joint plan is being used which would need updating to handle SLIC.
    // T17666 tracks this. For the time being, don't allow joint plans with
    // SLIC.
    return;
  }

  struct Candidate {
    unsigned groups;
    unsigned channels;
  };
  std::array<Candidate, 3> groupings{Candidate{1u, 4u}, Candidate{2u, 2u},
                                     Candidate{4u, 1u}};
  for (const auto convUnits : convUnitsCandidates) {
    for (const auto &grouping : groupings) {
      if (constrainedConvGroupsPerGroup &&
          *constrainedConvGroupsPerGroup !=
              popsolver::DataType{grouping.groups}) {
        continue;
      }

      if (constrainedChansPerGroup &&
          *constrainedChansPerGroup != popsolver::DataType{grouping.channels}) {
        continue;
      }

      candidates.emplace_back(Plan::Method::SLIC, inputType, ampPartialType,
                              grouping.groups, grouping.channels,
                              grouping.channels, slicWindowWidth, convUnits,
                              true);
    }
  }
}

static void getConvVertexOuterProductCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");

  const auto inChansPerGroup = 1u;
  const auto partialChansPerGroup = target.getVectorWidth(inputType);
  // Only one supported inChansPerGroup or partialChansPerGroup
  // for this method.
  if (constrainedInChansPerGroup &&
      *constrainedInChansPerGroup != popsolver::DataType{inChansPerGroup}) {
    return;
  }
  if (constrainedPartialChansPerGroup &&
      *constrainedPartialChansPerGroup !=
          popsolver::DataType{partialChansPerGroup}) {
    return;
  }
  // OuterProduct only implemented for when Tile.PartialType == input type.
  if (partialType != params.inputType) {
    return;
  }

  // The OuterProduct vertex does not require a grouping of the conv groups.
  const unsigned convGroupsPerGroup = 1;

  candidates.emplace_back(Plan::Method::OUTER_PRODUCT, inputType, inputType,
                          convGroupsPerGroup, inChansPerGroup,
                          partialChansPerGroup, 0, 0, true);
}

static std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            poplar::Type inputType, poplar::Type outputType,
                            poplar::Type partialType, const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedMethod = [&]() -> boost::optional<Plan::Method> {
    const auto constraint = planConstraints.get_optional<std::string>("method");
    if (constraint) {
      Plan::Method m;
      std::stringstream ss(*constraint);
      ss >> m;
      return m;
    }
    return boost::none;
  }();

  std::vector<Plan::Method> methodCandidates;
  if (constrainedMethod) {
    methodCandidates.push_back(*constrainedMethod);
  } else {

    // Disable SLIC until T18365 is fixed
    bool disableSLIC = options.pass == Pass::FC_INFERENCE_FWD ||
                       options.pass == Pass::FC_TRAINING_BWD ||
                       options.pass == Pass::FC_TRAINING_FWD ||
                       options.pass == Pass::FC_TRAINING_WU;

    // the order here should be in most-likely-best first for performance
    // because the planner constrains future models against the current best.
    methodCandidates = {
        Plan::Method::AMP,
        Plan::Method::SLIC,
        Plan::Method::MAC,
        Plan::Method::OUTER_PRODUCT,
    };

    if (disableSLIC) {
      methodCandidates.erase(methodCandidates.begin() + 1);
    }
  }

  // All the following methods assume half or float input/partial types.
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  assert(inputType == poplar::HALF || inputType == poplar::FLOAT);

  std::vector<ConvVertexType> convVertexTypeCandidates;
  for (const auto &method : methodCandidates) {
    switch (method) {
    case Plan::Method::MAC: {
      getConvVertexMACCandidates(target, inputType, outputType, partialType,
                                 params, options, isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::AMP: {
      getConvVertexAMPCandidates(target, inputType, outputType, partialType,
                                 params, options, isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Plan::Method::SLIC: {
      getConvVertexSLICCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Plan::Method::OUTER_PRODUCT: {
      getConvVertexOuterProductCandidates(
          target, inputType, outputType, partialType, params, options,
          isJointPlan, convVertexTypeCandidates);
      break;
    }
    default: {
      throw poputil::poplibs_error("Unknown Plan::Method");
    }
    }
  }
  return convVertexTypeCandidates;
}

static bool expandingDimChangesParams(const ConvParams &params, unsigned dim) {
  auto newParams = params;
  expandDim(newParams, dim);
  return newParams != params;
}

// Given a set return the set of all subsets. The set is specified as a
// vector that is assumed to have no duplicates. The relative order of
// items in each subset returned by this function matches the relative order
// of the items in the set of all items.
template <class T>
static std::vector<std::vector<T>> getPowerSet(const std::vector<T> &items) {
  const unsigned numItems = items.size();
  if (numItems >= std::numeric_limits<unsigned>::digits) {
    // Not handled.
    std::abort();
  }
  std::vector<std::vector<T>> subsets;
  subsets.reserve(1u << numItems);
  // We associate each subset with a number. The nth bit of the number indicates
  // whether the nth item is in the subset. We enumerate all subsets by
  // iterating over all numbers in the range [0, 1 << numItems).
  for (unsigned i = 0; i < (1u << numItems); ++i) {
    subsets.emplace_back();
    for (unsigned item = 0; item != numItems; ++item) {
      if ((i >> item) & 1)
        subsets.back().push_back(items[item]);
    }
  }
  return subsets;
}

static std::vector<std::vector<unsigned>>
getExpandDimsCandidates(unsigned ipuLevel, const ConvParams &params,
                        const ConvOptions &options) {
  const auto &planConstraints = options.planConstraints;
  const auto constraint = planConstraints.get_child_optional(
      std::to_string(ipuLevel) + ".transform.expandDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    forcedDims.reserve(constraint->size());
    for (const auto &child : *constraint) {
      forcedDims.push_back(child.second.get_value<unsigned>());
    }
    std::sort(forcedDims.begin(), forcedDims.end());
    forcedDims.erase(std::unique(forcedDims.begin(), forcedDims.end()),
                     forcedDims.end());
    std::reverse(forcedDims.begin(), forcedDims.end());
    candidateDimSets.emplace_back(std::move(forcedDims));
  } else {
    std::vector<unsigned> candidateDims;
    for (unsigned i = 0; i != params.getNumFieldDims(); ++i) {
      if (!expandingDimChangesParams(params, i)) {
        continue;
      }
      // Don't expand this dimension if the number of non zero kernel entries
      // is larger than the number of non zero input entries as it is unlikely
      // to be profitable. This heuristic cuts down the size of the search
      // space.
      //
      // TODO: T12884 Investigate better heuristics.
      if (params.inputFieldShape[i] < params.kernelShape[i])
        continue;
      candidateDims.push_back(i);
    }
    candidateDimSets = getPowerSet(candidateDims);
    for (auto &subset : candidateDimSets) {
      // The subsets returned by getPowerSet have the outermost dimension first
      // but it is more efficient to expand the innermost dimension first.
      std::reverse(subset.begin(), subset.end());
    }
  }
  return candidateDimSets;
}

static std::vector<std::vector<unsigned>>
getOutChanFlattenDimsCandidates(unsigned ipuLevel, const ConvParams &params,
                                const ConvOptions &options) {
  auto swappedParams = params;
  const auto &planConstraints = options.planConstraints;
  const auto constraint = planConstraints.get_child_optional(
      std::to_string(ipuLevel) + ".transform.outChanFlattenDims");
  std::vector<std::vector<unsigned>> candidateDimSets;
  if (constraint) {
    std::vector<unsigned> forcedDims;
    forcedDims.reserve(constraint->size());
    for (const auto &child : *constraint) {
      forcedDims.push_back(child.second.get_value<unsigned>());
    }
    std::sort(forcedDims.begin(), forcedDims.end());
    forcedDims.erase(std::unique(forcedDims.begin(), forcedDims.end()),
                     forcedDims.end());
    std::reverse(forcedDims.begin(), forcedDims.end());
    candidateDimSets.emplace_back(std::move(forcedDims));
  } else {
    if (params.outputChannelsPerConvGroup)
      poplin::swapOperands(swappedParams);
    std::vector<unsigned> candidateDims;
    for (unsigned i = 0; i != swappedParams.getNumFieldDims(); ++i) {
      // Don't flatten this dimension into the output channel dimension if it
      // wouldn't increase the number of output channels.
      if (params.getOutputSize(i) == 1)
        continue;
      // Don't flatten this dimension into the output channel dimension if the
      // number of non zero input entries is larger than the number of non zero
      // kernel entries as it is unlikely to be profitable. This heuristic cuts
      // down the size of the search space. TODO: T12884 Investigate better
      // heuristics.
      if (params.inputFieldShape[i] > params.kernelShape[i])
        continue;
      candidateDims.push_back(i);
    }
    candidateDimSets = getPowerSet(candidateDims);
    for (auto &subset : candidateDimSets) {
      // The subsets returned by getPowerSet have the outermost dimension first
      // but it is more efficient to expand the innermost dimension first.
      std::reverse(subset.begin(), subset.end());
    }
  }
  return candidateDimSets;
}

void swapOperands(ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> extraInputPadding(numFieldDims);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    const auto transformedInputSize = params.getTransformedInputSize(dim);
    const auto transformedKernelSize = params.getTransformedKernelSize(dim);
    extraInputPadding[dim] = transformedInputSize - transformedKernelSize;
  }
  std::swap(params.inputFieldShape, params.kernelShape);
  std::swap(params.inputTransform, params.kernelTransform);
  std::swap(params.batchSize, params.outputChannelsPerConvGroup);
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    params.inputTransform.flip[dim] = !params.inputTransform.flip[dim];
    params.kernelTransform.flip[dim] = !params.kernelTransform.flip[dim];
    params.inputTransform.paddingLower[dim] += extraInputPadding[dim];
    params.inputTransform.paddingUpper[dim] += extraInputPadding[dim];
  }
  params = params.canonicalize();
}

static std::vector<bool> getSwapOperandCandidates(const ConvParams &params,
                                                  const ConvOptions &options,
                                                  bool isJointPlan) {
  std::vector<bool> validValues;
  if (isJointPlan) {
    // The joint planning logic doesn't yet handle swapped operands.
    // TODO: T12885 Lift this restriction.
    validValues = {false};
  } else {
    validValues = {false, true};
  }

  // Check for explicitly forced swapped operands in the options.
  const auto &planConstraints = options.planConstraints;
  const auto constraint =
      planConstraints.get_optional<bool>("0.transform.swapOperands");
  if (constraint) {
    if (std::find(validValues.begin(), validValues.end(), *constraint) ==
        validValues.end()) {
      throw poputil::poplibs_error(
          "0.transform.swapOperands was constrained to be '" +
          std::string(*constraint ? "true" : "false") +
          "' but this is not valid for these parameters");
    }
    validValues = {*constraint};
  }

  return validValues;
}

static std::vector<ConvTypes> getConvTypes(const poplar::Target &target,
                                           unsigned numLevels,
                                           poplar::Type resultType,
                                           const ConvOptions &options) {
  std::vector<ConvTypes> types(numLevels);
  for (int level = numLevels - 1; level >= 0; --level) {
    types[level].partialType = options.partialsType;
    if (level == 0) {
      types[level].resultType = resultType;
    } else {
      bool isTileLevel = static_cast<unsigned>(level) == numLevels - 1;
      auto levelResultType = isTileLevel ? options.interTilePartialsType
                                         : options.interIpuPartialsType;
      // Use the result type of the previous level if it is smaller than the
      // requested result type. This means that if a user wants to use half
      // partials they only need to set the option for the first level that
      // should use half partials.
      if (!isTileLevel && target.getTypeSize(levelResultType) >
                              target.getTypeSize(types[level + 1].resultType)) {
        levelResultType = types[level + 1].resultType;
      }
      // There is no point in using a result type larger than the partial type.
      if (target.getTypeSize(levelResultType) >
          target.getTypeSize(types[level].partialType)) {
        levelResultType = types[level].partialType;
      }
      types[level].resultType = levelResultType;
    }
  }
  return types;
}

static std::vector<unsigned> getDilatePostConvDims(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  std::vector<unsigned> dilateAfterConv;
  for (std::size_t dim = 0; dim != numFieldDims; ++dim) {
    if (params.inputTransform.dilation[dim] != 1 &&
        canDeferDilation(params, dim)) {
      dilateAfterConv.push_back(dim);
    }
  }
  std::reverse(dilateAfterConv.begin(), dilateAfterConv.end());
  return dilateAfterConv;
}

#ifndef NDEBUG
static bool isPowerOf2(const unsigned n) {
  if (n == 0) {
    return false;
  }
  return (n & (n - 1)) == 0;
}
#endif

static std::vector<unsigned> getCombineConvGroupCandidates(
    const unsigned level, const ConvParams &params, const ConvOptions &options,
    const poplar::Target &target, const bool isJointPlan) {

  std::string transform = std::to_string(level) + ".transform.";
  std::vector<unsigned> validValues = [&] {
    // when we have more than one conv group and one input channel we want to
    // try this transformation.
    const auto ci = params.inputChannelsPerConvGroup;
    const bool validInputChannelSize =
        (params.inputType == poplar::FLOAT && ci == 1) ||
        (params.inputType == poplar::HALF && (ci == 1 || ci == 2));

    // Joint plans may invalidate this transformation if they, for example, swap
    // the input channels with the batch size and the batch size does not
    // satisfy the constraint above. TODO: T12886 With a more advanced check
    // here we could support cases like this.
    if (validInputChannelSize && params.numConvGroups > 1 && !isJointPlan) {
      const auto baseLoadElements =
          params.inputType == poplar::HALF
              ? target.getFp16ConvUnitInputLoadElemsPerCycle()
              : target.getFp32ConvUnitInputLoadElemsPerCycle();

      unsigned minFactor = convGroupCombineFactor(baseLoadElements, ci);
      const unsigned maxFactor =
          (params.inputType == poplar::HALF ? 16U : 8U) / ci;

      assert(isPowerOf2(baseLoadElements));
      assert(isPowerOf2(ci));
      assert(isPowerOf2(maxFactor));
      assert(minFactor > 0);
      std::vector<unsigned> result;

      // We call `result.push_back` until `minFactor * (2^n) <= maxFactor` is
      // false, where `n` is the loop counter. Solving for the loop counter
      // when the condition is false gives:
      //
      //    n > log2(maxFactor / minFactor)
      //
      // As n is an integer the greater than condition is first fulfilled by
      //
      //    n = floor(log2(maxFactor / minFactor)) + 1
      //      = 8 * sizeof(maxFactor) - __builtin_clz(maxFactor / minFactor)
      //
      // We also unconditionally push_back once and __builtin_clz (and log2)
      // is undefined for an input of zero, so we need to handle that case too.
      //
      // Some examples of the using the formula:
      //
      //     minFactor | maxFactor | expected | formula
      //    -----------+-----------+----------+---------
      //             1 |         2 |        3 |       3
      //             1 |         8 |        5 |       5
      //             3 |         8 |        3 |       3
      //             2 |        16 |        5 |       5
      //             4 |         2 |        1 |       1
      size_t n = 1;
      if (minFactor < maxFactor)
        n += 8 * sizeof(maxFactor) - __builtin_clz(maxFactor / minFactor);

      result.reserve(n);
      result.push_back(1U); // 1 is noop transform
      while (minFactor <= maxFactor) {
        result.push_back(minFactor);
        minFactor *= 2;
      }
      return result;
    } else {
      return std::vector<unsigned>{1U};
    }
  }();

  const auto &planConstraints = options.planConstraints;
  const auto constraint_ =
      planConstraints.get_child_optional(transform + "combineConvGroupsFactor");
  if (constraint_) {
    std::set<unsigned> constraints;
    for (const auto &child : *constraint_) {
      constraints.insert(child.second.get_value<unsigned>());
    }
    if (std::any_of(constraints.begin(), constraints.end(),
                    [](unsigned i) { return i != 1U; })) {
      const auto expandDimsConstraint =
          planConstraints.get_child_optional(transform + "expandDims");
      const auto outChanFlattenDimsConstraint =
          planConstraints.get_child_optional(transform + "outChanFlattenDims");
      if ((expandDimsConstraint && !expandDimsConstraint->empty()) ||
          (outChanFlattenDimsConstraint &&
           !outChanFlattenDimsConstraint->empty())) {
        throw poputil::poplibs_error(
            "The combineConvGroups transformation is only valid when there is "
            "there is not another transformation that can increase the number "
            "of input channels (ie. expandDims or outChanFlattenDims");
      }
    }

    auto constrainedValues =
        boost::adaptors::filter(validValues, [&](unsigned i) {
          return static_cast<bool>(constraints.count(i));
        });
    return std::vector<unsigned>(constrainedValues.begin(),
                                 constrainedValues.end());
  }

  return validValues;
}

/*
 * Function ensures:
 * 1. Each level specified in plan constraints is within range of hierarchy.
 * 2. Each value within transform.expandDims and transform.outChanFlattenDims
 *    arrays are a valid field dimension.
 * 3. The key of each child of partition.fieldSplit and partition.kernelSplit
 *    is a valid field or kernel dimension, respectively.
 */
void validatePlanConstraints(
    const ConvParams &params,
    const poplibs_support::PlanConstraints &planConstraints,
    const std::size_t numLevels) {
  const struct {
    std::string key;
    bool checkKey; // If false, each element of value array will be validated.
    std::size_t maximum;
  } keysToCheck[] = {
      {"transform.expandDims", false, params.getNumFieldDims()},
      {"transform.outChanFlattenDims", false, params.getNumFieldDims()},
      {"partition.fieldSplit", true, params.getNumFieldDims()},
      {"partition.kernelSplit", true, params.kernelShape.size()},
  };

  auto isNumeric = [](const std::string &text) -> bool {
    return !text.empty() && std::all_of(text.begin(), text.end(), ::isdigit);
  };

  auto isValidKey = [&isNumeric](const std::string &key,
                                 const std::size_t maximum) -> bool {
    if (!isNumeric(key)) {
      throw poputil::poplibs_error("Invalid key - must be numeric: " + key);
    }
    return std::stoul(key) >= maximum;
  };

  for (const auto &kv : planConstraints) {
    if (!isNumeric(kv.first)) {
      continue; // No further checks for non-numeric keys.
    }

    if (std::stoul(kv.first) >= numLevels) {
      throw poputil::poplibs_error("Plan constraint " + kv.first +
                                   " is not a valid level of hierarchy.");
    }
    for (const auto &entry : keysToCheck) {
      if (const auto &child = kv.second.get_child_optional(entry.key)) {
        for (const auto &childKV : *child) {
          if (entry.checkKey
                  ? isValidKey(childKV.first, entry.maximum)
                  : childKV.second.get_value<popsolver::DataType>() >=
                        popsolver::DataType{entry.maximum}) {
            throw poputil::poplibs_error(
                "Invalid plan constraint: " + kv.first + "." + entry.key + "." +
                childKV.first + (entry.checkKey ? " Key" : " Value") +
                " out-of-range -- maximum: " + std::to_string(entry.maximum));
          }
        }
      }
    }
  }
}

static void logPlanBreakdown(logging::Level l, const Plan &plan,
                             const Cost &cost,
                             const boost::optional<Cost> &referenceCost) {
  logging::log(l, "  breakdown of memory and cycle estimates:");
  logging::log(l, "   - total parallel split: {}", plan.totalParallelSplit());
  logging::log(l, "   - total serial split: {}", plan.totalSerialSplit());
  logging::log(l,
               "   - rearrangement before slice: {} cycles, {} bytes ({} "
               "overhead, {} per-loop iteration)",
               cost.rearrangeBeforeSliceCycles,
               cost.rearrangeBeforeSliceTempBytes +
                   cost.rearrangeBeforeSliceTempDuringRearrangeBytes,
               cost.rearrangeBeforeSliceTempBytes,
               cost.rearrangeBeforeSliceTempDuringRearrangeBytes);
  logging::log(l, "   - memsetZeroBeforeAddInPlace: {} cycles, unknown bytes",
               cost.memsetZeroBeforeAddInPlace);
  logging::log(l, "   - dynamic slice: {} cycles, unknown bytes",
               cost.dynamicSliceCycles);
  logging::log(l, "   - transform: {} cycles, {} bytes", cost.transformCycles,
               cost.transformTempBytes);
  logging::log(l,
               "   - exchange: {} cycles, n/a bytes. (Input {},"
               " Weight {}, Reduce {} + {})",
               cost.totalExchangeCycles,
               cost.itemisedExchangeCycles.inputExchangeCycles,
               cost.itemisedExchangeCycles.weightExchangeCycles,
               cost.itemisedExchangeCycles.reduceFirstStageExchangeCycles,
               cost.itemisedExchangeCycles.reduceRemainingStagesExchangeCycles);

  logging::log(l, "   - tile level transform: {} cycles, {} bytes",
               cost.tileLevelTransformCycles, cost.tileLevelTransformTempBytes);
  logging::log(l, "   - compute: {} cycles, {} bytes", cost.partialCalcCycles,
               cost.convTempBytes);
  logging::log(l, "   - reduction: {} cycles, {} bytes", cost.reduceCycles,
               cost.reduceTempBytes);
  logging::log(l, "   - dynamic update: {} cycles, unknown bytes",
               cost.dynamicUpdateCycles);
  logging::log(l, "   - add in-place: {} cycles, {} bytes",
               cost.addInPlaceCycles, cost.addInPlaceTempBytes);
  // The tensor generated on the final cast is not considered as part of the
  // temporary memory for the purposes of the Conv Planner.
  logging::log(l, "   - cast: {} cycles, 0 bytes", cost.castCycles, 0);
  logging::log(l, "   - total: {} cycles, {} bytes", cost.totalCycles,
               cost.totalTempBytes);
  if (referenceCost) {
    logging::log(l,
                 "   - cycle difference compared to reference ({} cycles): {}",
                 referenceCost->totalCycles, cost.totalPerStepCycleDiff);
  }
}

static std::vector<unsigned> getHierarchy(const ConvOptions &options) {
  return poplibs::getTileHierarchy(options.numIPUs, options.tilesPerIPU);
}

static std::pair<Plan, Cost>
createPlan(ConvParams params, const ConvOptions &options, bool isJointPlan,
           const PlanningObjective &objective, const poplar::Target &target,
           unsigned startTileIdxForVirtualHierarchy,
           const boost::optional<Plan> &referencePlan,
           const boost::optional<Cost> &referenceCost,
           PlanningCacheImpl::CycleEstimationImpl *cache) {
  logging::debug("Creating plan with objective {}", objective);
  validateLayerParams(params, options, target);

  // A coarse metric to measure the efficiency of the constraint solver
  popsolver::ConstraintEvaluationSummary totalConstraintsEvaluated{};

  // T8972: It is currently assumed that the parameters for all the training
  // passes can be derived from one pass, but this is no longer the case since a
  // different outputType can be specified for each pass. To avoid a costly
  // exchange of weights, we plan with the assumption that
  // outputType == inputType for FC_TRAINING.
  const auto originalOutputType = params.outputType;
  if (isJointPlan) {
    params.outputType = params.inputType;
  }

  // perLevelExchangeBytesPerCycle is indexed by hierarchy (not including the
  // tile level), lower indices to higher hierarchies.
  const auto perLevelExchangeBytesPerCycle =
      poplibs::getPerLevelExchangeBytesPerCycle(target, options.numIPUs);
  const auto hierarchy = getHierarchy(options);
  const auto numLevels = hierarchy.size() + 1;

  validatePlanConstraints(params, options.planConstraints, numLevels);

  Cost bestCost = highestCost;
  Plan bestPlan;
  std::vector<ConvTransform> transforms(numLevels);
  const auto convTypes =
      getConvTypes(target, numLevels, params.outputType, options);
  const auto ipuLevel = transforms.size() - 2;
  unsigned addedFieldDims = 0;
  auto numFieldDims = params.getNumFieldDims();
  auto paramsWithExtraDims = params;
  if (numFieldDims < 2) {
    // Various places assume there are at least two dimensions. In particular
    // code related to the nx1ConvPartial vertex has special handling for the
    // outermost dimension and special handling for the innermost dimension
    // and there is an assumption that these two dimensions are distinct.
    addedFieldDims = 2 - numFieldDims;
    addExtraDims(paramsWithExtraDims, addedFieldDims);
    numFieldDims = 2;
  }
  transforms[0].extraFieldDims = addedFieldDims;
  transforms[0].dilatePostConv = getDilatePostConvDims(paramsWithExtraDims);
  const auto paramsWithDeferredDilation = calculateParamsWithDeferredDilation(
      paramsWithExtraDims, transforms[0].dilatePostConv);

  for (bool swapOperands : getSwapOperandCandidates(paramsWithDeferredDilation,
                                                    options, isJointPlan)) {
    transforms[0].swapOperands = swapOperands;
    const auto swappedParams =
        calculateSwappedParams(paramsWithDeferredDilation, swapOperands);

    for (const std::vector<unsigned> &expandDims :
         getExpandDimsCandidates(ipuLevel, swappedParams, options)) {
      transforms[ipuLevel].expandDims = expandDims;
      auto expandedParams = calculateExpandedParams(swappedParams, expandDims);

      for (const std::vector<unsigned> &outChanFlattenDims :
           getOutChanFlattenDimsCandidates(ipuLevel, expandedParams, options)) {
        transforms[ipuLevel].outChanFlattenDims = outChanFlattenDims;
        auto flattenedParams =
            calculateFlattenedParams(expandedParams, outChanFlattenDims,
                                     transforms[ipuLevel].flattenDims);

        for (const unsigned combineConvGroups : getCombineConvGroupCandidates(
                 ipuLevel, flattenedParams, options, target, isJointPlan)) {
          transforms[ipuLevel].combineConvGroupsFactor = combineConvGroups;
          const auto groupedParams = calculateGroupedParams(
              flattenedParams, transforms[ipuLevel].combineConvGroupsFactor);

          const auto convVertexTypeCandidates = getConvVertexTypeCandidates(
              target, params.inputType, params.outputType,
              convTypes.back().partialType, groupedParams, options,
              isJointPlan);

          for (const auto &convVertexType : convVertexTypeCandidates) {
            std::vector<unsigned> fieldGrainSize(numFieldDims, 1);
            if (isJointPlan) {
              assert(options.pass == Pass::FC_TRAINING_FWD);
              // The innermost grain size becomes the inChansPerGroup in the
              // backward pass. For now assume the same grouping in both
              // passes.
              // TODO: T12887 Search for the optimal grouping in each pass.
              fieldGrainSize.back() = convVertexType.inChansPerGroup;
            } else if (groupedParams.outputType == poplar::HALF &&
                       convVertexType.partialChansPerGroup % 2 &&
                       groupedParams.getOutputSize(
                           groupedParams.getNumFieldDims() - 1) %
                               2 ==
                           0) {
              // If the number of output channels per group is odd then use a
              // field grain size of 2 to ensure the result has an even number
              // of elements on each tile since an odd number of elements
              // on a tile tends to cause costly rearrangements in the next
              // layer.
              fieldGrainSize.back() = 2;
            }
            Plan candidate;
            Cost candidateCost;
            // Override the partials type at the tile level with that chosen
            // for the vertex type as we may choose a lower precision to
            // implement the operation if we know the vertex can effectively
            // maintain the accuracy implied by the requested partials type.
            auto newConvTypes = convTypes;
            newConvTypes.back().partialType = convVertexType.partialType;
            popsolver::ConstraintEvaluationSummary constraintsEvaluated{};
            std::tie(candidate, candidateCost, constraintsEvaluated) =
                choosePlan(target, transforms, newConvTypes, hierarchy,
                           perLevelExchangeBytesPerCycle, fieldGrainSize,
                           convVertexType, params, isJointPlan, bestCost,
                           objective, startTileIdxForVirtualHierarchy,
                           referencePlan, referenceCost, cache, options);
            logging::trace("Evaluated {} constraints for candidate plan",
                           constraintsEvaluated);
            totalConstraintsEvaluated += constraintsEvaluated;
            if (candidateCost == highestCost) {
              continue;
            }

            if (objective.lowerCost(candidateCost, bestCost)) {
              bestPlan = candidate;
              bestCost = candidateCost;

              logging::debug("Found new best candidate plan using {}: {}",
                             candidate.method, candidateCost);
              logPlanBreakdown(logging::Level::Trace, bestPlan, bestCost,
                               referenceCost);
            }
          }
        }
      }
    }
  }

  const auto planIsValid = bestCost != highestCost;

  if (isJointPlan && planIsValid) {
    // If we created a plan with the assumption that inputType == outputType,
    // we now restore resultType to ensure bestPlan is valid.
    const auto numLevelsOfHierarchy = hierarchy.size() + 1;
    for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
      const auto outputTypeSize = target.getTypeSize(originalOutputType);
      auto &types = bestPlan.types[level];

      if (target.getTypeSize(types.resultType) < outputTypeSize || 0 == level) {
        types.resultType = originalOutputType;
      }
      if (target.getTypeSize(types.partialType) < outputTypeSize) {
        types.partialType = originalOutputType;
      }
    }
  }

  if (planIsValid) {
    logging::debug("Evaluated a total of {} constraints to find the best plan",
                   totalConstraintsEvaluated);
  } else {
    logging::debug(
        "Evaluated a total of {} constraints and could not find a valid plan",
        totalConstraintsEvaluated);
  }
  return {bestPlan, bestCost};
}
static CanonicalConvParams
getFullyConnectedPassParams(const CanonicalConvParams &params,
                            const ConvOptions &options, Pass pass) {
  assert(params->getNumFieldDims() == 1);
  assert(params->batchSize == 1);
  assert(params->inputTransform.flip[0] == false);
  assert(params->inputTransform.dilation[0] == 1);
  assert(params->kernelTransform.flip[0] == false);
  assert(params->kernelTransform.truncationLower[0] == 0);
  if (params->inputFieldShape[0] == 0) {
    // for a zero convolution the canonical form is to provide a kernel of size
    // 1 and then truncate it back to zero.
    assert(params->kernelTransform.truncationUpper[0] == 1);
    assert(params->outputTransform.truncationUpper[0] == 1);
  } else {
    assert(params->kernelTransform.truncationUpper[0] == 0);
    assert(params->outputTransform.truncationUpper[0] == 0);
  }
  assert(params->kernelShape[0] == 1);
  assert(params->outputTransform.stride[0] == 1);
  assert(params->outputTransform.paddingLower[0] == 0);
  assert(params->outputTransform.paddingUpper[0] == 0);

  // Translate convolution parameters to parameters of the fully connected layer
  // forward pass.
  unsigned fwdOutputSize, fwdInputSize, fwdBatchSize;
  switch (options.pass) {
  default:
    assert(0 && "Unexpected pass");
  case Pass::FC_TRAINING_FWD:
    fwdInputSize = params->getNumInputChansPerConvGroup();
    fwdBatchSize = params->getNumOutputChansPerConvGroup();
    fwdOutputSize = params->getInputSize(0);
    break;
  case Pass::FC_TRAINING_BWD:
    fwdInputSize = params->getInputSize(0);
    fwdBatchSize = params->getNumOutputChansPerConvGroup();
    fwdOutputSize = params->getNumInputChansPerConvGroup();
    break;
  case Pass::FC_TRAINING_WU:
    fwdOutputSize = params->getInputSize(0);
    fwdBatchSize = params->getNumInputChansPerConvGroup();
    fwdInputSize = params->getNumOutputChansPerConvGroup();
    break;
  }
  // Translate fully connected layer forward pass parameters back into
  // convolution parameters for the specified pass.
  unsigned convFieldSize, convInputChannels, convOutputChannels,
      inputPadding = 0, outputTruncation = 0;
  switch (pass) {
  default:
    assert(0 && "Unexpected pass");
  case Pass::FC_TRAINING_FWD:
    convInputChannels = fwdInputSize;
    convFieldSize = fwdOutputSize;
    convOutputChannels = fwdBatchSize;
    break;
  case Pass::FC_TRAINING_BWD:
    convInputChannels = fwdOutputSize;
    convFieldSize = fwdInputSize;
    convOutputChannels = fwdBatchSize;
    break;
  case Pass::FC_TRAINING_WU:
    convInputChannels = fwdBatchSize;
    convFieldSize = fwdOutputSize;
    convOutputChannels = fwdInputSize;
    break;
  }
  if (convFieldSize == 0) {
    // Transformed input must be greater than or equal to the transformed kernel
    // size.
    inputPadding = 1;
    outputTruncation = 1;
  }
  ConvParams newParams{
      params->inputType,
      params->outputType,
      1,                         // batchSize
      {convFieldSize},           // inputShape
      {1},                       // kernelShape
      convInputChannels,         // input channels
      convOutputChannels,        // output channels
      params->getNumConvGroups() // conv groups
  };
  newParams.inputTransform.paddingUpper = {inputPadding};
  newParams.outputTransform.truncationUpper = {outputTruncation};

  return newParams;
}

static ConvOptions getFullyConnectedPassOptions(const ConvOptions &options,
                                                Pass pass) {
  auto newOptions = options;
  newOptions.pass = pass;
  return newOptions;
}

static std::pair<Plan, Cost>
createPlan(const ConvParams &params, const ConvOptions &options,
           const PlanningObjective &objective, const poplar::Target &target,
           unsigned startTileIdxForVirtualHierarchy,
           const boost::optional<Plan> &referencePlan,
           const boost::optional<Cost> &referenceCost,
           PlanningCacheImpl::CycleEstimationImpl *cache,
           std::vector<std::pair<PlanningCacheImpl::Key, std::pair<Plan, Cost>>>
               *additionalPlansToCache) {
  const auto memBound = objective.getTileTempMemoryBound();
  const bool hasMemBound = memBound != popsolver::DataType::max();
  // we only support joint plans for fully connected layers for now.
  const bool isJointPlan =
      options.pass == Pass::FC_TRAINING_FWD && !referencePlan && !referenceCost;

  auto isSet = [](const Cost &cost) { return cost != highestCost; };

  auto print = [&](const Pass &pass, bool isSeparate) {
    const auto planDesc =
        !isJointPlan ? "non-joint" : isSeparate ? "separate joint" : "joint";
    logging::debug("Creating {} plan ({} pass)...", planDesc, pass);
  };

  auto createMyPlan = [&](const ConvParams &params, const ConvOptions &options,
                          bool isJointPlan, const PlanningObjective &objective,
                          const boost::optional<Cost> &referenceCost) {
    return createPlan(params, options, isJointPlan, objective, target,
                      startTileIdxForVirtualHierarchy, referencePlan,
                      referenceCost, cache);
  };

  auto minimizeCycles = [&](const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan) {
    print(options.pass, !isJointPlan);
    assert(objective.getType() !=
           PlanningObjective::Type::MINIMIZE_TILE_TEMP_MEMORY);
    auto planAndCost =
        createMyPlan(params, options, isJointPlan, objective, referenceCost);
    if (!isSet(planAndCost.second)) {
      logging::warn("Warning: convolution planner unable to meet memory "
                    "target. Optimizing for minimum memory...");
    }
    return planAndCost;
  };

  auto minimizeMemory = [&](const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan) {
    print(options.pass, !isJointPlan);
    if (hasMemBound) {
      // If we failed at minimising cycles, let's retry doubling temp memory a
      // few times before aiming at minimum cycles without memory bound (at this
      // point it is not expected to fit anyway)
      auto stepObjective = objective;
      auto stepMemBound = memBound;
      do {
        stepMemBound = stepMemBound * popsolver::DataType{2};
        stepObjective.setTileTempMemoryBound(stepMemBound);
        auto planAndCost = createMyPlan(params, options, isJointPlan,
                                        stepObjective, referenceCost);
        if (isSet(planAndCost.second)) {
          return planAndCost;
        }
      } while (stepMemBound <
               popsolver::DataType{target.getBytesPerTile() * 2});
    }
    // Minimise cycles without memory bound
    return createMyPlan(params, options, isJointPlan,
                        PlanningObjective::minimizeCycles(), boost::none);
  };

  if (!isJointPlan) {
    if (hasMemBound) {
      auto planAndCost = minimizeCycles(params, options, false);
      if (isSet(planAndCost.second)) {
        return planAndCost;
      }
    }
    return minimizeMemory(params, options, false);
  }

  // It doesn't make sense to compare joint and separate planning when the
  // number of cycles is bounded since we can't easily derive bounds for each
  // individual pass from a bound on the total number of cycles.
  assert(objective.getCyclesBound() == popsolver::DataType::max());
  assert(objective.getType() != PlanningObjective::MINIMIZE_COST_DIFF);

  // Plan joint and separate joint convolutions
  auto bwdParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_BWD);
  auto bwdOptions =
      getFullyConnectedPassOptions(options, Pass::FC_TRAINING_BWD);
  auto wuParams =
      getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_WU);
  auto wuOptions = getFullyConnectedPassOptions(options, Pass::FC_TRAINING_WU);
  Plan jointPlan, fwdPlan, bwdPlan, wuPlan;
  Cost jointCost, fwdCost, bwdCost, wuCost;
  if (hasMemBound) {
    std::tie(jointPlan, jointCost) = minimizeCycles(params, options, true);
    std::tie(fwdPlan, fwdCost) = minimizeCycles(params, options, false);
    std::tie(bwdPlan, bwdCost) =
        minimizeCycles(bwdParams.getParams(), bwdOptions, false);
    std::tie(wuPlan, wuCost) =
        minimizeCycles(wuParams.getParams(), wuOptions, false);
  }
  // Go for minimum memory if there was a bound and neither joint nor separate
  // plans couldn't fit. Decoupling cycle minimisation from memory minimisation
  // avoids doing the latter if it is not needed. For example, if only the joint
  // plan succeeded at minimising cycles, minimising memory for the separated
  // joint plan is pointless as it won't be picked.
  if (!hasMemBound || (!isSet(jointCost) &&
                       !(isSet(fwdCost) && isSet(bwdCost) && isSet(wuCost)))) {
    if (!isSet(jointCost)) {
      std::tie(jointPlan, jointCost) = minimizeMemory(params, options, true);
    }
    // Replan only those phases that couldn't fit
    if (!isSet(fwdCost)) {
      std::tie(fwdPlan, fwdCost) = minimizeMemory(params, options, false);
    }
    if (!isSet(bwdCost)) {
      std::tie(bwdPlan, bwdCost) =
          minimizeMemory(bwdParams.getParams(), bwdOptions, false);
    }
    if (!isSet(wuCost)) {
      std::tie(wuPlan, wuCost) =
          minimizeMemory(wuParams.getParams(), wuOptions, false);
    }
  }

  auto separateCost = fwdCost;
  for (const auto &cost : {bwdCost, wuCost}) {
    if (!isSet(separateCost) || !isSet(cost)) {
      separateCost = highestCost;
      break;
    }
    separateCost.totalCycles += cost.totalCycles;
    separateCost.totalTempBytes =
        std::max(separateCost.totalTempBytes, cost.totalTempBytes);
    separateCost.totalPerStepCycleDiff += cost.totalPerStepCycleDiff;
  }

  const bool separatePlanHasLowerCost =
      separateCost.totalTempBytes <= memBound
          ? (jointCost.totalTempBytes > memBound ||
             separateCost.totalCycles < jointCost.totalCycles)
          : (jointCost.totalTempBytes > memBound &&
             separateCost.totalTempBytes < jointCost.totalTempBytes);
  if (separatePlanHasLowerCost) {
    if (additionalPlansToCache) {
      PlanningCacheImpl::Key bwdKey{std::move(bwdParams),
                                    std::move(bwdOptions),
                                    boost::none,
                                    boost::none,
                                    false,
                                    boost::none,
                                    0};
      additionalPlansToCache->emplace_back(
          std::move(bwdKey),
          std::make_pair(std::move(bwdPlan), std::move(bwdCost)));

      PlanningCacheImpl::Key wuKey{std::move(wuParams),
                                   std::move(wuOptions),
                                   boost::none,
                                   boost::none,
                                   false,
                                   boost::none,
                                   0};
      additionalPlansToCache->emplace_back(
          std::move(wuKey),
          std::make_pair(std::move(wuPlan), std::move(wuCost)));
    }
    return {fwdPlan, fwdCost};
  }
  return {jointPlan, jointCost};
}

void writePlanConstraintsFile(const Plan &plan, const std::string filePath) {
  boost::property_tree::ptree constraints;
  const auto constrainValues = [&](const std::string &keySuffix,
                                   const std::vector<unsigned> &values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      constraints.add(keySuffix + "." + std::to_string(i), values[i]);
    }
  };

  const auto constrainArray = [&](const std::string &key,
                                  const std::vector<unsigned> &values) {
    using boost::property_tree::ptree;
    ptree array;
    for (const auto value : values) {
      array.push_back(ptree::value_type("", ptree(std::to_string(value))));
    }
    constraints.add_child(key, array);
  };

  // Transforms
  for (std::size_t i = 0; i < plan.transforms.size(); ++i) {
    const std::string keySuffix = std::to_string(i) + ".transform.";
    const ConvTransform &t = plan.transforms[i];
    constraints.add(keySuffix + "swapOperands", t.swapOperands);
    constrainArray(keySuffix + "expandDims", t.expandDims);
    constrainArray(keySuffix + "outChanFlattenDims", t.outChanFlattenDims);
    constraints.add(keySuffix + "combineConvGroups", t.combineConvGroupsFactor);
  }

  // Partitions
  for (std::size_t i = 0; i < plan.partitions.size(); ++i) {
    const std::string keySfx = std::to_string(i) + ".partition.";
    const Partition &p = plan.partitions[i];
    constrainValues(keySfx + "fieldSplit", p.fieldSplit);
    constraints.add(keySfx + "batchSplit", p.batchSplit);
    constraints.add(keySfx + "outChanSplit.serial", p.outChanSplit.serial);
    constraints.add(keySfx + "outChanSplit.parallel", p.outChanSplit.parallel);
    constrainValues(keySfx + "kernelSplit", p.kernelSplit);
    constraints.add(keySfx + "inChanSplit.serial", p.inChanSplit.serial);
    constraints.add(keySfx + "inChanSplit.parallel", p.inChanSplit.parallel);
    constraints.add(keySfx + "convGroupSplit", p.convGroupSplit);
  }

  // Other
  constraints.add("method", plan.method);
  constraints.add("convGroupsPerGroup", plan.convGroupsPerGroup);
  constraints.add("inChansPerGroup", plan.inChansPerGroup);
  constraints.add("partialChansPerGroup", plan.partialChansPerGroup);

  boost::property_tree::write_json(filePath, constraints);
}

std::string getPlanConstraintsOutputFile(const ConvOptions &options) {
  std::string path = options.planConstraintsOutputFilename;
  switch (options.pass) {
  case Pass::INFERENCE_FWD:
  case Pass::TRAINING_FWD:
  case Pass::FC_INFERENCE_FWD:
  case Pass::FC_TRAINING_FWD:
    path += "_FWD";
    break;
  case Pass::TRAINING_BWD:
  case Pass::FC_TRAINING_BWD:
    path += "_BWD";
    break;
  case Pass::TRAINING_WU:
  case Pass::FC_TRAINING_WU:
    path += "_WU";
    break;
  case Pass::NONE:
    break;
  }
  path += ".json";
  return path;
}

// Plan the specified convolution in one of three possible modes:
// cycle cost is the priority
// memory cost is the priority
// optimised for memory, constrained to have cycles cost no worse than some
// multiple of the minimum possible cycle cost.
// Planning a particular training pass (forward / backward / weight update) may
// create plans for the other training passes as a side effect. These plans
// are appended to the end of additionalPlansToCache if it is not null.
static std::pair<Plan, Cost> runPlanner(
    const CanonicalConvParams &ccParams, const ConvOptions &options,
    const poplar::Target &target, const boost::optional<Plan> &referencePlan,
    const boost::optional<Cost> &referenceCost, const bool minimizeForTiles,
    const boost::optional<popsolver::DataType> &cycleLimit,
    unsigned startTileIndicesForVirtualHierarchy,
    PlanningCacheImpl::CycleEstimationImpl *cache,
    std::vector<std::pair<PlanningCacheImpl::Key, std::pair<Plan, Cost>>>
        *additionalPlansToCache) {
  // we first attempt to find the fastest plan that we think will fit, if that
  // fails we replan, but minimising for memory instead. in an effort to fit in
  // memory we will apply an architecturally relevent memory limit to this first
  // plan. to calculate the limit we use a user-configured option called
  // `availableMemoryProportion` to state the proportion of memory which is
  // approximately available for this convolution. if the
  // `availableMemoryProportion` is 0 then we just optimise for memory.

  const auto availableTileMem =
      target.getBytesPerTile() * options.availableMemoryProportion;

  auto objective = [&] {
    if (!availableTileMem) {
      logging::debug("Planning convolution that uses the least amount of "
                     "temporary memory.");
      return PlanningObjective::minimizeTileTempMemory();
    } else {
      logging::debug("Planning convolution with a per-tile memory limit of {} "
                     "bytes across {} tiles.",
                     availableTileMem, options.tilesPerIPU);
      PlanningObjective objective;
      if (referenceCost) {
        logging::debug("  applying a reference cost: {}", *referenceCost);
        if (cycleLimit) {
          logging::warn("Planner was given both a reference cost and a cycle "
                        "limit. Ignoring the cycle limit.");
        }
        objective = PlanningObjective::minimizeCostDiff(minimizeForTiles);
      } else if (cycleLimit) {
        logging::debug("  applying a cycle limit: {}", *cycleLimit);
        objective = PlanningObjective::minimizeTiles();
        objective.setCyclesBound(*cycleLimit);
      } else {
        objective = PlanningObjective::minimizeCycles();
      }
      objective.setTileTempMemoryBound(popsolver::DataType{availableTileMem});
      return objective;
    }
  }();

  Plan plan;
  Cost cost = highestCost;
  const auto &params = ccParams.getParams();
  std::tie(plan, cost) = createPlan(
      params, options, objective, target, startTileIndicesForVirtualHierarchy,
      referencePlan, referenceCost, cache, nullptr);

  if (cost.totalCycles == popsolver::DataType::max()) {
    throw poputil::poplibs_error("No base plan found for unbounded plan");
  }

  logging::debug("Found best plan using {}: {}.", plan.method, cost);
  logging::debug(
      "  for input {}x({}x{}x{}), kernel {}, output = {}x({}x{}x{}), pass={}",
      params.inputFieldShape, params.getBatchSize(), params.getNumConvGroups(),
      params.getNumInputChansPerConvGroup(), params.kernelShape,
      params.getOutputFieldShape(), params.getBatchSize(),
      params.getNumConvGroups(), params.getNumOutputChansPerConvGroup(),
      options.pass);
  logPlanBreakdown(logging::Level::Debug, plan, cost, referenceCost);

  logging::debug("{}", plan);
  logging::trace("for params: {}", params);

  if (!options.planConstraintsOutputFilename.empty()) {
    writePlanConstraintsFile(plan, getPlanConstraintsOutputFile(options));
  }
  return std::make_pair(std::move(plan), std::move(cost));
}

static Plan getFullyConnectedWUPlan(const poplar::Target &target,
                                    const CanonicalConvParams &fwdParams,
                                    const ConvOptions &fwdOptions,
                                    const Plan &fwdPlan) {
  assert(fwdPlan.isJointPlan);
  assert(!fwdPlan.transforms[0].swapOperands);
  auto plan = fwdPlan;
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_WU;
  const auto numPartitions = plan.partitions.size();
  for (unsigned i = 0; i != numPartitions; ++i) {
    plan.partitions[i].inChanSplit = fwdPlan.partitions[i].outChanSplit;
    plan.partitions[i].outChanSplit = fwdPlan.partitions[i].inChanSplit;
    plan.partitions[i].outChanGrainSize = fwdPlan.partitions[i].inChanGrainSize;
    plan.partitions[i].inChanGrainSize = fwdPlan.partitions[i].outChanGrainSize;
  }
  plan.partialChansPerGroup = fwdPlan.inChansPerGroup;
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;

  plan.method = getFullyConnectedWUMethod(fwdParams.getParams(), fwdPlan.method,
                                          fwdPlan.partialChansPerGroup,
                                          fwdPlan.inChansPerGroup);
  // TODO: T12888 Make the forward pass aware that it would be good to use a
  // grouping of 16 if possible.
  plan.inChansPerGroup = fwdPlan.partialChansPerGroup;
  if (plan.method == Plan::Method::AMP &&
      !canUseConvolutionInstruction(
          fwdParams->inputType == poplar::FLOAT,
          fwdOptions.partialsType == poplar::FLOAT, plan.inChansPerGroup,
          plan.numConvUnitsRequired, plan.partialChansPerGroup, target)) {
    plan.inChansPerGroup =
        target.getWeightsPerConvUnit(fwdParams->inputType == poplar::FLOAT);
    plan.partitions.back().inChanGrainSize = plan.inChansPerGroup;
  }

  // If the result type is half and all the reduction is done within a single
  // pass of the AMP unit then there is no reason to use a higher precision
  // partial type.
  if (fwdParams->outputType == poplar::HALF &&
      fwdParams->getNumOutputChansPerConvGroup() == plan.inChansPerGroup &&
      target.getFp16InFp16OutConvUnitsPerTile() ==
          target.getFp16InFp32OutConvUnitsPerTile()) {
    for (auto &x : plan.types) {
      x.partialType = x.resultType = poplar::HALF;
    }
  }

  // Set the partials type to the output type as there are no reductions
  // required
  if (fwdParams->outputType == poplar::HALF &&
      plan.method == Plan::Method::OUTER_PRODUCT) {
    for (auto &x : plan.types) {
      x.partialType = x.resultType = poplar::HALF;
    }
  }
  return plan;
}

static Plan getFullyConnectedBwdPlan(const Plan &fwdPlan) {
  assert(fwdPlan.isJointPlan);
  assert(!fwdPlan.transforms[0].swapOperands);
  auto plan = fwdPlan;
  plan.method = getFullyConnectedBwdMethod(fwdPlan.method);
  plan.linearizeTileOrder = Plan::LinearizeTileOrder::FC_BWD_AS_CONV;
  for (auto &partition : plan.partitions) {
    // Input channel serial split cannot be swapped with Field Splitting as
    // serial Field Splitting is not supported yet.
    std::swap(partition.fieldSplit.back(), partition.inChanSplit.parallel);
    std::swap(partition.fieldAxisGrainSize.back(), partition.inChanGrainSize);
  }
  plan.inChansPerGroup = plan.partitions.back().inChanGrainSize;
  return plan;
}

void preplanConvolutionsImpl(const poplar::Target &target,
                             const std::set<ConvPlanKey> &paramSet,
                             PlanningCache &cache) {
  // convert to a vector for efficient tbb looping
  struct Job {
    const ConvPlanKey *input;
    std::vector<std::pair<PlanningCacheImpl::Key, std::pair<Plan, Cost>>>
        output;
  };
  std::vector<Job> jobs(paramSet.size());

  auto pIt = paramSet.cbegin();
  for (std::size_t i = 0u; i != paramSet.size(); ++i, ++pIt) {
    jobs[i].input = &*pIt;
  }
  // create plans in parallel

  tbb::parallel_for<std::size_t>(0u, paramSet.size(), [&](std::size_t i) {
    const auto &params = jobs[i].input->first;
    const auto &options = jobs[i].input->second;
    Plan plan;
    Cost cost;
    std::tie(plan, cost) = runPlanner(
        params, options, target, boost::none, boost::none, false, boost::none,
        0, &cache.impl->cycleEstimation, &jobs[i].output);
    auto key =
        PlanningCacheImpl::Key(jobs[i].input->first, jobs[i].input->second,
                               boost::none, boost::none, false, boost::none, 0);
    jobs[i].output.emplace_back(
        key, std::make_pair(std::move(plan), std::move(cost)));
  });
  // sequential insert into the cache
  for (std::size_t i = 0u; i != jobs.size(); ++i) {
    for (auto &entry : jobs[i].output) {
      cache.impl->addPlanToCache(std::move(entry.first),
                                 std::move(entry.second));
    }
  }
}

Plan getPlan(const poplar::Target &target, const CanonicalConvParams &params,
             const ConvOptions &options, PlanningCache *cache) {
  if (options.pass == Pass::FC_TRAINING_WU ||
      options.pass == Pass::FC_TRAINING_BWD) {
    auto fwdParams =
        getFullyConnectedPassParams(params, options, Pass::FC_TRAINING_FWD);
    auto fwdOptions =
        getFullyConnectedPassOptions(options, Pass::FC_TRAINING_FWD);
    const auto fwdPlan = getPlan(target, fwdParams, fwdOptions, cache);
    if (fwdPlan.isJointPlan) {
      if (options.pass == Pass::FC_TRAINING_WU) {
        return getFullyConnectedWUPlan(target, fwdParams, fwdOptions, fwdPlan);
      }
      assert(options.pass == Pass::FC_TRAINING_BWD);
      return getFullyConnectedBwdPlan(fwdPlan);
    }
  }

  auto temp = std::make_unique<PlanningCacheImpl>();
  auto &cacheImpl = cache ? cache->impl : temp;
  PlanningCacheImpl::Key key(params, options, boost::none, boost::none, false,
                             boost::none, 0);
  const auto cachedPlan = cacheImpl->getPlan(key);
  if (cachedPlan) {
    return cachedPlan->first;
  }

  std::vector<std::pair<PlanningCacheImpl::Key, std::pair<Plan, Cost>>>
      plansToCache;
  Plan plan;
  Cost cost;
  std::tie(plan, cost) =
      runPlanner(params, options, target, boost::none, boost::none, false,
                 boost::none, 0, &cacheImpl->cycleEstimation, &plansToCache);
  plansToCache.emplace_back(key, std::make_pair(plan, cost));
  for (const auto &entry : plansToCache) {
    cacheImpl->addPlanToCache({entry.first}, {entry.second});
  }
  return plan;
}

namespace {

enum class MultiPlanType { PARALLEL, SERIAL };

struct MultiPlanOptions {
  MultiPlanOptions(const poplar::OptionFlags &options) {
    using poplibs::OptionHandler;
    using poplibs::OptionSpec;

    static const std::map<std::string, MultiPlanType> planTypeMap{
        {"parallel", MultiPlanType::PARALLEL},
        {"serial", MultiPlanType::SERIAL}};

    const OptionSpec spec{
        {"planType", OptionHandler::createWithEnum(planType, planTypeMap)},
        {"perConvReservedTiles",
         OptionHandler::createWithInteger(perConvReservedTiles)},
        {"cycleBackOff", OptionHandler::createWithDouble(cycleBackOff)},
    };

    for (const auto &entry : options) {
      spec.parse(entry.first, entry.second);
    }
  }

  MultiPlanType planType = MultiPlanType::PARALLEL;
  unsigned perConvReservedTiles = 50;
  double cycleBackOff = 0.1;
};

} // unnamed namespace

static ParallelPlan
getParallelMultiPlan(const poplar::Target &target,
                     const std::vector<CanonicalConvParams> &params,
                     std::vector<ConvOptions> convOptions, PlanningCache *cache,
                     const MultiPlanOptions &options) {
  for (const auto &convOption : convOptions) {
    if (convOption.numIPUs != 1) {
      throw poputil::poplibs_error(
          "Multi plan is unsupported for more than 1 IPU");
    }
  }
  auto temp = std::make_unique<PlanningCacheImpl>();
  auto &cacheImpl = cache ? cache->impl : temp;

  const auto cachedRunPlanner =
      [&cacheImpl, &target](CanonicalConvParams params, ConvOptions convOptions,
                            boost::optional<Plan> referencePlan,
                            boost::optional<Cost> referenceCost,
                            bool minimizeForTiles,
                            boost::optional<popsolver::DataType> cycleLimit,
                            unsigned startTileIdxForVirtualHierarchy) {
        PlanningCacheImpl::Key key{std::move(params),
                                   std::move(convOptions),
                                   std::move(referencePlan),
                                   std::move(referenceCost),
                                   minimizeForTiles,
                                   cycleLimit,
                                   startTileIdxForVirtualHierarchy};
        if (auto cachedPlan = cacheImpl->getPlan(key)) {
          return *std::move(cachedPlan);
        } else {
          auto planAndCost =
              runPlanner(key.params, key.options, target, key.referencePlan,
                         key.referenceCost, key.minimizeForTiles,
                         key.cycleLimit, key.startTileIdxForVirtualHierarchy,
                         &cacheImpl->cycleEstimation, nullptr);
          cacheImpl->addPlanToCache(std::move(key), planAndCost);
          return planAndCost;
        }
      };

  // current multi-conv planning algorithm:
  //  1. plan largest first across all tiles, optimising for speed.
  //  2. re-plan with a % cycle backoff from fastest, optimising for tiles used.
  //  3. for the remaining convs from smallest to but not including 2nd largest:
  //      a. remove used tiles from the array
  //      b. plan, optimising for fitting in reference cost and then tiles used.
  //  4. for final conv plan, optimising to fit in reference but not limit tiles
  std::vector<Plan> plans;
  plans.resize(params.size());

  // indices into params, sorted in size order, smallest conv (by FLOPs)
  // to largest.
  auto idx = [&] {
    std::vector<std::size_t> idx(params.size());
    std::iota(std::begin(idx), std::end(idx), 0);

    std::vector<std::uint64_t> flops(idx.size());
    std::transform(std::begin(idx), std::end(idx), std::begin(flops),
                   [&](const auto i) { return getFwdFlops(*params[i]); });

    std::sort(std::begin(idx), std::end(idx),
              [&flops](const auto &lhs, const auto &rhs) {
                return flops[lhs] < flops[rhs];
              });
    return idx;
  }();

  logging::debug("multi-conv convolutions, smallest to largest: {}", idx);

  // The starting tile for the hierarchy is the same currently across every IPU
  unsigned startTileIdxForVirtualHierarchy = 0;

  // make sure each remaining conv gets at least N tiles.
  unsigned perConvReservedTiles = options.perConvReservedTiles;
  if (target.getNumTiles() < idx.size() * perConvReservedTiles) {
    logging::warn("Not enough tiles to reserve any for the multi-convolution.");
    perConvReservedTiles = std::max(1ul, target.getNumTiles() / idx.size());
  }

  // don't include first conv.
  unsigned reservedTiles = (idx.size() - 1) * perConvReservedTiles;

  // scale the cycle back off from the main conv based on how many other convs
  // need to share the remaining tiles.
  double cycleBackOff = 1 + (idx.size() - 1) * options.cycleBackOff;

  auto reference = [&] {
    const auto largestPlanIdx = idx.back();

    // step 1
    assert(convOptions[largestPlanIdx].tilesPerIPU >= reservedTiles);
    convOptions[largestPlanIdx].tilesPerIPU -= reservedTiles;
    if (convOptions[largestPlanIdx].tilesPerIPU == 0) {
      throw poputil::poplibs_error("Not enough tiles for multi-conv");
    }

    logging::debug("Planning largest convolution, optimising for speed");
    auto planAndCost = cachedRunPlanner(
        params[largestPlanIdx], convOptions[largestPlanIdx], boost::none,
        boost::none, false, boost::none, startTileIdxForVirtualHierarchy);

    // step 2
    logging::debug("Re-planning largest convolution, optimising for tiles");
    const auto cycleLimit =
        planAndCost.second.totalCycles.getAs<double>() * cycleBackOff;
    const popsolver::DataType integerCycleLimit{cycleLimit};
    planAndCost = cachedRunPlanner(
        params[largestPlanIdx], convOptions[largestPlanIdx], boost::none,
        boost::none, true, integerCycleLimit, startTileIdxForVirtualHierarchy);
    plans[largestPlanIdx] = planAndCost.first;

    startTileIdxForVirtualHierarchy += roundUp(
        *planAndCost.second.totalTiles, target.getTilesPerSharedExchangeBus());
    reservedTiles -= perConvReservedTiles;

    return planAndCost;
  }();

  if (idx.size() > 1) {
    // step 3
    for (std::size_t i = 0; i < idx.size() - 2; ++i) {
      const auto thisIdx = idx[i];

      // 3a.
      assert(target.getTilesPerIPU() >= reservedTiles);
      assert(target.getTilesPerIPU() - reservedTiles >=
             startTileIdxForVirtualHierarchy);
      convOptions[thisIdx].tilesPerIPU = target.getTilesPerIPU() -
                                         startTileIdxForVirtualHierarchy -
                                         reservedTiles;

      logging::debug("Planning convolution {} across {} tiles, optimising for "
                     "per-step cycle difference and then tiles used",
                     thisIdx, convOptions[thisIdx].tilesPerIPU);
      if (convOptions[thisIdx].tilesPerIPU == 0) {
        throw poputil::poplibs_error("Not enough tiles for multi-conv");
      }

      // 3b.
      auto planAndCost = cachedRunPlanner(
          params[thisIdx], convOptions[thisIdx], reference.first,
          reference.second, true, boost::none, startTileIdxForVirtualHierarchy);
      plans[thisIdx] = planAndCost.first;

      assert(reservedTiles >= perConvReservedTiles);
      reservedTiles -= perConvReservedTiles;
      startTileIdxForVirtualHierarchy +=
          roundUp(*planAndCost.second.totalTiles,
                  target.getTilesPerSharedExchangeBus());

      // if we weren't able to stay within the reference update it to record
      // where this conv has extended the limits.
      reference.second = maxPerStepCycles(reference.second, planAndCost.second);
    }

    // step 4
    const auto penultimateIdx = idx[idx.size() - 2];

    assert(reservedTiles == 0);
    assert(target.getTilesPerIPU() >= startTileIdxForVirtualHierarchy);
    convOptions[penultimateIdx].tilesPerIPU =
        target.getTilesPerIPU() - startTileIdxForVirtualHierarchy;

    logging::debug(
        "Planning final convolution on the remaining {} tiles, optimising for "
        "per-step cycle difference and then temporary memory used",
        convOptions[penultimateIdx].tilesPerIPU);
    if (convOptions[penultimateIdx].tilesPerIPU == 0) {
      throw poputil::poplibs_error("Not enough tiles for multi-conv");
    }

    auto planAndCost = cachedRunPlanner(
        params[penultimateIdx], convOptions[penultimateIdx], reference.first,
        reference.second, false, boost::none, startTileIdxForVirtualHierarchy);
    plans[penultimateIdx] = planAndCost.first;
  }

  return {std::move(plans)};
}

static SerialPlan
getSerialMultiPlan(const poplar::Target &target,
                   const std::vector<CanonicalConvParams> &params,
                   const std::vector<ConvOptions> &options,
                   PlanningCache *cache) {
  const auto totalPlans = params.size();

  std::vector<Plan> plans;
  plans.reserve(totalPlans);
  for (std::size_t i = 0; i < totalPlans; i++) {
    plans.push_back(getPlan(target, params[i], options[i], cache));
  }
  return {std::move(plans)};
}

MultiPlan getMultiPlan(const poplar::Target &target,
                       const std::vector<CanonicalConvParams> &params,
                       const std::vector<ConvOptions> &convOptions,
                       PlanningCache *cache,
                       const poplar::OptionFlags &options_) {
  assert(params.size() == convOptions.size());
  MultiPlanOptions options(options_);

  if (options.planType == MultiPlanType::PARALLEL) {
    try {
      return getParallelMultiPlan(target, params, convOptions, cache, options);
    } catch (poputil::poplibs_error) {
      logging::warn("Failed to find a parallel multiplan, falling back to "
                    "serial planning");
      return getSerialMultiPlan(target, params, convOptions, cache);
    }
  } else {
    assert(options.planType == MultiPlanType::SERIAL);
    return getSerialMultiPlan(target, params, convOptions, cache);
  }
}

template <typename T>
static void constrainVariable(popsolver::Model &m, popsolver::Variable v,
                              T value) {
  m.equal(v, popsolver::DataType{value});
}

template <typename T>
static void constrainVariable(popsolver::Model &m, Split<popsolver::Variable> v,
                              Split<T> value) {
  constrainVariable(m, v.parallel, popsolver::DataType{value.parallel});
  constrainVariable(m, v.serial, popsolver::DataType{value.serial});
}

static void constrainPartitionVars(popsolver::Model &m,
                                   const PartitionVariables &vars,
                                   const Partition &partition) {
  const auto numFieldDims = vars.fieldSplit.size();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    constrainVariable(m, vars.fieldSplit[dim], partition.fieldSplit[dim]);
    constrainVariable(m, vars.kernelSplit[dim], partition.kernelSplit[dim]);
  }
  constrainVariable(m, vars.batchSplit, partition.batchSplit);
  constrainVariable(m, vars.outChanSplit, partition.outChanSplit);
  constrainVariable(m, vars.inChanSplit, partition.inChanSplit);
  constrainVariable(m, vars.convGroupSplit, partition.convGroupSplit);
}

/// Estimate the cost of a convolution. This is not used by poplibs/enigma.
std::pair<std::uint64_t, std::uint64_t>
estimateConvCost(const poplar::Target &target, const ConvParams &params,
                 const ConvOptions &options, PlanningCache *cache,
                 const Plan &plan) {
  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  std::unique_ptr<PlanningCacheImpl> tempCache;
  if (!cache) {
    tempCache = std::unique_ptr<PlanningCacheImpl>(new PlanningCacheImpl);
    cacheImpl = tempCache.get();
  }
  const auto perLevelExchangeBytesPerCycle =
      poplibs::getPerLevelExchangeBytesPerCycle(target, options.numIPUs);
  const auto hierarchy =
      poplibs::getTileHierarchy(options.numIPUs, options.tilesPerIPU);
  assert(perLevelExchangeBytesPerCycle.size() == plan.partitions.size());
  auto objective = PlanningObjective::minimizeCycles();
  ConvVertexType convVertexType(
      plan.method, params.inputType, plan.types.back().partialType,
      plan.convGroupsPerGroup, plan.inChansPerGroup, plan.partialChansPerGroup,
      plan.slicWindowWidth, plan.numConvUnitsRequired, plan.useLimitedVersion);
  const auto fieldGrainSize = plan.partitions.back().fieldAxisGrainSize;
  // Check grain size is the same at each level.
#ifndef NDEBUG
  for (const auto &p : plan.partitions) {
    assert(p.fieldAxisGrainSize == fieldGrainSize);
  }
#endif
  popsolver::Model m;
  std::vector<PartitionVariables> partitionVars;
  const auto e = constructModel(
      target, plan.transforms, plan.types, hierarchy,
      perLevelExchangeBytesPerCycle, fieldGrainSize, convVertexType, params,
      plan.isJointPlan, highestCost, objective, boost::none, boost::none,
      &cacheImpl->cycleEstimation, options, m, partitionVars);
  const auto numLevelsOfHierarchy = plan.partitions.size();
  assert(partitionVars.size() == numLevelsOfHierarchy);
  for (unsigned level = 0; level != numLevelsOfHierarchy; ++level) {
    constrainPartitionVars(m, partitionVars[level], plan.partitions[level]);
  }
  popsolver::Solution s;
  s = m.minimize(e.totalCycles);
  if (!s.validSolution()) {
    return {*highestCost.totalCycles, *highestCost.totalTempBytes};
  }
  return {*s[e.totalCycles], *s[e.totalTempBytes]};
}

} // namespace poplin
