// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_ConvPlanTypes_hpp
#define poplin_ConvPlanTypes_hpp

#include "CanonicalConvParams.hpp"
#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include <popsolver/Model.hpp>
#include <vector>

namespace poplin {

// constraint variables that represent how each item is split for a particular
// level in the hierarchy.
struct PartitionVariables {
  // indexed by field dimension.
  std::vector<popsolver::Variable> fieldSplit;
  popsolver::Variable batchSplit;
  Split<popsolver::Variable> outChanSplit;
  // indexed by kernel dimension.
  std::vector<popsolver::Variable> kernelSplit;
  Split<popsolver::Variable> inChanSplit;
  popsolver::Variable convGroupSplit;
  std::vector<unsigned> fieldGrainSize;

  unsigned convGroupGrainSize;
  unsigned inChanGrainSize;
  unsigned outChanGrainSize;
};

// constraint variables that specify the grain sizes of each dimension.
struct ConvSizeVariables {
  // indexed by field dimension.
  std::vector<popsolver::Variable> numFieldGrains;
  popsolver::Variable batchSize;
  // indexed by kernel dimension.
  std::vector<popsolver::Variable> kernelSize;

  popsolver::Variable numConvGroupGrains;
  popsolver::Variable numInChanGrains;
  popsolver::Variable numOutChanGrains;
};

// a description of a (sub-)convolution at a particular level in the hierarchy.
template <typename T> struct ConvSize {
  T convGroupSize;
  T batchSize;
  std::vector<T> fieldSize;
  std::vector<T> kernelSize;
  T inChanSize;
  T outChanSize;
};

struct ConvVertexType {
  Plan::Method method;
  poplar::Type inputType;
  poplar::Type partialType;

  unsigned convGroupsPerGroup;
  unsigned inChansPerGroup;
  unsigned partialChansPerGroup;

  // TODO: these variables are only valid for certain methods, it might be
  // better to use a variant here instead.
  //
  // The width of the kernel that slides over the input. Only 4 is currently
  // supported in the software but the SLIC engine also supports 3.
  unsigned slicWindowWidth;
  // Number of engines enabled. Allowed options: 4 or 8
  unsigned numConvUnitsRequired;

  // If TRUE convolution library will use unsigned short type for vertex
  // states, otherwise will fallback into unsigned type
  bool useLimitedVersion;

  ConvVertexType(Plan::Method method, poplar::Type inputType,
                 poplar::Type partialType, unsigned convGroupsPerGroup,
                 unsigned inChansPerGroup, unsigned partialChansPerGroup,
                 unsigned slicWindowWidth, unsigned numConvUnitsRequired,
                 bool useLimitedVersion)
      : method(method), inputType(inputType), partialType(partialType),
        convGroupsPerGroup(convGroupsPerGroup),
        inChansPerGroup(inChansPerGroup),
        partialChansPerGroup(partialChansPerGroup),
        slicWindowWidth(slicWindowWidth),
        numConvUnitsRequired(numConvUnitsRequired),
        useLimitedVersion(useLimitedVersion) {}
};

template <typename T> struct ExchangeEstimates {
  T inputExchangeCycles;
  T weightExchangeCycles;
  T reduceFirstStageExchangeCycles;
  T reduceRemainingStagesExchangeCycles;
};

template <typename T>
inline bool operator<(const ExchangeEstimates<T> &a,
                      const ExchangeEstimates<T> &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &ExchangeEstimates<T>::inputExchangeCycles,
      &ExchangeEstimates<T>::weightExchangeCycles,
      &ExchangeEstimates<T>::reduceFirstStageExchangeCycles,
      &ExchangeEstimates<T>::reduceRemainingStagesExchangeCycles);

  return helper.lt(a, b);
}

template <typename T> struct Estimates {
  Estimates() = default;
  Estimates(const T totalTiles, const T totalCycles, const T totalTempBytes,
            const T totalPerStepCycleDiff)
      : totalTiles(totalTiles), totalCycles(totalCycles),
        totalTempBytes(totalTempBytes),
        totalPerStepCycleDiff(totalPerStepCycleDiff) {}

  // the four values we support minimizing on.
  T totalTiles;
  T totalCycles;
  T totalTempBytes;
  T totalPerStepCycleDiff;

  // break-down of the above totals
  T rearrangeBeforeSliceCycles;
  T memsetZeroBeforeAddInPlace;
  T dynamicSliceCycles;
  T transformCycles;

  T totalExchangeCycles;
  ExchangeEstimates<T> itemisedExchangeCycles;

  T tileLevelTransformCycles;
  T partialCalcCycles;
  T reduceCycles;
  T dynamicUpdateCycles;
  T addInPlaceCycles;
  T castCycles;

  T rearrangeBeforeSliceTempBytes;
  T rearrangeBeforeSliceTempDuringRearrangeBytes;
  T transformTempBytes;
  T tileLevelTransformTempBytes;
  T convTempBytes;
  T reduceTempBytes;
  T addInPlaceTempBytes;
};

using Cost = Estimates<popsolver::DataType>;

static Cost highestCost(popsolver::DataType::max(), popsolver::DataType::max(),
                        popsolver::DataType::max(), popsolver::DataType::max());

inline bool operator==(const Cost &a, const Cost &b) {
  return a.totalTiles == b.totalTiles && a.totalCycles == b.totalCycles &&
         a.totalTempBytes == b.totalTempBytes &&
         a.totalPerStepCycleDiff == b.totalPerStepCycleDiff;
}

inline bool operator!=(const Cost &a, const Cost &b) { return !(a == b); }

inline bool operator<(const Cost &a, const Cost &b) {
  constexpr static auto helper = poplibs_support::makeStructHelper(
      &Cost::totalTiles, &Cost::totalCycles, &Cost::totalTempBytes,
      &Cost::totalPerStepCycleDiff,

      &Cost::rearrangeBeforeSliceCycles, &Cost::memsetZeroBeforeAddInPlace,
      &Cost::dynamicSliceCycles, &Cost::transformCycles,

      &Cost::totalExchangeCycles, &Cost::itemisedExchangeCycles,

      &Cost::tileLevelTransformCycles, &Cost::partialCalcCycles,
      &Cost::reduceCycles, &Cost::dynamicUpdateCycles, &Cost::addInPlaceCycles,
      &Cost::castCycles,

      &Cost::rearrangeBeforeSliceTempBytes,
      &Cost::rearrangeBeforeSliceTempDuringRearrangeBytes,
      &Cost::transformTempBytes, &Cost::tileLevelTransformTempBytes,
      &Cost::convTempBytes, &Cost::reduceTempBytes, &Cost::addInPlaceTempBytes);

  return helper.lt(a, b);
}

// performs a max on the itemised cycle counts only.
inline Cost maxPerStepCycles(Cost a, const Cost &b) {
  a.rearrangeBeforeSliceCycles =
      std::max(a.rearrangeBeforeSliceCycles, b.rearrangeBeforeSliceCycles);
  a.memsetZeroBeforeAddInPlace =
      std::max(a.memsetZeroBeforeAddInPlace, b.memsetZeroBeforeAddInPlace);
  a.dynamicSliceCycles = std::max(a.dynamicSliceCycles, b.dynamicSliceCycles);
  a.transformCycles = std::max(a.transformCycles, b.transformCycles);

  // the MINIMIZE_COST_DIFF method currently using the totalExchangeCycles, if
  // that changes we would need to update this too.
  a.totalExchangeCycles =
      std::max(a.totalExchangeCycles, b.totalExchangeCycles);

  a.tileLevelTransformCycles =
      std::max(a.tileLevelTransformCycles, b.tileLevelTransformCycles);
  a.partialCalcCycles = std::max(a.partialCalcCycles, b.partialCalcCycles);
  a.reduceCycles = std::max(a.reduceCycles, b.reduceCycles);
  a.dynamicUpdateCycles =
      std::max(a.dynamicUpdateCycles, b.dynamicUpdateCycles);
  a.addInPlaceCycles = std::max(a.addInPlaceCycles, b.addInPlaceCycles);
  a.castCycles = std::max(a.castCycles, b.castCycles);

  return a;
}

inline std::ostream &operator<<(std::ostream &os, const Cost &c) {
  os << "Cost{cycles=" << c.totalCycles << ", memory=" << c.totalTempBytes;
  if (c.totalPerStepCycleDiff != popsolver::DataType::max()) {
    os << ", diff=" << c.totalPerStepCycleDiff;
  }
  os << ", tiles=" << c.totalTiles << "}";
  return os;
}

struct ConvDescription {
  // TODO pass only ConvDescriptions into the planner as the only source of
  // information to use, this will make sure the cache and planner are in
  // lockstep and we don't introduce more information accidently outside the
  // cache, e.g. target
  // TODO: derive information from target and include in the key.
  // Currently it's assumed to always have the same target universally.
  CanonicalConvParams params;
  ConvOptions options;
  boost::optional<Plan> referencePlan;
  boost::optional<Cost> referenceCost;
  bool minimizeForTiles;
  boost::optional<popsolver::DataType> cycleLimit;
  unsigned startTileIdxForVirtualHierarchy;

  ConvDescription(CanonicalConvParams params, ConvOptions options,
                  boost::optional<Plan> referencePlan,
                  boost::optional<Cost> referenceCost, bool minimizeForTiles,
                  boost::optional<popsolver::DataType> cycleLimit,
                  unsigned startTileIdxForVirtualHierarchy)
      : params{std::move(params)},
        options({std::move(options)}), referencePlan{std::move(referencePlan)},
        referenceCost{std::move(referenceCost)},
        minimizeForTiles{minimizeForTiles}, cycleLimit{std::move(cycleLimit)},
        startTileIdxForVirtualHierarchy{startTileIdxForVirtualHierarchy} {}

  ConvDescription(const ConvDescription &) = default;
  ConvDescription(ConvDescription &&) = default;

  bool operator<(const ConvDescription &other) const {
    constexpr static auto helper = poplibs_support::makeStructHelper(
        &ConvDescription::params, &ConvDescription::options,
        &ConvDescription::referenceCost, &ConvDescription::referencePlan,
        &ConvDescription::minimizeForTiles, &ConvDescription::cycleLimit,
        &ConvDescription::startTileIdxForVirtualHierarchy);

    return helper.lt(*this, other);
  }
};

} // namespace poplin

#endif // poplin_ConvPlanTypes_hpp
