// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#include "ConvVertexType.hpp"

#include "ConvModel.hpp"
#include "ConvOptions.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/Visitor.hpp"
#include "poplibs_support/logging.hpp"
#include "poplin/ConvParams.hpp"
#include "poputil/exceptions.hpp"

#include <gccs/popsolver/Model.hpp>

#include <gccs/Algorithm.hpp>

#include <vector>

using namespace poplibs_support;
namespace popsolver = gccs::popsolver;

namespace poplin {

static bool canUseConvolutionInstruction(const poplar::Type &actsType,
                                         const poplar::Type &partialsType,
                                         const poplar::Target &target) {
  // We expect partials type size to be larger or equal the activations type.
  if (target.getTypeSize(actsType) > target.getTypeSize(partialsType)) {
    return false;
  }

  // Half is the only supported partials type for quarter activations.
  if (actsType == poplar::QUARTER && partialsType != poplar::HALF) {
    return false;
  }

  return true;
}

bool canUseConvolutionInstruction(const poplar::Type &actsType,
                                  const poplar::Type &partialsType,
                                  unsigned inChansPerGroup,
                                  unsigned numConvUnitsRequired,
                                  unsigned convInputLoadElems,
                                  unsigned outChansPerGroup,
                                  const poplar::Target &target) {
  if (!canUseConvolutionInstruction(actsType, partialsType, target)) {
    return false;
  }
  const unsigned convChainLength =
      target.getConvUnitMaxPipelineDepth(partialsType);
  const unsigned usedWeightsPerConvUnit = convChainLength * convInputLoadElems;
  if (usedWeightsPerConvUnit % inChansPerGroup != 0) {
    return false;
  }
  if (actsType == poplar::QUARTER)
    if (outChansPerGroup != numConvUnitsRequired ||
        usedWeightsPerConvUnit != inChansPerGroup) {
      return false;
    }
  // Output channels grouping shall be great or equal to number of engines
  if ((outChansPerGroup % numConvUnitsRequired) != 0) {
    return false;
  }
  // Check we can use aligned loads (i.e. that the input channel grouping is
  // a multiple of the loaded input channels per-cycle).
  if (inChansPerGroup % convInputLoadElems != 0) {
    return false;
  }
  return true;
}

static void getConvVertexHMACCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {

  const auto vertexInputType =
      inputType == poplar::QUARTER ? poplar::HALF : inputType;

  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<popsolver::DataType>("convGroupsPerGroup");
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");
  const auto constrainedUseLimitedVersion =
      planConstraints.get_optional<bool>("method.useLimitedVersion");

  auto numConvUnits = target.getNumConvUnits(vertexInputType, partialType);
  Plan::Hmac method{};

  // For the test purposes constrain vertex to use unsigned type for
  // vertex states
  if (constrainedUseLimitedVersion) {
    method.useLimitedVersion = *constrainedUseLimitedVersion;
  }

  // Constrain the input channel grouping to a multiple of two if the activation
  // type is half. This ensures that we never need to apply padding when sending
  // activations over the exchange.
  auto grainSize = inputType == poplar::FLOAT ? 1u : 2u;
  const auto roundedNumInChans =
      gccs::alignNext(params.getNumInputChansPerConvGroup(), grainSize);

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
  // HMAC codelet for half partials processes 2 partials inside inner loop
  // to have most optimal load/store pipeline
  if (partialType != poplar::FLOAT) {
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
    if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
      // The input channels in the forward pass become the output channels of
      // the weight update pass. Make sure it is a multiple of the supported
      // output channels per group.
      if (inChansPerGroup != 1 && inChansPerGroup % numConvUnits != 0)
        continue;
    }

    // The HMAC vertex does not require a grouping of the conv groups.
    const unsigned convGroupsPerGroup = 1;

    candidates.emplace_back(method, vertexInputType, partialType,
                            convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup);
    previousInChanGroups = inChanGroups;
  }
}

static void getConvVertexVMACCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {

  const auto vertexInputType =
      inputType == poplar::QUARTER ? poplar::HALF : inputType;
  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<popsolver::DataType>("convGroupsPerGroup");

  // Assembly version only available for half activations and float partials
  if (vertexInputType == poplar::FLOAT) {
    return;
  }

  // Special exception for CPU target, where vector width is identified
  // differently for half types but our vertices assume half is 2 bytes
  // and vector width is 64-bits.
  if (vertexInputType != poplar::FLOAT &&
      target.getTypeSize(vertexInputType) != 2) {
    return;
  }

  // Every execution of the VMAC inner loop vertex processes a single input
  // channel.
  unsigned inChansPerGroup = 1;
  unsigned partialChansPerGroup = 1;
  auto vectorWidth = target.getVectorWidth(vertexInputType);
  const unsigned actsPer64Bits = vertexInputType == poplar::FLOAT ? 2u : 4u;
  std::vector<unsigned> convGroupsPerGroupCandidates = {vectorWidth};
  while (vectorWidth > actsPer64Bits) {
    vectorWidth >>= 1;
    convGroupsPerGroupCandidates.push_back(vectorWidth);
  }

  // Explicitly add groupings of 8 and 16
  if (partialType == poplar::HALF) {
    convGroupsPerGroupCandidates.push_back(8);
    convGroupsPerGroupCandidates.push_back(16);
  }

  for (auto convGroupsPerGroup : convGroupsPerGroupCandidates) {
    if (constrainedConvGroupsPerGroup &&
        *constrainedConvGroupsPerGroup !=
            popsolver::DataType{convGroupsPerGroup}) {
      continue;
    }

    if (options.experimentalSlicVmac16 && convGroupsPerGroup != 16) {
      continue;
    }

    candidates.emplace_back(Plan::Vmac{}, vertexInputType, partialType,
                            convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup);
  }
}

static void getConvVertexAMPCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &partialType_, const ConvOptions &options,
    bool isJointPlan, std::vector<ConvVertexType> &candidates) {
  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");
  const auto constrainedNumConvUnits =
      planConstraints.get_optional<popsolver::DataType>("method.convUnits");
  const auto constrainedConvInputLoadElems =
      planConstraints.get_optional<popsolver::DataType>(
          "method.convInputLoadElems");

  auto partialType = partialType_;
  if (inputType == poplar::FLOAT && partialType != poplar::FLOAT) {
    partialType = poplar::FLOAT;
  }
  const auto vertexInputType =
      (inputType == poplar::QUARTER &&
       target.getNumConvUnits(inputType, partialType) == 0)
          ? poplar::HALF
          : inputType;

  auto numConvUnitsOnIpu = target.getNumConvUnits(vertexInputType, partialType);
  if (canUseConvolutionInstruction(vertexInputType, partialType, target)) {
    const auto weightsPerConvUnit =
        target.getWeightsPerConvUnit(vertexInputType);

    const auto convChainLength =
        target.getConvUnitMaxPipelineDepth(partialType);
    const auto maxConvUnitInputLoadElems =
        target.getConvUnitInputLoadElemsPerCycle(vertexInputType);

    std::vector<unsigned> partialChansCandidates = {numConvUnitsOnIpu,
                                                    weightsPerConvUnit};
    std::vector<unsigned> numConvUnitsCandidates = {numConvUnitsOnIpu};
    std::vector<unsigned> convInputLoadElemsCandidates = {
        maxConvUnitInputLoadElems};

    // Ensure we have a 2/4/8 elem candidate for float/half/quarter
    // convInputLoadElems if supported by the target.
    {
      const unsigned smallestWidthLoad = vertexInputType == poplar::FLOAT  ? 2
                                         : vertexInputType == poplar::HALF ? 4
                                         : vertexInputType == poplar::QUARTER
                                             ? 8
                                             : 0;
      // Not expected to be possible for a user to trigger the condition in
      // this assert (vertexInputType is not FLOAT, HALF, or QUARTER) so this
      // is an assert rather than an exception.
      assert(smallestWidthLoad != 0);

      if (maxConvUnitInputLoadElems % smallestWidthLoad &&
          convInputLoadElemsCandidates.back() != smallestWidthLoad) {
        convInputLoadElemsCandidates.push_back(smallestWidthLoad);
      }
    }

    // On IPU2 with half inputs we need to enable 8 engines config as well
    if (vertexInputType != poplar::QUARTER && numConvUnitsOnIpu > 8) {
      numConvUnitsCandidates.push_back(8);
    }

    for (const auto convInputLoadElems : convInputLoadElemsCandidates) {
      if (constrainedConvInputLoadElems &&
          popsolver::DataType{convInputLoadElems} !=
              *constrainedConvInputLoadElems) {
        continue;
      }
      const auto weightsPerConvUnit = convChainLength * convInputLoadElems;
      std::vector<unsigned> partialChansCandidates = {numConvUnitsOnIpu};
      if (weightsPerConvUnit > numConvUnitsOnIpu) {
        partialChansCandidates.push_back(weightsPerConvUnit);
      }

      // On IPU2 we need to enable 8 engines config as well
      if (numConvUnitsOnIpu > 8) {
        partialChansCandidates.push_back(8);
      }

      for (const auto convUnits : numConvUnitsCandidates) {
        // Number of conv units constrain
        if (constrainedNumConvUnits &&
            popsolver::DataType{convUnits} != *constrainedNumConvUnits) {
          continue;
        }
        for (unsigned inputs = weightsPerConvUnit; inputs >= 1; inputs--) {
          // Input channels constraint
          if (constrainedInChansPerGroup &&
              popsolver::DataType{inputs} != *constrainedInChansPerGroup) {
            continue;
          }
          for (const auto partials : partialChansCandidates) {
            // Partial channels constrain
            if (constrainedPartialChansPerGroup &&
                popsolver::DataType{partials} !=
                    *constrainedPartialChansPerGroup) {
              continue;
            }

            if (partials != convUnits && partials != weightsPerConvUnit) {
              continue;
            }

            if (!canUseConvolutionInstruction(
                    vertexInputType, partialType, inputs, convUnits,
                    convInputLoadElems, partials, target)) {
              continue;
            }

            // There are two reasons we might choose to make
            // partialChansPerGroup not equal to numConvUnitsOnIpu:
            // - The output of a convolution is likely to be fed into another
            //   convolution that wants its input grouped by weightsPerConvUnit
            //   so there will be a small cost (estimated by the planner) if
            //   partialChansPerGroup != weightsPerConvUnit
            // - The output channel grouping of a fully connected forward pass
            //   becomes the input channel grouping of the fully connected
            //   weight update pass and so if partialChansPerGroup !=
            //   weightsPerConvUnit we can't fully utilize AMP in the weight
            //   update pass.
            // Neither of these reasons apply to fully connected inference (we
            // must always rearrange the output regardless of the grouping and
            // there is no weight update pass).
            if (options.pass == Pass::FC_INFERENCE_FWD &&
                partials != convUnits) {
              continue;
            }

            if (isJointPlan && options.pass == Pass::FC_TRAINING_FWD) {
              // The input channels in the forward pass become the output
              // channels of the weight update pass. Make sure it is a multiple
              // of the supported output channels per group.
              if (inputs % convUnits != 0) {
                continue;
              }
            }

            // AMP only supports a conv group grouping of 1.
            const unsigned convGroupsPerGroup = 1;

            Plan::Amp method{};
            method.convUnits = convUnits;
            method.convInputLoadElems = convInputLoadElems;
            candidates.emplace_back(method, vertexInputType, partialType,
                                    convGroupsPerGroup, inputs, partials);
          }
        }
      }
    }
  }
}

static void getConvVertexAMPCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  auto weightsPerConvUnit = target.getWeightsPerConvUnit(inputType);
  const auto isAll = [](const auto k, const auto &c) {
    return std::all_of(std::begin(c), std::end(c),
                       [k](const auto x) { return x == k; });
  };
  // The vertex output type can be smaller than the partial type if no reduction
  // is required afterwards.
  if (target.getTypeSize(outputType) < target.getTypeSize(partialType) &&
      params.inputChannelsPerConvGroup <= weightsPerConvUnit &&
      isAll(1U, params.getKernelShape())) {
    auto numCandidatesBefore = candidates.size();
    getConvVertexAMPCandidates(target, inputType, outputType, options,
                               isJointPlan, candidates);
    candidates.erase(
        std::remove_if(candidates.begin() + numCandidatesBefore,
                       candidates.end(),
                       [&](const ConvVertexType &type) {
                         return gccs::alignNext(
                                    params.inputChannelsPerConvGroup,
                                    type.inChansPerGroup) != weightsPerConvUnit;
                       }),
        candidates.end());
  }
  getConvVertexAMPCandidates(target, inputType, partialType, options,
                             isJointPlan, candidates);
}

static void getConvVertexSLICCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType_,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {

  if (inputType != poplar::HALF && inputType != poplar::QUARTER) {
    return;
  }

  const auto vertexInputType =
      (inputType == poplar::QUARTER &&
       target.getNumConvUnits(inputType, partialType_) == 0)
          ? poplar::HALF
          : inputType;

  const auto &planConstraints = options.planConstraints;
  const auto constrainedConvGroupsPerGroup =
      planConstraints.get_optional<popsolver::DataType>("convGroupsPerGroup");
  const auto constrainedSlicWindowWidth =
      planConstraints.get_optional<popsolver::DataType>("method.windowWidth");

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
  auto partialType = partialType_;
  if (vertexInputType == poplar::FLOAT && partialType != poplar::FLOAT) {
    partialType = poplar::FLOAT;
  }
  auto numConvUnits = target.getNumConvUnits(vertexInputType, partialType);
  // List the number of conv chains used in the candidate vertices which are
  // available - either on this hardware or implemented at present
  std::vector<unsigned> convChainsCandidates;
  if (partialType == poplar::FLOAT) {
    convChainsCandidates.push_back(2);
  } else {
    if (numConvUnits == 16) {
      convChainsCandidates.push_back(4);
    }
    // This is always available with 8, or 16 conv units - let cycle estimates
    // reject it in favour of the 16 conv unit version if that's available
    if (vertexInputType == poplar::HALF) {
      convChainsCandidates.push_back(2);
    }
  }

  const unsigned convChainLength =
      target.getConvUnitMaxPipelineDepth(partialType);
  const unsigned maxConvInputLoadElems =
      target.getConvUnitInputLoadElemsPerCycle(vertexInputType);

  // the numbers below are hardcoded but dependent on the expected machine
  // model that the real hardware models. ie. we expect 16 weights per conv unit
  if (convChainLength != 4) {
    throw poputil::poplibs_error(
        "Unsupported conv unit chain length for the SLIC instruction: " +
        std::to_string(convChainLength));
  }

  const unsigned minRequiredConvInputLoadElems =
      vertexInputType == poplar::FLOAT  ? 2
      : vertexInputType == poplar::HALF ? 4
      : vertexInputType == poplar::QUARTER
          ? 8
          : std::numeric_limits<unsigned>::max();
  if (maxConvInputLoadElems < minRequiredConvInputLoadElems) {
    throw poputil::poplibs_error(
        "Unsupported conv unit input load width for SLIC instruction: " +
        std::to_string(maxConvInputLoadElems));
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

  std::vector<Candidate> groupings;
  if (vertexInputType == poplar::QUARTER) {
    groupings.emplace_back(Candidate{1u, 8u});
  } else {
    assert(vertexInputType == poplar::HALF);
    groupings = {Candidate{1u, 4u}, Candidate{2u, 2u}, Candidate{4u, 1u}};
  }
  if (vertexInputType != poplar::QUARTER && partialType != poplar::FLOAT &&
      numConvUnits == 16) {
    groupings.emplace_back(Candidate{8u, 1u});
    groupings.emplace_back(Candidate{16u, 1u});
  }

  for (const auto convChains : convChainsCandidates) {
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
      if (options.experimentalSlicVmac16 && grouping.groups != 16) {
        continue;
      }
      Plan::Slic method{};
      method.windowWidth = slicWindowWidth;
      method.convUnitChainsRequired = convChains;
      candidates.emplace_back(method, vertexInputType, partialType,
                              grouping.groups, grouping.channels,
                              grouping.channels);
    }
  }
}

static void getConvVertexOuterProductCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {

  if (inputType == poplar::QUARTER) {
    return;
  }

  const auto &planConstraints = options.planConstraints;
  const auto constrainedInChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("inChansPerGroup");
  const auto constrainedPartialChansPerGroup =
      planConstraints.get_optional<popsolver::DataType>("partialChansPerGroup");

  const auto inChansPerGroup = 1u;
  // Only one supported inChansPerGroup for this method.
  if (constrainedInChansPerGroup &&
      *constrainedInChansPerGroup != popsolver::DataType{inChansPerGroup}) {
    return;
  }
  // Default to the vector width but allow a different value if it is forced
  // (used for joint plans).
  const auto partialChansPerGroup =
      constrainedPartialChansPerGroup
          .get_value_or(popsolver::DataType(target.getVectorWidth(inputType)))
          .getAs<unsigned>();

  const auto isAll = [](const auto k, const auto &c) {
    return std::all_of(std::begin(c), std::end(c),
                       [k](const auto x) { return x == k; });
  };
  // The vertex output type is the same the input type. This only allowed to be
  // smaller than the partial type if no reduction is required afterwards.
  if (target.getTypeSize(inputType) < target.getTypeSize(partialType) &&
      (params.inputChannelsPerConvGroup != 1 ||
       !isAll(1U, params.getKernelShape()))) {
    return;
  }

  // The OuterProduct vertex does not require a grouping of the conv groups.
  const unsigned convGroupsPerGroup = 1;

  candidates.emplace_back(Plan::OuterProduct{}, inputType, inputType,
                          convGroupsPerGroup, inChansPerGroup,
                          partialChansPerGroup);
}

// Order the candidates from most promising to least.
static void sortConvVertexTypeCandidates(
    const poplar::Target &target, const ConvParams &params,
    const ConvOptions &options, std::vector<ConvVertexType> &candidates) {
  auto numCandidates = candidates.size();
  struct ConvVertexTypeInfo {
    // Percentage of elements that are padding.
    double paddingRatio;
    // Maximum number of useful FLOPs on non-padding elements.
    double effectiveMaxFLOPs;
    // Partial type size.
    unsigned partialTypeSize;
    unsigned index;
  };
  std::vector<ConvVertexTypeInfo> candidatesInfo(numCandidates);
  for (std::size_t i = 0; i != numCandidates; ++i) {
    const auto &candidate = candidates[i];
    auto &candidateInfo = candidatesInfo[i];
    auto maxMACsPerCycle = getMaxMACsPerCyclePerTile(target, candidate);
    auto inChans = params.inputChannelsPerConvGroup;
    auto paddedInChans = gccs::alignNext(inChans, candidate.inChansPerGroup);
    auto outChans = params.outputChannelsPerConvGroup;
    auto paddedOutChans =
        gccs::alignNext(outChans, candidate.partialChansPerGroup);
    auto size = inChans * outChans;
    auto paddedSize = paddedInChans * paddedOutChans;
    candidateInfo.index = i;
    candidateInfo.paddingRatio =
        static_cast<double>(paddedSize - size) / paddedSize;
    candidateInfo.effectiveMaxFLOPs =
        maxMACsPerCycle * (1 - candidateInfo.paddingRatio);
    candidateInfo.partialTypeSize = target.getTypeSize(candidate.partialType);
  }
  std::sort(candidatesInfo.begin(), candidatesInfo.end(),
            [](const ConvVertexTypeInfo &a, const ConvVertexTypeInfo &b) {
              // Prefer candidates with more theoretical FLOPs
              if (a.effectiveMaxFLOPs != b.effectiveMaxFLOPs) {
                return a.effectiveMaxFLOPs > b.effectiveMaxFLOPs;
              }
              // Prefer candidates with less padding.
              if (a.paddingRatio != b.paddingRatio) {
                return a.paddingRatio < b.paddingRatio;
              }
              // Prefer candidates with a smaller partial size.
              if (a.partialTypeSize != b.partialTypeSize) {
                return a.partialTypeSize < b.partialTypeSize;
              }
              // Break ties with the index to ensure the sort is stable.
              return a.index < b.index;
            });
  std::vector<ConvVertexType> sortedCandidates;
  sortedCandidates.reserve(numCandidates);
  logging::poplin::trace("Convolution vertex candidates for {} pass:",
                         options.pass);
  for (auto &entry : candidatesInfo) {
    auto &candidate = candidates[entry.index];
    logging::poplin::trace(
        " - {} {}x{}x{}: "
        "partialTypeSize={}, effectiveMaxFLOPs={}, paddingRatio={}",
        candidate.method, candidate.convGroupsPerGroup,
        candidate.inChansPerGroup, candidate.partialChansPerGroup,
        entry.partialTypeSize, entry.effectiveMaxFLOPs, entry.paddingRatio);
    sortedCandidates.push_back(std::move(candidates[entry.index]));
  }
  candidates = std::move(sortedCandidates);
}

enum class Method { AMP, SLIC, HMAC, VMAC, OUTER_PRODUCT };

static std::vector<Method>
getConvMethodCandidates(const ConvOptions &options,
                        bool hasExpandDimsAtTileLevel) {
  // Constraints take priority over everything else.
  const auto constraint = options.planConstraints.get_child_optional("method");
  if (constraint) {
    std::stringstream ss;
    boost::property_tree::json_parser::write_json(ss, *constraint, false);
    Plan::Method m;
    ss >> m; // Reuse the plan serialiser
    auto visitor = poplibs_support::make_visitor<Method>(
        [&](const Plan::Amp &) { return Method::AMP; },
        [&](const Plan::Slic &) { return Method::SLIC; },
        [&](const Plan::Hmac &) { return Method::HMAC; },
        [&](const Plan::Vmac &) { return Method::VMAC; },
        [&](const Plan::OuterProduct &) { return Method::OUTER_PRODUCT; });
    return {boost::apply_visitor(visitor, m)};
  }

  // Only consider the AMP vertex when using the tile-level expand dims
  // transform, because it's the only vertex that supports optimising away
  // the broadcast of inputs.
  if (hasExpandDimsAtTileLevel) {
    return {Method::AMP};
  }

  // The order here should be in most-likely-best first for performance
  // because the planner constrains future models against the current best.
  // clang-format off
  std::vector<Method> methodCandidates = {
      Method::AMP,
      Method::SLIC,
      Method::HMAC,
      Method::VMAC,
      Method::OUTER_PRODUCT
  };
  // clang-format on

  // Disable SLIC until T18365 is fixed
  bool disableSLIC = options.pass == Pass::FC_INFERENCE_FWD ||
                     options.pass == Pass::FC_TRAINING_BWD ||
                     options.pass == Pass::FC_TRAINING_FWD ||
                     options.pass == Pass::FC_TRAINING_WU;
  if (disableSLIC) {
    methodCandidates.erase(methodCandidates.begin() + 1);
  }

  return methodCandidates;
}

std::vector<ConvVertexType>
getConvVertexTypeCandidates(const poplar::Target &target,
                            poplar::Type inputType, poplar::Type outputType,
                            poplar::Type partialType, const ConvParams &params,
                            const ConvOptions &options, bool isJointPlan,
                            bool hasExpandDimsAtTileLevel) {
  auto methodCandidates =
      getConvMethodCandidates(options, hasExpandDimsAtTileLevel);

  // All the following methods assume half or float partial types.
  assert(partialType == poplar::HALF || partialType == poplar::FLOAT);
  // All the following methods assume quarter, half or float input types,
  // or deal with the quarter type where unsupported
  assert(inputType == poplar::QUARTER || inputType == poplar::HALF ||
         inputType == poplar::FLOAT);

  std::vector<ConvVertexType> convVertexTypeCandidates;
  for (const auto &method : methodCandidates) {
    switch (method) {
    case Method::HMAC: {
      getConvVertexHMACCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Method::VMAC: {
      getConvVertexVMACCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Method::AMP: {
      getConvVertexAMPCandidates(target, inputType, outputType, partialType,
                                 params, options, isJointPlan,
                                 convVertexTypeCandidates);
      break;
    }
    case Method::SLIC: {
      getConvVertexSLICCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Method::OUTER_PRODUCT: {
      getConvVertexOuterProductCandidates(
          target, inputType, outputType, partialType, params, options,
          isJointPlan, convVertexTypeCandidates);
      break;
    }
    default: {
      throw poputil::poplibs_error("Unknown Conv vertex type method");
    }
    }
  }
  // Eliminate duplicate candidates
  std::sort(convVertexTypeCandidates.begin(), convVertexTypeCandidates.end());
  convVertexTypeCandidates.erase(std::unique(convVertexTypeCandidates.begin(),
                                             convVertexTypeCandidates.end()),
                                 convVertexTypeCandidates.end());
  sortConvVertexTypeCandidates(target, params, options,
                               convVertexTypeCandidates);
  return convVertexTypeCandidates;
}

static constexpr StructHelper vertexTypeHelper(
    &ConvVertexType::method, &ConvVertexType::inputType,
    &ConvVertexType::partialType, &ConvVertexType::convGroupsPerGroup,
    &ConvVertexType::inChansPerGroup, &ConvVertexType::partialChansPerGroup);

bool operator<(const ConvVertexType &a, const ConvVertexType &b) {
  return vertexTypeHelper.lt(a, b);
}

bool operator==(const ConvVertexType &a, const ConvVertexType &b) {
  return vertexTypeHelper.eq(a, b);
}

std::ostream &operator<<(std::ostream &os, const ConvVertexType &cvt) {
  os << "ConvVertexType{"
     << "method=" << cvt.method << ", inputType=" << cvt.inputType
     << ", partialType=" << cvt.partialType
     << ", convGroupsPerGroup=" << cvt.convGroupsPerGroup
     << ", inChansPerGroup=" << cvt.inChansPerGroup
     << ", partialChansPerGroup=" << cvt.partialChansPerGroup << "}";
  return os;
}

} // End namespace poplin
