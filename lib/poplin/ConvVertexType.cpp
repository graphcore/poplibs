// Copyright (c) 2016 Graphcore Ltd. All rights reserved.

#include "ConvVertexType.hpp"

#include "ConvModel.hpp"
#include "ConvOptions.hpp"
#include "PerformanceEstimation.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplin/ConvParams.hpp"
#include "popsolver/Model.hpp"
#include "poputil/exceptions.hpp"
#include <vector>

using namespace poplibs_support;

namespace poplin {

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

bool canUseConvolutionInstruction(bool floatActivations, bool floatPartials,
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

static void getConvVertexHMACCandidates(
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
  // HMAC codelet for half partials processes 2 partials inside inner loop
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

    // The HMAC vertex does not require a grouping of the conv groups.
    const unsigned convGroupsPerGroup = 1;

    candidates.emplace_back(Plan::Method::HMAC, inputType, partialType,
                            convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup, numConvUnits, numConvUnits,
                            useLimitedVersion);
    previousInChanGroups = inChanGroups;
  }
}

static void getConvVertexVMACCandidates(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &outputType, const poplar::Type &partialType,
    const ConvParams &params, const ConvOptions &options, bool isJointPlan,
    std::vector<ConvVertexType> &candidates) {
  bool floatActivations = inputType == poplar::FLOAT;
  bool floatPartials = partialType == poplar::FLOAT;

  // Assembly version only available for half activations and float partials
  if (floatActivations || !floatPartials) {
    return;
  }

  auto numConvUnits = getNumConvUnits(floatActivations, floatPartials, target);

  // Every execution of the VMAC inner loop vertex processes a single input
  // channel.
  unsigned inChansPerGroup = 1;
  unsigned partialChansPerGroup = 1;
  std::vector<unsigned> convGroupsPerGroupOptions = {4};
  for (auto convGroupsPerGroup : convGroupsPerGroupOptions) {
    candidates.emplace_back(Plan::Method::VMAC, inputType, partialType,
                            convGroupsPerGroup, inChansPerGroup,
                            partialChansPerGroup, numConvUnits, numConvUnits,
                            false);
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

  candidates.emplace_back(Plan::Method::OUTER_PRODUCT, inputType, inputType,
                          convGroupsPerGroup, inChansPerGroup,
                          partialChansPerGroup, 0, 0, true);
}

std::vector<ConvVertexType>
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
    // clang-format off
    methodCandidates = {
        Plan::Method::AMP,
        Plan::Method::SLIC,
        Plan::Method::HMAC,
        Plan::Method::VMAC,
        Plan::Method::OUTER_PRODUCT
    };
    // clang-format on

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
    case Plan::Method::HMAC: {
      getConvVertexHMACCandidates(target, inputType, outputType, partialType,
                                  params, options, isJointPlan,
                                  convVertexTypeCandidates);
      break;
    }
    case Plan::Method::VMAC: {
      getConvVertexVMACCandidates(target, inputType, outputType, partialType,
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

} // End namespace poplin
