// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef poplin_internal_ConvOptions_hpp
#define poplin_internal_ConvOptions_hpp

#include "poplibs_support/PlanConstraints.hpp"
#include "poplibs_support/StructHelper.hpp"
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <string>

namespace poplin {

enum class Pass {
  NONE,
  /// Unspecified matrix multiplication, this pass type is undocumented and
  /// exists so we compute exactly the same cost as we did before when the
  /// operands to a matrix multiplication were swapped (see T16758).
  /// TODO remove.
  NONE_MATMUL,
  INFERENCE_FWD,
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
  FC_INFERENCE_FWD,
  FC_TRAINING_FWD,
  FC_TRAINING_BWD,
  FC_TRAINING_WU
};

std::ostream &operator<<(std::ostream &, Pass p);

/** Options to control the implementation of a convolution */
class ConvOptions {
public:
  // proportion of tile memory available for this convolution.
  double availableMemoryProportion = .6;
  /// The pass this layer corresponds to.
  Pass pass = Pass::NONE;
  poplar::Type partialsType = poplar::FLOAT;
  poplar::Type interTilePartialsType = poplar::FLOAT;
  poplar::Type interIpuPartialsType = poplar::FLOAT;
  bool use128BitConvUnitLoad = false;
  // An optional set of constraints on the plan chosen to implement
  // this convolution.
  poplibs_support::PlanConstraints planConstraints;
  std::string planConstraintsOutputFilename; // Not including file extension
  // Allows convolution planner to use AMP vertices with only 4 engines
  // enabled to reduce paddings on small data sets
  bool enableAmpHalfEnginesPlan = false;
  // Do the reduction following the convolution in multiple stages if it
  // significantly reduce code size. This comes at the cost of increasing the
  // number of cycles.
  bool enableMultiStageReduce = true;
  // Enable a faster reduction vertex, but at the cost of partials being stored
  // in one contiguous block in interleaved memory
  bool enableFastReduce = false;
  // Remap output tensor if its layout is poor
  bool remapOutputTensor = true;
  // Use the ConvParams to pseudo-randomly select a start tile and direction
  // to lay out the convolution across the tiles.
  bool enableConvDithering = false;
  // Disable transformations.
  bool disableTransformations = false;
  // Enables insertion of cycle counts progs around convolution sequences
  bool insertTransformsCycleCountProgs = false;
  // Enables conversion table for transforms estimates
  bool enableTransformsConvTable = true;
  // Gather convolution output to a single variable
  bool gatherConvOutput = false;
  void parseConvOptions(const poplar::OptionFlags &options);

private:
  static constexpr auto helper = poplibs_support::makeStructHelper(
      &ConvOptions::availableMemoryProportion, &ConvOptions::pass,
      &ConvOptions::partialsType, &ConvOptions::interTilePartialsType,
      &ConvOptions::interIpuPartialsType, &ConvOptions::use128BitConvUnitLoad,
      &ConvOptions::planConstraints,
      &ConvOptions::planConstraintsOutputFilename,
      &ConvOptions::enableAmpHalfEnginesPlan,
      &ConvOptions::enableMultiStageReduce, &ConvOptions::enableFastReduce,
      &ConvOptions::remapOutputTensor, &ConvOptions::enableConvDithering,
      &ConvOptions::disableTransformations,
      &ConvOptions::insertTransformsCycleCountProgs,
      &ConvOptions::enableTransformsConvTable, &ConvOptions::gatherConvOutput);

public:
  bool operator<(const ConvOptions &other) const {
    return helper.lt(*this, other);
  }

  bool operator==(const ConvOptions &other) const {
    return helper.eq(*this, other);
  }

  ConvOptions() = default;

  ConvOptions(const poplar::OptionFlags &options) { parseConvOptions(options); }

  friend std::ostream &operator<<(std::ostream &os, const ConvOptions &opts);
};

// Options validation methods exposed for testing only.
namespace internal {

void validatePlanConstraintsPartitionVars(const std::string &,
                                          const boost::property_tree::ptree &);
void validatePlanConstraintsPartitionSplitVar(
    const std::string &, const boost::property_tree::ptree &);
void validatePlanConstraintsPartition(const std::string &,
                                      const boost::property_tree::ptree &);
void validatePlanConstraintsTransform(const std::string &,
                                      const boost::property_tree::ptree &);
void validatePlanConstraintsLevel(const std::string &,
                                  const boost::property_tree::ptree &);

} // namespace internal

// Validate the format. We don't know about further restrictions
// until we attempt to create a plan at which point other errors
// may be thrown.
struct ValidateConvPlanConstraintsOption {
  void operator()(const boost::property_tree::ptree &) const;
};

} // end namespace poplin

#endif // poplin_internal_ConvOptions_hpp
