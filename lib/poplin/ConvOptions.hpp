// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
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
  // These are only stored to allow ConvOptions to be used as the key for the
  // convolution cache. These values should be read from poplar::Target instead.
  unsigned numIPUs;
  unsigned tilesPerIPU;

public:
  // proportion of tile memory available for this convolution.
  double availableMemoryProportion = .6;
  unsigned startTileMultiplier = 0;
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
  // SLIC is currently only supported in the planner and so if enabled the
  // compilation will not complete. just enable this option to see what the
  // planner estimates would be if SLIC was fully supported.
  bool enableSLIC = false;

  void parseConvOptions(const poplar::OptionFlags &options);
  bool operator<(const ConvOptions &other) const {
    using poplibs_support::makeStructHelper;

    const auto helper = makeStructHelper(
        &ConvOptions::availableMemoryProportion,
        &ConvOptions::startTileMultiplier, &ConvOptions::numIPUs,
        &ConvOptions::tilesPerIPU, &ConvOptions::pass,
        &ConvOptions::partialsType, &ConvOptions::interTilePartialsType,
        &ConvOptions::interIpuPartialsType, &ConvOptions::use128BitConvUnitLoad,
        &ConvOptions::planConstraints,
        &ConvOptions::planConstraintsOutputFilename, &ConvOptions::enableSLIC);
    return helper.lt(*this, other);
  }
  ConvOptions(const poplar::Target &target)
      : numIPUs(target.getNumIPUs()), tilesPerIPU(target.getTilesPerIPU()) {}

  ConvOptions(const poplar::Target &target, const poplar::OptionFlags &options)
      : numIPUs(target.getNumIPUs()), tilesPerIPU(target.getTilesPerIPU()) {
    parseConvOptions(options);
  }

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
