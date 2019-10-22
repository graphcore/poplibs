// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef poplin_internal_ConvOptions_hpp
#define poplin_internal_ConvOptions_hpp

#include "poplibs_support/PlanConstraints.hpp"
#include "poplibs_support/StructHelper.hpp"
#include <poplar/Type.hpp>

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

/** Options to control the implementation of a convolution */
struct ConvOptions {
  // proportion of tile memory available for this convolution.
  double availableMemoryProportion = .6;
  unsigned startTileMultiplier = 0;
  unsigned numIPUs = 0;
  unsigned tilesPerIPU = 0;
  /// The pass this layer corresponds to.
  Pass pass = Pass::NONE;
  poplar::Type partialsType = poplar::FLOAT;
  poplar::Type interTilePartialsType = poplar::FLOAT;
  poplar::Type interIpuPartialsType = poplar::FLOAT;
  bool use128BitConvUnitLoad = false;
  // An optional set of constraints on the plan chosen to implement
  // this convolution.
  poplibs_support::PlanConstraints planConstraints;
  ConvOptions(unsigned numIPUs, unsigned tilesPerIPU)
      : numIPUs(numIPUs), tilesPerIPU(tilesPerIPU) {}

  unsigned getNumTiles() const { return numIPUs * tilesPerIPU; }
};

inline bool operator<(const ConvOptions &a, const ConvOptions &b) {
  using poplibs_support::makeStructHelper;

  const auto helper = makeStructHelper(
      &ConvOptions::availableMemoryProportion,
      &ConvOptions::startTileMultiplier, &ConvOptions::numIPUs,
      &ConvOptions::tilesPerIPU, &ConvOptions::pass, &ConvOptions::partialsType,
      &ConvOptions::interTilePartialsType, &ConvOptions::interIpuPartialsType,
      &ConvOptions::use128BitConvUnitLoad, &ConvOptions::planConstraints);
  return helper.lt(a, b);
}

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
