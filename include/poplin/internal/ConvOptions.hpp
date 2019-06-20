// Copyright (c) 2018, Graphcore Ltd, All rights reserved.
#ifndef poplin_internal_ConvOptions_hpp
#define poplin_internal_ConvOptions_hpp

#include <poplar/Type.hpp>
#include "poplibs_support/OptionParsing.hpp"
#include <boost/property_tree/ptree.hpp>

namespace poplin {

enum class WeightUpdateMethod {
  AMP,
  AUTO
};

const char *asString(const WeightUpdateMethod &method);
std::ostream &operator<<(std::ostream &os, const WeightUpdateMethod &method);
std::istream &operator>>(std::istream &is, WeightUpdateMethod &method);

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

// Wraps ptree only in order to add custom comparison operators.
class ConvPlanConstraints : public boost::property_tree::ptree {
  using BaseTreeType = boost::property_tree::ptree;
public:
  ConvPlanConstraints() = default;
  ConvPlanConstraints(BaseTreeType t) : BaseTreeType(std::move(t)) {}
  ConvPlanConstraints &operator=(BaseTreeType t) {
    static_cast<BaseTreeType &>(*this) = std::move(t);
    return *this;
  }
};

bool operator<(const ConvPlanConstraints &a, const ConvPlanConstraints &b);

// Make an option handler that will parse ConvPlanConstraints
poplibs::OptionHandler
makePlanConstraintsOptionHandler(ConvPlanConstraints &output);

/** Options to control the implementation of a convolution */
struct ConvOptions {
  WeightUpdateMethod weightUpdateMethod = WeightUpdateMethod::AUTO;
  // Only one of tempMemoryBudget and cycleBackoffPercent may be non-zero
  unsigned tempMemoryBudget = 0;
  unsigned cycleBackoffPercent = 20;
  unsigned startTileMultiplier = 0;
  unsigned numIPUs = 0;
  unsigned tilesPerIPU = 0;
  // setting maxOutputMemoryProportion to zero disables splitting the output
  // channels serially.
  double maxOutputMemoryProportion = 0.1;
  /// The pass this layer corresponds to.
  Pass pass = Pass::NONE;
  poplar::Type partialsType = poplar::FLOAT;
  poplar::Type interTilePartialsType = poplar::FLOAT;
  poplar::Type interIpuPartialsType = poplar::FLOAT;
  bool use128BitConvUnitLoad = false;
  // An optional set of constraints on the plan chosen to implement
  // this convolution.
  ConvPlanConstraints planConstraints;
  // set this to attempt regrouping for both activations and weights in the
  // convolution
  bool useAggressiveRegrouping = false;
  ConvOptions(unsigned numIPUs, unsigned tilesPerIPU) :
    numIPUs(numIPUs), tilesPerIPU(tilesPerIPU) {}

  unsigned getNumTiles() const {
    return numIPUs * tilesPerIPU;
  }
};

inline bool operator<(const ConvOptions &a, const ConvOptions &b) {
  return std::tie(a.weightUpdateMethod,
                  a.tempMemoryBudget,
                  a.cycleBackoffPercent,
                  a.startTileMultiplier,
                  a.numIPUs,
                  a.tilesPerIPU,
                  a.pass,
                  a.partialsType,
                  a.interTilePartialsType,
                  a.interIpuPartialsType,
                  a.use128BitConvUnitLoad,
                  a.planConstraints) <
           std::tie(b.weightUpdateMethod,
                    b.tempMemoryBudget,
                    a.cycleBackoffPercent,
                    b.startTileMultiplier,
                    b.numIPUs,
                    b.tilesPerIPU,
                    b.pass,
                    b.partialsType,
                    b.interTilePartialsType,
                    b.interIpuPartialsType,
                    b.use128BitConvUnitLoad,
                    b.planConstraints);
}

// Options validation methods exposed for testing only.
namespace internal {
  void
  validatePlanConstraintsBoolean(const std::string &,
                                 const boost::property_tree::ptree &);
  void
  validatePlanConstraintsUnsigned(const std::string &,
                                  const boost::property_tree::ptree &);
  void
  validatePlanConstraintsPartitionVars(const std::string &,
                                       const boost::property_tree::ptree &);
  void
  validatePlanConstraintsPartitionSplitVar(const std::string &,
                                           const boost::property_tree::ptree &);
  void
  validatePlanConstraintsPartition(const std::string &,
                                   const boost::property_tree::ptree &);
  void
  validatePlanConstraintsTransform(const std::string &,
                                   const boost::property_tree::ptree &);
  void
  validatePlanConstraintsLevel(const std::string &,
                               const boost::property_tree::ptree &);
  void
  validatePlanConstraintsOption(const boost::property_tree::ptree &);
}

} // end namespace poplin

#endif // poplin_internal_ConvOptions_hpp
