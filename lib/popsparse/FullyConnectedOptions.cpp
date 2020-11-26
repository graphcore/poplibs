// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "FullyConnectedOptions.hpp"

#include <poputil/OptionParsing.hpp>

#include "poplibs_support/StructHelper.hpp"

#include <map>
#include <string>
#include <unordered_set>

using namespace poplar;
using namespace poplibs;

namespace popsparse {
namespace fullyconnected {

std::ostream &operator<<(std::ostream &os, const Options &o) {
  os << "{availableMemoryProportion: " << o.availableMemoryProportion
     << ",\n metaInfoBucketOversizeProportion: "
     << o.metaInfoBucketOversizeProportion
     << ",\n doGradAPass: " << o.doGradAPass
     << ",\n doGradWPass: " << o.doGradWPass
     << ",\n partialsType: " << o.partialsType
     << ",\n sharedBuckets: " << o.sharedBuckets
     << ",\n enableGradWStructuredRearrangements: "
     << o.enableStructuredRearrangements
     << ",\n partitioner.optimiseForSpeed: " << o.partitioner.optimiseForSpeed
     << ",\n partitioner.forceBucketSpills: " << o.partitioner.forceBucketSpills
     << ",\n partitioner.useActualWorkerSplitCosts: "
     << o.partitioner.useActualWorkerSplitCosts
     << ",\n planConstraints: " << o.planConstraints << "}";
  return os;
}

static std::map<std::string, poplar::Type> partialsTypeMap{
    {"half", poplar::HALF}, {"float", poplar::FLOAT}};

using boost::property_tree::ptree;
using poplibs_support::validatePlanConstraintsBoolean;
using poplibs_support::validatePlanConstraintsUnsigned;

static std::unordered_set<std::string> validPartitionConstraintVar = {"x", "y",
                                                                      "z"};

static std::unordered_set<std::string> validExchangeConstraintBool = {
    "gradWExchangeBuckets"};

void validatePlanConstraintsExchange(const std::string &path, const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const std::string subPath = path + "." + child.first;
    if (validExchangeConstraintBool.count(child.first) > 0) {
      validatePlanConstraintsBoolean(subPath, child.second);
    } else {
      throw poplar::invalid_option(
          "'" + subPath + "': " + child.first +
          " is not currently handled or does not exist");
    }
  }
}

void validatePlanConstraintsPartition(const std::string &path, const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const std::string subPath = path + "." + child.first;
    if (validPartitionConstraintVar.count(child.first) > 0) {
      validatePlanConstraintsUnsigned(subPath, child.second);
    } else {
      throw poplar::invalid_option("'" + subPath + "': " + child.first +
                                   " is not currently handled or does not "
                                   "exist");
    }
  }
}

static void validatePlanConstraintsUseDense(const std::string &path,
                                            const ptree &t) {
  validatePlanConstraintsBoolean(path, t);
}

namespace {

struct ValidatePlanConstraintsOption {
  void operator()(const boost::property_tree::ptree &t) const {
    if (t.empty() && !t.data().empty()) {
      throw poplar::invalid_option("Plan constraints must be an object");
    }

    for (const auto &child : t) {
      if (child.first == "exchange") {
        validatePlanConstraintsExchange(child.first, child.second);
      } else if (child.first == "partition") {
        validatePlanConstraintsPartition(child.first, child.second);
      } else if (child.first == "useDense") {
        validatePlanConstraintsUseDense(child.first, child.second);
      } else {
        throw poplar::invalid_option(
            child.first + " is not currently handled or does not exist");
      }
    }
  }
};

} // end anonymous namespace

Options parseOptionFlags(const OptionFlags &flags) {
  Options options;

  using poplibs_support::makePlanConstraintsOptionHandler;
  const auto makeSparseFCPlanConstraintsOptionHandler =
      &poplibs_support::makePlanConstraintsOptionHandler<
          ValidatePlanConstraintsOption>;

  const OptionSpec optSpec{
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(options.availableMemoryProportion)},
      {"metaInfoBucketOversizeProportion",
       OptionHandler::createWithDouble(
           options.metaInfoBucketOversizeProportion)},
      {"doGradAPass", OptionHandler::createWithBool(options.doGradAPass)},
      {"doGradWPass", OptionHandler::createWithBool(options.doGradWPass)},
      {"partialsType",
       OptionHandler::createWithEnum(options.partialsType, partialsTypeMap)},
      {"sharedBuckets", OptionHandler::createWithBool(options.sharedBuckets)},
      {"enableStructuredRearrangements",
       OptionHandler::createWithBool(options.enableStructuredRearrangements)},
      {"partitioner.optimiseForSpeed",
       OptionHandler::createWithBool(options.partitioner.optimiseForSpeed)},
      {"partitioner.forceBucketSpills",
       OptionHandler::createWithBool(options.partitioner.forceBucketSpills)},
      {"partitioner.useActualWorkerSplitCosts",
       OptionHandler::createWithBool(
           options.partitioner.useActualWorkerSplitCosts)},
      {"planConstraints",
       makeSparseFCPlanConstraintsOptionHandler(options.planConstraints)}};
  for (const auto &entry : flags) {
    optSpec.parse(entry.first, entry.second);
  }
  return options;
}

static constexpr auto optionsHelper = poplibs_support::makeStructHelper(
    &Options::availableMemoryProportion,
    &Options::metaInfoBucketOversizeProportion, &Options::doGradAPass,
    &Options::doGradWPass, &Options::partialsType, &Options::sharedBuckets,
    &Options::enableStructuredRearrangements, &Options::partitioner);

bool operator<(const Options &a, const Options &b) {
  return optionsHelper.lt(a, b);
}

bool operator==(const Options &a, const Options &b) {
  return optionsHelper.eq(a, b);
}

bool operator!=(const Options &a, const Options &b) { return !(a == b); }

} // end namespace fullyconnected
} // end namespace popsparse
