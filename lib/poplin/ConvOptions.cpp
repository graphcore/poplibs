#include "ConvOptions.hpp"
#include "ConvPlan.hpp"
#include "poputil/exceptions.hpp"

#include <iostream>
#include <unordered_set>

namespace poplin {

using boost::property_tree::ptree;

using poplibs_support::validatePlanConstraintsBoolean;
using poplibs_support::validatePlanConstraintsUnsigned;
using poplibs_support::validatePlanConstraintsUnsignedArray;

namespace internal {

// Listings of currently handled plan constraints of different types.
// TODO: Add more as these are handled.
static std::unordered_set<std::string> validPartitionConstraintVar = {
    "convGroupSplit",
    "batchSplit",
    "inChanSplit",
};
static std::unordered_set<std::string> validPartitionConstraintVars = {
    "fieldSplit",
    "kernelSplit",
};
static std::unordered_set<std::string> validPartitionConstraintSplitVar = {
    "outChanSplit",
};
static std::unordered_set<std::string> validTransformConstraintBool = {
    "swapOperands",
};
static std::unordered_set<std::string> validTransformConstraintDims = {
    "expandDims",
    "outChanFlattenDims",
};

static void validatePlanConstraintsIndex(const std::string &path,
                                         const std::string &indexStr) {
  std::stringstream s(indexStr);
  std::int64_t level;
  s >> level;
  if (s.fail()) {
    throw poplar::invalid_option("'" + path + "': Index not an integer");
  }
  if (level < 0) {
    throw poplar::invalid_option("'" + path + "': Index is negative");
  }
}

void validatePlanConstraintsTransform(const std::string &path, const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const std::string subPath = path + "." + child.first;
    if (validTransformConstraintBool.count(child.first) > 0) {
      validatePlanConstraintsBoolean(subPath, child.second);
    } else if (validTransformConstraintDims.count(child.first) > 0) {
      validatePlanConstraintsUnsignedArray(subPath, child.second);
    } else {
      throw poplar::invalid_option("'" + subPath + "': " + child.first +
                                   " is not currently handled or does "
                                   "not exist");
    }
  }
}

void validatePlanConstraintsPartitionVars(const std::string &path,
                                          const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const auto subPath = path + "." + child.first;
    validatePlanConstraintsIndex(subPath, child.first);
    validatePlanConstraintsUnsigned(subPath, child.second);
  }
}

void validatePlanConstraintsPartitionSplitVar(const std::string &path,
                                              const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const auto subPath = path + "." + child.first;
    if (child.first == "parallel") {
      validatePlanConstraintsUnsigned(subPath, child.second);
    } else if (child.first == "serial") {
      validatePlanConstraintsUnsigned(subPath, child.second);
    } else {
      throw poplar::invalid_option("'" + subPath + "': " + child.first +
                                   " is not either 'parallel' or 'serial'");
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
    } else if (validPartitionConstraintVars.count(child.first)) {
      validatePlanConstraintsPartitionVars(subPath, child.second);
    } else if (validPartitionConstraintSplitVar.count(child.first)) {
      validatePlanConstraintsPartitionSplitVar(subPath, child.second);
    } else {
      throw poplar::invalid_option("'" + subPath + "': " + child.first +
                                   " is not currently handled or does not "
                                   "exist");
    }
  }
}

void validatePlanConstraintsLevel(const std::string &path, const ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const std::string subPath = path + "." + child.first;
    if (child.first == "transform") {
      validatePlanConstraintsTransform(subPath, child.second);
    } else if (child.first == "partition") {
      validatePlanConstraintsPartition(subPath, child.second);
    } else {
      throw poplar::invalid_option("'" + subPath + "': " + child.first +
                                   " is not a valid sub-domain of the plan. "
                                   "Must be either 'transform' or "
                                   "'partition'");
    }
  }
}

void validatePlanConstraintsMethod(const std::string &path, const ptree &t) {
  Plan::Method m;
  try {
    std::stringstream ss(t.data());
    ss >> m;
  } catch (const poputil::poplibs_error &e) {
    throw poplar::invalid_option("'" + path + "': " + e.what());
  }
}

} // end namespace internal

// Validate the format. We don't know about further restrictions
// until we attempt to create a plan at which point other errors
// may be thrown.
void ValidateConvPlanConstraintsOption::operator()(const ptree &t) const {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("Plan constraints must be an object");
  }

  for (const auto &child : t) {
    if (child.first == "method") {
      internal::validatePlanConstraintsMethod(child.first, child.second);
    } else if (child.first == "inChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "partialChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else {
      internal::validatePlanConstraintsIndex(child.first, child.first);
      internal::validatePlanConstraintsLevel(child.first, child.second);
    }
  }
}

} // end namespace poplin
