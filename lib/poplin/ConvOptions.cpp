#include "ConvOptions.hpp"
#include "poputil/exceptions.hpp"
#include "ConvPlan.hpp"

#include <unordered_set>
#include <iostream>

#include <boost/property_tree/json_parser.hpp>

static inline bool pTreeLessThan(const boost::property_tree::ptree &a,
                                 const boost::property_tree::ptree &b) {
  auto aIt = a.ordered_begin();
  auto bIt = b.ordered_begin();
  auto aEnd = a.not_found();
  auto bEnd = b.not_found();
  auto aN = std::distance(aIt, aEnd);
  auto bN = std::distance(bIt, bEnd);

  if (aN != bN) {
    return aN < bN;
  }

  for (; aIt != aEnd; ++aIt, ++bIt) {
    const auto &aSub = aIt->second;
    const auto &bSub = bIt->second;
    if (aSub.empty() != bSub.empty()) {
      return aSub.empty() < bSub.empty();
    } else if (!aSub.empty()) {
      return pTreeLessThan(aSub, bSub);
    } else if (auto aStr = aSub.get_value_optional<std::string>()) {
      auto bStr = bSub.get_value_optional<std::string>();
      if (bool(aStr) != bool(bStr)) {
        return bool(aStr) < bool(bStr);
      }
      if (*aStr != *bStr) {
        return *aStr < *bStr;
      }
    } else if (auto aNum = aSub.get_value_optional<unsigned>()) {
      auto bNum = bSub.get_value_optional<unsigned>();
      if (bool(aNum) != bool(bNum)) {
        return bool(aNum) < bool(bNum);
      }
      if (*aNum != *bNum) {
        return *aNum < *bNum;
      }
    } else if (auto aBool = aSub.get_value_optional<bool>()) {
      auto bBool = bSub.get_value_optional<bool>();
      if (bool(aBool) != bool(bBool)) {
        return bool(aBool) < bool(bBool);
      }
      if (*aBool != *bBool) {
        return *aBool < *bBool;
      }
    } else {
      throw poputil::poplibs_error("Unhandled child type in property "
                                   "tree comparison operator");
    }
  }

  return false;
}

namespace poplin {

using namespace internal;

// Compare a property_tree in an ordered way
bool operator<(const ConvPlanConstraints &a, const ConvPlanConstraints &b) {
  return pTreeLessThan(a, b);
}

poplibs::OptionHandler
makePlanConstraintsOptionHandler(ConvPlanConstraints &output) {
  return poplibs::OptionHandler{
    [&output](const std::string &value) {
      if (!value.empty()) {
        std::stringstream ss(value);
        boost::property_tree::ptree t;
        boost::property_tree::json_parser::read_json(ss, t);
        // Validate the format. We don't know about further restrictions
        // until we attempt to create a plan at which point other errors
        // may be thrown.
        validatePlanConstraintsOption(t);
        output = std::move(t);
      } else {
        output.clear();
      }
    }
  };
}

namespace internal {

// Listings of currently handled plan constraints of different types.
// TODO: Add more as these are handled.
static std::unordered_set<std::string>
  validPartitionConstraintVar = {
  "convGroupSplit",
};
static std::unordered_set<std::string>
  validPartitionConstraintVars = {
  "fieldSplit",
  "kernelSplit",
};
static std::unordered_set<std::string>
  validPartitionConstraintSplitVar = {
  "batchSplit",
  "outChanSplit",
  "inChanSplit",
};
static std::unordered_set<std::string>
  validTransformConstraintBool = {
  "swapOperands",
};
static std::unordered_set<std::string>
  validTransformConstraintDims = {
  "expandDims",
  "outChanFlattenDims",
};

static void
validatePlanConstraintsIndex(const std::string &path,
                             const std::string &indexStr) {
  std::stringstream s(indexStr);
  std::int64_t level;
  s >> level;
  if (s.fail()) {
    throw poplar::invalid_option("'" + path +
                                 "': Index not an integer");
  }
  if (level < 0) {
    throw poplar::invalid_option("'" + path +
                                 "': Index is negative");
  }
}

void
validatePlanConstraintsBoolean(
    const std::string &path,
    const boost::property_tree::ptree &t) {
  const auto val = t.get_value_optional<bool>();
  if (!val) {
    throw poplar::invalid_option("'" + path + "': Not a boolean value");
  }
}

void
validatePlanConstraintsUnsigned(
    const std::string &path,
    const boost::property_tree::ptree &t) {
  const auto val = t.get_value_optional<double>();
  if (!val || *val < 0 || *val > std::numeric_limits<unsigned>::max()) {
    throw poplar::invalid_option("'" + path + "': Not a valid unsigned "
                                 "integer");
  }
}

void
validatePlanConstraintsUnsignedArray(
    const std::string &path,
    const boost::property_tree::ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an array");
  }
  for (const auto &child : t) {
    if (!child.first.empty()) {
      throw poplar::invalid_option("'" + path + "': Must be an array");
    }
    validatePlanConstraintsUnsigned(path, child.second);
  }
}

void
validatePlanConstraintsTransform(const std::string &path,
                                 const boost::property_tree::ptree &t) {
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

void
validatePlanConstraintsPartitionVars(
    const std::string &path,
    const boost::property_tree::ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("'" + path + "': Must be an object");
  }
  for (const auto &child : t) {
    const auto subPath = path + "." + child.first;
    validatePlanConstraintsIndex(subPath, child.first);
    validatePlanConstraintsUnsigned(subPath, child.second);
  }
}

void
validatePlanConstraintsPartitionSplitVar(
    const std::string &path,
    const boost::property_tree::ptree &t) {
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

void
validatePlanConstraintsPartition(const std::string &path,
                                 const boost::property_tree::ptree &t) {
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

void
validatePlanConstraintsLevel(const std::string &path,
                             const boost::property_tree::ptree &t) {
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

void
validatePlanConstraintsMethod(const std::string &path,
                              const boost::property_tree::ptree &t) {
  Plan::Method m;
  try {
    std::stringstream ss(t.data());
    ss >> m;
  } catch (const poputil::poplibs_error &e) {
    throw poplar::invalid_option("'" + path + "': " + e.what());
  }
}

void
validatePlanConstraintsOption(const boost::property_tree::ptree &t) {
  if (t.empty() && !t.data().empty()) {
    throw poplar::invalid_option("Plan constraints must be an object");
  }
  for (const auto &child : t) {
    if (child.first == "method") {
      validatePlanConstraintsMethod(child.first, child.second);
    } else if (child.first == "inChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "partialChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else {
      validatePlanConstraintsIndex(child.first,
                                   child.first);
      validatePlanConstraintsLevel(child.first, child.second);
    }
  }
}

} // end namespace internal

} // end namespace poplin
