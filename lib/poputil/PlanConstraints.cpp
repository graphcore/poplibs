// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poputil/PlanConstraints.hpp"

#include <poplar/exceptions.hpp>

#include <ostream>

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
      throw poplar::runtime_error(
          "Unhandled child type in property tree comparison operator");
    }
  }

  return false;
}

namespace poputil {

std::ostream &operator<<(std::ostream &os, const PlanConstraints &pt) {
  boost::property_tree::json_parser::write_json(os, pt, false);
  return os;
}

// Compare a property_tree in an ordered way
bool operator<(const PlanConstraints &a, const PlanConstraints &b) {
  return pTreeLessThan(a, b);
}

void validatePlanConstraintsBoolean(const std::string &path,
                                    const boost::property_tree::ptree &t) {
  const auto val = t.get_value_optional<bool>();
  if (!val) {
    throw poplar::invalid_option("'" + path + "': Not a boolean value");
  }
}

void validatePlanConstraintsUnsigned(const std::string &path,
                                     const boost::property_tree::ptree &t) {
  const auto val = t.get_value_optional<double>();
  if (!val || *val < 0 || *val > std::numeric_limits<unsigned>::max()) {
    throw poplar::invalid_option("'" + path +
                                 "': Not a valid unsigned "
                                 "integer");
  }
}

void validatePlanConstraintsUnsignedArray(
    const std::string &path, const boost::property_tree::ptree &t) {
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

} // namespace poputil
