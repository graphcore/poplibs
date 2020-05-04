// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
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

std::map<std::string, Pass> passMap{
    {"NONE", Pass::NONE},
    {"INFERENCE_FWD", Pass::INFERENCE_FWD},
    {"TRAINING_FWD", Pass::TRAINING_FWD},
    {"TRAINING_BWD", Pass::TRAINING_BWD},
    {"TRAINING_WU", Pass::TRAINING_WU},
    {"FC_INFERENCE_FWD", Pass::FC_INFERENCE_FWD},
    {"FC_TRAINING_FWD", Pass::FC_TRAINING_FWD},
    {"FC_TRAINING_BWD", Pass::FC_TRAINING_BWD},
    {"FC_TRAINING_WU", Pass::FC_TRAINING_WU}};

std::map<std::string, poplar::Type> partialsTypeMap{{"half", poplar::HALF},
                                                    {"float", poplar::FLOAT}};

std::ostream &operator<<(std::ostream &os, const Pass p) {
  switch (p) {
  case Pass::NONE:
    return os << "NONE";
  case Pass::INFERENCE_FWD:
    return os << "INFERENCE_FWD";
  case Pass::TRAINING_FWD:
    return os << "TRAINING_FWD";
  case Pass::TRAINING_BWD:
    return os << "TRAINING_BWD";
  case Pass::TRAINING_WU:
    return os << "TRAINING_WU";
  case Pass::FC_INFERENCE_FWD:
    return os << "FC_INFERENCE_FWD";
  case Pass::FC_TRAINING_FWD:
    return os << "FC_TRAINING_FWD";
  case Pass::FC_TRAINING_BWD:
    return os << "FC_TRAINING_BWD";
  case Pass::FC_TRAINING_WU:
    return os << "FC_TRAINING_WU";
  }

  const auto id = static_cast<std::underlying_type_t<Pass>>(p);
  throw poputil::poplibs_error("Unknown pass <" + std::to_string(id) + ">");
}

std::ostream &operator<<(std::ostream &os, const ConvOptions &opts) {
  os << "Option: availableMemoryProportion     ";
  os << opts.availableMemoryProportion << "\n";
  os << "        pass                          ";
  os << opts.pass << "\n";
  os << "        partialsType                  ";
  os << opts.partialsType << "\n";
  os << "        interTilePartialsType         ";
  os << opts.interTilePartialsType << "\n";
  os << "        interIpuPartialsType          ";
  os << opts.interIpuPartialsType << "\n";
  os << "        use128BitConvUnitLoad         ";
  os << opts.use128BitConvUnitLoad << "\n";
  os << "        planConstraints               ";
  os << opts.planConstraints; // No newline needed
  os << "        planConstraintsOutputFilename ";
  os << opts.planConstraintsOutputFilename << "\n";
  os << "        enableAmpHalfEnginesPlan      ";
  os << opts.enableAmpHalfEnginesPlan << "\n";
  os << "        enableMultiStageReduce        ";
  os << opts.enableMultiStageReduce << "\n";
  os << "        remapOutputTensor             ";
  os << opts.remapOutputTensor;
  return os;
}

// Parse the passed options, taking default numIPUs and tilesPerIPU from the
// target
void ConvOptions::parseConvOptions(const poplar::OptionFlags &options) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  using poplibs_support::makePlanConstraintsOptionHandler;

  const auto makeConvPlanConstraintsOptionHandler =
      &makePlanConstraintsOptionHandler<ValidateConvPlanConstraintsOption>;

  const OptionSpec convSpec{
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(availableMemoryProportion)},
      {"pass", OptionHandler::createWithEnum(pass, passMap)},
      {"partialsType",
       OptionHandler::createWithEnum(partialsType, partialsTypeMap)},
      {"partialsType.interTile",
       OptionHandler::createWithEnum(interTilePartialsType, partialsTypeMap)},
      {"partialsType.interIPU",
       OptionHandler::createWithEnum(interIpuPartialsType, partialsTypeMap)},
      {"use128BitConvUnitLoad",
       OptionHandler::createWithBool(use128BitConvUnitLoad)},
      {"planConstraints",
       makeConvPlanConstraintsOptionHandler(planConstraints)},
      {"planConstraintsOutputFilename",
       OptionHandler::createWithString(planConstraintsOutputFilename)},
      {"enableAmpHalfEnginesPlan",
       OptionHandler::createWithBool(enableAmpHalfEnginesPlan)},
      {"enableMultiStageReduce",
       OptionHandler::createWithBool(enableMultiStageReduce)},
      {"remapOutputTensor", OptionHandler::createWithBool(remapOutputTensor)}};
  for (const auto &entry : options) {
    convSpec.parse(entry.first, entry.second);
  }
}

decltype(ConvOptions::helper) ConvOptions::helper;

namespace internal {

// Listings of currently handled plan constraints of different types.
// TODO: Add more as these are handled.
static std::unordered_set<std::string> validPartitionConstraintVar = {
    "convGroupSplit",
    "batchSplit",
};
static std::unordered_set<std::string> validPartitionConstraintVars = {
    "fieldSplit",
    "kernelSplit",
};
static std::unordered_set<std::string> validPartitionConstraintSplitVar = {
    "inChanSplit",
    "outChanSplit",
};
static std::unordered_set<std::string> validTransformConstraintBool = {
    "swapOperands"};
static std::unordered_set<std::string> validTransformConstraintUnsignedArray = {
    "expandDims",
    "outChanFlattenDims",
    "combineConvGroupsFactor",
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
    } else if (validTransformConstraintUnsignedArray.count(child.first) > 0) {
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
    } else if (child.first == "convGroupsPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "inChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "partialChansPerGroup") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "numAmpConvUnits") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else if (child.first == "slicWindowWidth") {
      validatePlanConstraintsUnsigned(child.first, child.second);
    } else {
      internal::validatePlanConstraintsIndex(child.first, child.first);
      internal::validatePlanConstraintsLevel(child.first, child.second);
    }
  }
}

} // end namespace poplin
