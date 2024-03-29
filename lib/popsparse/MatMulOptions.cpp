// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include "MatMulOptions.hpp"

#include <poputil/OptionParsing.hpp>

#include <gccs/StructHelper.hpp>

using namespace poplar;
using namespace poplibs;

namespace popsparse {
namespace dynamic {

static constexpr auto comparisonHelper = gccs::makeStructHelper(
    &MatMulOptions::availableMemoryProportion,
    &MatMulOptions::metaInfoBucketOversizeProportion,
    &MatMulOptions::partialsType, &MatMulOptions::sharedBuckets,
    &MatMulOptions::partitioner);

bool operator<(const MatMulOptions &a, const MatMulOptions &b) {
  return comparisonHelper.lt(a, b);
}

bool operator==(const MatMulOptions &a, const MatMulOptions &b) {
  return comparisonHelper.eq(a, b);
}

bool operator!=(const MatMulOptions &a, const MatMulOptions &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const MatMulOptions &o) {
  os << "{availableMemoryProportion: " << o.availableMemoryProportion
     << ",\n metaInfoBucketOversizeProportion: "
     << o.metaInfoBucketOversizeProportion
     << ",\n partialsType: " << o.partialsType
     << ",\n sharedBuckets: " << o.sharedBuckets
     << ",\n partitioner.optimiseForSpeed: " << o.partitioner.optimiseForSpeed
     << ",\n partitioner.forceBucketSpills: " << o.partitioner.forceBucketSpills
     << ",\n partitioner.useActualWorkerSplitCosts: "
     << o.partitioner.useActualWorkerSplitCosts << "}";
  return os;
}

static std::map<std::string, poplar::Type> partialsTypeMap{
    {"half", poplar::HALF}, {"float", poplar::FLOAT}};

MatMulOptions parseMatMulOptionFlags(const OptionFlags &flags) {
  MatMulOptions options;
  const OptionSpec optSpec{
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(options.availableMemoryProportion, 0.)},
      {"metaInfoBucketOversizeProportion",
       OptionHandler::createWithDouble(
           options.metaInfoBucketOversizeProportion)},
      {"partialsType",
       OptionHandler::createWithEnum(options.partialsType, partialsTypeMap)},
      {"sharedBuckets", OptionHandler::createWithBool(options.sharedBuckets)},
      {"partitioner.optimiseForSpeed",
       OptionHandler::createWithBool(options.partitioner.optimiseForSpeed)},
      {"partitioner.forceBucketSpills",
       OptionHandler::createWithBool(options.partitioner.forceBucketSpills)},
      {"partitioner.useActualWorkerSplitCosts",
       OptionHandler::createWithBool(
           options.partitioner.useActualWorkerSplitCosts)}};
  for (const auto &entry : flags) {
    optSpec.parse(entry.first, entry.second);
  }
  return options;
}

} // end namespace dynamic

namespace static_ {

static constexpr auto comparisonHelper = gccs::makeStructHelper(
    &MatMulOptions::availableMemoryProportion, &MatMulOptions::numBands,
    &MatMulOptions::nSplit, &MatMulOptions::verboseLogging);

bool operator<(const MatMulOptions &a, const MatMulOptions &b) {
  return comparisonHelper.lt(a, b);
}

bool operator==(const MatMulOptions &a, const MatMulOptions &b) {
  return comparisonHelper.eq(a, b);
}

bool operator!=(const MatMulOptions &a, const MatMulOptions &b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream &os, const MatMulOptions &o) {
  os << "{numBands=" << (o.numBands ? std::to_string(o.numBands) : "none")
     << ", nSplit=" << (o.nSplit ? std::to_string(o.nSplit) : "none")
     << ", availableMemoryProportion=" << o.availableMemoryProportion
     << ", verbose logging=" << (o.verboseLogging ? "true" : "false") << "}";
  return os;
}

MatMulOptions parseMatMulOptionFlags(const OptionFlags &optionFlags) {
  static std::map<std::string, poplar::Type> partialsTypeMap{
      {"half", poplar::HALF}, {"float", poplar::FLOAT}};

  MatMulOptions options;
  using namespace poplibs;
  const poplibs::OptionSpec spec{
      {"numBands", OptionHandler::createWithInteger(options.numBands)},
      {"nSplit", OptionHandler::createWithInteger(options.nSplit)},
      {"verboseLogging", OptionHandler::createWithBool(options.verboseLogging)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(options.availableMemoryProportion)},
      {"enablePreprocessing",
       OptionHandler::createWithBool(options.enablePreprocessing)}};

  for (const auto &entry : optionFlags) {
    spec.parse(entry.first, entry.second);
  }
  return options;
}

} // end namespace static_
} // end namespace popsparse
