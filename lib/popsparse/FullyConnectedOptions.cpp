// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "FullyConnectedOptions.hpp"

#include <poputil/OptionParsing.hpp>

#include <map>
#include <string>

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
     << ",\n partitioner.optimiseForSpeed: " << o.partitioner.optimiseForSpeed
     << ",\n partitioner.forceBucketSpills: " << o.partitioner.forceBucketSpills
     << ",\n partitioner.useActualWorkerSplitCosts: "
     << o.partitioner.useActualWorkerSplitCosts << "}";
  return os;
}

static std::map<std::string, poplar::Type> partialsTypeMap{
    {"half", poplar::HALF}, {"float", poplar::FLOAT}};

Options parseOptionFlags(const OptionFlags &flags) {
  Options options;

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

bool operator<(const Options &a, const Options &b) {
  return std::tie(
             a.availableMemoryProportion, a.metaInfoBucketOversizeProportion,
             a.doGradAPass, a.doGradWPass, a.partialsType, a.sharedBuckets,
             a.partitioner.optimiseForSpeed, a.partitioner.forceBucketSpills,
             a.partitioner.useActualWorkerSplitCosts) <
         std::tie(
             b.availableMemoryProportion, b.metaInfoBucketOversizeProportion,
             b.doGradAPass, b.doGradWPass, b.partialsType, b.sharedBuckets,
             b.partitioner.optimiseForSpeed, a.partitioner.forceBucketSpills,
             b.partitioner.useActualWorkerSplitCosts);
}

} // end namespace fullyconnected
} // end namespace popsparse
