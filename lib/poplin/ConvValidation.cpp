// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "ConvValidation.hpp"

#include "ConvOptions.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/exceptions.hpp"
#include <string>

using namespace poplibs_support;

void poplin::validateLayerParams(const ConvParams &params,
                                 const poplar::Target &target,
                                 ConvOptions &options) {
  const struct {
    poplar::Type type;
    const char *name;
  } typesToCheck[] = {
      {params.inputType, "input element type"},
      {params.outputType, "output element type"},
      {options.partialsType, "partial type"},
      {options.interTilePartialsType, "inter-tile partial type"},
      {options.interIpuPartialsType, "inter-ipu partial type"},
  };
  for (const auto &entry : typesToCheck) {
    if (entry.type != poplar::HALF && entry.type != poplar::FLOAT &&
        entry.type != poplar::QUARTER) {
      throw poputil::poplibs_error(std::string("Unsupported ") + entry.name +
                                   " (must be float, half or quarter)");
    }
  }
  const struct {
    poplar::Type &type;
    const char *name;
  } partialTypes[] = {
      {options.partialsType, "partial type"},
      {options.interTilePartialsType, "inter-tile partial type"},
      {options.interIpuPartialsType, "inter-ipu partial type"},
  };
  // Partial types must be at least as big as the output type.
  for (const auto &partialType : partialTypes) {
    if (target.getTypeSize(partialType.type) <
        target.getTypeSize(params.outputType)) {
      logging::poplin::warn(
          "Ignoring {} ({}) which is smaller than the output type ({})",
          partialType.name, partialType.type.toString(),
          params.outputType.toString());
      partialType.type = params.outputType;
    }
  }
  if (params.inputType == poplar::QUARTER) {
    if (params.outputType != poplar::HALF) {
      throw poputil::poplibs_error("When the input type is quarter, the output"
                                   " type must be half, not " +
                                   params.outputType.toString());
    }
    for (const auto &partialType : partialTypes) {
      logging::poplin::warn(
          "Selecting partial type HALF which is the only valid option where"
          " the input is of type QUARTER");
      partialType.type = poplar::HALF;
    }
  }
}
