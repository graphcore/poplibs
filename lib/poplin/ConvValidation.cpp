// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#include "ConvValidation.hpp"
#include "ConvOptions.hpp"
#include <poputil/exceptions.hpp>
#include <string>

void poplin::validateLayerParams(const ConvParams &params,
                                 const ConvOptions &options,
                                 const poplar::Target &target) {
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
    if (entry.type != poplar::HALF && entry.type != poplar::FLOAT) {
      throw poputil::poplibs_error(std::string("Unsupported ") + entry.name +
                                   " (must be float or half)");
    }
  }
}
