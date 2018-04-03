#include "ConvValidation.hpp"
#include "popconv/internal/ConvOptions.hpp"
#include <poputil/exceptions.hpp>
#include <string>

void popconv::
validateLayerParams(const ConvParams &params, const ConvOptions &options) {
  const struct {
    poplar::Type type;
    const char *name;
  } typesToCheck[] = {
    { params.dType, "element type" },
    { options.partialsType, "partial type" },
    { options.interTilePartialsType, "inter-tile partial type" },
    { options.interIpuPartialsType, "inter-ipu partial type" },
  };
  for (const auto &entry : typesToCheck) {
    if (entry.type != poplar::HALF && entry.type != poplar::FLOAT) {
      throw poputil::poplib_error(std::string("Unsupported ") + entry.name +
                                  " (must be float or half)");
    }
  }
}
