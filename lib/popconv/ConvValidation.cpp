#include "ConvValidation.hpp"
#include <poputil/exceptions.hpp>

void popconv::
validateLayerParams(const ConvParams &params) {
  if (params.dType != poplar::HALF && params.dType != poplar::FLOAT) {
    throw poputil::poplib_error("Unknown element type");
  }
}
