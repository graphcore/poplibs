#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(const ConvParams &params) {
  if (params.dType != poplar::HALF && params.dType != poplar::FLOAT) {
    throw popstd::poplib_error("Unknown element type");
  }
}
