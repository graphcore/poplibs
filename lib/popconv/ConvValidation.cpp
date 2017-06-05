#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(const ConvParams &params) {
  std::vector<std::string> dimName = {"height", "width"};
  for (const auto &dim : {0, 1}) {
    if (params.getPaddedDilatedInputSize(dim) < 0) {
      throw popstd::poplib_error("Negative " + dimName[dim] + " padding " +
                                 "truncates more than the input size");
    }
  }
  if (params.dType != "half" && params.dType != "float") {
    throw popstd::poplib_error("Unknown element type");
  }
}
