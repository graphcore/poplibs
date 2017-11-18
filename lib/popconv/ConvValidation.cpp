#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(const ConvParams &params) {
  const auto numFieldDims = params.getNumFieldDims();
  for (unsigned dim = 0; dim != numFieldDims; ++dim) {
    if (params.getPaddedDilatedInputSize(dim) < 0) {
      throw popstd::poplib_error("Negative padding in dimension " +
                                 std::to_string(dim) +
                                 " truncates more than the input size");
    }
  }
  if (params.dType != poplar::HALF && params.dType != poplar::FLOAT) {
    throw popstd::poplib_error("Unknown element type");
  }
}
