#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(const ConvParams &params) {

  if (params.paddingLower[0] >= params.kernelShape[0]) {
    throw popstd::poplib_error(
      "Lower edge height padding is greater than or equal to the kernel size"
    );
  }

  if (params.paddingUpper[0] >= params.kernelShape[0]) {
    throw popstd::poplib_error(
      "Upper edge height padding is greater than or equal to the kernel size"
    );
  }

  if (params.paddingLower[1] >= params.kernelShape[1]) {
    throw popstd::poplib_error(
      "Lower edge width padding of width is greater than or equal "
        "to the kernel size"
    );
  }

  if (params.paddingUpper[1] >= params.kernelShape[1]) {
    throw popstd::poplib_error(
      "Upper edge width padding of width is greater than or equal "
        "to the kernel size"
    );
  }

  if (params.dType != "half" && params.dType != "float") {
    throw popstd::poplib_error("Unknown element type");
  }
}
