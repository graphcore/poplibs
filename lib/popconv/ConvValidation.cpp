#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    const std::vector<unsigned> &stride,
                    const std::vector<unsigned> &paddingLower,
                    const std::vector<unsigned> &paddingUpper,
                    unsigned numChannels, const std::string &dType) {

  if (paddingLower[0] >= kernelSizeY) {
    throw popstd::poplib_error(
      "Lower edge height padding is greater than or equal to the kernel size"
    );
  }

  if (paddingUpper[0] >= kernelSizeY) {
    throw popstd::poplib_error(
      "Upper edge height padding is greater than or equal to the kernel size"
    );
  }

  if (paddingLower[1] >= kernelSizeX) {
    throw popstd::poplib_error(
      "Lower edge width padding of width is greater than or equal "
        "to the kernel size"
    );
  }

  if (paddingUpper[1] >= kernelSizeX) {
    throw popstd::poplib_error(
      "Upper edge width padding of width is greater than or equal "
        "to the kernel size"
    );
  }

  if (dType != "half" && dType != "float") {
    throw popstd::poplib_error("Unknown element type");
  }
}
