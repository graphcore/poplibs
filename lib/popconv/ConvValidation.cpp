#include "ConvValidation.hpp"
#include <popstd/exceptions.hpp>

void popconv::
validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX,
                    unsigned numChannels, const std::string &dType) {

  if (paddingY >= kernelSizeY) {
    throw popstd::poplib_error(
      "Padding is greater than or equal to the kernel size"
    );
  }

  if (paddingX >= kernelSizeX) {
    throw popstd::poplib_error(
      "Padding of width is greater than or equal to the kernel size"
    );
  }

  if (dType != "half" && dType != "float") {
    throw popstd::poplib_error("Unknown element type");
  }
}
