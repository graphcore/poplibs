#include "ConvValidation.hpp"
#include <popnn/exceptions.hpp>

void conv::
validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSizeY, unsigned kernelSizeX,
                    unsigned strideY, unsigned strideX,
                    unsigned paddingY, unsigned paddingX,
                    unsigned numChannels, const std::string &dType) {

  if (paddingY >= kernelSizeY) {
    throw popnn::popnn_error(
      "Padding is greater than or equal to the kernel size"
    );
  }

  if (paddingX >= kernelSizeX) {
    throw popnn::popnn_error(
      "Padding of width is greater than or equal to the kernel size"
    );
  }

  if (dType != "half" && dType != "float") {
    throw popnn::popnn_error("Unknown element type");
  }
}
