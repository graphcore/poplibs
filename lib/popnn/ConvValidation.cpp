#include "ConvValidation.hpp"

#include <popnn/exceptions.hpp>

void conv::
validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                    unsigned kernelSize, unsigned stride, unsigned padding,
                    unsigned numChannels, const std::string &dType) {
  if (padding >= kernelSize) {
    throw net_creation_error(
      "Padding is greater than or equal to the kernel size"
    );
  }
  if (dType != "half" && dType != "float") {
    throw net_creation_error("Unknown element type");
  }
}
