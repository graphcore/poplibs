#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <string>

namespace conv {

void validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                         unsigned kernelSizeY, unsigned kernelSizeX,
                         unsigned strideY, unsigned strideX,
                         unsigned paddingY, unsigned paddingX,
                         unsigned numChannels, const std::string &dType);

} // End namespace conv

#endif // _ConvValidation_hpp_
