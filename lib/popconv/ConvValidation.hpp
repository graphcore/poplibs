#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <string>
#include <vector>

namespace popconv {

void validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                         unsigned kernelSizeY, unsigned kernelSizeX,
                         const std::vector<unsigned> &stride,
                         const std::vector<unsigned> &paddingLower,
                         const std::vector<unsigned> &paddingUpper,
                         unsigned numChannels, const std::string &dType);

} // End namespace conv

#endif // _ConvValidation_hpp_
