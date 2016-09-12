#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <string>

namespace conv {

void validateLayerParams(unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                         unsigned kernelSize, unsigned stride, unsigned padding,
                         unsigned numChannels, const std::string &dType);

} // End namespace conv

#endif // _ConvValidation_hpp_
