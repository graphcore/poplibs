#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <string>
#include <vector>
#include <popconv/Convolution.hpp>

namespace popconv {

void validateLayerParams(const ConvParams &params, const ConvOptions &options);

} // End namespace conv

#endif // _ConvValidation_hpp_
