#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <string>
#include <vector>
#include <poplin/Convolution.hpp>

namespace poplin {

struct ConvOptions;

void validateLayerParams(const ConvParams &params, const ConvOptions &options);

} // End namespace conv

#endif // _ConvValidation_hpp_
