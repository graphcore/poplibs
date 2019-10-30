#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <poplin/Convolution.hpp>
#include <string>
#include <vector>

namespace poplin {

class ConvOptions;

void validateLayerParams(const ConvParams &params, const ConvOptions &options,
                         const poplar::Target &target);

} // namespace poplin

#endif // _ConvValidation_hpp_
