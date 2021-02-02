// Copyright (c) 2016 Graphcore Ltd. All rights reserved.
#ifndef _ConvValidation_hpp_
#define _ConvValidation_hpp_

#include <poplin/Convolution.hpp>
#include <string>
#include <vector>

namespace poplin {

class ConvOptions;

// Check the parameters and the options for a layer. The options may be
// updated as a side effect.
void validateLayerParams(const ConvParams &params, const poplar::Target &target,
                         ConvOptions &options);

} // namespace poplin

#endif // _ConvValidation_hpp_
