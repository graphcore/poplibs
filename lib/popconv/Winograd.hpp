#ifndef __Winograd_hpp__
#define __Winograd_hpp__

#include <poplar/Program.hpp>
#include <popconv/Convolution.hpp>
#include "popconv/internal/ConvOptions.hpp"

namespace popconv {

poplar::program::Program winogradConvolution(poplar::Graph &graph,
            const ConvParams &params,
            const poplar::Tensor &in, const poplar::Tensor &weights,
            const poplar::Tensor &out,
            unsigned patchSizeX, unsigned patchSizeY,
            const poplar::Type &partialsType,
            const std::string &debugPrefix = "",
            const ConvOptions &options = ConvOptions());

}

#endif //__Winograd_hpp__
