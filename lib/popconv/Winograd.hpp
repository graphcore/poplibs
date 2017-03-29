#ifndef __Winograd_hpp__
#define __Winograd_hpp__

#include <poplar/Program.hpp>

namespace popconv {

poplar::program::Program winogradConvolution(poplar::Graph &graph,
            unsigned strideY, unsigned strideX,
            unsigned paddingY, unsigned paddingX,
            poplar::Tensor in, poplar::Tensor weights, poplar::Tensor biases,
            poplar::Tensor out,
            const std::string &partialsType,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string &debugPrefix = "");

}

#endif //__Winograd_hpp__
