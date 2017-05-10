#ifndef __Winograd_hpp__
#define __Winograd_hpp__

#include <poplar/Program.hpp>

namespace popconv {

poplar::program::Program winogradConvolution(poplar::Graph &graph,
            const std::vector<unsigned> &stride,
            const std::vector<unsigned> &paddingLower,
            const std::vector<unsigned> &paddingUpper,
            poplar::Tensor in, poplar::Tensor weights,
            poplar::Tensor out,
            const std::string &partialsType,
            unsigned patchSizeX, unsigned patchSizeY,
            const std::string &debugPrefix = "");

}

#endif //__Winograd_hpp__
