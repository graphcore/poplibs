#ifndef __popstd_DynamicSlice_hpp__
#define __popstd_DynamicSlice_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace poplar {
class Tensor;
}

namespace popstd {


/** Slice a tensor based on offsets read from a tensor.
 *  \a dims gives the dimensions to slice, \a sizes defines the size of the
 *  slice in those dimensions and \a offset gives the base offsets on each
 *  execution.
 *  \a offset, \a dims and \a sizes must have the same rank/size
 *  \param graph       The poplar graph
 *  \param t           The source tensor
 *  \param offset      A tensor of offsets at which the output is extracted
 *  \param outShape    The shape of the output Tensor
 *  \param prog        The program to be extended
 *  \param debugPrefix The prefix prepended to debugging info
 *  \returns           The specified subtensor
 **/
poplar::Tensor dynamicSlice(poplar::Graph &graph,
                            const poplar::Tensor &t,
                            const poplar::Tensor &offset,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &sizes,
                            poplar::program::Sequence &prog,
                            const std::string &debugPrefix = "");
} // end namespace popstd

#endif //__popstd_DynamicSlice_hpp__
