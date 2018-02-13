#ifndef __popstd_DynamicSlice_hpp__
#define __popstd_DynamicSlice_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

namespace poplar {
class Tensor;
}

namespace popops {


/** Slice a tensor based on offsets specified by a tensor.
 *  \a dims gives the dimensions to slice, \a sizes defines the size of the
 *  slice in those dimensions and \a offset gives the base offsets on each
 *  execution.
 *  \a offset[0], \a dims and \a sizes must have the same size. \a offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The poplar graph
 *  \param t           The source tensor
 *  \param offset      A tensor of offsets at which the output is extracted
 *  \param dims        The dimensions of \a t to slice
 *  \param sizes       The size of the slice in each of \a dims
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


/** Update a subtensor at offsets read from a tensor
 *  \a dims gives the dimensions that are partialy updated, by \a sizes elements
 *  at offsets \a offset. Unspecified dimensions are copied in full with zero
 *  offset.
 *  \a offset[0], \a dims and \a sizes must have the same size. \a offset may
 *  have a second dimension with an element per tile, which can eliminate
 *  exchange.
 *  \param graph       The poplar graph
 *  \param t           The tensor to update
 *  \param s           The updates
 *  \param offset      The offset within \a t to be updated
 *  \param dims        The dimensions to be dynamically updated
 *  \param sizes       The size of the update in each of \a dims
 *  \param prog        The program to be extended
 *  \param debugPrefix The prefix prepended to debugging info
 **/
void dynamicUpdate(poplar::Graph &graph,
                   const poplar::Tensor &t,
                   const poplar::Tensor &s,
                   const poplar::Tensor &offset,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "");

}

#endif //__popstd_DynamicSlice_hpp__
