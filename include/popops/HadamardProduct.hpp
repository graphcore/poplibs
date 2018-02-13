#ifndef __popstd_HadamardProduct_hpp__
#define __popstd_HadamardProduct_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/** Execute pointwise multiplication (hadamard product).
 *
 *
 *
 */
void hadamardProduct(poplar::Graph &graph,
                     poplar::Tensor A, poplar::Tensor B,
                     poplar::program::Sequence &prog,
                     const std::string &debugPrefix = "");

}

#endif // __popstd_HadamardProduct_hpp__
