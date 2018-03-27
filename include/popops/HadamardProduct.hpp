// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_HadamardProduct_hpp
#define popops_HadamardProduct_hpp
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

#endif // popops_HadamardProduct_hpp
