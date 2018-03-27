// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_AllTrue_hpp
#define popops_AllTrue_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popops {

/**
 * Takes a boolean tensor, produces a logical AND of all elements
 * and returns the result through the exit status
 * \param graph         The poplar graph
 * \param A             The boolean tensor
 * \param prog          The program sequence to add this operation to
 * \param debugPrefix   A debug name for the operation
 */
void allTrue(poplar::Graph &graph,
             poplar::Tensor A,
             poplar::program::Sequence &prog,
             const std::string &debugPrefix = "");

}

#endif // popops_AllTrue_hpp
