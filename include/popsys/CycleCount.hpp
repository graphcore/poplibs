// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popsys_CycleCount_hpp
#define popsys_CycleCount_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popsys {

/**
 * Given a sequence program type, times the program and returns the 64 bit
 * value in a tensor of 2 unsigned integers. Sequence is timed by adding
 * a timing program to the original sequence. Must also specify the tile
 * on which the program is timed.
 * \param graph         The poplar graph
 * \param prog          The program sequence to time
 * \param tile          The tile on which the program is timed
 * \returns             A unsigned integer tensor of length 2
 */
poplar::Tensor cycleCount(poplar::Graph &graph,
                          poplar::program::Sequence &prog,
                          unsigned tile,
                          const std::string &debugPrefix = "");

}

#endif // popsys_CycleCount_hpp
