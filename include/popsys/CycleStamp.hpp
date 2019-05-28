// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popsys_CycleStamp_hpp
#define popsys_CycleStamp_hpp

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popsys {

/**
 * Add a sequence program to record an absolute Hw cycle stamp on a given tile.
 * The stamp is a snapshot of a continuously running h/w counter on a tile and
 * to have consistent results, measurements must be done on the same tile.
 *
 * The result is a tensor containing two 32-bit elements os a 64-bit snapshot
 * of the h/w counter. The first element of the tensor is the lower 32-bits and
 * the second the upper 32-bits.
 *
 * The timestamp is added after an internal sync is executed.
 *
 * \param graph         The poplar graph
 * \param prog          The program sequence to which the time stamp is added
 * \param tile          The tile on which the time stamp is added
 * \returns             A unsigned integer tensor of length 2
 */
poplar::Tensor cycleStamp(poplar::Graph &graph, poplar::program::Sequence &prog,
                          unsigned tile, const std::string &debugPrefix = "");

/**
 * Add a compute set to record an absolute Hw cycle stamp on the specified
 * tiles.
 * \param graph         The poplar graph
 * \param prog          The program sequence to which the time stamp is added
 * \param tiles         The tiles on which the time stamp is added
 * \returns             A vector of tensors of 2 integers
 */
std::vector<poplar::Tensor> cycleStamp(poplar::Graph &graph,
                                       poplar::program::Sequence &prog,
                                       const std::vector<unsigned> &tiles,
                                       const std::string &debugPrefix = "");

}

#endif // popsys_CycleStamp_hpp
