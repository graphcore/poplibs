#ifndef __alltrue_hpp__
#define __alltrue_hpp__

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>

namespace popstd {

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

} // namespace popstd

#endif // __operations_hpp__
