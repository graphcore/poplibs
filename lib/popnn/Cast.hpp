#ifndef __Cast_hpp__
#define __Cast_hpp__
#include "poplar/Program.hpp"
#include <vector>

/** Create a program to copy tensor casting between types (e.g. half->float).
 */
poplar::program::Program
cast(poplar::Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     poplar::Tensor src, poplar::Tensor dst);

#endif // __Cast_hpp__
