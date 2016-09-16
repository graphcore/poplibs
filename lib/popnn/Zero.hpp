#ifndef __Zero_hpp__
#define __Zero_hpp__
#include "poplar/Program.hpp"
#include <vector>

/// Construct a program to zero the specified tensor.
poplar::program::Program
zero(poplar::Graph &graph,
     poplar::Tensor t,
     const std::vector<unsigned> &tileMapping);

#endif // __Zero_hpp__
