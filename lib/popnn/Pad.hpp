#ifndef __Pad_hpp__
#define __Pad_hpp__
#include "poplar/Program.hpp"
#include <vector>

/// Create a new tensor adding zero padding if necessary. The
/// \a beforePadding vector specifies, for each dimension, the amount of zero
/// padding to add at the start of that dimension. Additional zero padding
/// is added at the end of each dimension such that the size after padding
/// equals the size of the provided dimensions.
poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &beforePadding);

#endif // __Pad_hpp__
