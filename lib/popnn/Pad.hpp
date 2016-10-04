#ifndef __Pad_hpp__
#define __Pad_hpp__
#include "poplar/Program.hpp"
#include <vector>

/// Copy one tensor to another adding zero padding if necessary. The
/// \a beforePadding vector specifies, for each dimension, the amount of zero
/// padding to add at the start of that dimension. Additional zero padding
/// is added at the end of each dimension such that the size after padding
/// equals the size of the corresponding dimension in the \a out tensor.
poplar::program::Program
pad(poplar::Graph &graph,
    poplar::Tensor in, poplar::Tensor out,
    const std::vector<std::size_t> &beforePadding,
    const std::vector<unsigned> &outTileMapping);

#endif // __Pad_hpp__
