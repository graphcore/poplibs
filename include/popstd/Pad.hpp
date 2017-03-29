#ifndef __popstd_Pad_hpp__
#define __popstd_Pad_hpp__
#include "poplar/Program.hpp"
#include <vector>

namespace popstd {

/// Create a new tensor adding zero padding if necessary. The
/// \a beforePadding vector specifies, for each dimension, the amount of zero
/// padding to add at the start of that dimension. Additional zero padding
/// is added at the end of each dimension such that the size after padding
/// equals the size of the provided dimensions.
poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &beforePadding);

/// Zero pad the specified dimension. \a newSize specifies the size along
/// dimension after padding is applied and \a beforePadding specifies the
/// amount of zero padding to add at the start of the dimension.
poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::size_t newSize,
    std::size_t beforePadding, unsigned dim);

} // end namespace popstd

#endif // __popstd_Pad_hpp__
