#ifndef __popstd_Pad_hpp__
#define __popstd_Pad_hpp__
#include "poplar/Program.hpp"
#include <vector>

namespace popstd {

/// Return a tensor with zero padding added.
/// \param graph        The graph containing the tensor.
/// \param t            The tensor to pad.
/// \param paddingLower A vector specifying the amount of padding to add at the
///                     start of each dimension.
/// \param paddingUpper A vector specifying the amount of padding to add at the
///                     end of each dimension.
/// \return The tensor with zero padding added.
poplar::Tensor
pad(poplar::Graph &graph, poplar::Tensor t,
    const std::vector<std::size_t> &paddingLower,
    const std::vector<std::size_t> &paddingUpper);

/// Return a tensor with zero padding added to one dimension.
/// \param t            The tensor to pad.
/// \param paddingLower The amount of padding to add at the start of the
///                     dimension.
/// \param paddingUpper The amount of padding to add at the end of the
///                     dimension.
/// \param dim          The dimension to pad.
/// \return The tensor with zero padding added.
poplar::Tensor
pad(poplar::Graph &graph, const poplar::Tensor &t, std::size_t paddingLower,
    std::size_t paddingUpper, unsigned dim);

} // end namespace popstd

#endif // __popstd_Pad_hpp__
