#ifndef Reduction_hpp
#define Reduction_hpp

#include <cstddef>
#include <vector>

#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include "popops/Reduce.hpp"
#include "ReductionDebug.hpp"

// List of suboptimal things / potential optimisations:

// 1. Currently we nearly always do an on-tile reduction stage. In some cases
//    this may not be optimal (e.g. if there are 2 values on a tile that could
//    be reduced, there's no point adding a whole extra compute/exchange cycle
//    to save like, 2 cycles sending it.
//
// 2. When we are a doing a reduction after an exchange we should round up
//    the size of the output regions to a multiple of 4 or 8 (depending on
//    the operation and data type), and then paste each row in aligned.
//    The extra bits of data can be ignored. For example if we have 5
//    incoming partials of length 3, we should arrange them in a Vector like
//    this:
//
//    A1 A2 A3 # B1 B2 B3 # C1 C2 C3 # D1 D2 D3 # E1 E2 E3
//
//    That way the reduction can be done quickly and the extra data can be
//    ignored. I have no idea how to do that though.
//
// 3. IntermediatePartials stores all of the partials for one tensor in a single
//    1D tensor but it is probably better to use a different tensor for each
//    output. That would be simpler and allow filling memory gaps, and allow
//    arbitrary alignment. On the other hand a lot of the tensors would be
//    very small - often 1, 2 or 4 elements.
//
// 4. Deciding on types is kind of broken. You should be able to specify
//    the input type, the output type, and intermediate types depending on the
//    exchange level. This is similar to convolutions.
//
// Also search the code for "Optimisation", where I listed a few more things.

namespace popops {

/// A tensor that has been reshaped to 2D, and the shape that it should
/// be reshape()'d to to retrieve the original.
struct MangledTensor {
  poplar::Tensor tensor;
  std::vector<std::size_t> inflatedShape;
};

/// Flatten, dimroll and reshape A so that it is 2D and the dimensions given
/// by `dims` are in the first dimension, and all the others are in the second.
MangledTensor mangleTo2D(const poplar::Tensor &A,
                         std::set<unsigned> &reducedDims);

}

#endif // Reduction_hpp
