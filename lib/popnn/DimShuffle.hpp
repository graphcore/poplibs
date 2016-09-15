#ifndef __DimShuffle_hpp__
#define __DimShuffle_hpp__
#include "poplar/Program.hpp"
#include <vector>

/// Copy data from one tensor to another, permuting the order of dimensions.
/// The permutation vector specifies a mapping from the output dimension to the
/// input dimension. For example the permutation of {2, 0, 1} specifies that
/// element in[a][b][c] is copied to out[c][a][b].
void
dimShuffle(poplar::Graph &graph, const poplar::ComputeSet &cs,
           poplar::Tensor in, poplar::Tensor out,
           const std::vector<unsigned> &permutation,
           const std::vector<unsigned> &outTileMapping);

#endif // __DimShuffle_hpp__
