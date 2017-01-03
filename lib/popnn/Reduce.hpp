#ifndef __Reduce_hpp__
#define __Reduce_hpp__

#include "poplar/Graph.hpp"
#include <vector>

/// Perform a reduction over the first dimension of the partials tensor, writing
/// the result to the reduced tensor. The dimensions of the reduced tensor must
/// be the same as the dimensions of the partials tensor with the first
/// dimension removed.
void reduce(poplar::Graph &graph,
            poplar::Tensor partials,
            poplar::Tensor reduced,
            const std::vector<
              std::vector<std::pair<unsigned, unsigned>>
            > &reducedMapping,
            poplar::ComputeSet reduceCS);

#endif // __Reduce_hpp__
