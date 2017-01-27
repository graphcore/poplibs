#ifndef __Reduce_hpp__
#define __Reduce_hpp__

#include "poplar/Graph.hpp"
#include <vector>

/// Perform a reduction over the first dimension of the partials tensor, writing
/// the result to the reduced tensor. The dimensions of the reduced tensor must
/// be the same as the dimensions of the partials tensor with the first
/// dimension removed. reduceMapping specifies a mapping from tiles to regions
/// of the reduced tensor that should be calculated on the tile.
void reduce(poplar::Graph &graph,
            poplar::Tensor partials,
            poplar::Tensor reduced,
            const std::vector<
              std::vector<poplar::Interval<std::size_t>>
            > &reduceMapping,
            poplar::ComputeSet reduceCS);

/// Perform a reduction over the first dimension of the partials tensor, writing
/// the result to the reduced tensor. The dimensions of the reduced tensor must
/// be the same as the dimensions of the partials tensor with the first
/// dimension removed. reducedMapping specifies the tile mapping of the reduced
/// tensor. The mapping for the reduced tensor influences the mapping fo the
/// computation but it does not dictate it - elements may be reduced on
/// a different tile if that improves balance.
void reduceByDstMapping(poplar::Graph &graph,
                        poplar::Tensor partials,
                        poplar::Tensor reduced,
                        const std::vector<
                          std::vector<poplar::Interval<std::size_t>>
                        > &reducedMapping,
                        poplar::ComputeSet reduceCS);

#endif // __Reduce_hpp__
