#ifndef __Cast_hpp__
#define __Cast_hpp__

#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <vector>

/** Create a program to copy tensor casting between types (e.g. half->float).
 */
poplar::program::Program
cast(poplar::Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     poplar::Tensor src, poplar::Tensor dst,
     const std::string &debugPrefix = "");

void
cast(poplar::Graph &graph, const std::vector<unsigned> &dstActivationMapping,
     poplar::Tensor src, poplar::Tensor dst, poplar::ComputeSet cs);

/// Create vertices to copy element wise from the src tensor to the dst tensor
/// casting between types (e.g. half->float). The vertices are added to the
/// specified compute set.
void cast(poplar::Graph &graph,
          const std::vector<
            std::vector<poplar::Interval<std::size_t>>
          > &mapping,
          poplar::Tensor src, poplar::Tensor dst,
          poplar::ComputeSet cs);

#endif // __Cast_hpp__
