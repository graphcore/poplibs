#ifndef __popstd_Cast_hpp__
#define __popstd_Cast_hpp__

#include <poplar/Interval.hpp>
#include <poplar/Program.hpp>
#include <vector>

namespace popstd {

/// Create a program to copy tensor casting between types (e.g. half->float).
poplar::program::Program
cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
     const std::string &debugPrefix = "");

/// Create vertices to copy element wise from the src tensor to the dst tensor
/// casting between types (e.g. half->float). The vertices are added to the
/// specified compute set.
void
cast(poplar::Graph &graph, poplar::Tensor src, poplar::Tensor dst,
     poplar::ComputeSet cs);

} // end namespace popstd

#endif // __posptd_Cast_hpp__
