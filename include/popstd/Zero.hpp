#ifndef __popstd_Zero_hpp__
#define __popstd_Zero_hpp__

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popstd {

void
zero(poplar::Graph &graph,
     poplar::Tensor t,
     const std::vector<poplar::Interval> &tileRegions,
     unsigned tile,
     poplar::ComputeSet zeroCS);

void
zero(poplar::Graph &graph,
     const poplar::Tensor &t,
     unsigned tile,
     poplar::ComputeSet zeroCS);

void
zero(poplar::Graph &graph,
     const poplar::Tensor &t,
     const std::vector<
       std::vector<poplar::Interval>
     > &mapping,
     poplar::ComputeSet zeroCS);

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "");

} // end namespace popstd

#endif // __popstd_Zero_hpp__
