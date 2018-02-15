#ifndef popops_Zero_hpp
#define popops_Zero_hpp

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popops {

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

}

#endif // popops_Zero_hpp
