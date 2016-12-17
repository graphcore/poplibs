#ifndef __Zero_hpp__
#define __Zero_hpp__

#include "poplar/Graph.hpp"
#include <vector>

void
zero(poplar::Graph &graph,
     poplar::Tensor t,
     const std::vector<std::pair<unsigned, unsigned>> &tileRegions,
     unsigned tile,
     poplar::ComputeSet zeroCS);

void
zero(poplar::Graph &graph,
     const poplar::Tensor &t,
     const std::vector<
       std::vector<std::pair<unsigned, unsigned>>
     > &mapping,
     poplar::ComputeSet zeroCS);

#endif // __Zero_hpp__
