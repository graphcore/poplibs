#ifndef __Regroup_hpp__
#define __Regroup_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <string>
#include <vector>

void
regroup(poplar::Graph &graph, const poplar::ComputeSet &cs,
        const std::string &inType, const std::string &outType,
        const std::vector<unsigned> &outTileMapping,
        poplar::Tensor in, poplar::Tensor out);

#endif //__Regroup_hpp__
