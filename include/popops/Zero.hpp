// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef popops_Zero_hpp
#define popops_Zero_hpp

#include "poplar/Graph.hpp"
#include "poplar/Program.hpp"
#include <vector>

namespace popops {

void zero(poplar::Graph &graph, const poplar::Tensor &t,
          poplar::program::Sequence &prog,
          const std::string &debugPrefix = "");

}

#endif // popops_Zero_hpp
