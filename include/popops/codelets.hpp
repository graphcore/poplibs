// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popops_codelets_hpp
#define popops_codelets_hpp
#include <poplar/Graph.hpp>

/// Common functions, such as elementwise and reductions.
namespace popops {
void addCodelets(poplar::Graph &graph);
}

#endif // popops_codelets_hpp
