// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplin_codelets_hpp
#define poplin_codelets_hpp
#include <poplar/Graph.hpp>

/// Linear algebra functions.
namespace poplin {
void addCodelets(poplar::Graph &graph);
}

#endif // poplin_codelets_hpp
