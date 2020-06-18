// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poprand_codelets_hpp
#define poprand_codelets_hpp
#include <poplar/Graph.hpp>

/// Pseudo-random number generator (PRNG) functions.
namespace poprand {
void addCodelets(poplar::Graph &graph);
}

#endif // poprand_codelets_hpp
