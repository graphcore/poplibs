// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef popnn_codelets_hpp
#define popnn_codelets_hpp
#include <poplar/Graph.hpp>

/// Functions used in neural networks.
namespace popnn {
void addCodelets(poplar::Graph &graph);
}

#endif // popnn_codelets_hpp
