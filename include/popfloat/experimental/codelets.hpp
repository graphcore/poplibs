// Copyright (c) 2019 Graphcore Ltd, All rights reserved.

#ifndef popfloat_codelets_hpp
#define popfloat_codelets_hpp
#include <poplar/Graph.hpp>

namespace popfloat {
namespace experimental {
void addCodelets(poplar::Graph &graph);
} // end namespace experimental
} // end namespace popfloat

#endif // popfloat_codelets_hpp
