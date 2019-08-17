// Copyright (c) 2019, Graphcore Ltd, All rights reserved.

#ifndef popfloat_codelets_hpp
#define popfloat_codelets_hpp
#include <poplar/Graph.hpp>

namespace experimental {
namespace popfloat {
void addCodelets(poplar::Graph &graph);
} // end namespace popfloat
} // end namespace experimental

#endif // popfloat_codelets_hpp
