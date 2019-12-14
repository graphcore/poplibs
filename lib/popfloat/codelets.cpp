// Copyright (c) Graphcore Ltd, All rights reserved.
#include "poplibs_support/codelets.hpp"
#include "popfloatCycleEstimators.hpp"
#include <experimental/popfloat/codelets.hpp>

namespace experimental {
namespace popfloat {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popfloat", "popfloat.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // end namespace popfloat
} // end namespace experimental
