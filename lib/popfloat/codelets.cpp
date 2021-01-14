// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplibs_support/codelets.hpp"
#include "popfloatCycleEstimators.hpp"
#include <popfloat/experimental/codelets.hpp>

namespace popfloat {
namespace experimental {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popfloat", "popfloat.gp", loc));
  poplibs::registerPerfFunctions(graph, makePerfFunctionTable());
}

} // end namespace experimental
} // end namespace popfloat
