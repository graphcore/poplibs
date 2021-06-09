// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poprandCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>
#include <poprand/codelets.hpp>

namespace poprand {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("poprand", "poprand.gp", loc));
  poputil::registerPerfFunctions(graph, makePerfFunctionTable());
}

} // namespace poprand
