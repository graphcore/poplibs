// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparseCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>
#include <popsparse/codelets.hpp>

using namespace poplar;

namespace popsparse {

void addCodelets(Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popsparse", "popsparse.gp", loc));
  poputil::internal::registerPerfFunctions(graph, makePerfFunctionTable());
}

} // end namespace popsparse
