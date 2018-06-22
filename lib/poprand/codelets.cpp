#include <poprand/codelets.hpp>
#include "poprandCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>

namespace poprand {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("poprand", "poprand.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace poprand
