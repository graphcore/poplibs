#include <poprand/codelets.hpp>
#include "poprandCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>

namespace poprand {

void addCodelets(poplar::Graph &graph) {
  graph.addCodelets(poplibs::getCodeletsPath("poprand", "poprand"));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace poprand
