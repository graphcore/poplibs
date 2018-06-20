#include <popnn/codelets.hpp>
#include "popnnCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>

namespace popnn {

void addCodelets(poplar::Graph &graph) {
  graph.addCodelets(poplibs::getCodeletsPath("popnn", "popnn.gp"));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace popnn
