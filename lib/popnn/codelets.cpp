#include <popnn/codelets.hpp>
#include "popnnCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>

namespace popnn {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popnn", "popnn.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace popnn
