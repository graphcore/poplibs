#include <popfloat/codelets.hpp>
#include "popfloatCycleEstimators.hpp"
#include "poplibs_support/codelets.hpp"

namespace popfloat {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popfloat", "popfloat.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace popreduce
