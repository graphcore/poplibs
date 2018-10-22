#include <popsys/codelets.hpp>
#include <poplibs_support/codelets.hpp>
#include "popsysCycleEstimators.hpp"
#include <poputil/exceptions.hpp>


namespace popsys {

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popsys", "popsys.gp", loc));
  const auto &target = graph.getTarget();
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable(target));
}

} // namespace popsys
