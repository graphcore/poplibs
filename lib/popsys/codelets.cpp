#include <popsys/codelets.hpp>
#include <poplibs_support/codelets.hpp>
#include "popsysCycleEstimators.hpp"
#include <poputil/exceptions.hpp>


namespace popsys {

void addCodelets(poplar::Graph &graph) {
  if (graph.getTarget().getTargetType() != poplar::TargetType::IPU) {
    throw poputil::poplib_error("popsys is only valid for ipu targets");
  }
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popsys", "popsys.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
}

} // namespace popsys
