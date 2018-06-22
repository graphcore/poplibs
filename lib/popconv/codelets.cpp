#include <popconv/codelets.hpp>
#include "popconvCycleEstimators.hpp"
#include <poplibs_support/codelets.hpp>

namespace popconv {

const std::vector<std::string> winogradCodelets =
{{ "popconv::WgdKernelTransform<float,4,4,3,3>",
   "popconv::WgdKernelTransform<half,4,4,3,3>",
   "popconv::WgdPartials<float>",
   "popconv::WgdPartials<half>",
   "popconv::WgdReduce<float,4,4>",
   "popconv::WgdReduce<half,4,4>",
   "popconv::WgdInverseTransform<float,4,4,3,3>",
   "popconv::WgdInverseTransform<half,4,4,3,3>",
   "popconv::WgdConvComplete<float>",
   "popconv::WgdConvComplete<half>"
 }};


void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popconv", "popconv.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());

  // The winograd codelets are not currently supported and do not have correct
  // cycle estimators.
  auto zeroEstimator = [](const poplar::VertexIntrospector &v,
                          const poplar::Target &device) {
                              return std::uint64_t(0);
                           };
  for (const auto &codelet : winogradCodelets) {
    graph.registerCycleEstimator(codelet, zeroEstimator);
  }
}

} // namespace popconv
