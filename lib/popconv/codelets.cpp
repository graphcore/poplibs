#include <popconv/codelets.hpp>
#include "popconvCycleEstimators.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include <fstream>

namespace popconv {

static std::string findGraphProg() {
  // TODO: This needs to be replaced with a proper object search mechanism
  // in poplar.
  const auto env = std::getenv("IPU_POPCONV_GP");
  if (env && std::ifstream(env).good())
    return env;

#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + "popconv.gp";
    return path;
  }
#endif

  std::string path = "lib/popconv/popconv.gp";
  if (std::ifstream(path).good())
    return path;

  path = "../" + path;
  return path;
}

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
  graph.addCodelets(findGraphProg());
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
