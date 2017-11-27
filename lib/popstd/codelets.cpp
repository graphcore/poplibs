#include <popstd/codelets.hpp>
#include "popstdCycleEstimators.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include <fstream>

namespace popstd {

static std::string findGraphProg() {
  // TODO: This needs to be replaced with a proper object search mechanism
  // in poplar.
  const auto env = std::getenv("IPU_POPSTD_GP");
  if (env && std::ifstream(env).good())
    return env;

#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + "popstd.gp";
    return path;
  }
#endif

  std::string path = "lib/popstd/popstd.gp";
  if (std::ifstream(path).good())
    return path;

  path = "../" + path;
  return path;
}

void addCodelets(poplar::Graph &graph) {
  graph.addCodelets(findGraphProg());
  poplibs::registerCyclesFunctions(graph, popstd::cyclesFunctionTable);
}

} // namespace popstd
