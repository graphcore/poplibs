#include <poplibs_support/codelets.hpp>
#include <algorithm>
#include <locale>
#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include <fstream>

namespace poplibs {

std::string getCodeletsPath(const std::string &libName,
                            const std::string &codeletsFile) {
  auto upperName = libName;
  std::transform(upperName.begin(), upperName.end(), upperName.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  std::string envVar= "IPU_" + upperName + "_GP";

  const auto env = std::getenv(envVar.c_str());
  if (env && std::ifstream(env).good())
    return env;

#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of( '/' ) + 1);
    path = path + codeletsFile;
    return path;
  }
#endif

  std::string path = "lib/" + libName + "/" + codeletsFile;
  if (std::ifstream(path).good())
    return path;

  path = "../" + path;
  return path;
}

} // end namespace poplibs
