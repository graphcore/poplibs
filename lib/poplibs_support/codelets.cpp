// Copyright (c) Graphcore Ltd, All rights reserved.
#include <algorithm>
#include <locale>
#include <poplibs_support/codelets.hpp>
#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif
#include <fstream>

namespace poplibs {

std::string getCodeletsPath(const std::string &libName,
                            const std::string &codeletsFile_,
                            const CurrentLibLocator &locator) {
  auto codeletsFile = codeletsFile_;
  auto upperName = libName;
  std::transform(upperName.begin(), upperName.end(), upperName.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  std::string envVar = "IPU_" + upperName + "_GP";

  const auto env = std::getenv(envVar.c_str());
  if (env && std::ifstream(env).good())
    return env;

  const auto suffixVar = "POPLIBS_CODELET_SUFFIX";
  const auto suffixEnv = std::getenv(suffixVar);
  if (suffixEnv) {
    auto pos = codeletsFile.find(".gp");
    if (pos == std::string::npos) {
      codeletsFile += suffixEnv;
    } else {
      codeletsFile.insert(pos, suffixEnv);
    }
  }

#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  if (dladdr(&locator.dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of('/') + 1);
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
