#ifndef poplibs_support_codelets_hpp_
#define poplibs_support_codelets_hpp_
#include <string>

namespace poplibs {

struct CurrentLibLocator {
  void *dummy;
};

std::string getCodeletsPath(const std::string &libName,
                            const std::string &codeletsFile,
                            const CurrentLibLocator &locator);

} // end namespace poplibs

#endif // poplibs_support_codelets_hpp_
