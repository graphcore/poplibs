// Copyright (c) 2018 Graphcore Ltd, All rights reserved.
#include "poputil/exceptions.hpp"

namespace poputil {

poplibs_error::poplibs_error(const std::string &s) : std::runtime_error(s) {
  type = __FUNCTION__;
}
poplibs_error::poplibs_error(const char *s) : std::runtime_error(s) {
  type = __FUNCTION__;
}

} // namespace poputil
