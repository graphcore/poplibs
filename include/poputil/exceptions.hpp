// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poputil_exceptions_hpp
#define poputil_exceptions_hpp

#include <stdexcept>
#include <string>

namespace poputil {

struct poplib_error : std::logic_error {
  std::string type;
  explicit poplib_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit poplib_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace popstd.

#endif // poputil_exceptions_hpp
