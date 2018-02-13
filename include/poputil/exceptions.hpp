// Copyright (c) 2016, Graphcore Ltd, All rights reserved.

#ifndef popstd_exceptions_hpp_
#define popstd_exceptions_hpp_

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

#endif // popstd_exceptions_hpp_
