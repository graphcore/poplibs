// Copyright (c) 2016, Graphcore Ltd, All rights reserved.

#ifndef poplib_test_exceptions_hpp_
#define poplib_test_exceptions_hpp_

#include <stdexcept>
#include <string>

namespace poplib_test {

struct poplib_test_error : std::logic_error {
  std::string type;
  explicit poplib_test_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit poplib_test_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace poplib_test.

#endif // poplib_test_exceptions_hpp_
