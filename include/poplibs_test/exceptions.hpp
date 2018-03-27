// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_test_exceptions_hpp
#define poplibs_test_exceptions_hpp

#include <stdexcept>
#include <string>

namespace poplibs_test {

struct poplibs_test_error : std::logic_error {
  std::string type;
  explicit poplibs_test_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit poplibs_test_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace poplibs_test.

#endif // poplibs_test_exceptions_hpp
