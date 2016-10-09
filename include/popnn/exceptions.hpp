// Copyright (c) 2016, Graphcore Ltd, All rights reserved.

#ifndef popnn_exceptions_hpp_
#define popnn_exceptions_hpp_

#include <stdexcept>
#include <string>

namespace popnn {

struct popnn_error : std::logic_error {
  std::string type;
  explicit popnn_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit popnn_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace popnn.

#endif // popnn_exceptions_hpp_
