// Copyright (c) 2016, Graphcore Ltd, All rights reserved.

#ifndef popnn_ref_exceptions_hpp_
#define popnn_ref_exceptions_hpp_

#include <stdexcept>
#include <string>

namespace popnn_ref {

struct popnn_ref_error : std::logic_error {
  std::string type;
  explicit popnn_ref_error(const std::string &s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
  explicit popnn_ref_error(const char *s) : std::logic_error(s) {
    type = __FUNCTION__;
  }
};

} // End namespace popnn_ref.

#endif // popnn_ref_exceptions_hpp_
